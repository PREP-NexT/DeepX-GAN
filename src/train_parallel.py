import os
import json
import argparse
import time
from datetime import datetime

import cv2
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import time     # for optimization
from spatial_utils import *
from data_utils import *
from gan_utils import * 
from models_parallel import *


def train(args):
    """
        Main training function for the DeepX-GAN model.

    :param args: command line arguments
    """
    # test = args.test
    dname = args.dname
    time_steps = args.time_steps
    batch_size = args.batch_size
    batch_size_half = args.batch_size//2
    seed = args.seed  # Random seed for reproducibility
    save_freq = args.save_freq

    u = args.u
    pre_trained_path = args.pre_trained_path
    iter_final = args.iter_final
    x_size = args.x_size
    sample_method = args.sample_method
    cuda_num = args.cuda_num
    season = args.season
    scratch_dir = args.path   
    parallel_ids = args.parallel_ids
    theta1 = args.theta1
    theta2 = args.theta2

    if dname == 'lgcp':
        dataset, x_height, x_width = fetch_lgcp(time_steps=time_steps, x_size=x_size, method=sample_method)
    elif (dname in ['air', 'tmax']):
        dataset, x_height, x_width = fetch_climate(dname, time_steps=time_steps, season=season)

    # Calculate spatio-temporal embedding
    embedding_op = args.embedding_op
    stx_method = args.stx_method
    b = args.dec_weight
    #b = torch.exp(-torch.arange(1, time_steps).flip(0).float() / b).view(1, 1, -1)
    b = temporal_weights(time_steps,b)
    if stx_method=="kw":
      b = 1 / -torch.log(b[0,0,0]) * time_steps-1 # Get temporal weight b from computed weight tensor of length n
      b = torch.exp(-torch.stack([torch.abs(torch.arange(0, time_steps) - t) for t in range(0,time_steps)]) / b)

    w_sparse = make_sparse_weight_matrix(x_height, x_width)
    if embedding_op == "moran":
        data_emb = make_mis(dataset.data, w_sparse)
    elif embedding_op == "spate":
        data_emb = make_spates(dataset.data, w_sparse, b, stx_method, u, theta1=theta1, theta2=theta2)   # (x, w_sparse, b, method="skw", u=None)
    else:
        data_emb = dataset.data
    # Concatenate data
    data = torch.cat((dataset.data, data_emb), dim=2)    ## concatenate to the nchannel dim
    dataset_full = MyDataset(data)

    scale = args.scale
    para_lam = args.lam

    # filter size for (de)convolutional layers
    g_state_size = args.g_state_size
    d_state_size = args.d_state_size
    g_filter_size = args.g_filter_size
    d_filter_size = args.d_filter_size
    reg_penalty = args.reg_penalty
    nlstm = args.n_lstm
    channels = args.n_channels
    bn = args.batch_norm
    # Number of RNN layers stacked together
    gen_lr = args.lr
    disc_lr = args.lr
    np.random.seed(seed)

    it_counts = 0
    sinkhorn_eps = args.sinkhorn_eps
    sinkhorn_l = args.sinkhorn_l
    # scaling_coef = 1.0
    # loader = DataLoader(dataset_full, batch_size=batch_size * 2, drop_last=True, shuffle=True)
    loader = DataLoader(dataset_full, batch_size=batch_size, drop_last=True, shuffle=True)
    # #, num_workers=2, add shuffle=True on 2023/8/16, shuffle is on the batch size but does not influence the nchannel dim

    # Decide which device we want to run on
    device = torch.device(f"cuda:{cuda_num}" if (torch.cuda.is_available()) else "cpu")

    # Create instances of generator, discriminator_h and
    # discriminator_m CONV VERSION
    z_width = args.z_dims_t
    z_height = args.z_dims_t
    y_dim = args.y_dims
    j_dims = 16

#### Consume pre-trained models if available
    if pre_trained_path is not None:
        if os.listdir(pre_trained_path):
            for dirpath, dirnames, filenames in os.walk(pre_trained_path):
                for filename in sorted([f for f in filenames if f.endswith("generator{}.pt".format(iter_final))]):
                    file_path = os.path.join(dirpath, filename)
                    generator = torch.load(file_path, map_location=device)
                    print("Load pretrained generator from: ", file_path)
                for filename in sorted([f for f in filenames if f.endswith("discriminatorH{}.pt".format(iter_final))]):
                    file_path = os.path.join(dirpath, filename)
                    discriminator_h = torch.load(file_path, map_location=device)
                    print("Load pretrained discriminator_h from: ", file_path)
                for filename in sorted([f for f in filenames if f.endswith("discriminatorM{}.pt".format(iter_final))]):
                    file_path = os.path.join(dirpath, filename)
                    discriminator_m = torch.load(file_path, map_location=device)
                    print("Load pretrained discriminator_m from: ", file_path)
        generator = handle_tuple_err(generator)
        discriminator_h = handle_tuple_err(discriminator_h)
        discriminator_m = handle_tuple_err(discriminator_m)

#### Create new instances of generator, discriminator_h and discriminator_m
    else:
        generator = VideoDCG(time_steps, x_h=x_height, x_w=x_width, filter_size=g_filter_size,
                             state_size=g_state_size, bn=bn, output_act='sigmoid', nchannel=channels)
        if torch.cuda.device_count() > 1:
            print("You have", torch.cuda.device_count(), "GPUs, and will use", parallel_ids, "for parallel training.")
            generator = nn.DataParallel(generator, device_ids=parallel_ids)
        generator.to(device)

        if embedding_op == "none":
            discriminator_h = VideoDCD(x_h=x_height, x_w=x_width, filter_size=d_filter_size, j=j_dims,
                                       nchannel=channels, bn=bn).to(device)
            discriminator_m = VideoDCD(x_h=x_height, x_w=x_width, filter_size=d_filter_size, j=j_dims,
                                       nchannel=channels, bn=bn).to(device)
        else:
            discriminator_h = VideoDCD(x_h=x_height, x_w=x_width, filter_size=d_filter_size, j=j_dims,
                                       nchannel=channels*2, bn=bn)
            if torch.cuda.device_count() > 1:
                discriminator_h = nn.DataParallel(discriminator_h, device_ids=parallel_ids)
            discriminator_h.to(device)

            discriminator_m = VideoDCD(x_h=x_height, x_w=x_width, filter_size=d_filter_size, j=j_dims,
                                       nchannel=channels*2, bn=bn)
            if torch.cuda.device_count() > 1:
                discriminator_m = nn.DataParallel(discriminator_m, device_ids=parallel_ids)
            discriminator_m.to(device)

    test_ = dname + "-" + args.loss_func + '-' + embedding_op
    
    if embedding_op=="spate":   # changed from "bea" to "spate"
      test_ = test_ + '-' + stx_method

    saved_file = "{}_{}{}-{}.{}".format(test_,
                                              datetime.now().strftime("%h"),
                                              datetime.now().strftime("%d"),
                                              datetime.now().strftime("%H"),
                                              datetime.now().strftime("%M"))

    log_dir = "{}/trained/{}/log".format(scratch_dir, saved_file)

    # Create directories for storing images later.
    if not os.path.exists("{}/trained/{}/data".format(scratch_dir, saved_file)):
        os.makedirs("{}/trained/{}/data".format(scratch_dir, saved_file))
    if not os.path.exists("{}/trained/{}/images".format(scratch_dir, saved_file)):
        os.makedirs("{}/trained/{}/images".format(scratch_dir, saved_file))

    # GAN train notes
    with open("{}/trained/{}/train_notes.txt".format(scratch_dir, saved_file), 'w') as f:
        # Include any experiment notes here:
        f.write("Experiment notes: .... \n\n")
        f.write("MODEL_DATA: {}\nSEQ_LEN: {}\n".format(
            test_,
            time_steps))
        f.write("STATE_SIZE: {}\nLAMBDA: {}\n".format(
            g_state_size,
            reg_penalty))
        f.write("BATCH_SIZE: {}\nCRITIC_ITERS: {}\nGenerator LR: {}\n".format(
            batch_size,
            gen_lr,
            disc_lr))
        f.write("SINKHORN EPS: {}\nSINKHORN L: {}\n\n".format(
            sinkhorn_eps,
            sinkhorn_l))
        f.write("Image size: {}*{}\nSample method: {}\n\n".format(
            x_size, x_size,
            sample_method)),
        f.write("All parameters: {}\n".format(args))


    writer = SummaryWriter(log_dir)

    beta1 = 0.5
    beta2 = 0.9
    optimizerG = optim.Adam(generator.parameters(), lr=gen_lr, betas=(beta1, beta2))
    optimizerDH = optim.Adam(discriminator_h.parameters(), lr=disc_lr, betas=(beta1, beta2))
    optimizerDM = optim.Adam(discriminator_m.parameters(), lr=disc_lr, betas=(beta1, beta2))

    epochs = args.n_epochs

    w_sparse = w_sparse.to(device)
    b = b.to(device)
    print('For reference, b.is_cuda: ', b.is_cuda)

    for e in range(epochs):

        for x in loader:
            it_counts += 1
            # Train D
            x1 = x[:, :, :(channels//2)+1, :, :].reshape(batch_size, time_steps, channels, x_height, x_width).to(device)
            if (stx_method == "skw") | (stx_method == "tdc") | (stx_method == "tdc_masked"):
                x2 = x[:, 1:, (channels//2)+1:, :, :].reshape(batch_size, time_steps - 1, channels, x_height, x_width).to(device)
            else:
                x2 = x[:, :, (channels//2)+1:, :, :].reshape(batch_size, time_steps, channels, x_height, x_width).to(device)
            z = torch.randn(batch_size_half, time_steps, z_height * z_width).to(device)
            y = torch.randn(batch_size_half, y_dim).to(device)
            z_p = torch.randn(batch_size_half, time_steps, z_height * z_width).to(device)
            y_p = torch.randn(batch_size_half, y_dim).to(device)
            real_data = x1[:batch_size_half, ...]
            real_data_p = x1[batch_size_half:, ...]
            real_data_emb = x2[:batch_size_half, ...]
            real_data_p_emb = x2[batch_size_half:, ...]

            fake_data = generator(z, y).reshape(batch_size_half, time_steps, channels, x_height, x_width)
            fake_data_p = generator(z_p, y_p).reshape(batch_size_half, time_steps, channels, x_height, x_width)

            if embedding_op == "moran":
                fake_data_emb = make_mis(fake_data, w_sparse)#[:, 1:, :, :, :]
                fake_data_p_emb = make_mis(fake_data_p, w_sparse)#[:, 1:, :, :, :]
            elif embedding_op == "spate":
                if (stx_method == "skw") | (stx_method == "tdc") | (stx_method == "tdc_masked"):
                    fake_data_emb = make_spates(fake_data, w_sparse, b, stx_method, u, theta1=theta1, theta2=theta2)[:, 1:, :, :, :]
                    fake_data_p_emb = make_spates(fake_data_p, w_sparse, b, stx_method, u, theta1=theta1, theta2=theta2)[:, 1:, :, :, :]
                else:
                    fake_data_emb = make_spates(fake_data, w_sparse, b, stx_method, theta1=theta1, theta2=theta2)#[:, 1:, :, :, :]
                    fake_data_p_emb = make_spates(fake_data_p, w_sparse, b, stx_method, theta1=theta1, theta2=theta2)#[:, 1:, :, :, :]
            else:
                fake_data_emb = None
                fake_data_p_emb = None

            if fake_data_emb is not None:
                if (stx_method == "skw") | (stx_method == "tdc") | (stx_method == "tdc_masked"):
                  real_emb = torch.cat((torch.unsqueeze(real_data[:, 0, :, :, :], 1), real_data_emb), 1)
                  fake_emb = torch.cat((torch.unsqueeze(fake_data[:, 0, :, :, :], 1), fake_data_emb), 1)
                else:
                  real_emb = real_data_emb
                  fake_emb = fake_data_emb
                concat_real = torch.cat((real_data, real_emb), dim=2)
                concat_fake = torch.cat((fake_data, fake_emb), dim=2)

                if args.loss_func == "sinkhorngan":
                    concat_real = concat_real.reshape(batch_size_half, time_steps, -1)
                    concat_fake = concat_fake.reshape(batch_size_half, time_steps, -1)
                    loss_d = original_sinkhorn_loss(concat_real, concat_fake, sinkhorn_eps, sinkhorn_l, scale=scale)
                    disc_loss = -loss_d
                else:
                    if (stx_method == "skw") | (stx_method == "tdc") | (stx_method == "tdc_masked"):
                        real_emb_p = torch.cat((torch.unsqueeze(real_data_p[:, 0, :, :, :], 1), real_data_p_emb), 1)
                        fake_emb_p = torch.cat((torch.unsqueeze(fake_data_p[:, 0, :, :, :], 1), fake_data_p_emb), 1)
                    else:
                        real_emb_p = real_data_p_emb
                        fake_emb_p = fake_data_p_emb
                    #return real_data_p, real_emb_p
                    concat_real_p = torch.cat((real_data_p, real_emb_p), dim=2)
                    concat_fake_p = torch.cat((fake_data_p, fake_emb_p), dim=2)

                    # second returned output isn't used.
                    h_fake, h_fake_emb = discriminator_h(concat_fake, concat_fake_p)
                    m_real, m_real_emb = discriminator_m(concat_real, concat_real_p)
                    m_fake, m_fake_emb = discriminator_m(concat_fake, concat_fake_p)
                    h_real_p, h_real_p_emb = discriminator_h(concat_real_p, concat_real)
                    h_fake_p, h_fake_p_emb = discriminator_h(concat_fake_p, concat_fake)
                    m_real_p, m_real_p_emb = discriminator_m(concat_real_p, concat_real)

                    loss_d = compute_mixed_sinkhorn_loss(concat_real, concat_fake, m_real, m_fake, h_fake,
                                                         sinkhorn_eps, sinkhorn_l, concat_real_p, concat_fake_p,
                                                         m_real_p, h_real_p, h_fake_p, scale=scale)
                    pm1 = scale_invariante_martingale_regularization(m_real, reg_penalty, scale=scale)
                    pm2 = scale_invariante_martingale_regularization(m_real_emb, reg_penalty, scale=scale)
                    disc_loss = -loss_d + pm1 + pm2
            else:
                if args.loss_func == "sinkhorngan":
                    real_data = real_data.reshape(batch_size_half, time_steps, -1)
                    fake_data = fake_data.reshape(batch_size_half, time_steps, -1)

                    loss_d = original_sinkhorn_loss(real_data, fake_data, sinkhorn_eps, sinkhorn_l, scale=scale)
                    disc_loss = -loss_d
                else:
                    h_fake = discriminator_h(fake_data)

                    m_real = discriminator_m(real_data)
                    m_fake = discriminator_m(fake_data)

                    h_real_p = discriminator_h(real_data_p)
                    h_fake_p = discriminator_h(fake_data_p)

                    m_real_p = discriminator_m(real_data_p)

                    real_data = real_data.reshape(batch_size_half, time_steps, -1)
                    fake_data = fake_data.reshape(batch_size_half, time_steps, -1)
                    real_data_p = real_data_p.reshape(batch_size_half, time_steps, -1)
                    fake_data_p = fake_data_p.reshape(batch_size_half, time_steps, -1)

                    loss_d = compute_mixed_sinkhorn_loss(real_data, fake_data, m_real, m_fake, h_fake,
                                                         sinkhorn_eps, sinkhorn_l, real_data_p, fake_data_p,
                                                         m_real_p, h_real_p, h_fake_p, scale=scale)

                    pm1 = scale_invariante_martingale_regularization(m_real, reg_penalty, scale=scale)
                    disc_loss = -loss_d + pm1

            # torch.autograd.set_detect_anomaly(True)

            # updating Discriminators
            discriminator_h.zero_grad()
            discriminator_m.zero_grad()
            disc_loss.backward()
            optimizerDH.step()
            optimizerDM.step()

            # Train G
            z = torch.randn(batch_size_half, time_steps, z_height * z_width).to(device)
            y = torch.randn(batch_size_half, y_dim).to(device)
            z_p = torch.randn(batch_size_half, time_steps, z_height * z_width).to(device)
            y_p = torch.randn(batch_size_half, y_dim).to(device)

            fake_data = generator(z, y).reshape(batch_size_half, time_steps, channels, x_height, x_width)
            fake_data_p = generator(z_p, y_p).reshape(batch_size_half, time_steps, channels, x_height, x_width)

            if embedding_op == "moran":
                fake_data_emb = make_mis(fake_data, w_sparse)#[:, 1:, :, :, :]
                fake_data_p_emb = make_mis(fake_data_p, w_sparse)#[:, 1:, :, :, :]
            elif embedding_op == "spate":
                if (stx_method == "skw") | (stx_method == "tdc") | (stx_method == "tdc_masked"):   # merge with 'tdc'
                    fake_data_emb = make_spates(fake_data, w_sparse, b, stx_method, u, theta1=theta1, theta2=theta2)[:, 1:, :, :, :]
                    fake_data_p_emb = make_spates(fake_data_p, w_sparse, b, stx_method, u, theta1=theta1, theta2=theta2)[:, 1:, :, :, :]
                else:
                    fake_data_emb = make_spates(fake_data, w_sparse, b, stx_method, theta1=theta1, theta2=theta2)#[:, 1:, :, :, :]
                    fake_data_p_emb = make_spates(fake_data_p, w_sparse, b, stx_method, theta1=theta1, theta2=theta2)#[:, 1:, :, :, :]
            else:
                fake_data_emb = None
                fake_data_p_emb = None

            if fake_data_emb is not None:
                if (stx_method == "skw") | (stx_method == "tdc") | (stx_method == "tdc_masked"):
                    real_emb = torch.cat((torch.unsqueeze(real_data[:, 0, :, :, :], 1), real_data_emb), 1)
                    fake_emb = torch.cat((torch.unsqueeze(fake_data[:, 0, :, :, :], 1), fake_data_emb), 1)
                else:
                    real_emb = real_data_emb
                    fake_emb = fake_data_emb
                concat_real = torch.cat((real_data, real_emb), dim=2)
                concat_fake = torch.cat((fake_data, fake_emb), dim=2)

                if args.loss_func == "sinkhorngan":
                    concat_real = concat_real.reshape(batch_size_half, time_steps, -1)
                    concat_fake = concat_fake.reshape(batch_size_half, time_steps, -1)
                    loss_g = original_sinkhorn_loss(concat_real, concat_fake, sinkhorn_eps, sinkhorn_l, scale=scale)
                else:
                    if (stx_method == "skw") | (stx_method == "tdc") | (stx_method == "tdc_masked"):
                        real_emb_p = torch.cat((torch.unsqueeze(real_data_p[:, 0, :, :, :], 1), real_data_p_emb), 1)
                        fake_emb_p = torch.cat((torch.unsqueeze(fake_data_p[:, 0, :, :, :], 1), fake_data_p_emb), 1)
                    else:
                        real_emb_p = real_data_p_emb
                        fake_emb_p = fake_data_p_emb
                    concat_real_p = torch.cat((real_data_p, real_emb_p), dim=2)
                    concat_fake_p = torch.cat((fake_data_p, fake_emb_p), dim=2)

                    # second returned output isn't used.
                    h_fake, h_fake_emb = discriminator_h(concat_fake, concat_fake_p)
                    m_real, m_real_emb = discriminator_m(concat_real, concat_real_p)
                    m_fake, m_fake_emb = discriminator_m(concat_fake, concat_fake_p)
                    h_real_p, h_real_p_emb = discriminator_h(concat_real_p, concat_real)
                    h_fake_p, h_fake_p_emb = discriminator_h(concat_fake_p, concat_fake)
                    m_real_p, m_real_p_emb = discriminator_m(concat_real_p, concat_real)

                    loss_g = compute_mixed_sinkhorn_loss(concat_real, concat_fake, m_real, m_fake, h_fake,
                                                         sinkhorn_eps, sinkhorn_l, concat_real_p, concat_fake_p,
                                                         m_real_p, h_real_p, h_fake_p, scale=scale)
            else:
                if args.loss_func == "sinkhorngan":
                    real_data = real_data.reshape(batch_size_half, time_steps, -1)
                    fake_data = fake_data.reshape(batch_size_half, time_steps, -1)
                    loss_g = original_sinkhorn_loss(real_data, fake_data, sinkhorn_eps, sinkhorn_l, scale=scale)
                else:
                    h_fake = discriminator_h(fake_data)

                    m_real = discriminator_m(real_data)
                    m_fake = discriminator_m(fake_data)

                    h_real_p = discriminator_h(real_data_p)
                    h_fake_p = discriminator_h(fake_data_p)

                    m_real_p = discriminator_m(real_data_p)

                    real_data = real_data.reshape(batch_size_half, time_steps, -1)
                    fake_data = fake_data.reshape(batch_size_half, time_steps, -1)
                    real_data_p = real_data_p.reshape(batch_size_half, time_steps, -1)
                    fake_data_p = fake_data_p.reshape(batch_size_half, time_steps, -1)

                    loss_g = compute_mixed_sinkhorn_loss(real_data, fake_data, m_real, m_fake, h_fake,
                                                         sinkhorn_eps, sinkhorn_l, real_data_p, fake_data_p,
                                                         m_real_p, h_real_p, h_fake_p, scale=scale)
            gen_loss = loss_g

            # updating Generator
            generator.zero_grad()
            gen_loss.backward()
            optimizerG.step()
            # it.set_postfix(loss=float(gen_loss))
            # it.update(1)

            # ...log the running loss
            writer.add_scalar('Sinkhorn training loss', gen_loss, it_counts)
            if args.loss_func == "cotgan":
                writer.add_scalar('pM for real', pm1, it_counts)
                if not embedding_op == "none":
                    writer.add_scalar('pM for embedding', pm2, it_counts)
                writer.flush()

            if torch.isinf(gen_loss):
                print('%s Loss exploded!' % test_)
                # Open the existing file with mode a - append
                with open("./trained/{}/train_notes.txt".format(saved_file), 'a') as f:
                    # Include any experiment notes here:
                    f.write("\n Training failed! ")
                break
            else:
                if it_counts % save_freq == 0 or it_counts == 1:
                    print("\nEpoch [%d/%d] - Iter [%d/%d] - Generator Loss: %f - Discriminator Loss: %f" % (
                    e, epochs, it_counts%(dataset.data.size(0)//batch_size), dataset.data.size(0)//batch_size, gen_loss.item(), disc_loss.item()))
                    z = torch.randn(batch_size_half, time_steps, z_height * z_width).to(device)
                    y = torch.randn(batch_size_half, y_dim).to(device)
                    samples = generator(z, y)

                    n_show = min(batch_size_half, 5)
                    samples = samples[:n_show, :, 0, :, :].permute(0, 2, 1, 3)
                    img = samples.reshape(1, n_show * x_height, time_steps * x_width)
                    writer.add_image('Generated images', img, global_step=it_counts)
                    #  save model to file
                    save_path = "{}/trained/{}/ckpts".format(scratch_dir, saved_file)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    # torch.save(generator, save_path + '/' + 'generator.pt') #Only save final generator

                    torch.save(generator, save_path + '/' + 'generator{}.pt'.format(it_counts)) #Save steps
                    torch.save(discriminator_h, save_path + '/' + 'discriminatorH{}.pt'.format(it_counts))
                    torch.save(discriminator_m, save_path + '/' + 'discriminatorM{}.pt'.format(it_counts))
                    print("Saved all models to {}".format(save_path))
            continue

    writer.close()



if __name__ == '__main__':
          parser = argparse.ArgumentParser(description='cot')
          parser.add_argument('-d', '--dname', type=str, default="lgcp",
                              choices=['lgcp', 'extreme_weather', 'turbulent_flows'])
          parser.add_argument('-lf', '--loss_func', type=str, default="cotgan", choices=["sinkhorngan", "cotgan"])
          parser.add_argument('-eo', '--embedding_op', type=str, default="spate", choices=["moran", "spate", "none"])
          parser.add_argument('-stx', '--stx_method', type=str, default="skw", choices=["skw", "k", "kw"])
          parser.add_argument('-t', '--test', type=str, default='cot', choices=['cot'])
          parser.add_argument('-s', '--seed', type=int, default=1)
          parser.add_argument('-b', '--dec_weight', type=int, default=20)
          parser.add_argument('-gss', '--g_state_size', type=int, default=32)
          parser.add_argument('-gfs', '--g_filter_size', type=int, default=32)
          parser.add_argument('-dss', '--d_state_size', type=int, default=32)
          parser.add_argument('-dfs', '--d_filter_size', type=int, default=32)
          parser.add_argument('-ts', '--time_steps', type=int, default=10)
          parser.add_argument('-sinke', '--sinkhorn_eps', type=float, default=0.8)
          parser.add_argument('-reg_p', '--reg_penalty', type=float, default=1.5)
          parser.add_argument('-sinkl', '--sinkhorn_l', type=int, default=100)
          parser.add_argument('-Dx', '--Dx', type=int, default=1)
          parser.add_argument('-Dz', '--z_dims_t', type=int, default=5)
          parser.add_argument('-Dy', '--y_dims', type=int, default=20)
          parser.add_argument('-g', '--gen', type=str, default="fc", choices=["lstm", "fc"])
          parser.add_argument('-bs', '--batch_size', type=int, default=32)
          parser.add_argument('-p', '--path', type=str, default='./')
          parser.add_argument('-save', '--save_freq', type=int, default=5)
          parser.add_argument('-ne', '--n_epochs', type=int, default=30)
          parser.add_argument('-lr', '--lr', type=float, default=1e-4)
          parser.add_argument('-bn', '--batch_norm', type=bool, default=True)
          parser.add_argument('-sl', '--scale', type=bool, default=True)
          parser.add_argument('-nlstm', '--n_lstm', type=int, default=1)
          parser.add_argument('-lam', '--lam', type=float, default=1.0)
      
          parser.add_argument('-nch', '--n_channels', type=int, default=1)
          parser.add_argument('-rt', '--read_tfrecord', type=bool, default=True)
          parser.add_argument('-f')  # Dummy to get parser to run in Colab
      
          args = parser.parse_args()
          print("TRAINING - Dataset: " + args.dname + " Emb: " + args.embedding_op + " Loss: " + args.loss_func)
          train(args)