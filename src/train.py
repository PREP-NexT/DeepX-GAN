"""
train.py — Training loop for DeepX-GAN
========================================
This module implements the end-to-end training procedure for DeepX-GAN.

Training follows the standard GAN alternating optimisation:

1. **Train Discriminators (D_H and D_M)**: Freeze the generator; update the
   two discriminators using the COT-GAN loss (maximise the Sinkhorn distance
   minus the martingale penalty).

2. **Train Generator (G)**: Freeze the discriminators; update the generator to
   minimise the same Sinkhorn distance.

Every ``save_freq`` iterations, model checkpoints and sample images are logged
to TensorBoard for visual inspection.

Checkpoints can be resumed: pass ``--pre_trained_path`` and ``--iter_final``
to restart training from a saved checkpoint.

Reference
---------
[Paper citation here]
"""

import os
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_utils import MyDataset, fetch_climate_data, DATASET_REGISTRY
from spatial_utils import make_sparse_weight_matrix, temporal_weights, make_spates
from gan_utils import (
    compute_mixed_sinkhorn_loss,
    scale_invariante_martingale_regularization,
)
from models import VideoDCG, VideoDCD


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    """
    Run DeepX-GAN training with the given configuration.

    All hyper-parameters are passed via the ``args`` namespace (from
    ``argparse``).  See ``main.py`` for a full description of each argument.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """

    # ------------------------------------------------------------------
    # 1. Unpack hyper-parameters
    # ------------------------------------------------------------------
    time_steps    = args.time_steps
    batch_size    = args.batch_size
    save_freq     = args.save_freq
    scratch_dir   = args.output_dir
    season        = args.season
    debug_run     = args.debug_run
    pre_trained_path = args.pre_trained_path
    iter_final    = args.iter_final
    cuda_num      = args.cuda_num

    # DeepX embedding parameters
    u       = args.u          # tail-dependence quantile threshold
    theta1  = args.theta1     # weight for classic SPATE component
    theta2  = args.theta2     # weight for masked DeepX component

    # ------------------------------------------------------------------
    # 2. Load training dataset
    # ------------------------------------------------------------------
    print("\n[1/4] Loading training data...")
    dataset, x_height, x_width = fetch_climate_data(
        dname=args.dname,
        data_dir=args.data_dir,
        time_steps=time_steps,
        season=season,
        debug_run=debug_run,
    )
    print(f"  Training samples: {len(dataset)},  Spatial grid: {x_height} × {x_width}")

    # ------------------------------------------------------------------
    # 3. Compute DeepX embedding for the training data
    #
    # The DeepX embedding is a spatiotemporal autocorrelation statistic
    # (a modified local Moran's I) that captures extreme co-occurrence
    # patterns.  It is pre-computed on the real data and concatenated with
    # the raw climate field as a second channel fed to the discriminator.
    # ------------------------------------------------------------------
    print("\n[2/4] Computing DeepX embedding for training data...")
    stx_method = args.stx_method   # 'tdc_masked' is the default DeepX method

    # Build exponential temporal decay weights
    b = temporal_weights(time_steps, args.dec_weight)

    # Build the queen-contiguity spatial weight matrix (H*W × H*W sparse)
    w_sparse = make_sparse_weight_matrix(x_height, x_width)

    # Compute DeepX embedding: output shape same as dataset.data
    data_emb = make_spates(
        dataset.data, w_sparse, b, method=stx_method, u=u, theta1=theta1, theta2=theta2
    )

    # Concatenate raw climate field and embedding along the channel dimension
    # Result: (N, T, 2, H, W) — channel 0 = temperature, channel 1 = DeepX
    data_combined = torch.cat((dataset.data, data_emb), dim=2)
    dataset_full = MyDataset(data_combined)

    # ------------------------------------------------------------------
    # 4. Initialise models and optimisers
    # ------------------------------------------------------------------
    print("\n[3/4] Initialising models...")
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    # Noise dimensions
    z_dim   = args.z_dims_t ** 2   # temporal latent noise (z_dims_t × z_dims_t grid)
    z_width = args.z_dims_t
    z_height = args.z_dims_t
    y_dim   = args.y_dims           # static latent noise
    j_dims  = 16                    # discriminator output feature dimension J

    channels = args.n_channels   # = 1 (single climate variable)

    # Generator: maps Gaussian noise → synthetic climate sequences
    generator = VideoDCG(
        batch_size=batch_size,
        time_steps=time_steps,
        x_h=x_height,
        x_w=x_width,
        filter_size=args.g_filter_size,
        state_size=args.g_state_size,
        bn=args.batch_norm,
        output_act='sigmoid',
        nchannel=channels,
        z_dim=z_dim,
        y_dim=y_dim,
    ).to(device)

    # Discriminators D_H and D_M: both use 2*channels input because the
    # DeepX embedding is concatenated as an extra channel
    discriminator_h = VideoDCD(
        batch_size=batch_size,
        x_h=x_height,
        x_w=x_width,
        filter_size=args.d_filter_size,
        j=j_dims,
        nchannel=channels * 2,   # raw field + DeepX embedding
        bn=args.batch_norm,
    ).to(device)

    discriminator_m = VideoDCD(
        batch_size=batch_size,
        x_h=x_height,
        x_w=x_width,
        filter_size=args.d_filter_size,
        j=j_dims,
        nchannel=channels * 2,
        bn=args.batch_norm,
    ).to(device)

    # Adam optimisers with β₁ = 0.5, β₂ = 0.9 (standard GAN setting)
    beta1, beta2 = 0.5, 0.9
    optimizerG  = optim.Adam(generator.parameters(),      lr=args.lr, betas=(beta1, beta2))
    optimizerDH = optim.Adam(discriminator_h.parameters(), lr=args.lr, betas=(beta1, beta2))
    optimizerDM = optim.Adam(discriminator_m.parameters(), lr=args.lr, betas=(beta1, beta2))

    # ------------------------------------------------------------------
    # 6. Optionally resume from a saved checkpoint
    # ------------------------------------------------------------------
    start_epoch = 0
    it_counts   = -1

    if pre_trained_path is not None:
        ckpt_path = os.path.join(pre_trained_path, f'checkpoint{iter_final}.pt')
        print(f"\n  Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        # Handle checkpoints saved with DataParallel (remove 'module.' prefix)
        def strip_module(state_dict):
            if any(k.startswith('module.') for k in state_dict):
                return {k.replace('module.', ''): v for k, v in state_dict.items()}
            return state_dict

        generator.load_state_dict(strip_module(ckpt['generator_state_dict']))
        discriminator_h.load_state_dict(strip_module(ckpt['discriminator_h_state_dict']))
        discriminator_m.load_state_dict(strip_module(ckpt['discriminator_m_state_dict']))
        optimizerG.load_state_dict(ckpt['optimizerG_state_dict'])
        optimizerDH.load_state_dict(ckpt['optimizerDH_state_dict'])
        optimizerDM.load_state_dict(ckpt['optimizerDM_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        it_counts   = ckpt['global_iteration']
        print(f"  Resumed from epoch {start_epoch}, iteration {it_counts + 1}")

    # ------------------------------------------------------------------
    # 7. Prepare output directories and TensorBoard logger
    # ------------------------------------------------------------------
    # Build a timestamp-based run identifier using the dataset name
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"{ts}_{args.dname}-{stx_method}"

    save_root = os.path.join(scratch_dir, 'trained', run_name)
    ckpt_dir  = os.path.join(save_root, 'ckpts')
    log_dir   = os.path.join(save_root, 'log')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Save experiment configuration as a structured JSON file for reproducibility
    train_notes = {
        "run_name":  run_name,
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "dataset": {
            "dname":          args.dname,
            "data_dir":       args.data_dir,
            "description":    DATASET_REGISTRY[args.dname]['description'],
            "season":         season,
            "time_steps":     time_steps,
        },
        "embedding": {
            "method":         stx_method,
            "u":              u,
            "dec_weight":     args.dec_weight,
            "theta1":         theta1,
            "theta2":         theta2,
        },
        "model": {
            "g_state_size":   args.g_state_size,
            "g_filter_size":  args.g_filter_size,
            "d_filter_size":  args.d_filter_size,
            "z_dims_t":       args.z_dims_t,
            "y_dims":         args.y_dims,
            "n_channels":     args.n_channels,
            "batch_norm":     args.batch_norm,
        },
        "training": {
            "n_epochs":       args.n_epochs,
            "batch_size":     batch_size,
            "lr":             args.lr,
            "sinkhorn_eps":   args.sinkhorn_eps,
            "sinkhorn_l":     args.sinkhorn_l,
            "reg_penalty":    args.reg_penalty,
            "scale":          args.scale,
            "save_freq":      save_freq,
        },
        "hardware": {
            "device":         str(device),
        },
    }
    with open(os.path.join(save_root, 'train_notes.json'), 'w') as f:
        json.dump(train_notes, f, indent=4)
    print(f"  Config saved → {os.path.join(save_root, 'train_notes.json')}")

    writer = SummaryWriter(log_dir)
    print(f"\n[4/4] Starting training.  Outputs → {save_root}")

    # ------------------------------------------------------------------
    # 8. Data loader
    # ------------------------------------------------------------------
    loader = DataLoader(
        dataset_full,
        batch_size=batch_size * 2,   # load 2× for the mixed Sinkhorn loss (x and x')
        drop_last=True,
        shuffle=True,
    )

    # Move spatial utilities to the training device
    w_sparse = w_sparse.to(device)
    b        = b.to(device)

    sinkhorn_eps = args.sinkhorn_eps
    sinkhorn_l   = args.sinkhorn_l
    reg_penalty  = args.reg_penalty
    scale        = args.scale
    epochs       = args.n_epochs

    # ===================================================================
    # Training loop
    # ===================================================================
    for epoch in range(start_epoch, epochs):
        for x in loader:
            it_counts += 1

            # ---- Split batch into two halves (x and x') for mixed Sinkhorn ----
            # x has shape (2*batch_size, T, 2*C, H, W)
            # Channel 0   = raw temperature field
            # Channel 1   = DeepX embedding
            real_all = x.to(device)

            # Raw climate field (channel 0)
            x1 = real_all[:, :, :channels, :, :]    # (2N, T, C, H, W)

            # DeepX embedding (channels 1 onwards), shifted by 1 time step
            # because the embedding at t uses observations up to t-1 (causal)
            x2 = real_all[:, 1:, channels:, :, :]   # (2N, T-1, C, H, W)

            # ---- Sample generator inputs -----------------------------------
            z   = torch.randn(batch_size, time_steps, z_height * z_width).to(device)
            y   = torch.randn(batch_size, y_dim).to(device)
            z_p = torch.randn(batch_size, time_steps, z_height * z_width).to(device)
            y_p = torch.randn(batch_size, y_dim).to(device)

            # Split real data into x and x' (two independent real batches)
            real_data     = x1[:batch_size, ...]          # (N, T, C, H, W)
            real_data_p   = x1[batch_size:, ...]          # (N, T, C, H, W)
            real_data_emb   = x2[:batch_size, ...]        # (N, T-1, C, H, W)
            real_data_p_emb = x2[batch_size:, ...]        # (N, T-1, C, H, W)

            # ---- Generate fake data ----------------------------------------
            fake_data   = generator(z,   y  ).reshape(batch_size, time_steps, channels, x_height, x_width)
            fake_data_p = generator(z_p, y_p).reshape(batch_size, time_steps, channels, x_height, x_width)

            # ---- Compute DeepX embedding for fake data -------------------
            # The embedding uses only past values (causal) so we take [1:] steps
            fake_data_emb   = make_spates(fake_data,   w_sparse, b, stx_method, u, theta1, theta2)[:, 1:, :, :, :]
            fake_data_p_emb = make_spates(fake_data_p, w_sparse, b, stx_method, u, theta1, theta2)[:, 1:, :, :, :]

            # ---- Build concatenated (field + embedding) tensors -----------
            # For the tdc/tdc_masked method the embedding starts at t=1, so we
            # prepend the t=0 frame of the raw field to make lengths match.
            real_emb   = torch.cat((real_data[:, :1, :, :, :],   real_data_emb),   dim=1)
            fake_emb   = torch.cat((fake_data[:, :1, :, :, :],   fake_data_emb),   dim=1)
            real_emb_p = torch.cat((real_data_p[:, :1, :, :, :], real_data_p_emb), dim=1)
            fake_emb_p = torch.cat((fake_data_p[:, :1, :, :, :], fake_data_p_emb), dim=1)

            concat_real   = torch.cat((real_data,   real_emb),   dim=2)   # (N, T, 2C, H, W)
            concat_fake   = torch.cat((fake_data,   fake_emb),   dim=2)
            concat_real_p = torch.cat((real_data_p, real_emb_p), dim=2)
            concat_fake_p = torch.cat((fake_data_p, fake_emb_p), dim=2)

            # ==============================================================
            # Train Discriminators (D_H and D_M)
            # ==============================================================
            for p in generator.parameters():
                p.requires_grad = False
            for p in list(discriminator_h.parameters()) + list(discriminator_m.parameters()):
                p.requires_grad = True

            # Each discriminator forward pass with two inputs returns (output1, output2),
            # where output1 uses LSTM head 1 on inputs1, and output2 uses LSTM head 2 on inputs2.
            h_fake,   h_fake_emb   = discriminator_h(concat_fake,   concat_fake_p)
            m_real,   m_real_emb   = discriminator_m(concat_real,   concat_real_p)
            m_fake,   m_fake_emb   = discriminator_m(concat_fake,   concat_fake_p)
            h_real_p, h_real_p_emb = discriminator_h(concat_real_p, concat_real)
            h_fake_p, h_fake_p_emb = discriminator_h(concat_fake_p, concat_fake)
            m_real_p, m_real_p_emb = discriminator_m(concat_real_p, concat_real)

            loss_d = compute_mixed_sinkhorn_loss(
                concat_real, concat_fake, m_real, m_fake, h_fake,
                sinkhorn_eps, sinkhorn_l,
                concat_real_p, concat_fake_p,
                m_real_p, h_real_p, h_fake_p, scale=scale,
            )
            # Martingale penalty applied to both the raw-field and embedding outputs of D_M.
            # pm1 uses m_real (discriminator response to the raw temperature field).
            # pm2 uses m_real_emb (discriminator response to the DeepX embedding channel),
            # which is already available from the same forward pass above.
            pm1 = scale_invariante_martingale_regularization(m_real,     reg_penalty, scale=scale)
            pm2 = scale_invariante_martingale_regularization(m_real_emb, reg_penalty, scale=scale)
            disc_loss = -loss_d + pm1 + pm2

            discriminator_h.zero_grad()
            discriminator_m.zero_grad()
            disc_loss.backward()
            optimizerDH.step()
            optimizerDM.step()

            # ==============================================================
            # Train Generator (G)
            # ==============================================================
            for p in generator.parameters():
                p.requires_grad = True
            for p in list(discriminator_h.parameters()) + list(discriminator_m.parameters()):
                p.requires_grad = False

            # Fresh generator samples for the generator update step
            z   = torch.randn(batch_size, time_steps, z_height * z_width).to(device)
            y   = torch.randn(batch_size, y_dim).to(device)
            z_p = torch.randn(batch_size, time_steps, z_height * z_width).to(device)
            y_p = torch.randn(batch_size, y_dim).to(device)

            fake_data   = generator(z,   y  ).reshape(batch_size, time_steps, channels, x_height, x_width)
            fake_data_p = generator(z_p, y_p).reshape(batch_size, time_steps, channels, x_height, x_width)

            fake_data_emb   = make_spates(fake_data,   w_sparse, b, stx_method, u, theta1, theta2)[:, 1:, :, :, :]
            fake_data_p_emb = make_spates(fake_data_p, w_sparse, b, stx_method, u, theta1, theta2)[:, 1:, :, :, :]

            fake_emb   = torch.cat((fake_data[:, :1, :, :, :],   fake_data_emb),   dim=1)
            fake_emb_p = torch.cat((fake_data_p[:, :1, :, :, :], fake_data_p_emb), dim=1)
            concat_fake   = torch.cat((fake_data,   fake_emb),   dim=2)
            concat_fake_p = torch.cat((fake_data_p, fake_emb_p), dim=2)

            h_fake,   _ = discriminator_h(concat_fake,   concat_fake_p)
            m_real,   _ = discriminator_m(concat_real,   concat_real_p)
            m_fake,   _ = discriminator_m(concat_fake,   concat_fake_p)
            h_real_p, _ = discriminator_h(concat_real_p, concat_real)
            h_fake_p, _ = discriminator_h(concat_fake_p, concat_fake)
            m_real_p, _ = discriminator_m(concat_real_p, concat_real)

            # Generator loss: maximise the Sinkhorn distance between real and fake.
            # (Discriminators are frozen here, so no penalty terms are needed.)
            gen_loss = compute_mixed_sinkhorn_loss(
                concat_real, concat_fake, m_real, m_fake, h_fake,
                sinkhorn_eps, sinkhorn_l,
                concat_real_p, concat_fake_p,
                m_real_p, h_real_p, h_fake_p, scale=scale,
            )

            generator.zero_grad()
            gen_loss.backward()
            optimizerG.step()

            # ---- TensorBoard logging --------------------------------------
            writer.add_scalar('Loss/generator',     gen_loss.item(),  it_counts)
            writer.add_scalar('Loss/discriminator', disc_loss.item(), it_counts)
            writer.add_scalar('Penalty/pM_real',    pm1.item(),       it_counts)
            writer.add_scalar('Penalty/pM_emb',     pm2.item(),       it_counts)
            writer.flush()

            # ---- Divergence check -----------------------------------------
            if torch.isinf(gen_loss):
                print(f"[WARNING] Generator loss exploded at iteration {it_counts}. Stopping.")
                train_notes['training_failed_at_iteration'] = it_counts
                with open(os.path.join(save_root, 'train_notes.json'), 'w') as f:
                    json.dump(train_notes, f, indent=4)
                writer.close()
                return

            # ==============================================================
            # Periodic checkpoint saving
            # ==============================================================
            if it_counts in (1, 5, 10, 50) or it_counts % save_freq == 0:
                print(f"Epoch [{epoch}/{epochs}] Iter {it_counts} | "
                      f"G Loss: {gen_loss.item():.4f} | D Loss: {disc_loss.item():.4f}")

                # ---- Save checkpoint --------------------------------------
                checkpoint = {
                    'epoch':                    epoch,
                    'global_iteration':         it_counts,
                    'generator_state_dict':     generator.state_dict(),
                    'discriminator_h_state_dict': discriminator_h.state_dict(),
                    'discriminator_m_state_dict': discriminator_m.state_dict(),
                    'optimizerG_state_dict':    optimizerG.state_dict(),
                    'optimizerDH_state_dict':   optimizerDH.state_dict(),
                    'optimizerDM_state_dict':   optimizerDM.state_dict(),
                }
                ckpt_file = os.path.join(ckpt_dir, f'checkpoint{it_counts}.pt')
                torch.save(checkpoint, ckpt_file)
                print(f"  Checkpoint saved → {ckpt_file}")

    writer.close()
    print("\nTraining complete.")
