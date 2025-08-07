from data_utils import *
from spatial_utils import *
from gan_utils import *
from models_parallel import *
from train_parallel import *
import sys

#Check if GPU is available
torch.cuda.is_available()

# # launch tensorboard
# %load_ext tensorboard
# %tensorboard --logdir=trained

if __name__ == '__main__':
    ## path
    print(f"\nCurrent working directory: {os.getcwd()}")
    if not os.getcwd().endswith("src"):
        # os.chdir("/path_to_project_folder/src/")  ## !!! MODIFY THIS PATH !!!
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        print(f"\nChanged working directory: {os.getcwd()}")

    parser = argparse.ArgumentParser(description='cot')
    parser.add_argument('-lf', '--loss_func', type=str, default="cotgan",
                        choices=["sinkhorngan", "cotgan"])  # Loss function
    parser.add_argument('-eo', '--embedding_op', type=str, default="spate",
                        choices=["moran", "spate", "none"])  # Embedding loss
    parser.add_argument('-stx', '--stx_method', type=str, default="tdc_masked",    # "skw" for original spate-gan. "tdc_masked" for DeepX-GAN
                        choices=["skw", "k", "kw", "tdc", "tdc_masked"])  # Spatiotemporal expectation method for SPATE
    parser.add_argument('-t', '--test', type=str, default='cot', choices=['cot'])
    parser.add_argument('-s', '--seed', type=int, default=1)  # Random seed
    parser.add_argument('-b', '--dec_weight', type=int, default=20)  # Temporal weight (for SPATE^kw and SPATE^skw)

    # changed
    parser.add_argument('-ts', '--time_steps', type=int, default=30)  # Number of time steps
    parser.add_argument('-bs', '--batch_size', type=int, default=4)  # Batch size
    parser.add_argument('-gss', '--g_state_size', type=int, default=16)  # Generator state size
    parser.add_argument('-gfs', '--g_filter_size', type=int, default=16)  # Generator filter size
    parser.add_argument('-dss', '--d_state_size', type=int, default=16)  # Discriminator state size
    parser.add_argument('-dfs', '--d_filter_size', type=int, default=16)  # Discriminator filter size
    parser.add_argument('-ne', '--n_epochs', type=int, default=10000)  # Number of training epochs
    parser.add_argument('-d', '--dname', type=str, default="lgcp",
                        choices=['air', 'tmax', 'prate', 'lgcp', 't2m'])  # Dataset

    parser.add_argument('-sinke', '--sinkhorn_eps', type=float, default=0.8)  # Sinkhorn epsilon
    parser.add_argument('-reg_p', '--reg_penalty', type=float, default=1.5)  # Regularization penalty
    parser.add_argument('-sinkl', '--sinkhorn_l', type=int, default=100)  # Sinkhorn l
    parser.add_argument('-Dx', '--Dx', type=int, default=1)  # Noise dimensions
    parser.add_argument('-Dz', '--z_dims_t', type=int, default=5)  # Noise dimensions
    parser.add_argument('-Dy', '--y_dims', type=int, default=20)  # Noise dimensions
    parser.add_argument('-g', '--gen', type=str, default="fc", choices=["lstm", "fc"])  # Generator layers
    parser.add_argument('-p', '--path', type=str, default='../')  # Model save path   ## !!! MODIFY THIS PATH !!!
    parser.add_argument('-save', '--save_freq', type=int, default=20)  # Model save frequency
    parser.add_argument('-lr', '--lr', type=float, default=1e-4)  # Learning rate
    parser.add_argument('-bn', '--batch_norm', type=bool, default=True)  # Enable batch normalization
    parser.add_argument('-sl', '--scale', type=bool, default=True)  # Enable loss scaling
    parser.add_argument('-nlstm', '--n_lstm', type=int, default=1)  # Number of LSTM layers
    parser.add_argument('-lam', '--lam', type=float, default=1.0)
    parser.add_argument('-nch', '--n_channels', type=int, default=1)  # Number of channels
    parser.add_argument('-rt', '--read_tfrecord', type=bool, default=True)
    parser.add_argument('-f')  # Dummy to get parser to run in Colab
    # added
    parser.add_argument('-u', '--u', type=float, default=0.7)  # Threshold for empirical upper tail dependence coefficient
    parser.add_argument('-pre_path', '--pre_trained_path', type=str, default=None)
        # None for no load, otherwise give the path, e.g."./trained/lgcp-cotgan-spate-tdc_Mar04-06.22.45.465254/ckpts"
    # "../initial"
    parser.add_argument('-iter_final', '--iter_final', type=int, default=None)  ## 330
    parser.add_argument('-image_size', '--x_size', type=int, default=16)
    parser.add_argument('-sample_method', '--sample_method', type=str, default='pyrDown', choices=['pyrDown', 'resize', 'crop'])
    parser.add_argument('-cuda_num', '--cuda_num', type=int, default=0)  ## choose which GPU to be run on
    parser.add_argument('-season', '--season', type=str, default='JJA', choices=['full_year', 'JJA'])
    parser.add_argument('-parallel_ids', '--parallel_ids', type=list, default=[0, 1])
    parser.add_argument('-theta1', '--theta1', type=float, default=0.8)
    parser.add_argument('-theta2', '--theta2', type=float, default=0.2)

    args = parser.parse_args()
    print("TRAINING - Dataset: " + args.dname + " Emb: " + args.embedding_op + " Loss: " + args.loss_func)
    train(args)
    print("All finished!")

