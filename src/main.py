"""
main.py — Entry point for DeepX-GAN training on ERA5 temperature data
=======================================================================
DeepX-GAN is a Causal Optimal Transport GAN designed to reproduce realistic
spatiotemporal patterns of daily maximum 2-m temperature (t2m), including
extreme co-occurrence events.

The key innovation over standard video GANs is the **DeepX embedding**: a
modified local Moran's I statistic that uses the empirical upper
tail-dependence coefficient (TDC) as spatial weights.  This makes the
discriminator explicitly sensitive to spatially co-occurring extreme events.

Usage
-----
Train with default settings (ERA5 JJA, tdc_masked embedding)::

    python main.py

Train with custom settings::

    python main.py \\
        --dname era5-t2m-daymax \\
        --data_dir ../DATA/ \\
        --n_epochs 50000 \\
        --batch_size 32 \\
        --time_steps 30 \\
        --stx_method tdc_masked \\
        --save_freq 100

Quick debug run (64 samples, 10 epochs)::

    python main.py --debug_run True --n_epochs 10

Resume from checkpoint::

    python main.py \\
        --pre_trained_path ./trained/<run_name>/ckpts \\
        --iter_final <checkpoint_iteration>

Reference
---------
[Paper citation here]
"""

import os
import argparse

import torch

from train import train


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for DeepX-GAN training.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description='DeepX-GAN: Extreme Spatiotemporal GAN for Climate Data'
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    parser.add_argument(
        '--dname', type=str, default='era5-t2m-daymax',
        help=(
            'Dataset name.  Must be a key in the DATASET_REGISTRY dictionary '
            'defined in data_utils.py (e.g. "era5-t2m-daymax").  '
            'The name is also embedded in the output folder name so each run '
            'is self-documenting.'
        )
    )
    parser.add_argument(
        '--data_dir', type=str, default='../DATA/',
        help=(
            'Base directory that contains the NetCDF data files.  '
            'The filename for the selected --dname is appended automatically.'
        )
    )
    parser.add_argument(
        '--output_dir', type=str, default='../',
        help='Root directory for saving checkpoints, logs, and images.'
    )

    # ------------------------------------------------------------------
    # Data settings
    # ------------------------------------------------------------------
    parser.add_argument(
        '--time_steps', '-ts', type=int, default=30,
        help=(
            'Number of consecutive days in each training sequence window.  '
            'A value of 30 covers approximately one summer month.'
        )
    )
    parser.add_argument(
        '--season', type=str, default='JJA', choices=['JJA', 'full_year'],
        help=(
            "Season to train on.  'JJA' keeps only June–August days "
            "(summer extremes); 'full_year' uses all days."
        )
    )
    parser.add_argument(
        '--debug_run', type=bool, default=True,
        help=(
            'If True, restrict to 64 training samples for a fast sanity check '
            'before launching a full run.'
        )
    )

    # ------------------------------------------------------------------
    # Embedding settings
    # ------------------------------------------------------------------
    parser.add_argument(
        '--stx_method', type=str, default='tdc_masked',
        choices=['tdc_masked', 'tdc', 'skw'],
        help=(
            "Spatio-temporal embedding method used as the discriminator auxiliary channel:\n"
            "  tdc_masked : DeepX metric with joint-exceedance-masked TDC weights (recommended).\n"
            "  tdc        : DeepX metric with time-averaged TDC weights (no masking).\n"
            "  skw        : Classic SPATE (sequential Kulldorff weights; no extreme focus)."
        )
    )
    parser.add_argument(
        '--u', type=float, default=0.7,
        help=(
            'Quantile threshold for defining extremes (tail-dependence coefficient). '
            'u=0.7 means the top 30%% of values are considered extreme.'
        )
    )
    parser.add_argument(
        '--dec_weight', '-b', type=int, default=20,
        help=(
            'Exponential temporal decay parameter b.  Larger values give more '
            'weight to remote past time steps when computing the space-time expectation.'
        )
    )
    parser.add_argument(
        '--theta1', type=float, default=0.5,
        help='Weight for the classic SPATE component in the tdc_masked combined embedding.'
    )
    parser.add_argument(
        '--theta2', type=float, default=0.5,
        help='Weight for the masked DeepX component in the tdc_masked combined embedding.'
    )

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    parser.add_argument(
        '--g_state_size', '-gss', type=int, default=16,
        help='LSTM hidden size in the generator.'
    )
    parser.add_argument(
        '--g_filter_size', '-gfs', type=int, default=16,
        help='Number of feature maps in the last generator deconv layer.'
    )
    parser.add_argument(
        '--d_state_size', '-dss', type=int, default=16,
        help='(Unused; discriminator feature maps are set by d_filter_size.)'
    )
    parser.add_argument(
        '--d_filter_size', '-dfs', type=int, default=16,
        help='Number of feature maps in the first discriminator conv layer.'
    )
    parser.add_argument(
        '--z_dims_t', '-Dz', type=int, default=5,
        help=(
            'Square root of the temporal noise dimension z. '
            'The actual z dimension is z_dims_t² = 25 by default.'
        )
    )
    parser.add_argument(
        '--y_dims', '-Dy', type=int, default=20,
        help='Dimensionality of the static (per-sequence) latent noise y.'
    )
    parser.add_argument(
        '--n_channels', '-nch', type=int, default=1,
        help='Number of climate variable channels (1 for univariate temperature).'
    )
    parser.add_argument(
        '--batch_norm', '-bn', type=bool, default=True,
        help='Enable Batch Normalisation in generator and discriminator.'
    )

    # ------------------------------------------------------------------
    # Training hyper-parameters
    # ------------------------------------------------------------------
    parser.add_argument(
        '--n_epochs', '-ne', type=int, default=50000,
        help='Total number of training epochs (passes over the dataset).'
    )
    parser.add_argument(
        '--batch_size', '-bs', type=int, default=32,
        help=(
            'Number of real samples per batch (the data loader fetches 2× '
            'this number to provide both x and x\' for the mixed Sinkhorn loss).'
        )
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Learning rate for the Adam optimisers (same for G, D_H, D_M).'
    )
    parser.add_argument(
        '--sinkhorn_eps', '-sinke', type=float, default=0.8,
        help='Sinkhorn entropy regularisation coefficient ε.'
    )
    parser.add_argument(
        '--sinkhorn_l', '-sinkl', type=int, default=100,
        help='Maximum number of Sinkhorn iterations.'
    )
    parser.add_argument(
        '--reg_penalty', '-reg_p', type=float, default=1.5,
        help='Coefficient λ for the martingale regularisation penalty p_M.'
    )
    parser.add_argument(
        '--scale', '-sl', type=bool, default=True,
        help='Divide Sinkhorn costs by the number of time steps T.'
    )

    # ------------------------------------------------------------------
    # Checkpoint and logging
    # ------------------------------------------------------------------
    parser.add_argument(
        '--save_freq', '-save', type=int, default=100,
        help='Save a checkpoint every this many iterations.'
    )
    parser.add_argument(
        '--pre_trained_path', type=str, default=None,
        help=(
            'Path to a checkpoint directory to resume training from.  '
            'Must be used together with --iter_final.'
        )
    )
    parser.add_argument(
        '--iter_final', type=int, default=None,
        help='Checkpoint iteration number to load when resuming training.'
    )
    parser.add_argument(
        '--cuda_num', type=int, default=0,
        help='CUDA device index to use for training (e.g. 0 for the first GPU).'
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected — running on CPU (training will be slow).")

    # Ensure we are in the script's directory so relative paths work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    args = parse_args()
    print("\n=== DeepX-GAN Training Configuration ===")
    for k, v in vars(args).items():
        print(f"  {k:30s}: {v}")
    print("========================================\n")

    train(args)
    print("\nAll done!")
