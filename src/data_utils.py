"""
data_utils.py — Data loading utilities for DeepX-GAN
=====================================================
This module handles loading and preprocessing of gridded climate data for
DeepX-GAN training.  Currently supported: ERA5 daily maximum 2-m temperature
(t2m daymax) for the MENA (Middle East and North Africa) region.

Data are organised as overlapping sliding-window sequences of consecutive summer days
(JJA: June-July-August) for each year, then min-max normalised to [0, 1].

"""

import os
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

# Maps a user-facing dataset name (passed as --dname) to the NetCDF filename,
# the variable name inside the file, and the name of the time coordinate.
# Add new datasets here to make them available without changing any other code.
DATASET_REGISTRY: dict = {
    'era5-t2m-daymax': {
        'filename':    'era5.reanalysis.t2m.daymax.32x64.1979-2014.MENA.nc',
        'nc_var':      't2m',
        'time_dim':    'valid_time',
        'description': 'ERA5 daily maximum 2-m temperature, 1979-2014, MENA (32 x 64 grid)',
    },
    # ---- Template for adding a new dataset --------------------------------
    # 'my-dataset-name': {
    #     'filename':    'my_data.nc',
    #     'nc_var':      'var_name_in_netcdf',
    #     'time_dim':    'time',
    #     'description': 'Short description shown in logs',
    # },
}


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class MyDataset(Dataset):
    """
    Simple map-style PyTorch Dataset that wraps a 5-D tensor of climate sequences.

    Each item is a single spatiotemporal sequence window of shape
    ``(time_steps, n_channels, height, width)``.
    """

    def __init__(self, data: torch.Tensor):
        """
        Parameters
        ----------
        data : torch.Tensor, shape (N, T, C, H, W)
            N   - number of sequence samples
            T   - number of time steps per sample
            C   - number of channels (1 for univariate temperature)
            H,W - spatial height and width of the domain
        """
        self.data = data

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, item: int) -> torch.Tensor:
        # Returns one spatiotemporal sequence of shape (T, C, H, W)
        return self.data[item]


# ---------------------------------------------------------------------------
# Season extraction helper
# ---------------------------------------------------------------------------

def season_preprocess(data: torch.Tensor, time, season: str, time_steps: int = 30) -> torch.Tensor:
    """
    Extract sliding-window sequences of consecutive summer days (JJA) for every year.

    For each year in the dataset we select the June-August days and build all
    overlapping windows of length ``time_steps`` with stride 1.  Windows from
    different years are concatenated along the first (batch) dimension.

    Parameters
    ----------
    data : torch.Tensor, shape (total_days, H, W)
        Full time series of daily gridded temperature values.
    time : xarray DataArray
        Time coordinate corresponding to the first axis of ``data``.
    season : str
        Target season.  Currently only ``'JJA'`` is supported.
    time_steps : int
        Length of each sequence window (number of days).

    Returns
    -------
    data_seq : torch.Tensor, shape (N_windows, T, H, W)
        All extracted sliding-window sequences stacked along the first axis.
    """
    if season == 'JJA':
        month_want = [6, 7, 8]   # June, July, August
    else:
        raise ValueError(f"Unsupported season '{season}'. Use 'JJA'.")

    # Build a helper DataFrame for year/month indexing
    df = pd.DataFrame({'time': time.values})
    df['month'] = pd.to_datetime(df['time']).dt.month
    df['year'] = pd.to_datetime(df['time']).dt.year

    data_seq = []
    unique_years = df['year'].unique()

    for year in unique_years:
        # Select days in the target months for this year
        mask = (df['year'] == year) & (df['month'].isin(month_want))
        data_yr = data[mask.values, ...]   # shape (n_days_yr, H, W)

        if data_yr.shape[0] < time_steps:
            # Skip years with fewer days than the requested window length
            continue

        # Build all overlapping windows of length time_steps
        windows = [data_yr[i:i + time_steps, ...] for i in range(data_yr.shape[0] - time_steps + 1)]
        data_seq.append(torch.stack(windows, dim=0))   # (n_windows, T, H, W)

    return torch.vstack(data_seq)   # (total_windows, T, H, W)


# ---------------------------------------------------------------------------
# Main data loading function
# ---------------------------------------------------------------------------

def fetch_climate_data(
    dname: str,
    data_dir: str = '../DATA/',
    time_steps: int = 30,
    season: str = 'JJA',
    debug_run: bool = False,
    minmax_stats: dict = None,
    return_minmax_stats: bool = False,
):
    """
    Load a registered climate dataset and prepare spatiotemporal sequence
    windows for DeepX-GAN training.

    The dataset is resolved from ``DATASET_REGISTRY`` using ``dname``.
    To add a new dataset, insert a new entry into ``DATASET_REGISTRY`` at the
    top of this file — no other code changes are required.

    Parameters
    ----------
    dname : str
        Dataset identifier, must be a key in ``DATASET_REGISTRY``
        (e.g. ``'era5-t2m-daymax'``).
    data_dir : str
        Base directory that contains the NetCDF files.  The filename from
        ``DATASET_REGISTRY`` is appended to form the full path.
        Default: ``'../DATA/'``.
    time_steps : int
        Number of consecutive days in each sequence window fed to the GAN.
        Default is 30 (approximately one summer month).
    season : str
        Season to extract.  Use ``'JJA'`` to keep only June-August days,
        or ``'full_year'`` to use all days with sliding windows.
    debug_run : bool
        If ``True``, only the first 64 samples are kept.  Useful for quick
        sanity-checks without running the full dataset.
    minmax_stats : dict or None
        Pre-computed normalisation bounds ``{'min': val, 'max': val}``.
        If ``None``, statistics are computed from the current dataset.
    return_minmax_stats : bool
        If ``True``, also return the normalisation statistics dictionary.

    Returns
    -------
    dataset : MyDataset
        PyTorch dataset with items of shape ``(T, C=1, H, W)`` normalised to [0, 1].
    x_height : int
        Spatial height (number of latitude grid points).
    x_width : int
        Spatial width (number of longitude grid points).
    minmax_stats : dict  (only when ``return_minmax_stats=True``)
        ``{'min': torch.Tensor, 'max': torch.Tensor}`` used for normalisation.
    """

    # ------------------------------------------------------------------
    # 1. Resolve file path and variable metadata from the registry
    # ------------------------------------------------------------------
    if dname not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dname}'. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    entry      = DATASET_REGISTRY[dname]
    data_path  = os.path.join(data_dir, entry['filename'])
    nc_var     = entry['nc_var']
    time_dim   = entry['time_dim']
    print(f"  Dataset : {entry['description']}")
    print(f"  File    : {data_path}")

    # ------------------------------------------------------------------
    # 2. Open NetCDF and extract the climate variable array
    # ------------------------------------------------------------------
    ds = xr.open_dataset(data_path)
    var        = ds[nc_var].values      # numpy ndarray, shape (T_total, H, W)
    time_coord = ds[time_dim]           # xarray DataArray with datetime values

    data = torch.from_numpy(var)    # Convert to PyTorch tensor

    # Remove singleton level dimension if present (some NCEP variables have it)
    if data.dim() == 4 and data.shape[1] == 1:
        data = data.squeeze(1)
    if data.dim() != 3:
        raise ValueError(f"Expected 3-D data (time, H, W), got shape {data.shape}.")

    total_days, x_height, x_width = data.shape

    # ------------------------------------------------------------------
    # 3. Extract seasonal windows
    # ------------------------------------------------------------------
    if season == 'full_year':
        # Sliding windows over the entire time axis (no month filtering)
        windows = [data[i:i + time_steps, ...] for i in range(total_days - time_steps + 1)]
        data_seq = torch.stack(windows, dim=0)   # (N, T, H, W)
    else:
        # Extract JJA windows year by year
        data_seq = season_preprocess(data, time_coord, season, time_steps=time_steps)

    # ------------------------------------------------------------------
    # 4. Add channel dimension: (N, T, H, W) → (N, T, C=1, H, W)
    # ------------------------------------------------------------------
    data_seq = data_seq.unsqueeze(2)   # insert channel axis

    # ------------------------------------------------------------------
    # 5. Optional: limit to a small subset for debugging
    # ------------------------------------------------------------------
    if debug_run:
        data_seq = data_seq[:64, ...]
        print(f"  [debug_run] Restricting to {data_seq.shape[0]} samples.")

    # ------------------------------------------------------------------
    # 6. Min-max normalisation to [0, 1]
    # ------------------------------------------------------------------
    if minmax_stats is None:
        minmax_stats = {
            'min': torch.min(data_seq),
            'max': torch.max(data_seq),
        }
    data_seq = (data_seq - minmax_stats['min']) / (minmax_stats['max'] - minmax_stats['min'])

    dataset = MyDataset(data_seq)
    print(f"  Samples : {data_seq.shape[0]}  |  Grid: {x_height} × {x_width}")

    if return_minmax_stats:
        return dataset, x_height, x_width, minmax_stats
    return dataset, x_height, x_width
