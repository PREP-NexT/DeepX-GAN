"""
spatial_utils.py — Spatial embedding utilities for DeepX-GAN
==============================================================
This module implements the **DeepX** embedding used
as an auxiliary channel in the GAN discriminator.

DeepX is a modification of SPATE (Zellner et al. 2022) that replaces the
classical Kulldorff space-time expectation with an extreme-value-aware version
based on the empirical upper tail-dependence coefficient (TDC).  This makes the
discriminator explicitly sensitive to co-occurring spatial extremes, which is
the key novelty of this paper.

Mathematical background
-----------------------
Given a spatiotemporal field  x[h, w, t]  (height × width × time), DeepX
proceeds in three steps:

1. **Tail dependence weights**:  For each pair of grid cells (i, j), estimate
   the probability that both cells are simultaneously above their respective
   ``u``-th quantile.  This yields a symmetric weight matrix TDC[i, j].

2. **Space-time expectation**:  For each cell i and time t, compute a weighted
   space-time expected value  E_i(t) that combines the TDC weights with an
   exponential temporal decay.

3. **Local Moran's I (SPATE statistic)**:  Compute the local spatial
   autocorrelation  MI_i(t) between the observed field and the expectation,
   using a first-order queen-contiguity spatial weight matrix.

The ``tdc_masked`` variant additionally masks the expectation by the joint
exceedance indicator, down-weighting non-extreme time steps.

Reference
---------
[Paper citation here]
"""

import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse
from libpysal.weights import lat2W


# ---------------------------------------------------------------------------
# Sparse matrix utilities
# ---------------------------------------------------------------------------

def crs_to_torch_sparse(x: scipy.sparse.csr_matrix) -> torch.Tensor:
    """
    Convert a SciPy CSR sparse matrix to a PyTorch sparse FloatTensor.

    Parameters
    ----------
    x : scipy.sparse.csr_matrix
        Input sparse matrix.

    Returns
    -------
    torch.sparse.FloatTensor
    """
    x = x.tocoo()
    indices = torch.LongTensor(np.vstack((x.row, x.col)))
    values = torch.FloatTensor(x.data)
    return torch.sparse.FloatTensor(indices, values, torch.Size(x.shape))


def make_sparse_weight_matrix(h: int, w: int, rook: bool = False) -> torch.Tensor:
    """
    Build the binary queen (or rook) contiguity spatial weight matrix for an
    h × w regular grid and return it as a PyTorch sparse tensor.

    Queen contiguity (default) assigns weight 1 to all eight neighbours
    (including diagonals); rook contiguity assigns weight 1 only to the four
    orthogonal neighbours.

    Parameters
    ----------
    h : int
        Number of rows (latitude grid cells).
    w : int
        Number of columns (longitude grid cells).
    rook : bool
        If ``True``, use rook contiguity; otherwise use queen contiguity.

    Returns
    -------
    torch.sparse.FloatTensor of shape (h*w, h*w)
    """
    W = lat2W(h, w, rook=rook)
    return crs_to_torch_sparse(W.sparse)


# ---------------------------------------------------------------------------
# Temporal decay weights
# ---------------------------------------------------------------------------

def temporal_weights(n: int, b: float) -> torch.Tensor:
    """
    Compute exponentially decaying temporal weights for a window of n time steps.

    The weight for lag τ (looking backward from the current time step) is::

        w(τ) = exp(−τ / b),   τ = 1, …, n−1

    arranged so that the most recent lag (τ = 1) appears last in the tensor.

    Parameters
    ----------
    n : int
        Number of time steps.
    b : float
        Temporal decay parameter; larger ``b`` means slower decay (more memory).

    Returns
    -------
    torch.Tensor of shape (1, 1, n−1)
    """
    return torch.exp(-torch.arange(1, n).flip(0) / b).view(1, 1, -1)


# ---------------------------------------------------------------------------
# Upper tail dependence coefficient (TDC)
# ---------------------------------------------------------------------------

def get_tdc(a: torch.Tensor, u: float = 0.8, correct_diag: bool = True) -> torch.Tensor:
    """
    Estimate the empirical upper tail-dependence coefficient (TDC) matrix.

    For each pair of grid cells (i, j), TDC[i, j] estimates the conditional
    probability that cell j is above its u-th quantile given that cell i is
    above its u-th quantile::

        TDC[i, j] ≈ P(X_j > Q_j(u) | X_i > Q_i(u))
                   = P(X_i > Q_i(u), X_j > Q_j(u)) / (1 − u)

    The computation is vectorised via matrix multiplication for efficiency.

    Parameters
    ----------
    a : torch.Tensor, shape (n_time, n_cells)
        Flattened spatiotemporal field; each column is one grid cell's time series.
    u : float
        Quantile threshold (e.g. 0.8 means top-20 % extremes).
    correct_diag : bool
        Force diagonal to 1 (a cell is perfectly dependent with itself).

    Returns
    -------
    torch.Tensor of shape (n_cells, n_cells)
        Symmetric TDC weight matrix.
    """
    n, hw = a.shape
    # Boolean indicator: True where cell value exceeds its u-th quantile at that time step
    in_tail = a > torch.quantile(a, q=u, dim=0)   # (n_time, n_cells)

    # Count simultaneous exceedances for each cell pair (vectorised dot product)
    probs = torch.matmul(in_tail.t().float(), in_tail.float())  # (n_cells, n_cells)
    probs = probs / (n * (1 - u))

    if correct_diag:
        probs.fill_diagonal_(1.0)
    return probs


def get_weights_tdc(x: torch.Tensor, u: float = 0.8) -> torch.Tensor:
    """
    Compute the TDC weight matrix for a single spatiotemporal field.

    Parameters
    ----------
    x : torch.Tensor, shape (H, W, T)
        A single spatiotemporal sequence (height × width × time steps).
    u : float
        Quantile threshold for extreme identification.

    Returns
    -------
    torch.Tensor of shape (H, W, H*W)
        TDC weights reshaped for use in the space-time expectation.
    """
    h, w, n = x.shape
    # Reshape to (T, H*W) so each column is one cell's time series
    x_flat = x.reshape(h * w, n).permute(1, 0)   # (T, H*W)
    weights_tdc = get_tdc(x_flat, u, correct_diag=True)   # (H*W, H*W)
    return weights_tdc.reshape(h, w, h * w)


def paired_multiply(mat: torch.Tensor) -> torch.Tensor:
    """
    Compute the outer product of a flattened 2-D matrix with itself.

    For an H × W boolean mask, this returns an (H*W × H*W) matrix where
    entry [i, j] = mask_flat[i] * mask_flat[j], i.e. 1 iff both cells are in
    the extreme tail at this time step.

    Parameters
    ----------
    mat : torch.Tensor, shape (H, W)
        2-D mask (typically a boolean or float extreme indicator).

    Returns
    -------
    torch.Tensor of shape (H*W, H*W)
    """
    h, w = mat.shape
    mat_flat = mat.reshape(h * w, 1).float()
    return torch.mul(mat_flat, mat_flat.t())   # outer product


def get_weights_tdc_masked(x: torch.Tensor, u: float = 0.8) -> torch.Tensor:
    """
    Compute the TDC weight matrix masked by the joint exceedance indicator.

    Unlike ``get_weights_tdc``, each time step receives its own masked weight:
    the TDC coefficient between cells i and j at time t is zeroed out unless
    *both* cells exceed their respective u-th quantile at time t.  This focuses
    the embedding strictly on time steps that are jointly extreme.

    Parameters
    ----------
    x : torch.Tensor, shape (H, W, T)
        Spatiotemporal sequence.
    u : float
        Quantile threshold.

    Returns
    -------
    torch.Tensor of shape (H, W, H*W, T)
        Time-varying masked TDC weights.
    """
    h, w, n = x.shape

    # Boolean mask: True where each cell exceeds its own u-th quantile at each time step
    # Shape: (H, W, T)
    mask_ele = x > torch.quantile(x, u, dim=2, keepdim=True)

    # Build the outer-product mask for every time step: shape (T, H*W, H*W)
    mask_paired = torch.stack([paired_multiply(mask_ele[:, :, t]) for t in range(n)])

    # Global TDC matrix (time-averaged)
    x_flat = x.reshape(h * w, n).permute(1, 0)   # (T, H*W)
    weights_tdc = get_tdc(x_flat, u, correct_diag=True)   # (H*W, H*W)

    # Mask by joint exceedance: broadcast (H*W, H*W) against (T, H*W, H*W)
    weights_tdc_masked = weights_tdc * mask_paired          # (T, H*W, H*W)

    # Reshape to (H, W, H*W, T)  for use in st_ex_tdc
    weights_tdc_masked = weights_tdc_masked.permute(1, 2, 0)          # (H*W, H*W, T)
    weights_tdc_masked = weights_tdc_masked.reshape(h, w, h * w, n)
    return weights_tdc_masked


# ---------------------------------------------------------------------------
# Space-time expectation functions
# ---------------------------------------------------------------------------

def st_ex_tdc(x: torch.Tensor, weights: torch.Tensor, weights_tdc, mask_on: bool = False) -> torch.Tensor:
    """
    Compute the DeepX space-time expectation using TDC weights.

    For each grid cell i and time step t, the expected value is::

        E_i(t) = [Σ_{τ<t} w(t-τ) * x_i(τ)] *
                 [Σ_j TDC(i,j) * x_j(t)] /
                 [Σ_{τ<t} w(t-τ) * x_i(τ)]   (integrated over space j)

    When ``mask_on=True`` (the ``tdc_masked`` method), the second factor uses
    the time-varying masked TDC weights so that only jointly extreme time steps
    contribute.

    Parameters
    ----------
    x : torch.Tensor, shape (H, W, T)
        Spatiotemporal sequence.
    weights : torch.Tensor, shape (1, 1, T−1)
        Exponential temporal decay weights (from ``temporal_weights()``).
    weights_tdc : torch.Tensor
        TDC weight tensor.  Shape (H, W, H*W) for ``mask_on=False``;
        shape (H, W, H*W, T) for ``mask_on=True``.
    mask_on : bool
        If ``True``, use the time-varying masked TDC weights.

    Returns
    -------
    torch.Tensor of shape (H, W, T−1)
        DeepX space-time expectation for each cell and time step.
    """
    h, w, n = x.shape

    if mask_on:
        # weights_tdc shape: (H, W, H*W, T) → reshape to (H*W, H*W, T)
        weights_tdc_resize = weights_tdc.view(h * w, h * w, n)
        x_resize = x.view(h * w, n)   # (H*W, T)
        # For each t, compute Σ_j TDC_masked(i,j,t) * x_j(t) for all cells i
        # Result is a list of (H*W,) vectors for t=1,...,T-1
        term1 = [torch.matmul(weights_tdc_resize[:, :, t].t(), x_resize[:, 1:])[:, t - 1]
                 for t in range(1, n)]
        term1 = torch.stack(term1).t()   # (H*W, T-1)
    else:
        # weights_tdc shape: (H, W, H*W) → (H*W, H*W) after view + transpose
        # term1[i, t] = Σ_j TDC(i, j) * x_j(t+1),  shape (H*W, T-1)
        term1 = torch.matmul(weights_tdc.view(h * w, -1).t(), x.view(h * w, n)[:, 1:])

    # Temporal component: weighted average of past values, shape (H*W, 1) for each t
    exp_val = [
        (weights[:, :, -t:] * x[:, :, :t]).sum(dim=2).reshape(-1) *
        term1[:, t - 1] /
        (weights[:, :, -t:] * x[:, :, :t]).reshape(-1).sum()
        for t in range(1, n)
    ]
    exp_val = torch.stack(exp_val).permute(1, 0).reshape(h, w, n - 1)
    return exp_val


def st_ex(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Classic sequential space-time expectation (original SPATE, no TDC).

    This is the baseline expectation assuming spatial and temporal independence::

        E_i(t) = [Σ_{τ<t} w(t-τ) * x_i(τ)] * Σ_j x_j(t) /
                 [Σ_{τ<t} w(t-τ) * x_i(τ)]   (integrated over all cells j)

    Parameters
    ----------
    x : torch.Tensor, shape (H, W, T)
    weights : torch.Tensor, shape (1, 1, T−1)

    Returns
    -------
    torch.Tensor of shape (H, W, T−1)
    """
    h, w, n = x.shape
    exp_val = [
        (weights[:, :, -t:] * x[:, :, :t]).sum(dim=2).reshape(-1) *
        x[:, :, t].reshape(-1).sum() /
        (weights[:, :, -t:] * x[:, :, :t]).reshape(-1).sum()
        for t in range(1, n)
    ]
    return torch.stack(exp_val).permute(1, 0).reshape(h, w, n - 1)


# ---------------------------------------------------------------------------
# Local Moran's I helper
# ---------------------------------------------------------------------------

def mi_mean(x: torch.Tensor, x_mean: torch.Tensor, w_sparse: torch.Tensor) -> torch.Tensor:
    """
    Compute local Moran's I with a custom (externally provided) mean.

    Local Moran's I quantifies how similar a cell's deviation from expectation
    is to its spatial neighbours' deviations::

        MI_i = (n−1) * z_i * Σ_j w_{ij} z_j / Σ_i z_i²

    where  z_i = (x_i − x_mean_i) / std(x).

    Parameters
    ----------
    x : torch.Tensor, shape (H*W,) or (H, W)
        Observed field (will be flattened).
    x_mean : torch.Tensor, shape (H*W,)
        Expected value used as the local mean (replaces the global mean).
    w_sparse : torch.Tensor (sparse), shape (H*W, H*W)
        Spatial weight matrix.

    Returns
    -------
    torch.Tensor of shape (H*W,)
        Local Moran's I for each grid cell.
    """
    x = x.reshape(-1)
    x_mean = x_mean.reshape(-1)
    n = len(x)
    z = x - x_mean
    sx = x.std()
    z = z / sx
    den = (z * z).sum()
    # Spatially lagged z: Σ_j w_{ij} z_j
    zl = torch.sparse.mm(w_sparse, z.reshape(-1, 1)).reshape(-1)
    return (n - 1) * z * zl / den


# ---------------------------------------------------------------------------
# Per-sample SPATE / DeepX computation
# ---------------------------------------------------------------------------

def spate(
    x: torch.Tensor,
    w_sparse: torch.Tensor,
    b: torch.Tensor,
    method: str = "tdc_masked",
    b_tdc=None,
) -> torch.Tensor:
    """
    Compute SPATE or DeepX for a single spatiotemporal sequence.

    Parameters
    ----------
    x : torch.Tensor, shape (H, W, T)
        Single spatiotemporal field (one sample, one channel).
    w_sparse : torch.Tensor (sparse), shape (H*W, H*W)
        Queen-contiguity spatial weight matrix.
    b : torch.Tensor, shape (1, 1, T−1)
        Temporal decay weights.
    method : str
        Embedding method.  Options:
        - ``'tdc'``        : DeepX with time-averaged TDC weights.
        - ``'tdc_masked'`` : DeepX with time-varying masked TDC weights
                             (default; captures joint extremes best).
        - ``'skw'``        : Classic SPATE with sequential Kulldorff weights.
    b_tdc : torch.Tensor or None
        Pre-computed TDC weight tensor (pass from ``get_weights_tdc`` or
        ``get_weights_tdc_masked``).  Must be provided for ``'tdc'`` and
        ``'tdc_masked'``.

    Returns
    -------
    torch.Tensor of shape (H, W, T−1)
        SPATE / DeepX values for each cell and time step.
    """
    h, w, n = x.shape

    if method == "tdc":
        x_means = st_ex_tdc(x, b, b_tdc, mask_on=False)
    elif method == "tdc_masked":
        x_means = st_ex_tdc(x, b, b_tdc, mask_on=True)
    else:
        # Fallback to classic sequential SPATE
        x_means = st_ex(x, b)

    # Compute local Moran's I between x[:,t+1] and expectation x_means[:,t]
    spates = torch.stack([
        mi_mean(x[:, :, i + 1].reshape(-1), x_means[:, :, i].reshape(-1), w_sparse).reshape(h, w)
        for i in range(n - 1)
    ])
    return spates.permute(1, 2, 0)   # (H, W, T-1)


# ---------------------------------------------------------------------------
# Batch DeepX computation
# ---------------------------------------------------------------------------

def make_spates(
    x: torch.Tensor,
    w_sparse: torch.Tensor,
    b: torch.Tensor,
    method: str = "tdc_masked",
    u: float = 0.8,
    theta1: float = 0.5,
    theta2: float = 0.5,
) -> torch.Tensor:
    """
    Compute DeepX embeddings for a full batch of spatiotemporal sequences.

    The output shape matches the input shape but with T replaced by T−1 for
    the time-sequential methods (``tdc``, ``tdc_masked``, ``skw``).  A zero
    frame is prepended and the whole tensor is rolled by one step so that the
    embedding at time t uses only information up to t−1 (causal embedding).

    Parameters
    ----------
    x : torch.Tensor, shape (N, T, C, H, W)
        Batch of real or generated spatiotemporal sequences.
        N – batch size, T – time steps, C – channels, H – height, W – width.
    w_sparse : torch.Tensor (sparse), shape (H*W, H*W)
        Spatial weight matrix (should be on the same device as ``x``).
    b : torch.Tensor, shape (1, 1, T−1)
        Temporal decay weights.
    method : str
        Embedding method (see ``spate()`` for options).
    u : float
        TDC threshold (quantile); only used for ``'tdc'`` and ``'tdc_masked'``.
    theta1 : float
        Weight for the classic SPATE component in ``'tdc_masked'``.
    theta2 : float
        Weight for the masked DeepX component in ``'tdc_masked'``.

    Returns
    -------
    torch.Tensor of shape (N, T, C, H, W)
        DeepX embedding, normalised per-sample to [0, 1].
        The first time step contains zeros (no history available at t=0).
    """
    n, t, nc, h, w = x.shape

    if method == "tdc_masked":
        # Compute the masked DeepX component for each sample and channel
        spates_masked = torch.stack([
            spate(
                x[i, :, j, :, :].reshape(t, h, w).permute(1, 2, 0),  # (H, W, T)
                w_sparse, b, method,
                get_weights_tdc_masked(x[i, :, j, :, :].reshape(t, h, w).permute(1, 2, 0), u)
            ).permute(2, 0, 1)   # (T-1, H, W)
            for j in range(nc) for i in range(n)
        ]).reshape(n, t - 1, nc, h, w)

        # Also compute the classic SPATE component (for the convex combination)
        spates_original = torch.stack([
            spate(
                x[i, :, j, :, :].reshape(t, h, w).permute(1, 2, 0),
                w_sparse, b, method="skw"
            ).permute(2, 0, 1)
            for j in range(nc) for i in range(n)
        ]).reshape(n, t - 1, nc, h, w)

        # Convex combination: theta1 * SPATE + theta2 * DeepX_masked
        spates = theta1 * spates_original + theta2 * spates_masked

    elif method == "tdc":
        spates = torch.stack([
            spate(
                x[i, :, j, :, :].reshape(t, h, w).permute(1, 2, 0),
                w_sparse, b, method,
                get_weights_tdc(x[i, :, j, :, :].reshape(t, h, w).permute(1, 2, 0), u)
            ).permute(2, 0, 1)
            for j in range(nc) for i in range(n)
        ]).reshape(n, t - 1, nc, h, w)

    else:
        # Classic SPATE (skw, k, kw)
        spates = torch.stack([
            spate(
                x[i, :, j, :, :].reshape(t, h, w).permute(1, 2, 0),
                w_sparse, b, method
            ).permute(2, 0, 1)
            for j in range(nc) for i in range(n)
        ]).reshape(n, t - 1, nc, h, w)

    # Per-sample min-max normalisation to [0, 1]
    spates = torch.stack([
        (spates[i, :, j, :, :] - spates[i, :, j, :, :].min()) /
        (spates[i, :, j, :, :].max() - spates[i, :, j, :, :].min())
        for j in range(nc) for i in range(n)
    ]).reshape(n, t - 1, nc, h, w)

    # Prepend a zero frame (no embedding available at t=0) and roll so index 0 contains zeros
    spates = F.pad(spates, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])   # pad one step on time axis
    spates = torch.roll(spates, 1, 1)                          # shift right by 1

    return spates   # (N, T, C, H, W)
