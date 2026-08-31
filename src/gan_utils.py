"""
gan_utils.py — GAN loss functions for DeepX-GAN
==================================================
This module implements the **Causal Optimal Transport GAN (COT-GAN)** loss
(Xu et al. 2020) used to train DeepX-GAN.

The COT-GAN objective extends the classical Sinkhorn divergence with a
martingale-aware cost function that respects the causal (temporal) structure
of the data.  The key building blocks are:

1. **Sinkhorn algorithm**: An efficient approximation to the Wasserstein-1
   optimal transport cost via entropy regularisation.

2. **Modified cost function**: The pairwise transport cost between two
   trajectories is augmented with a term involving the discriminator outputs
   h (from D_H) and M (from D_M), capturing the causal dependence structure.

3. **Martingale regularisation** (p_M): An auxiliary penalty that encourages
   the discriminator output M to satisfy the discrete martingale condition,
   which is required for the theoretical guarantees of COT-GAN.

Reference
---------
Xu, T., et al. (2020). "COT-GAN: Generating Sequential Data via Causal
Optimal Transport." NeurIPS 2020.

"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Pairwise cost matrices
# ---------------------------------------------------------------------------

def cost_matrix(x: torch.Tensor, y: torch.Tensor, p: int = 2, scale: bool = False) -> torch.Tensor:
    """
    Compute the L^p pairwise cost matrix between two batches of trajectories.

    Each entry C[i, j] is the sum of p-th powers of element-wise absolute
    differences between trajectory i in ``x`` and trajectory j in ``y``,
    accumulated over all time steps and features::

        C[i, j] = Σ_t ||x_i(t) - y_j(t)||^p

    Parameters
    ----------
    x : torch.Tensor, shape (N, T, F)
        Batch of N trajectories of length T with feature dimension F.
    y : torch.Tensor, shape (M, T, F)
        Batch of M trajectories.
    p : int
        Power of the Lp norm.  Default is p=2 (squared Euclidean).
    scale : bool
        If ``True``, divide the cost by the number of time steps T to make
        the loss independent of sequence length.

    Returns
    -------
    torch.Tensor of shape (N, M)
    """
    # Broadcasting: x_col is (N, 1, T, F), y_lin is (1, M, T, F)
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    T = x.shape[1]
    # Sum over features first, then over time steps
    b = torch.sum(torch.abs(x_col - y_lin) ** p, dim=-1)   # (N, M, T)
    c = torch.sum(b, dim=-1)                                 # (N, M)
    if scale:
        c = c / T
    return c


def modified_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    h: torch.Tensor,
    M: torch.Tensor,
    scale: bool = False,
) -> torch.Tensor:
    """
    Compute the **martingale-aware modified cost** used in COT-GAN.

    The cost between trajectory x_i and trajectory y_j is::

        C_hM(x_i, y_j) = L²(x_i, y_j) + Σ_{t=1}^{T−1} h_i(t) · ΔM_j(t+1)

    where ΔM_j(t) = M_j(t) − M_j(t−1) is the increment of the discriminator
    output M along trajectory j, and h_i(t) is the discriminator output H for
    trajectory i.  This coupling term captures the causal structure by making
    the cost sensitive to how the marginals evolve over time.

    Parameters
    ----------
    x : torch.Tensor, shape (N, T, F)
        Real trajectories.
    y : torch.Tensor, shape (M, T, F)
        Fake (generated) trajectories.
    h : torch.Tensor, shape (N, T, J)
        Discriminator H output for the batch of trajectories.
    M : torch.Tensor, shape (M, T, J)
        Discriminator M output for the batch of trajectories.
    scale : bool
        Divide by T−1 for length-independent loss.

    Returns
    -------
    torch.Tensor of shape (N, M)
        Modified pairwise cost matrix.
    """
    # Martingale increments of M along the sequence: shape (M, T-1, J)
    DeltaMt = M[:, 1:, :] - M[:, :-1, :]
    ht = h[:, :-1, :]          # h at all but the last time step: (N, T-1, J)
    T_minus1 = ht.shape[1]

    # Cross-term: h_i(t) · ΔM_j(t);  broadcasting over the two batch dimensions
    # ht[:, None]: (N, 1, T-1, J),  DeltaMt[None, :]: (1, M, T-1, J)
    sum_over_j = torch.sum(ht[:, None, :, :] * DeltaMt[None, :, :, :], dim=-1)  # (N, M, T-1)
    C_hM = torch.sum(sum_over_j, dim=-1)   # (N, M)
    if scale:
        C_hM = C_hM / T_minus1

    return cost_matrix(x, y, scale=scale) + C_hM


# ---------------------------------------------------------------------------
# Sinkhorn algorithm
# ---------------------------------------------------------------------------

def compute_sinkhorn(
    x: torch.Tensor,
    y: torch.Tensor,
    h: torch.Tensor,
    M: torch.Tensor,
    epsilon: float = 0.1,
    niter: int = 10,
    scale: bool = False,
    benchmark: bool = False,
) -> torch.Tensor:
    """
    Compute the Sinkhorn divergence between two empirical distributions.

    Uses the log-domain Sinkhorn algorithm with Nesterov-like acceleration for
    numerical stability.  The entropy-regularised OT cost is::

        OT_ε(μ, ν) = min_{π ∈ Π(μ,ν)} <C, π> + ε KL(π | μ⊗ν)

    Parameters
    ----------
    x : torch.Tensor, shape (N, T, F)
        Samples from distribution μ (e.g. real data).
    y : torch.Tensor, shape (N, T, F)
        Samples from distribution ν (e.g. generated data).
    h : torch.Tensor, shape (N, T, J)
        Discriminator H output (used in the modified cost).
    M : torch.Tensor, shape (N, T, J)
        Discriminator M output (used in the modified cost).
    epsilon : float
        Entropy regularisation coefficient.  Smaller values give a closer
        approximation to true OT but may be less stable.
    niter : int
        Maximum number of Sinkhorn iterations.
    scale : bool
        Divide costs by T.
    benchmark : bool
        If ``True``, use the standard L² cost (no h, M modification); useful
        for computing the SinkhornGAN baseline.

    Returns
    -------
    torch.Tensor (scalar)
        Sinkhorn transport cost.
    """
    n = x.shape[0]

    # Build the pairwise cost matrix
    if benchmark:
        C = cost_matrix(x, y, scale=scale)               # standard L² cost
    else:
        C = modified_cost(x, y, h, M, scale=scale)        # COT-GAN modified cost

    # Uniform marginals (equal weight to every sample)
    mu = torch.ones(n, requires_grad=False, device=x.device) / n
    nu = torch.ones(n, requires_grad=False, device=x.device) / n

    # Sinkhorn acceleration parameters
    tau = -0.8     # Nesterov extrapolation coefficient

    # Helper closures for the log-domain Sinkhorn iterations
    def M_fn(u, v):
        """Log-domain modified cost: (-C + u_i + v_j) / ε"""
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        """Log-sum-exp over the last dimension."""
        return torch.logsumexp(A, dim=-1, keepdim=True)

    def ave(u, u1):
        """Nesterov extrapolation (barycentric update)."""
        return tau * u + (1 - tau) * u1

    # Initialise dual variables
    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)
    thresh = 1e-4    # convergence tolerance

    for _ in range(niter):
        u_prev = u
        u = epsilon * (torch.log(mu) - lse(M_fn(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M_fn(u, v).t()).squeeze()) + v
        err = (u - u_prev).abs().sum()
        if err.item() < thresh:
            break

    # Recover the optimal transport plan and compute the primal cost
    pi = torch.exp(M_fn(u, v))        # (N, N) transport plan
    cost = torch.sum(pi * C)
    return cost


# ---------------------------------------------------------------------------
# Martingale regularisation
# ---------------------------------------------------------------------------

def scale_invariante_martingale_regularization(
    M: torch.Tensor, reg_lam: float, scale: bool = False
) -> torch.Tensor:
    """
    Compute the martingale regularisation penalty p_M.

    p_M penalises the discriminator output M for deviating from the martingale
    condition (i.e. E[M(t+1) | history] = M(t)).  Specifically, it encourages
    the *mean* increment ΔM across the batch to be zero at every time step::

        p_M = reg_lam * Σ_t |  (1/N) Σ_i ΔM_i(t) / std(M) |

    Parameters
    ----------
    M : torch.Tensor, shape (N, T, J)
        Discriminator M output for a batch of N trajectories.
    reg_lam : float
        Scaling coefficient for the martingale penalty.
    scale : bool
        Divide by T.

    Returns
    -------
    torch.Tensor (scalar)
        Martingale penalty p_M.
    """
    m, t, j = M.shape
    N = M[:, 1:, :] - M[:, :-1, :]                       # increments: (N, T-1, J)
    N_std = N / (torch.std(M, dim=(0, 1)) + 1e-6)         # standardised increments

    # Mean increment across batch: shape (T-1, J)
    sum_m_std = torch.sum(N_std, dim=0) / m

    # Sum of absolute mean increments across time and feature dimensions
    sum_across_paths = torch.sum(torch.abs(sum_m_std))
    if scale:
        sum_across_paths = sum_across_paths / t

    return reg_lam * sum_across_paths


# ---------------------------------------------------------------------------
# Combined COT-GAN loss (mixed Sinkhorn)
# ---------------------------------------------------------------------------

def compute_mixed_sinkhorn_loss(
    f_real: torch.Tensor,
    f_fake: torch.Tensor,
    m_real: torch.Tensor,
    m_fake: torch.Tensor,
    h_fake: torch.Tensor,
    sinkhorn_eps: float,
    sinkhorn_l: int,
    f_real_p: torch.Tensor,
    f_fake_p: torch.Tensor,
    m_real_p: torch.Tensor,
    h_real_p: torch.Tensor,
    h_fake_p: torch.Tensor,
    scale: bool = False,
) -> torch.Tensor:
    """
    Compute the **mixed Sinkhorn COT-GAN loss**.

    The loss is an unbiased estimator of the squared Cauchy-Sinkhorn distance::

        L = S(x, y) + S(x', y') - S(x, x') - S(y, y')

    where x, x' are two independent real batches and y, y' are two independent
    generated batches.  This mixed formulation reduces variance compared to the
    naïve ``2*S(x,y) - S(x,x') - S(y,y')``.

    Parameters
    ----------
    f_real, f_real_p : torch.Tensor, shape (N, T, F)
        Two independent real data batches (x and x').
    f_fake, f_fake_p : torch.Tensor, shape (N, T, F)
        Two independent generated data batches (y and y').
    m_real, m_real_p : torch.Tensor, shape (N, T, J)
        Discriminator M output for real batches (x and x').
    m_fake : torch.Tensor, shape (N, T, J)
        Discriminator M output for generated batch y.
    h_fake, h_fake_p : torch.Tensor, shape (N, T, J)
        Discriminator H output for generated batches y and y'.
    h_real_p : torch.Tensor, shape (N, T, J)
        Discriminator H output for real batch x'.
    sinkhorn_eps : float
        Sinkhorn epsilon regularisation.
    sinkhorn_l : int
        Number of Sinkhorn iterations.
    scale : bool
        Divide costs by T.

    Returns
    -------
    torch.Tensor (scalar)
        Mixed Sinkhorn COT-GAN loss.
    """
    # Flatten spatial dimensions into features for Sinkhorn
    f_real   = f_real.reshape(f_real.shape[0], f_real.shape[1], -1)
    f_fake   = f_fake.reshape(f_fake.shape[0], f_fake.shape[1], -1)
    f_real_p = f_real_p.reshape(f_real_p.shape[0], f_real_p.shape[1], -1)
    f_fake_p = f_fake_p.reshape(f_fake_p.shape[0], f_fake_p.shape[1], -1)

    loss_xy  = compute_sinkhorn(f_real,   f_fake,   h_fake,   m_real,   sinkhorn_eps, sinkhorn_l, scale=scale)
    loss_xyp = compute_sinkhorn(f_real_p, f_fake_p, h_fake_p, m_real_p, sinkhorn_eps, sinkhorn_l, scale=scale)
    loss_xx  = compute_sinkhorn(f_real,   f_real_p, h_real_p, m_real,   sinkhorn_eps, sinkhorn_l, scale=scale)
    loss_yy  = compute_sinkhorn(f_fake,   f_fake_p, h_fake_p, m_fake,   sinkhorn_eps, sinkhorn_l, scale=scale)

    return loss_xy + loss_xyp - loss_xx - loss_yy
