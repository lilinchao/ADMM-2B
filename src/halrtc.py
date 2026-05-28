"""
HaLRTC: High-accuracy Low-Rank Tensor Completion.

Implements the HaLRTC algorithm from:
  Liu et al. (2013), "Tensor Completion for Estimating Missing Values
  in Visual Data," IEEE TPAMI.

Minimizes sum of nuclear norms of unfoldings (convex relaxation of tensor rank).
Uses ADMM on each unfolding via matrix singular value thresholding.
"""

import numpy as np


def unfold(X, mode):
    """Mode-n unfolding of tensor X. Returns matrix (n_mode, rest)."""
    return np.reshape(
        np.moveaxis(X, mode, 0),
        (X.shape[mode], -1),
    )


def fold(M, mode, shape):
    """Inverse of unfold: reconstruct tensor from mode-n unfolding."""
    full_shape = [shape[mode]] + [shape[i] for i in range(len(shape)) if i != mode]
    T = np.reshape(M, full_shape)
    return np.moveaxis(T, 0, mode)


def svt(M, tau):
    """Singular value thresholding: shrink singular values by tau."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    return U @ np.diag(s_thresh) @ Vt


def halrtc(
    X,
    mask,
    alpha=None,
    rho=1e-4,
    max_iter=500,
    tol=1e-4,
    verbose=False,
):
    """
    HaLRTC tensor completion via ADMM on nuclear norm of unfoldings.

    Parameters
    ----------
    X    : ndarray (I, J, K) — tensor with any values at missing positions
    mask : ndarray (I, J, K) bool — True = observed
    alpha : list of 3 weights summing to 1 (default: uniform [1/3, 1/3, 1/3])
    rho  : float — ADMM penalty / step size
    max_iter : int
    tol  : float — convergence threshold on relative change of M
    verbose : bool

    Returns
    -------
    M : ndarray (I, J, K) — completed tensor
    history : list of objective values (approximation)
    """
    shape = X.shape
    n_modes = len(shape)
    if alpha is None:
        alpha = [1.0 / n_modes] * n_modes

    # Initialize
    M = X.copy()
    M[~mask] = 0.0

    # Y_n: auxiliary unfolding matrices (one per mode)
    Y = [unfold(M, n) for n in range(n_modes)]
    # Z_n: dual variables
    Z = [np.zeros_like(y) for y in Y]

    history = []

    for it in range(max_iter):
        M_prev = M.copy()

        # Update each Y_n via SVT
        for n in range(n_modes):
            M_unfold = unfold(M, n)
            tau = alpha[n] / rho
            Y[n] = svt(M_unfold + Z[n], tau)

        # Update M: average of (Y_n - Z_n) folded back, then enforce observed data
        M_new = np.zeros(shape)
        for n in range(n_modes):
            M_new += fold(Y[n] - Z[n], n, shape)
        M_new /= n_modes
        M_new[mask] = X[mask]  # enforce observed entries
        M = M_new

        # Update dual variables
        for n in range(n_modes):
            Z[n] = Z[n] + unfold(M, n) - Y[n]

        # Convergence
        rel_change = np.linalg.norm(M - M_prev) / (np.linalg.norm(M_prev) + 1e-12)
        # Approximate objective: sum of nuclear norms
        obj = sum(
            alpha[n] * np.sum(np.linalg.svd(unfold(M, n), compute_uv=False))
            for n in range(n_modes)
        )
        history.append(obj)

        if verbose:
            print(f"  HaLRTC iter {it+1:3d}: rel_change={rel_change:.2e}  obj={obj:.4f}")
        if rel_change < tol and it > 5:
            break

    return M, history
