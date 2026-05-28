"""
CP Tensor Decomposition via Alternating Least Squares (ALS).

Implements:
  - CP-ALS for masked tensor completion (observed entries only)
  - L2 regularization on factor matrices
  - Multiple random restarts
  - Returns reconstructed tensor

Reference: Kolda & Bader (2009), SIAM Review.
"""

import numpy as np


def cp_als(
    X,
    mask,
    rank=10,
    beta=0.01,
    max_iter=200,
    tol=1e-4,
    seed=0,
    verbose=False,
):
    """
    CP decomposition via ALS on observed entries (masked tensor completion).

    Minimizes:
        sum_{(i,j,k) in Omega} (X_ijk - sum_r A_ir B_jr C_kr)^2
        + beta * (||A||_F^2 + ||B||_F^2 + ||C||_F^2)

    Parameters
    ----------
    X    : ndarray (I, J, K)
    mask : ndarray (I, J, K) bool — True = observed
    rank : int — CP rank R
    beta : float — L2 regularization
    max_iter : int
    tol : float — relative objective change for convergence
    seed : int
    verbose : bool

    Returns
    -------
    X_hat : ndarray (I, J, K) — reconstructed tensor
    factors : (A, B, C) each ndarray
    history : list of objective values
    """
    I, J, K = X.shape
    rng = np.random.default_rng(seed)

    # Initialize factors
    A = rng.standard_normal((I, rank))
    B = rng.standard_normal((J, rank))
    C = rng.standard_normal((K, rank))

    # Observed values
    obs_idx = np.where(mask)
    X_obs = X[obs_idx]  # (N_obs,)

    history = []
    prev_obj = np.inf

    for it in range(max_iter):
        # Reconstruct full tensor
        X_hat = _cp_reconstruct(A, B, C)

        # -- Update A --
        # For each i: A[i,:] minimizes sum_{j,k in Omega_i} (X_ijk - A[i] @ (B[j] * C[k]).T)^2 + beta ||A[i]||^2
        for i in range(I):
            jk_mask = mask[i]  # (J, K)
            if not jk_mask.any():
                continue
            js, ks = np.where(jk_mask)
            # Khatri-Rao: (N_i, R) — one row per observed (j,k) pair
            V = B[js] * C[ks]  # (N_i, R)
            rhs = X[i][jk_mask]  # (N_i,)
            A[i] = np.linalg.solve(V.T @ V + beta * np.eye(rank), V.T @ rhs)

        # -- Update B --
        for j in range(J):
            ik_mask = mask[:, j, :]
            if not ik_mask.any():
                continue
            is_, ks = np.where(ik_mask)
            V = A[is_] * C[ks]
            rhs = X[:, j, :][ik_mask]
            B[j] = np.linalg.solve(V.T @ V + beta * np.eye(rank), V.T @ rhs)

        # -- Update C --
        for k in range(K):
            ij_mask = mask[:, :, k]
            if not ij_mask.any():
                continue
            is_, js = np.where(ij_mask)
            V = A[is_] * B[js]
            rhs = X[:, :, k][ij_mask]
            C[k] = np.linalg.solve(V.T @ V + beta * np.eye(rank), V.T @ rhs)

        # Objective
        X_hat = _cp_reconstruct(A, B, C)
        resid = X_obs - X_hat[obs_idx]
        obj = 0.5 * np.dot(resid, resid) + 0.5 * beta * (
            np.sum(A ** 2) + np.sum(B ** 2) + np.sum(C ** 2)
        )
        history.append(obj)

        rel_change = abs(prev_obj - obj) / (abs(prev_obj) + 1e-12)
        if verbose:
            print(f"  ALS iter {it+1:3d}: obj={obj:.6f}  rel_change={rel_change:.2e}")
        if rel_change < tol and it > 0:
            break
        prev_obj = obj

    return X_hat, (A, B, C), history


def cp_als_best_of(
    X,
    mask,
    rank=10,
    beta=0.01,
    max_iter=200,
    tol=1e-4,
    n_restarts=3,
    verbose=False,
):
    """Run cp_als with n_restarts random seeds; return best (lowest obj)."""
    best_Xhat, best_factors, best_history = None, None, None
    best_obj = np.inf
    for seed in range(n_restarts):
        Xhat, factors, history = cp_als(
            X, mask, rank=rank, beta=beta,
            max_iter=max_iter, tol=tol, seed=seed, verbose=verbose,
        )
        if history and history[-1] < best_obj:
            best_obj = history[-1]
            best_Xhat = Xhat
            best_factors = factors
            best_history = history
    return best_Xhat, best_factors, best_history


def _cp_reconstruct(A, B, C):
    """Reconstruct tensor from CP factors via einsum."""
    return np.einsum("ir,jr,kr->ijk", A, B, C)
