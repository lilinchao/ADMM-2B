"""
Direct interpolation baselines for spatio-temporal tensor imputation.

Tensor shape: (I, J, K) = (road segments, days, time slots per day).

Methods
-------
mean_fill          : fill all missing positions with global observed mean
daily_profile      : for each (i, k), fill missing days with mean of observed days
linear_time        : linear interp along k (time-of-day) per (i, j) slice;
                     fallback to daily_profile then mean_fill for whole-day gaps
locf_time          : last-observation-carried-forward along k per (i, j) slice,
                     with backward fill for leading gaps; same fallback chain
"""

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _global_mean(X_obs: np.ndarray, mask: np.ndarray) -> float:
    obs = X_obs[mask]
    return float(obs.mean()) if obs.size > 0 else 0.0


def _daily_profile_matrix(X_obs: np.ndarray, mask: np.ndarray,
                           fallback: float) -> np.ndarray:
    """Return (I, K) matrix of mean observed values per (sensor, time-slot)."""
    I, J, K = X_obs.shape
    profile = np.full((I, K), fallback)
    for i in range(I):
        for k in range(K):
            obs_j = np.where(mask[i, :, k])[0]
            if len(obs_j) > 0:
                profile[i, k] = float(X_obs[i, obs_j, k].mean())
    return profile


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mean_fill(X_obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill all missing positions with global observed mean."""
    Z = X_obs.copy()
    mu = _global_mean(X_obs, mask)
    Z[~mask] = mu
    return Z


def daily_profile(X_obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    For each (road i, time-slot k), fill missing days j with the mean of
    observed days for that (i, k) pair.
    Falls back to global mean when no day in column (i, k) is observed.
    """
    Z = X_obs.copy()
    mu = _global_mean(X_obs, mask)
    profile = _daily_profile_matrix(X_obs, mask, fallback=mu)

    miss_i, miss_j, miss_k = np.where(~mask)
    Z[miss_i, miss_j, miss_k] = profile[miss_i, miss_k]
    return Z


def linear_time(X_obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Linear interpolation along the time-slot (k) axis, per (road i, day j) slice.

    Cascade:
    1. Linear interp along k if the slice has ≥ 2 observed points.
       Constant extrapolation at both ends (clamp to nearest observed value).
    2. Single-point slices: fill with that one observed value.
    3. Whole-day gaps (no observed k in that slice): fill from daily_profile.
    4. Any remaining gaps (whole (i,k) column unobserved): global mean.
    """
    I, J, K = X_obs.shape
    Z = X_obs.copy()
    filled = mask.copy()                        # track filled positions
    mu = _global_mean(X_obs, mask)

    k_idx = np.arange(K, dtype=float)

    # Step 1 & 2: per (i, j) slice
    for i in range(I):
        for j in range(J):
            obs_k = np.where(mask[i, j, :])[0]
            miss_k = np.where(~mask[i, j, :])[0]
            if miss_k.size == 0:
                continue
            if obs_k.size >= 2:
                f = interp1d(
                    obs_k.astype(float),
                    X_obs[i, j, obs_k],
                    kind="linear",
                    bounds_error=False,
                    fill_value=(X_obs[i, j, obs_k[0]], X_obs[i, j, obs_k[-1]]),
                )
                Z[i, j, miss_k] = f(miss_k.astype(float))
                filled[i, j, miss_k] = True
            elif obs_k.size == 1:
                Z[i, j, miss_k] = X_obs[i, j, obs_k[0]]
                filled[i, j, miss_k] = True
            # else: whole-day gap — handled next

    # Step 3: daily profile for whole-day gaps
    unfilled = ~filled
    if unfilled.any():
        profile = _daily_profile_matrix(X_obs, mask, fallback=mu)
        uf_i, uf_j, uf_k = np.where(unfilled)
        Z[uf_i, uf_j, uf_k] = profile[uf_i, uf_k]
        filled[uf_i, uf_j, uf_k] = True

    # Step 4: global mean for anything still unfilled
    unfilled2 = ~filled
    if unfilled2.any():
        Z[unfilled2] = mu

    return Z


def locf_time(X_obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Last-observation-carried-forward (LOCF) along time-slot (k) axis,
    then backward fill for leading gaps, per (road i, day j) slice.

    Same fallback cascade as linear_time for whole-day gaps.
    """
    I, J, K = X_obs.shape
    Z = X_obs.copy()
    filled = mask.copy()
    mu = _global_mean(X_obs, mask)

    for i in range(I):
        for j in range(J):
            obs_k = np.where(mask[i, j, :])[0]
            miss_k = np.where(~mask[i, j, :])[0]
            if miss_k.size == 0 or obs_k.size == 0:
                continue

            # Forward fill
            last_val = None
            for k in range(K):
                if mask[i, j, k]:
                    last_val = X_obs[i, j, k]
                elif last_val is not None:
                    Z[i, j, k] = last_val
                    filled[i, j, k] = True

            # Backward fill (leading missing values)
            first_val = None
            for k in range(K - 1, -1, -1):
                if mask[i, j, k]:
                    first_val = X_obs[i, j, k]
                elif not filled[i, j, k] and first_val is not None:
                    Z[i, j, k] = first_val
                    filled[i, j, k] = True

    # Daily profile fallback for whole-day gaps
    unfilled = ~filled
    if unfilled.any():
        profile = _daily_profile_matrix(X_obs, mask, fallback=mu)
        uf_i, uf_j, uf_k = np.where(unfilled)
        Z[uf_i, uf_j, uf_k] = profile[uf_i, uf_k]
        filled[uf_i, uf_j, uf_k] = True

    if (~filled).any():
        Z[~filled] = mu

    return Z
