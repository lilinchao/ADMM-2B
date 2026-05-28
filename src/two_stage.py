"""
Two-stage pipeline baselines.

Two-stage A: Kriging → CP-TD
  1. Fill missing entries with kriging (variogram or legacy)
  2. Run CP-ALS on the filled tensor (all entries now observed)

Two-stage B: CP-TD → Kriging
  1. Fill missing entries with CP-ALS (masked tensor completion)
  2. Smooth with kriging at original missing positions

Supports sensor_coords and variogram_params for data-driven kriging.
"""

import numpy as np
from .kriging import SpatioTemporalKriging
from .cp_als   import cp_als_best_of


def two_stage_kriging_then_td(
    X, mask, locs, kriging_params,
    rank=10, beta=0.01, max_iter=200,
    n_neighbors=50, n_restarts=3,
    variogram_model="exponential",
    sensor_coords=None,
    variogram_params=None,
    kriging_mode="variogram",
    tau_j=3.0,
    tau_k=1.0,
):
    """
    Two-stage A: Kriging fill → CP-ALS refinement.

    Parameters
    ----------
    X, mask        : tensor and observed mask (I, J, K)
    locs           : (I, 2) — kept for API compat
    kriging_params : dict   — kept for API compat
    sensor_coords  : ndarray (I, 2) or None — MDS-embedded coordinates
    variogram_params : dict or None — fitted variogram
    kriging_mode   : str — "variogram", "kkt_index", or "idw"
    tau_j, tau_k   : float — temporal scale factors
    """
    krig = SpatioTemporalKriging(
        sensor_coords=sensor_coords,
        variogram_model=variogram_model,
        variogram_params=variogram_params,
        n_neighbors=n_neighbors,
        kriging_mode=kriging_mode,
        tau_j=tau_j,
        tau_k=tau_k,
    )
    krig.fit(X, mask)
    X_krig = krig.predict_full(X, mask)

    # Stage 2: CP-ALS on the fully-filled tensor
    full_mask = np.ones_like(mask)
    X_hat, _, _ = cp_als_best_of(
        X_krig, full_mask,
        rank=rank, beta=beta,
        max_iter=max_iter, tol=1e-4,
        n_restarts=n_restarts,
    )
    return X_hat


def two_stage_td_then_kriging(
    X, mask, locs, kriging_params,
    rank=10, beta=0.01, max_iter=200,
    n_neighbors=50, n_restarts=3,
    variogram_model="exponential",
    sensor_coords=None,
    variogram_params=None,
    kriging_mode="variogram",
    tau_j=3.0,
    tau_k=1.0,
):
    """
    Two-stage B: CP-ALS fill → Kriging smoothing.

    Parameters
    ----------
    Same as two_stage_kriging_then_td.
    """
    # Stage 1: masked CP-ALS
    X_td, _, _ = cp_als_best_of(
        X, mask,
        rank=rank, beta=beta,
        max_iter=max_iter, tol=1e-4,
        n_restarts=n_restarts,
    )

    # Stage 2: kriging smooth at original missing positions only
    # Use the original observed entries as reference for kriging,
    # and let kriging re-estimate the originally-missing entries.
    krig = SpatioTemporalKriging(
        sensor_coords=sensor_coords,
        variogram_model=variogram_model,
        variogram_params=variogram_params,
        n_neighbors=n_neighbors,
        kriging_mode=kriging_mode,
        tau_j=tau_j,
        tau_k=tau_k,
    )
    # Build a mask where only original observed entries are marked as observed
    # The CP-filled values at missing positions are treated as missing for kriging
    krig_mask = mask.copy()
    krig.fit(X_td, krig_mask)
    X_hat = krig.predict_full(X_td, krig_mask)

    # Preserve observed data exactly
    X_hat[mask] = X[mask]
    return X_hat
