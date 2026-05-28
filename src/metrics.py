"""Metrics for tensor imputation evaluation."""

import numpy as np


def mae(X_true, X_pred, mask_missing):
    """Mean Absolute Error at missing positions."""
    diff = np.abs(X_true[mask_missing] - X_pred[mask_missing])
    return float(diff.mean()) if diff.size > 0 else 0.0


def rmse(X_true, X_pred, mask_missing):
    """Root Mean Squared Error at missing positions."""
    diff = (X_true[mask_missing] - X_pred[mask_missing]) ** 2
    return float(np.sqrt(diff.mean())) if diff.size > 0 else 0.0


def mape(X_true, X_pred, mask_missing, eps=1e-8):
    """Mean Absolute Percentage Error at missing positions (skip near-zero true values)."""
    yt = X_true[mask_missing]
    yp = X_pred[mask_missing]
    valid = np.abs(yt) > eps
    if valid.sum() == 0:
        return float("nan")
    return float((np.abs(yt[valid] - yp[valid]) / np.abs(yt[valid])).mean())


def evaluate(X_true, X_pred, mask, data_range=1.0):
    """
    Compute all metrics at missing positions, optionally denormalized.

    Parameters
    ----------
    X_true : ndarray (I, J, K) — ground truth (normalized to [0,1])
    X_pred : ndarray (I, J, K) — imputed tensor (normalized)
    mask   : ndarray (I, J, K) bool — True = originally observed
    data_range : float — (xmax - xmin) for denormalization.
                 If 1.0, returns normalized-scale metrics.

    Returns
    -------
    dict with mae, rmse, mape (in original data scale if data_range != 1.0)
    """
    missing = ~mask
    mae_val = mae(X_true, X_pred, missing)
    rmse_val = rmse(X_true, X_pred, missing)
    mape_val = mape(X_true, X_pred, missing)
    return {
        "mae":  mae_val * data_range,
        "rmse": rmse_val * data_range,
        "mape": mape_val,
    }
