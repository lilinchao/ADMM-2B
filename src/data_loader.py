"""
Data loading and preprocessing for spatio-temporal tensor imputation experiments.

Supports:
  - METR-LA traffic speed dataset (207 sensors × 288 time-slots × days)
  - Synthetic data for sanity checks
  - Beijing PM2.5 air quality data (secondary dataset)

Tensor shape convention: (I, J, K) = (sensors/locations, time_slots_per_day, days)
"""

import numpy as np
import os
import urllib.request
import zipfile
import h5py
import json
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Missing mask generators
# ---------------------------------------------------------------------------

def random_missing_mask(shape, missing_rate, seed=0):
    """Return boolean mask: True = observed, False = missing (random pattern)."""
    rng = np.random.default_rng(seed)
    mask = rng.random(shape) >= missing_rate
    # Ensure at least one observed entry per slice to avoid degenerate kriging
    for i in range(shape[0]):
        if not mask[i].any():
            idx = rng.integers(0, mask[i].size)
            mask[i].flat[idx] = True
    return mask


def block_missing_mask(shape, missing_rate, block_axis=1, seed=0):
    """Return boolean mask with contiguous block removed along block_axis."""
    rng = np.random.default_rng(seed)
    mask = np.ones(shape, dtype=bool)
    n = shape[block_axis]
    block_len = int(n * missing_rate)
    start = rng.integers(0, n - block_len + 1)
    slices = [slice(None)] * len(shape)
    slices[block_axis] = slice(start, start + block_len)
    mask[tuple(slices)] = False
    return mask


def block_missing_mask_multi(shape, missing_rate, seed=0):
    """
    Return boolean mask with multiple random 3D blocks set as missing.

    Blocks are placed repeatedly until the target missing rate is reached.
    Each block has random dimensions (max 1/3 of each dimension, min 2).
    Follows the approach in ADMMProject-2 _generate_block_mask.

    True = observed, False = missing.
    """
    rng = np.random.RandomState(seed)
    mask = np.ones(shape, dtype=bool)
    total_elements = mask.size
    target_missing = int(total_elements * missing_rate)
    current_missing = 0

    max_attempts = 10000
    attempts = 0

    while current_missing < target_missing and attempts < max_attempts:
        attempts += 1

        block_dims = []
        for dim_size in shape:
            max_size = max(1, dim_size // 3)
            min_size = min(2, max_size)
            if max_size <= min_size:
                size = max_size
            else:
                size = rng.randint(min_size, max_size + 1)
            block_dims.append(size)

        start_indices = []
        for dim_size, block_dim in zip(shape, block_dims):
            high = dim_size - block_dim + 1
            start_index = rng.randint(0, max(1, high))
            start_indices.append(start_index)

        block_slices = tuple(
            slice(start, start + length)
            for start, length in zip(start_indices, block_dims)
        )
        mask[block_slices] = False
        current_missing = int((~mask).sum())

    actual_rate = current_missing / total_elements
    print(
        f"Block mask: target={missing_rate:.2f}, actual={actual_rate:.4f} "
        f"({current_missing}/{total_elements})"
    )
    return mask


# ---------------------------------------------------------------------------
# METR-LA dataset
# ---------------------------------------------------------------------------

METRLA_URL = (
    "https://zenodo.org/record/5724979/files/METR-LA.h5?download=1"
)
METRLA_LOCAL = DATA_DIR / "METR-LA.h5"


def download_metrla():
    """Download METR-LA h5 file if not already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if METRLA_LOCAL.exists():
        return
    print(f"Downloading METR-LA dataset to {METRLA_LOCAL} ...")
    urllib.request.urlretrieve(METRLA_URL, METRLA_LOCAL)
    print("Download complete.")


def load_metrla(max_days=None, n_slots=288):
    """
    Load METR-LA as a 3-D tensor (sensors × time_slots_per_day × days).

    Parameters
    ----------
    max_days : int or None
        Truncate to this many days (for fast sanity runs).
    n_slots : int
        Number of 5-minute intervals per day (288 = 24 h × 12 per hour).

    Returns
    -------
    X : ndarray, shape (I, J, K) = (207, 288, n_days)
    locs : ndarray, shape (I, 2) — sensor lat/lon (if available, else None)
    """
    download_metrla()
    with h5py.File(METRLA_LOCAL, "r") as f:
        # Shape: (time_steps, sensors) — values in mph
        data = f["df"]["block0_values"][:]   # may vary by h5 structure
        # Try alternative key structure
        if data.ndim == 1:
            speeds = f["speed"][:]
        else:
            speeds = data  # (T, I)

    I = speeds.shape[1]  # number of sensors
    T = speeds.shape[0]  # total time steps
    n_days = T // n_slots
    if max_days is not None:
        n_days = min(n_days, max_days)
    T_use = n_days * n_slots

    # Reshape → (I, J, K) = (sensors, slots/day, days)
    X = speeds[:T_use, :I].T  # (I, T_use)
    X = X.reshape(I, n_slots, n_days)  # (I, J, K)

    # Normalize to [0, 1] range
    xmin, xmax = X.min(), X.max()
    X = (X - xmin) / (xmax - xmin + 1e-8)

    # Sensor locations — not in this h5, return placeholder grid
    locs = None
    locs_file = DATA_DIR / "metrla_locs.npy"
    if locs_file.exists():
        locs = np.load(locs_file)
    else:
        # Placeholder: arrange sensors on a regular grid
        side = int(np.ceil(np.sqrt(I)))
        grid = np.array([[r, c] for r in range(side) for c in range(side)])[:I]
        locs = grid.astype(float)

    return X, locs


def load_metrla_fallback(max_days=None, n_slots=288):
    """
    Fallback loader: tries multiple known METR-LA h5 key structures.
    Returns X (I, J, K) normalized, locs (I, 2).
    """
    download_metrla()
    with h5py.File(METRLA_LOCAL, "r") as f:
        keys = list(f.keys())
        speeds = None
        # Common structures
        for path in [
            ("df", "block0_values"),
            ("speed",),
            ("data",),
        ]:
            try:
                obj = f
                for k in path:
                    obj = obj[k]
                arr = obj[:]
                if arr.ndim == 2:
                    speeds = arr
                    break
            except (KeyError, TypeError):
                continue

        if speeds is None:
            # Last resort: grab first 2D dataset
            def find_2d(obj, results):
                if hasattr(obj, "keys"):
                    for k in obj:
                        find_2d(obj[k], results)
                elif hasattr(obj, "shape") and len(obj.shape) == 2:
                    results.append(obj[:])
            results = []
            find_2d(f, results)
            if results:
                speeds = results[0]

    if speeds is None:
        raise RuntimeError(
            "Could not parse METR-LA h5 file. "
            "Please place a valid METR-LA.h5 in data/ directory."
        )

    # Ensure shape is (T, I)
    if speeds.shape[0] < speeds.shape[1]:
        speeds = speeds.T  # (T, I)

    I = speeds.shape[1]
    T = speeds.shape[0]
    n_days = T // n_slots
    if max_days is not None:
        n_days = min(n_days, max_days)
    T_use = n_days * n_slots

    X = speeds[:T_use, :].T  # (I, T_use)
    X = X.reshape(I, n_slots, n_days)
    xmin, xmax = X.min(), X.max()
    X = (X - xmin) / (xmax - xmin + 1e-8)

    side = int(np.ceil(np.sqrt(I)))
    grid = np.array([[r, c] for r in range(side) for c in range(side)])[:I]
    locs = grid.astype(float)
    return X, locs


# ---------------------------------------------------------------------------
# Synthetic dataset (for sanity / unit tests)
# ---------------------------------------------------------------------------

def make_synthetic(I=20, J=24, K=10, rank=3, noise=0.05, seed=0):
    """
    Create a low-rank synthetic tensor with spatial structure.

    Returns
    -------
    X_true : ndarray (I, J, K) — ground truth
    locs   : ndarray (I, 2)    — spatial locations on [0,1]^2 grid
    """
    rng = np.random.default_rng(seed)
    # Factor matrices
    A = rng.standard_normal((I, rank))
    B = rng.standard_normal((J, rank))
    C = rng.standard_normal((K, rank))
    # CP tensor
    X = np.einsum("ir,jr,kr->ijk", A, B, C)
    X += noise * rng.standard_normal(X.shape)
    # Normalize
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)
    # Spatial locations
    side = int(np.ceil(np.sqrt(I)))
    grid = np.mgrid[0:1:complex(side), 0:1:complex(side)].reshape(2, -1).T
    locs = grid[:I]
    return X, locs


# ---------------------------------------------------------------------------
# Beijing PM2.5 secondary dataset (public UCI / KDD Cup 2018 style)
# ---------------------------------------------------------------------------

def make_pm25_synthetic(I=35, J=24, K=60, rank=4, seed=42):
    """
    Synthetic secondary dataset mimicking PM2.5 spatial structure.
    (Real data download is dataset-specific; synthetic stands in for testing.)
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((I, rank))
    B = rng.standard_normal((J, rank))
    C = rng.standard_normal((K, rank))
    X = np.einsum("ir,jr,kr->ijk", A, B, C)
    X += 0.1 * rng.standard_normal(X.shape)
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)
    locs = np.random.default_rng(seed).random((I, 2))
    return X, locs


# ---------------------------------------------------------------------------
# Data-driven spatial distance and MDS embedding
# ---------------------------------------------------------------------------

def compute_sensor_distance_matrix(X, method="pearson", min_obs_overlap=10):
    """
    Compute I x I sensor distance matrix from data correlations.

    For each pair of sensors (i1, i2), compute Pearson correlation of their
    full temporal profiles (J*K length vectors), using only positions where
    both sensors have non-zero observations. Distance = 1 - |r|.

    Parameters
    ----------
    X : ndarray (I, J, K) — raw or normalized tensor
    method : str — "pearson" (default) or "cosine"
    min_obs_overlap : int — minimum jointly-observed positions per pair

    Returns
    -------
    D : ndarray (I, I) — symmetric distance matrix, D[i,i] = 0
    """
    I, J, K = X.shape
    # Reshape to (I, J*K)
    X_flat = X.reshape(I, -1)
    nonzero = X_flat != 0

    D = np.ones((I, I), dtype=float)

    for i1 in range(I):
        mask1 = nonzero[i1]
        for i2 in range(i1 + 1, I):
            mask2 = nonzero[i2]
            overlap = mask1 & mask2
            n_overlap = int(overlap.sum())
            if n_overlap < min_obs_overlap:
                continue  # keep D[i1,i2] = 1.0 (max distance)

            v1 = X_flat[i1, overlap]
            v2 = X_flat[i2, overlap]

            if method == "cosine":
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 < 1e-12 or norm2 < 1e-12:
                    continue
                r = np.dot(v1, v2) / (norm1 * norm2)
            else:  # pearson
                v1c = v1 - v1.mean()
                v2c = v2 - v2.mean()
                s1 = np.linalg.norm(v1c)
                s2 = np.linalg.norm(v2c)
                if s1 < 1e-12 or s2 < 1e-12:
                    continue
                r = np.dot(v1c, v2c) / (s1 * s2)

            dist = 1.0 - abs(r)
            D[i1, i2] = dist
            D[i2, i1] = dist

    np.fill_diagonal(D, 0.0)
    return D


def embed_sensor_coordinates(D, n_dims=2, normalize=True):
    """
    Embed sensor distance matrix into Euclidean coordinates via classical MDS.

    Parameters
    ----------
    D : ndarray (I, I) — pairwise distance matrix
    n_dims : int — target embedding dimension (2 recommended)
    normalize : bool — scale coordinates to [0, 1] per axis

    Returns
    -------
    coords : ndarray (I, n_dims) — sensor coordinates in embedded space
    """
    I = D.shape[0]
    # Double-center the squared distance matrix
    D2 = D ** 2
    H = np.eye(I) - np.ones((I, I)) / I  # centering matrix
    B = -0.5 * H @ D2 @ H  # inner product matrix

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Sort in descending order, take top n_dims
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clamp negative eigenvalues to 0
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Take top n_dims
    lam = np.sqrt(eigenvalues[:n_dims])
    V = eigenvectors[:, :n_dims]
    coords = V * lam[np.newaxis, :]  # (I, n_dims)

    if normalize:
        for d in range(n_dims):
            lo, hi = coords[:, d].min(), coords[:, d].max()
            if hi - lo > 1e-12:
                coords[:, d] = (coords[:, d] - lo) / (hi - lo)
            else:
                coords[:, d] = 0.5

    return coords


def build_meaningful_coords(X, tau_j=3.0, tau_k=1.0,
                            spatial_weight=1.0, n_spatial_dims=2,
                            distance_method="pearson"):
    """
    Build data-driven sensor coordinates for spatio-temporal kriging.

    Parameters
    ----------
    X : ndarray (I, J, K)
    tau_j : float — day-axis scale factor
    tau_k : float — time-slot-axis scale factor
    spatial_weight : float — scaling for spatial dims vs temporal
    n_spatial_dims : int — dimensionality of MDS embedding
    distance_method : str — "pearson" or "cosine"

    Returns
    -------
    sensor_coords : ndarray (I, n_spatial_dims) — MDS-embedded coordinates
    """
    D = compute_sensor_distance_matrix(X, method=distance_method)
    sensor_coords = embed_sensor_coordinates(D, n_dims=n_spatial_dims)
    sensor_coords *= spatial_weight
    return sensor_coords


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

def load_tensor_mat(max_days=None, compute_spatial=True):
    """
    Load data/tensor.mat — shape (214, 61, 144): sensors × time_slots × days.

    Returns
    -------
    X : ndarray (I, J, K) normalized to [0, 1]
    locs : ndarray (I, 2) — data-driven MDS coordinates if compute_spatial=True,
            else placeholder grid
    """
    import scipy.io
    mat_path = DATA_DIR / "tensor.mat"
    mat = scipy.io.loadmat(str(mat_path))
    X_raw = mat["tensor"].astype(np.float64)  # (214, 61, 144)

    # Compute spatial coordinates from FULL tensor before truncation
    if compute_spatial:
        cache_path = DATA_DIR / "sensor_coords_mds.npy"
        if cache_path.exists():
            locs = np.load(str(cache_path))
        else:
            locs = build_meaningful_coords(X_raw)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), locs)
    else:
        I = X_raw.shape[0]
        side = int(np.ceil(np.sqrt(I)))
        grid = np.array([[r, c] for r in range(side)
                         for c in range(side)])[:I]
        locs = grid.astype(float)

    if max_days is not None:
        X_raw = X_raw[:, :, :max_days]

    # Normalize
    xmin, xmax = X_raw.min(), X_raw.max()
    data_range = float(xmax - xmin)
    X = (X_raw - xmin) / (data_range + 1e-8)

    return X, locs, data_range


DATASETS = {
    "synthetic_small": lambda: (*make_synthetic(I=20, J=24, K=10), 1.0),
    "synthetic_medium": lambda: (*make_synthetic(I=50, J=48, K=30), 1.0),
    "pm25_synthetic": lambda: (*make_pm25_synthetic(), 1.0),
    "metrla": lambda: (*load_metrla_fallback(max_days=30), 1.0),
    "metrla_full": lambda: (*load_metrla_fallback(), 1.0),
    "metrla_toy": lambda: (*make_synthetic(I=50, J=48, K=7, rank=5), 1.0),
    "tensor_mat": load_tensor_mat,
    "tensor_mat_small": lambda: load_tensor_mat(max_days=30),
}


def get_dataset(name):
    """Return (X_true, locs, data_range) for named dataset."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}")
    return DATASETS[name]()
