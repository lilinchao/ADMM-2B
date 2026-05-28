"""
Dataset configuration for multi-dataset experiments.

Provides a unified interface for loading different datasets
(Guangzhou traffic, PeMSD7 traffic, EPA CA air quality)
across all experiment scripts.
"""

import numpy as np
import scipy.io
from pathlib import Path


DATASETS = {
    "guangzhou": {
        "data_dir": "data",
        "coords_file": "sensor_coords_mds.npy",
        "default_deploy_numbers": [20, 50, 100, 150, 190],
        "hpo_cache_dir": "results/hpo_cache",
        "dl_cache_dir": "results/dl_model_cache",
        "output_suffix": "",
        "description": "Guangzhou traffic speed (214 sensors x 61 slots x 144 days)",
    },
    "pemsd7": {
        "data_dir": "data/PeMSD7",
        "coords_file": "sensor_coords.npy",
        "default_deploy_numbers": [20, 50, 100, 150, 200],
        "hpo_cache_dir": "results/hpo_cache_pemsd7",
        "dl_cache_dir": "results/dl_model_cache_pemsd7",
        "output_suffix": "_pemsd7",
        "description": "PeMSD7 traffic speed (228 sensors x 96 slots x 44 days)",
    },
    "epa_ca": {
        "data_dir": "data/EPA_CA",
        "coords_file": "sensor_coords.npy",
        "default_deploy_numbers": [10, 25, 50, 75, 100],
        "hpo_cache_dir": "results/hpo_cache_epa_ca",
        "dl_cache_dir": "results/dl_model_cache_epa_ca",
        "output_suffix": "_epa_ca",
        "description": "EPA CA PM2.5 air quality (~127 sensors x 24 hours x ~90 days)",
    },
}


def add_dataset_arg(parser):
    """Add --dataset argument to an argparse parser."""
    names = list(DATASETS.keys())
    descs = [f"{k}: {v['description']}" for k, v in DATASETS.items()]
    parser.add_argument(
        "--dataset", choices=names, default="guangzhou",
        help="Dataset to use: " + "; ".join(descs),
    )


def get_config(dataset_name):
    """Get configuration dict for a dataset."""
    return DATASETS[dataset_name]


def default_output_path(dataset_name, base_name):
    """Return default output path for a dataset and experiment base name.

    E.g., default_output_path("pemsd7", "experiment1") -> "results/experiment1_pemsd7.json"
    """
    suffix = DATASETS[dataset_name]["output_suffix"]
    stem = Path(base_name).stem
    ext = Path(base_name).suffix
    return f"results/{stem}{suffix}{ext}"


def load_dataset(project_root, dataset_name):
    """Load a dataset and return (X_raw, sensor_coords, data_range, xmin, config).

    For datasets with intrinsic NaN values (e.g., EPA), NaN positions are
    tracked separately and the raw tensor is filled with 0 at NaN positions.
    The original NaN mask is stored as an attribute on the returned X_raw
    (X_raw.nan_mask) for downstream use.

    Parameters
    ----------
    project_root : Path
    dataset_name : str
        One of the keys in DATASETS.

    Returns
    -------
    X_raw : ndarray, shape (I, J, K)
    sensor_coords : ndarray, shape (I, 2)
    data_range : float
    xmin : float
    config : dict
    """
    config = DATASETS[dataset_name]
    data_dir = project_root / config["data_dir"]

    mat = scipy.io.loadmat(str(data_dir / "tensor.mat"))
    X_raw = mat["tensor"].astype(np.float64)

    # Handle intrinsic NaN values
    nan_mask = np.isnan(X_raw)
    intrinsic_missing_rate = nan_mask.sum() / nan_mask.size
    if intrinsic_missing_rate > 0:
        print(f"  Intrinsic missing rate: {intrinsic_missing_rate:.1%}, filling with 0 for normalization")
        X_raw[nan_mask] = 0.0

    xmin = float(X_raw.min())
    data_range = float(X_raw.max() - xmin)

    sensor_coords = np.load(str(data_dir / config["coords_file"]))

    print(f"  Dataset: {dataset_name} — {config['description']}")
    print(f"  Shape: {X_raw.shape}, data_range={data_range:.2f}")
    print(f"  Coords: {sensor_coords.shape}")
    if intrinsic_missing_rate > 0:
        print(f"  NaN mask: {nan_mask.sum():,} intrinsic missing entries")

    return X_raw, sensor_coords, data_range, xmin, config
