"""
Data splitting for sensor deployment experiments.

Splits sensors into train/val/test sets with strict separation:
- Test sensors: never observed, used only for final evaluation
- Val sensors: from deployment pool, temporarily hidden for hyperparameter tuning
- Train sensors: observed data for algorithm input

This ensures no information leakage during hyperparameter optimization.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SensorSplit:
    """Sensor index sets for train/val/test splitting."""
    train: np.ndarray  # sensors available to the algorithm
    val: np.ndarray    # sensors hidden from algorithm, used for HPO
    test: np.ndarray   # sensors never observed, used for final reporting
    deploy_pool: np.ndarray  # train + val = candidate deployment sensors

    def summary(self):
        return (f"SensorSplit(train={len(self.train)}, val={len(self.val)}, "
                f"test={len(self.test)}, deploy_pool={len(self.deploy_pool)})")


def split_sensors(n_sensors, test_frac=0.3, val_frac_of_deploy=0.1, seed=42):
    """
    Split sensor indices into train/val/test.

    Parameters
    ----------
    n_sensors : int
        Total number of sensors (axis-0 size of tensor).
    test_frac : float
        Fraction of sensors reserved for testing (never observed).
    val_frac_of_deploy : float
        Fraction of deployment pool reserved for validation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    SensorSplit
    """
    rng = np.random.default_rng(seed)
    all_sensors = np.arange(n_sensors)

    # Step 1: split test from deploy pool
    n_test = max(1, int(n_sensors * test_frac))
    test_sensors = rng.choice(all_sensors, size=n_test, replace=False)
    deploy_pool = np.setdiff1d(all_sensors, test_sensors)

    # Step 2: split val from train within deploy pool
    n_val = max(1, int(len(deploy_pool) * val_frac_of_deploy))
    val_sensors = rng.choice(deploy_pool, size=n_val, replace=False)
    train_sensors = np.setdiff1d(deploy_pool, val_sensors)

    return SensorSplit(
        train=np.sort(train_sensors),
        val=np.sort(val_sensors),
        test=np.sort(test_sensors),
        deploy_pool=np.sort(deploy_pool),
    )


def make_deployment_mask(tensor_shape, split, n_deploy, seed=0):
    """
    Create observation mask for a given number of deployed sensors.

    Randomly selects n_deploy sensors from the deploy_pool (train+val).
    Test sensors are always unobserved. Val sensors are also unobserved
    (they are only used for HPO evaluation).

    Parameters
    ----------
    tensor_shape : tuple (I, J, K)
    split : SensorSplit
    n_deploy : int
        Number of sensors to deploy (sampled from deploy_pool).
    seed : int
        Random seed for deployment selection.

    Returns
    -------
    mask : ndarray (I, J, K) bool — True = observed
    deployed_sensors : ndarray — indices of deployed sensors
    """
    I, J, K = tensor_shape
    rng = np.random.default_rng(seed)

    n_deploy = min(n_deploy, len(split.deploy_pool))
    deployed_sensors = rng.choice(split.deploy_pool, size=n_deploy, replace=False)
    deployed_set = set(deployed_sensors)

    mask = np.zeros(tensor_shape, dtype=bool)
    for s in deployed_set:
        mask[s, :, :] = True

    return mask, deployed_sensors


def make_hpo_masks(tensor_shape, split, seed=0):
    """
    Create masks for hyperparameter optimization.

    For HPO, all train sensors are observed, val sensors are hidden.
    The algorithm runs on train_mask and we evaluate predictions at
    val sensor positions.

    Parameters
    ----------
    tensor_shape : tuple (I, J, K)
    split : SensorSplit
    seed : int (unused, kept for API consistency)

    Returns
    -------
    train_mask : ndarray bool — only train sensors observed
    val_positions : ndarray bool — True at val sensor positions (for evaluation)
    """
    I, J, K = tensor_shape

    train_mask = np.zeros(tensor_shape, dtype=bool)
    for s in split.train:
        train_mask[s, :, :] = True

    val_positions = np.zeros(tensor_shape, dtype=bool)
    for s in split.val:
        val_positions[s, :, :] = True

    return train_mask, val_positions


def make_test_mask(tensor_shape, split, n_deploy, seed=0):
    """
    Create mask for final test evaluation.

    Deploys n_deploy sensors from deploy_pool (train+val both observed),
    test sensors always unobserved.

    Parameters
    ----------
    tensor_shape : tuple (I, J, K)
    split : SensorSplit
    n_deploy : int
    seed : int

    Returns
    -------
    mask : ndarray bool — True at deployed sensor positions
    test_positions : ndarray bool — True at test sensor positions (for evaluation)
    deployed_sensors : ndarray
    """
    I, J, K = tensor_shape
    rng = np.random.default_rng(seed)

    n_deploy = min(n_deploy, len(split.deploy_pool))
    deployed_sensors = rng.choice(split.deploy_pool, size=n_deploy, replace=False)

    mask = np.zeros(tensor_shape, dtype=bool)
    for s in deployed_sensors:
        mask[s, :, :] = True

    test_positions = np.zeros(tensor_shape, dtype=bool)
    for s in split.test:
        test_positions[s, :, :] = True

    return mask, test_positions, deployed_sensors
