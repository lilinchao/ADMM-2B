"""
Deep learning baselines (BRITS, SAITS) via PyPOTS for spatial extrapolation.

Both methods are real PyTorch models trained with self-supervised learning on
observed sensor data. For unobserved sensors (spatial extrapolation), they have
zero training signal and can only extrapolate from cross-feature patterns
learned on observed sensors — demonstrating the fundamental limitation of
DL imputation methods in the sensor deployment scenario.
"""

import numpy as np
from pathlib import Path


def _identify_sensors(mask):
    """Return observed and unobserved sensor indices from mask (I, J, K)."""
    observed = np.where(mask[:, 0, 0])[0]
    unobserved = np.where(~mask[:, 0, 0])[0]
    return observed, unobserved


def _tensor_to_pypots(X_obs, mask):
    """Reshape (I, J, K) tensor to PyPOTS format (J, K, I) with NaN for missing.

    PyPOTS expects (n_samples, n_steps, n_features) with NaN indicating missing.
    We map: n_samples=J (days), n_steps=K (time_slots), n_features=I (sensors).
    """
    data = X_obs.copy().astype(np.float64)
    data[~mask] = np.nan
    # (I, J, K) -> (J, K, I)
    data_pypots = np.transpose(data, (1, 2, 0))
    return data_pypots


def _pypots_to_tensor(imputed, X_obs, mask, I, J, K):
    """Reshape PyPOTS output (J, K, I) back to (I, J, K) and restore observed entries."""
    # (J, K, I) -> (I, J, K)
    X_hat = np.transpose(imputed, (2, 0, 1))
    # Restore observed entries from original data
    X_hat[mask] = X_obs[mask]
    return X_hat


def _create_train_val_sets(data_pypots, observed_ids, holdout_ratio=0.1, seed=0):
    """Create train/val sets for PyPOTS self-supervised training.

    Artificially masks `holdout_ratio` of observed entries for validation.
    Unobserved sensor columns remain entirely NaN.
    """
    rng = np.random.default_rng(seed)
    train_data = data_pypots.copy()
    X_ori = data_pypots.copy()

    # Only mask entries that are currently non-NaN in observed sensor columns
    for fid in observed_ids:
        col = train_data[:, :, fid]
        non_nan = ~np.isnan(col)
        indices = np.argwhere(non_nan)
        n_holdout = max(1, int(len(indices) * holdout_ratio))
        chosen = rng.choice(len(indices), size=n_holdout, replace=False)
        for idx in chosen:
            j, k = indices[idx]
            train_data[j, k, fid] = np.nan

    train_set = {"X": train_data}
    val_set = {"X": train_data.copy(), "X_ori": X_ori}
    return train_set, val_set


def _get_device():
    """Auto-detect CUDA availability."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _try_load_or_train(model, train_set, val_set, cache_dir, model_name, seed, params):
    """Load cached model if available, otherwise train and save."""
    if cache_dir is not None:
        param_str = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
        cache_path = Path(cache_dir) / f"{model_name}_seed{seed}_{param_str}"
        model_file = cache_path / "model.pypots"
        if model_file.exists():
            print(f"  Loading cached {model_name} from {cache_path}")
            model.load(str(cache_path))
            return model
        else:
            model.fit(train_set, val_set)
            cache_path.mkdir(parents=True, exist_ok=True)
            model.save(str(cache_path))
            return model
    else:
        model.fit(train_set, val_set)
        return model


def run_brits(X_obs, mask, sensor_coords=None, variogram_params=None,
              max_iter=100, seed=0, rnn_hidden_size=64, lr=1e-3,
              epochs=50, batch_size=32, cache_dir=None, **kwargs):
    """BRITS: Bidirectional Recurrent Imputation for Time Series (via PyPOTS).

    Trains bidirectional RNN on observed sensor data with self-supervised
    artificial masking. For unobserved sensors, the model has no training
    signal and can only extrapolate from learned temporal patterns.
    """
    try:
        from pypots.imputation import BRITS
        from pypots.optim import Adam
    except ImportError:
        raise ImportError(
            "PyPOTS is required for BRITS baseline. Install with: pip install pypots"
        )

    I, J, K = X_obs.shape
    observed_ids, unobserved_ids = _identify_sensors(mask)
    if len(unobserved_ids) == 0:
        return X_obs.copy()

    data_pypots = _tensor_to_pypots(X_obs, mask)
    train_set, val_set = _create_train_val_sets(
        data_pypots, observed_ids, holdout_ratio=0.1, seed=seed
    )
    test_set = {"X": data_pypots}

    params = {"rnn_hidden_size": rnn_hidden_size, "lr": lr}
    optimizer = Adam(lr=lr)
    model = BRITS(
        n_steps=K,
        n_features=I,
        rnn_hidden_size=rnn_hidden_size,
        epochs=epochs,
        optimizer=optimizer,
        batch_size=batch_size,
        device=_get_device(),
        saving_path=None,
        verbose=True,
    )
    model = _try_load_or_train(
        model, train_set, val_set, cache_dir, "brits", seed, params
    )

    imputed = model.impute(test_set)
    X_hat = _pypots_to_tensor(imputed, X_obs, mask, I, J, K)
    return X_hat


def run_saits(X_obs, mask, sensor_coords=None, variogram_params=None,
              max_iter=100, seed=0, d_model=64, n_heads=4, n_layers=2,
              lr=1e-3, epochs=50, batch_size=32, cache_dir=None, **kwargs):
    """SAITS: Self-Attention-based Imputation for Time Series (via PyPOTS).

    Trains diagonal-masked self-attention model on observed sensor data.
    For unobserved sensors, cross-feature attention cannot produce meaningful
    estimates without any observed values at those positions.
    """
    try:
        from pypots.imputation import SAITS
        from pypots.optim import Adam
    except ImportError:
        raise ImportError(
            "PyPOTS is required for SAITS baseline. Install with: pip install pypots"
        )

    I, J, K = X_obs.shape
    observed_ids, unobserved_ids = _identify_sensors(mask)
    if len(unobserved_ids) == 0:
        return X_obs.copy()

    data_pypots = _tensor_to_pypots(X_obs, mask)
    train_set, val_set = _create_train_val_sets(
        data_pypots, observed_ids, holdout_ratio=0.1, seed=seed
    )
    test_set = {"X": data_pypots}

    params = {"d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "lr": lr}
    optimizer = Adam(lr=lr)
    model = SAITS(
        n_steps=K,
        n_features=I,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_k=d_model // n_heads,
        d_v=d_model // n_heads,
        d_ffn=d_model * 2,
        dropout=0.1,
        epochs=epochs,
        optimizer=optimizer,
        batch_size=batch_size,
        device=_get_device(),
        saving_path=None,
        verbose=True,
    )
    model = _try_load_or_train(
        model, train_set, val_set, cache_dir, "saits", seed, params
    )

    imputed = model.impute(test_set)
    X_hat = _pypots_to_tensor(imputed, X_obs, mask, I, J, K)
    return X_hat
