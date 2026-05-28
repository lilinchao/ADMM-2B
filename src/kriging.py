"""
Kriging module for tensor imputation.

Supports three modes:
  - "variogram": ordinary kriging with fitted variogram in data-driven
    coordinate space (sensor MDS coords + scaled temporal dims)
  - "kkt_index": original 3D tensor-index KDTree + analytic KKT weights
  - "idw": inverse-distance weighting in data-driven coordinate space

Variogram functions are now real implementations (replacing previous stubs).
"""

import numpy as np
from scipy.spatial import KDTree, distance as sdist
from scipy.optimize import minimize
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# Variogram: empirical computation and parametric fitting
# ---------------------------------------------------------------------------

def empirical_variogram_spatial(X, mask, sensor_coords, n_lags=15,
                                max_lag=None, n_sample_pairs=5000,
                                temporal_mode="same_slot", seed=42):
    """
    Compute empirical variogram from observed data using data-driven
    spatial coordinates.

    Subsamples sensor pairs, computes semivariance, bins by spatial distance.

    Parameters
    ----------
    X : ndarray (I, J, K)
    mask : ndarray (I, J, K) bool — True = observed
    sensor_coords : ndarray (I, 2) — MDS-embedded spatial coordinates
    n_lags : int — number of distance bins
    max_lag : float or None — max distance; if None, half of max pairwise
    n_sample_pairs : int — number of sensor pairs to sample
    temporal_mode : str
        "same_slot" — pair points at same (j, k) across different observations
    seed : int

    Returns
    -------
    lag_centers : ndarray (n_lags,)
    gamma_vals  : ndarray (n_lags,)
    counts      : ndarray (n_lags,)
    """
    I, J, K = X.shape
    rng = np.random.default_rng(seed)

    # Compute all pairwise spatial distances
    from scipy.spatial.distance import pdist, squareform
    D_spatial = squareform(pdist(sensor_coords))
    if max_lag is None:
        max_lag = float(D_spatial[D_spatial > 0].max()) * 0.5

    # Sample sensor pairs
    all_pairs = []
    for i1 in range(I):
        for i2 in range(i1 + 1, I):
            all_pairs.append((i1, i2))
    all_pairs = np.array(all_pairs)

    if len(all_pairs) > n_sample_pairs:
        idx = rng.choice(len(all_pairs), size=n_sample_pairs, replace=False)
        sampled = all_pairs[idx]
    else:
        sampled = all_pairs

    # Compute variogram cloud
    h_list = []
    gamma_list = []
    obs_mask = mask  # True = observed

    for i1, i2 in sampled:
        i1, i2 = int(i1), int(i2)
        h = D_spatial[i1, i2]

        # For each day j and time-slot k where both sensors are observed,
        # compute semivariance
        if temporal_mode == "same_slot":
            both_obs = obs_mask[i1] & obs_mask[i2]  # (J, K)
            if not both_obs.any():
                continue
            js, ks = np.where(both_obs)
            if len(js) > 50:
                sub = rng.choice(len(js), size=50, replace=False)
                js, ks = js[sub], ks[sub]
            diffs_sq = (X[i1, js, ks] - X[i2, js, ks]) ** 2
            gamma = 0.5 * float(diffs_sq.mean())
            h_list.append(h)
            gamma_list.append(gamma)

    if not h_list:
        return (np.linspace(0, max_lag, n_lags),
                np.zeros(n_lags),
                np.zeros(n_lags, dtype=int))

    h_arr = np.array(h_list)
    g_arr = np.array(gamma_list)

    # Bin
    bin_edges = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    gamma_vals = np.zeros(n_lags)
    counts = np.zeros(n_lags, dtype=int)

    for b in range(n_lags):
        in_bin = (h_arr >= bin_edges[b]) & (h_arr < bin_edges[b + 1])
        if in_bin.any():
            gamma_vals[b] = g_arr[in_bin].mean()
            counts[b] = int(in_bin.sum())

    # Only return bins with data
    valid = counts > 0
    return lag_centers[valid], gamma_vals[valid], counts[valid]


def _variogram_func(h, params):
    """Evaluate parametric variogram at distance(s) h."""
    nugget = params["nugget"]
    sill = params["sill"]
    range_ = max(params["range_"], 1e-12)
    model = params.get("model", "exponential")

    h = np.asarray(h, dtype=float)
    hr = h / range_

    if model == "spherical":
        g = np.where(
            h <= range_,
            nugget + (sill - nugget) * (1.5 * hr - 0.5 * hr ** 3),
            sill,
        )
    else:  # exponential (default)
        g = nugget + (sill - nugget) * (1.0 - np.exp(-hr))

    return g


def fit_variogram(h, gamma_emp, model="exponential", counts=None):
    """
    Fit parametric variogram model via weighted least squares.

    Parameters
    ----------
    h : ndarray (n_lags,) — lag center distances
    gamma_emp : ndarray (n_lags,) — empirical semivariances
    model : str — "exponential" or "spherical"
    counts : ndarray or None — bin counts for weighting (more pairs = more weight)

    Returns
    -------
    params : dict with nugget, sill, range_, model, rmse
    """
    h = np.asarray(h, dtype=float)
    gamma_emp = np.asarray(gamma_emp, dtype=float)
    valid = h > 0
    if valid.sum() < 3:
        return {"nugget": 0.01, "sill": float(gamma_emp.max()) if gamma_emp.size else 1.0,
                "range_": float(h.max()) if h.size else 1.0, "model": model, "rmse": 0.0}

    h_v = h[valid]
    g_v = gamma_emp[valid]
    w = np.ones(len(h_v))
    if counts is not None:
        c = np.asarray(counts, dtype=float)[valid]
        c = np.maximum(c, 1.0)
        w = c / c.max()

    # Initial guesses
    nugget0 = max(float(g_v[0]) * 0.5, 1e-6)
    sill0 = max(float(g_v[-1]), nugget0 * 2)
    range0 = float(h_v[len(h_v) // 2])

    def objective(params_flat):
        nugget, sill, range_ = params_flat
        if sill <= nugget or range_ <= 0:
            return 1e12
        pred = _variogram_func(h_v, {"nugget": nugget, "sill": sill,
                                      "range_": range_, "model": model})
        return float(np.sum(w * (pred - g_v) ** 2))

    try:
        result = minimize(
            objective,
            x0=[nugget0, sill0, range0],
            method="L-BFGS-B",
            bounds=[(1e-8, sill0 * 2), (nugget0 * 1.01, sill0 * 4), (h_v.min() * 0.1, h_v.max() * 3)],
        )
        if result.success:
            nugget, sill, range_ = result.x
        else:
            nugget, sill, range_ = nugget0, sill0, range0
    except Exception:
        nugget, sill, range_ = nugget0, sill0, range0

    # Ensure sill > nugget
    if sill <= nugget:
        sill = nugget * 2

    params = {"nugget": float(nugget), "sill": float(sill),
              "range_": float(range_), "model": model}
    pred = _variogram_func(h_v, params)
    params["rmse"] = float(np.sqrt(np.mean((pred - g_v) ** 2)))
    return params


def fit_variogram_from_tensor(X, mask, sensor_coords, model="exponential",
                               n_lags=15, cache_path=None):
    """
    Compute empirical variogram from tensor data and fit a parametric model.

    Called ONCE before ADMM iterations.

    Parameters
    ----------
    X : ndarray (I, J, K)
    mask : ndarray (I, J, K) bool
    sensor_coords : ndarray (I, 2)
    model : str
    n_lags : int
    cache_path : str or None — if set, cache/load from this .npz file

    Returns
    -------
    variogram_params : dict
    """
    if cache_path is not None:
        import os
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            return dict(data["params"].item())

    h, gamma, counts = empirical_variogram_spatial(
        X, mask, sensor_coords, n_lags=n_lags,
    )
    params = fit_variogram(h, gamma, model=model, counts=counts)

    if cache_path is not None:
        np.savez(cache_path, params=np.array(params))

    return params


# ---------------------------------------------------------------------------
# Ordinary kriging solver
# ---------------------------------------------------------------------------

def _solve_ordinary_kriging(target_coords, nbr_coords, nbr_vals,
                            variogram_params):
    """
    Solve ordinary kriging system for one estimation point.

    Kriging system:
        [Gamma  1] [lambda]   [gamma_0]
        [1'     0] [  mu  ] = [  1    ]

    Parameters
    ----------
    target_coords : ndarray (D,)
    nbr_coords : ndarray (L, D)
    nbr_vals : ndarray (L,)
    variogram_params : dict

    Returns
    -------
    lam : ndarray (L,) — kriging weights (sum = 1)
    w_val : float — kriging estimate
    sigma2 : float — kriging variance (negative = fallback used)
    """
    L = len(nbr_vals)
    if L == 0:
        return np.array([]), 0.0, -1.0

    if L == 1:
        return np.array([1.0]), float(nbr_vals[0]), 0.0

    # Pairwise distances between neighbors
    h_lm = sdist.cdist(nbr_coords, nbr_coords)
    # Distances from neighbors to target
    h_l0 = sdist.cdist(nbr_coords, target_coords.reshape(1, -1)).ravel()

    # Evaluate variogram
    gamma_lm = _variogram_func(h_lm, variogram_params)  # (L, L)
    gamma_l0 = _variogram_func(h_l0, variogram_params)   # (L,)

    # Build kriging system: (L+1) x (L+1)
    K_mat = np.zeros((L + 1, L + 1))
    K_mat[:L, :L] = gamma_lm
    K_mat[:L, L] = 1.0
    K_mat[L, :L] = 1.0

    rhs = np.zeros(L + 1)
    rhs[:L] = gamma_l0
    rhs[L] = 1.0

    # Solve with regularization for numerical stability
    try:
        cond = np.linalg.cond(K_mat)
        if cond > 1e10:
            reg = max(variogram_params.get("nugget", 1e-6) * 0.1, 1e-4)
            K_mat[:L, :L] += np.eye(L) * reg
        sol = np.linalg.solve(K_mat, rhs)
        lam = sol[:L]
        mu = sol[L]
        sigma2 = float(np.dot(gamma_l0, lam) + mu)
    except np.linalg.LinAlgError:
        # Fallback to IDW
        dists = np.maximum(h_l0, 1e-12)
        w = 1.0 / dists
        lam = w / w.sum()
        sigma2 = -1.0  # flag: fallback used

    # Safeguard: if any weight is very negative, fall back to IDW
    if np.any(lam < -0.5) or np.any(np.isnan(lam)):
        dists = np.maximum(h_l0, 1e-12)
        w = 1.0 / dists
        lam = w / w.sum()
        sigma2 = -1.0

    w_val = float(np.dot(lam, nbr_vals))
    return lam, w_val, sigma2


def _solve_kriging_chunk(target_coords, nbr_coords, nbr_vals, variogram_params,
                         _return_weights=False):
    """Solve a chunk of kriging systems (vectorized within chunk)."""
    M = target_coords.shape[0]
    K = nbr_coords.shape[1]
    w_vals = np.empty(M, dtype=float)
    sigma2_vals = np.empty(M, dtype=float)
    lam_all = np.empty((M, K), dtype=float) if _return_weights else None

    diff_lm = nbr_coords[:, :, np.newaxis, :] - nbr_coords[:, np.newaxis, :, :]
    h_lm = np.sqrt((diff_lm ** 2).sum(axis=-1) + 1e-30)
    diff_l0 = nbr_coords - target_coords[:, np.newaxis, :]
    h_l0 = np.sqrt((diff_l0 ** 2).sum(axis=-1) + 1e-30)

    gamma_lm = _variogram_func(h_lm, variogram_params)
    gamma_l0 = _variogram_func(h_l0, variogram_params)

    K_mat = np.zeros((M, K + 1, K + 1))
    K_mat[:, :K, :K] = gamma_lm
    K_mat[:, :K, K] = 1.0
    K_mat[:, K, :K] = 1.0

    rhs = np.zeros((M, K + 1))
    rhs[:, :K] = gamma_l0
    rhs[:, K] = 1.0

    try:
        # Add regularization for numerical stability
        # Use max of nugget-based and absolute minimum to ensure robustness
        reg = max(variogram_params.get("nugget", 1e-6) * 0.1, 1e-4)
        K_mat[:, :K, :K] += np.eye(K) * reg
        # Use np.linalg.solve (more stable than explicit inverse)
        sol = np.linalg.solve(K_mat, rhs)  # (M, K+1)
        lam = sol[:, :K]
        mu = sol[:, K]
        # Check for unreasonable negative weights (indicates solver failure)
        bad_weights = np.any(lam < -0.5, axis=1) | np.any(~np.isfinite(lam), axis=1)
        w_vals = np.einsum("mk,mk->m", lam, nbr_vals)
        sigma2_vals = np.einsum("mk,mk->m", gamma_l0, lam) + mu
        if _return_weights:
            lam_all[:] = lam
    except (np.linalg.LinAlgError, ValueError):
        bad_weights = np.ones(M, dtype=bool)  # all failed
        if _return_weights:
            lam_all[:] = np.nan

    # Fix problematic points: use single-point solver with IDW fallback
    bad = ~np.isfinite(w_vals) | bad_weights
    if bad.any():
        for m in np.where(bad)[0]:
            lam_m, w, s2 = _solve_ordinary_kriging(
                target_coords[m], nbr_coords[m], nbr_vals[m], variogram_params)
            w_vals[m] = w
            sigma2_vals[m] = s2
            if _return_weights:
                lam_all[m] = lam_m

    if _return_weights:
        return w_vals, sigma2_vals, lam_all
    return w_vals, sigma2_vals


def solve_ordinary_kriging_batch(target_coords_all, nbr_coords_all,
                                  nbr_vals_all, variogram_params,
                                  chunk_size=5000, return_weights=False):
    """
    Batch ordinary kriging for all missing positions, processed in chunks.

    Parameters
    ----------
    target_coords_all : ndarray (M, D)
    nbr_coords_all : ndarray (M, K, D)
    nbr_vals_all : ndarray (M, K)
    variogram_params : dict
    chunk_size : int — number of points per chunk (controls memory)
    return_weights : bool — if True, also return weight matrix (M, K)

    Returns
    -------
    w_vals : ndarray (M,) — kriging estimates
    sigma2_vals : ndarray (M,) — kriging variances
    lam_all : ndarray (M, K) — only if return_weights=True
    """
    M = target_coords_all.shape[0]
    K = nbr_coords_all.shape[1]
    w_vals = np.empty(M, dtype=float)
    sigma2_vals = np.empty(M, dtype=float)
    lam_all = np.empty((M, K), dtype=float) if return_weights else None

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        s, s2, lam_chunk = _solve_kriging_chunk(
            target_coords_all[start:end],
            nbr_coords_all[start:end],
            nbr_vals_all[start:end],
            variogram_params,
            _return_weights=True,
        )
        w_vals[start:end] = s
        sigma2_vals[start:end] = s2
        if return_weights:
            lam_all[start:end] = lam_chunk

    if return_weights:
        return w_vals, sigma2_vals, lam_all
    return w_vals, sigma2_vals


# ---------------------------------------------------------------------------
# Legacy KKT solver (kept for backward compatibility)
# ---------------------------------------------------------------------------

def _solve_kriging_weights(z_tilde, X_l, gamma=1.0, beta=0.01):
    """
    Analytic KKT solution for kriging weights (legacy, not true kriging):
        min  (z_tilde - sum lambda_l * X_l)^2  s.t. sum lambda_l = 1
    """
    L = len(X_l)
    if L == 0:
        return np.array([]), 0.0

    mu = X_l.mean()
    dX = X_l - mu
    denom = np.dot(dX, dX)

    if denom < 1e-12:
        return np.ones(L) / L, float(mu)

    lam = np.ones(L) / L + (z_tilde - mu) / denom * dX
    w_val = float(np.dot(lam, X_l))
    return lam, w_val


# ---------------------------------------------------------------------------
# SpatioTemporalKriging class
# ---------------------------------------------------------------------------

class SpatioTemporalKriging:
    """
    Spatio-temporal kriging for tensor imputation.

    Supports three kriging modes:
      - "variogram": ordinary kriging with fitted variogram (recommended)
      - "kkt_index": original 3D tensor-index KDTree + KKT weights (legacy)
      - "idw": inverse-distance weighting in data-driven coordinate space

    When sensor_coords is provided, the KDTree operates in a meaningful
    coordinate space [sensor_x, sensor_y, j/tau_j, k/tau_k].
    """

    def __init__(
        self,
        sensor_coords=None,
        locs=None,
        variogram_model="exponential",
        variogram_params=None,
        n_neighbors=50,
        fit_variogram_from_data=False,
        gamma=1.0,
        beta=0.01,
        n_jobs=-1,
        kriging_mode="variogram",
        tau_j=3.0,
        tau_k=1.0,
    ):
        self.sensor_coords = sensor_coords  # (I, 2) or None
        self.locs = locs          # kept for API compat
        self.model = variogram_model
        self.variogram_params = variogram_params or {}
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.beta = beta
        self.n_jobs = n_jobs
        self.kriging_mode = kriging_mode
        self.tau_j = tau_j
        self.tau_k = tau_k

    def _build_spatial_coords(self, indices, mask):
        """Convert tensor indices (N, 3) to meaningful spatial coordinates (N, D)."""
        if self.sensor_coords is None:
            return indices.astype(float)  # fallback to 3D index space

        i_idx = indices[:, 0]
        spatial = np.column_stack([
            self.sensor_coords[i_idx, 0],
            self.sensor_coords[i_idx, 1],
            indices[:, 1].astype(float) / self.tau_j,
            indices[:, 2].astype(float) / self.tau_k,
        ])
        return spatial

    def fit(self, X_obs_full, obs_mask):
        """Fit variogram from data if requested."""
        if self.kriging_mode == "variogram" and self.sensor_coords is not None:
            if not self.variogram_params or "nugget" not in self.variogram_params:
                self.variogram_params = fit_variogram_from_tensor(
                    X_obs_full, obs_mask, self.sensor_coords,
                    model=self.model,
                )
        return self

    def predict(self, X, mask):
        """
        Predict all missing entries.

        Parameters
        ----------
        X    : ndarray (I, J, K)
        mask : ndarray (I, J, K) bool — True = observed

        Returns
        -------
        X_filled       : ndarray (I, J, K)
        weights        : dict {(i,j,k): ndarray}
        neighbor_indices : dict {(i,j,k): list}
        """
        obs_indices = np.column_stack(np.where(mask)).astype(int)  # (N_obs, 3)
        miss_indices = np.column_stack(np.where(~mask)).astype(int)  # (N_miss, 3)
        miss_positions = list(zip(*np.where(~mask)))

        if len(miss_indices) == 0:
            return X.copy(), {}, {}

        # Build KDTree in appropriate coordinate space
        obs_spatial = self._build_spatial_coords(obs_indices, mask)
        obs_tree = KDTree(obs_spatial)

        X_out = X.copy()
        weights = {}
        neighbor_indices = {}

        k_actual = min(self.n_neighbors, len(obs_indices))

        # Build miss spatial coords
        miss_spatial = self._build_spatial_coords(miss_indices, mask)

        if self.kriging_mode == "variogram" and self.sensor_coords is not None:
            # Ordinary kriging with variogram
            _, nbr_idx_all = obs_tree.query(miss_spatial, k=k_actual)

            # Gather neighbor coordinates and values
            M = len(miss_indices)
            K = k_actual
            nbr_coords = np.empty((M, K, obs_spatial.shape[1]))
            nbr_vals = np.empty((M, K))

            for m in range(M):
                for l in range(K):
                    idx = int(nbr_idx_all[m, l]) if K > 1 else int(nbr_idx_all[m])
                    nbr_coords[m, l] = obs_spatial[idx]
                    oi, oj, ok = obs_indices[idx]
                    nbr_vals[m, l] = X[oi, oj, ok]

            w_vals, _ = solve_ordinary_kriging_batch(
                miss_spatial, nbr_coords, nbr_vals, self.variogram_params,
            )

            for m, (i, j, k) in enumerate(miss_positions):
                X_out[i, j, k] = w_vals[m]
                weights[(i, j, k)] = np.zeros(K)  # not stored per-point for efficiency
                neighbor_indices[(i, j, k)] = []

        elif self.kriging_mode == "idw" and self.sensor_coords is not None:
            # IDW in meaningful coordinate space (vectorized)
            dists_all, nbr_idx_all = obs_tree.query(miss_spatial, k=k_actual)
            if k_actual == 1:
                dists_all = dists_all.reshape(-1, 1)
                nbr_idx_all = nbr_idx_all.reshape(-1, 1)

            M = len(miss_indices)
            K = k_actual
            # Gather neighbor values: (M, K)
            nbr_vals = X[obs_indices[nbr_idx_all.ravel(), 0],
                         obs_indices[nbr_idx_all.ravel(), 1],
                         obs_indices[nbr_idx_all.ravel(), 2]].reshape(M, K)
            # Weights: inverse distance, normalized
            d_safe = np.maximum(dists_all, 1e-12)  # (M, K)
            w = 1.0 / d_safe
            w /= w.sum(axis=1, keepdims=True)  # (M, K)
            # Weighted sum
            pred_vals = np.einsum("mk,mk->m", w, nbr_vals)  # (M,)
            for m, (i, j, k) in enumerate(miss_positions):
                X_out[i, j, k] = pred_vals[m]
                weights[(i, j, k)] = w[m]
                neighbor_indices[(i, j, k)] = []

        else:
            # Legacy: 3D index-space KDTree + KKT weights
            obs_tree_legacy = KDTree(obs_indices)
            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._kriging_one_entry_legacy)(
                    i, j, k, X, obs_indices, obs_tree_legacy, self.n_neighbors,
                )
                for (i, j, k) in miss_positions
            )
            for i, j, k, lam, w_val in results:
                X_out[i, j, k] = w_val
                weights[(i, j, k)] = lam
                neighbor_indices[(i, j, k)] = []

        return X_out, weights, neighbor_indices

    def predict_full(self, X, mask):
        """Convenience: return only the filled tensor."""
        X_filled, _, _ = self.predict(X, mask)
        return X_filled

    @staticmethod
    def _kriging_one_entry_legacy(i, j, k, X, obs_coords, obs_tree,
                                    n_neighbors, gamma=1.0, beta=0.01):
        """Legacy single-entry kriging in 3D index space."""
        k_actual = min(n_neighbors, len(obs_coords))
        distances, indices = obs_tree.query([i, j, k], k=k_actual)
        if np.isscalar(indices):
            indices = [indices]

        X_l = np.array([X[int(obs_coords[idx][0]),
                           int(obs_coords[idx][1]),
                           int(obs_coords[idx][2])]
                        for idx in indices])

        z_tilde = float(np.mean(X_l)) if len(X_l) > 0 else 0.0
        lam, w_val = _solve_kriging_weights(z_tilde, X_l, gamma, beta)
        return i, j, k, lam, w_val
