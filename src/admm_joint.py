"""
ADMM Joint Kriging + CP Tensor Decomposition for missing data imputation.

Kriging: 3D KDTree on tensor (i,j,k) coords + analytic KKT weights.

Objective:
    min  (1/2)||P_Omega(X - Z)||_F^2
         + (gamma/2)||P_Omega_bar(Z - W)||_F^2
         + beta * R_Tensor(Theta)
    s.t. Z = Y,  Z = W,  sum_l lambda_ijk^(l) = 1  for (i,j,k) in Omega_bar

Bug-fixed (2026-04-03):
    W update: z_tilde = Z + Lambda2 / (gamma + rho)   [was / rho]
    Z update (missing): denom = gamma + 2*rho           [was 2*rho]
                        numerator includes (gamma+rho)*W [was rho*W]

Adaptive gamma v9 (adaptive_gamma=True, 2026-04-15):
    EM closed-form update with low-start curriculum + RM-adaptive cap.

    Theory (Bayesian optimal fusion):
        gamma* = rho * (sigma_Y^2 / sigma_W^2 - 1)

    v6 flaw: W ≈ Z by KKT construction → W-Z residuals → 0 → positive
             feedback → gamma saturates at fixed cap (500) regardless of RM.

    v7 flaw: holdout at observed positions → CP beats 3D-index IDW (CP
             captures temporal structure IDW ignores) → sigma_Y << sigma_W
             → gamma → minimum → kriging disabled.

    v9 fix: restore v6 curriculum effect + RM-adaptive cap.

    Why v6 EM worked: starting at gamma_init=1 gave a "curriculum" effect —
    early iterations with low gamma let ALS converge freely, then positive-
    feedback EM drove gamma toward cap=500. This outperformed fixed gamma=100
    even though 500 is nominally "above optimal" at low missing rates.

    Why v8 broke it: v8 set gamma_init = empirical_gamma (already at target),
    so the curriculum effect disappeared and results degraded.

    Why Method B failed: Same flaw as v7 — at holdout (observed) positions,
    CP (trained on 95% of observed data) almost always beats IDW → alpha ≤ 0.5
    → gamma → gamma_min → kriging disabled. Method B is preserved in code but
    disabled by default (use_method_b=False).

    v9 design:
        gamma_init = rho   (low start, restores curriculum)
        gamma_max  = _empirical_gamma(RM) * em_factor  (RM-adaptive cap)
        EM positive feedback → gamma climbs from rho toward gamma_max

    Three-phase schedule:
        warmup   (t < warmup_steps)   : gamma fixed at gamma_init
        adaptive (warmup <= t < freeze): gamma updated every update_freq steps
        frozen   (t >= freeze)        : gamma fixed at last update value

    Parameters (adaptive_gamma_cfg dict):
        warmup_steps : int   = 15
        update_freq  : int   = 5
        ema_alpha    : float = 0.15   (log-space EMA)
        freeze_frac  : float = 0.5    (freeze at 50% of max_iter)
        em_factor    : float = 1.3    (Method A cap: empirical_gamma * factor)
        gamma_min    : float = 0.1 * rho
        gamma_max    : float = None   (if set explicitly, overrides em_factor)
        use_method_b : bool  = True   (True=B primary, False=A only)
        holdout_frac : float = 0.05   (fraction of obs to hold out for B)
        max_holdout  : int   = 1000   (cap on holdout set size)
        tau_j        : float = 3.0    (day-axis scale in same-sensor IDW)
        tau_k        : float = 1.0    (slot-axis scale in same-sensor IDW)
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix, diags, eye as speye
from scipy.sparse.linalg import spsolve
from joblib import Parallel, delayed
from src.interpolation_baselines import linear_time as _interp_linear_time, daily_profile as _interp_daily_profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cp_reconstruct(A, B, C):
    return np.einsum("ir,jr,kr->ijk", A, B, C)


def _project_simplex(v):
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_arr = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    if len(rho_arr) == 0:
        return np.ones(n) / n
    rho = rho_arr[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def _solve_kriging_weights(z_tilde, X_l, gamma, beta):
    """
    Analytic KKT solution for kriging weights:
        min  (z_tilde - sum_l lambda_l * X_l)^2
        s.t. sum_l lambda_l = 1

    lambda* = 1/L + (z_tilde - mean(X_l)) / ||X_l - mean||^2 * (X_l - mean)
    Degenerates to uniform weights when all neighbors identical.
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


def _update_one_missing(i, j, k, z_tilde_val, gamma, beta,
                        X_np, obs_coords, obs_tree, n_neighbors):
    """Kriging update for one missing position."""
    k_actual = min(n_neighbors, len(obs_coords))
    distances, indices = obs_tree.query([i, j, k], k=k_actual)
    if np.isscalar(indices):
        indices = [indices]

    X_l = np.array([X_np[int(obs_coords[idx][0]),
                         int(obs_coords[idx][1]),
                         int(obs_coords[idx][2])]
                    for idx in indices])

    lam, w_val = _solve_kriging_weights(z_tilde_val, X_l, gamma, beta)
    return i, j, k, lam, w_val, indices


def _gamma_cap_from_alpha(alpha):
    """Max gamma such that kriging weight in Z-update <= alpha."""
    alpha = float(np.clip(alpha, 0.500001, 0.999999))
    return (2.0 * alpha - 1.0) / (1.0 - alpha)


def _empirical_gamma(n_obs, n_miss, rho=1.0):
    """
    Empirical gamma via piecewise-linear interpolation on 4 calibration anchors.

    Calibration anchors from full gamma sweep (seed=0, 2026-04-05):
        RM=20% -> optimal gamma=100
        RM=40% -> optimal gamma=500
        RM=60% -> optimal gamma=20
        RM=80% -> optimal gamma=100

    The optimal gamma is non-monotone in RM: it peaks at RM=40% (kriging most
    helpful) then drops sharply at RM=60% (W noisy, over-weighting hurts) and
    partially recovers at RM=80%.

    Implementation: log-linear interpolation between adjacent anchors, with
    linear extrapolation beyond the range.

    Returns
    -------
    gamma : float  (already multiplied by rho)
    """
    n_obs  = max(int(n_obs), 1)
    n_miss = max(int(n_miss), 0)
    if n_miss == 0:
        return 0.1 * rho

    rm = n_miss / (n_obs + n_miss)   # missing rate in [0,1)

    # Calibration anchors: (RM, log10(optimal_gamma))
    anchors_rm  = np.array([0.20, 0.40, 0.60, 0.80])
    anchors_log = np.log10(np.array([100.0, 500.0, 20.0, 100.0]))

    # Piecewise-linear interpolation in log space
    log_gamma = float(np.interp(rm, anchors_rm, anchors_log))
    gamma = 10.0 ** log_gamma

    # Hard floor: never go below 10
    gamma = max(gamma, 10.0)
    return rho * gamma


def _estimate_adaptive_gamma(
    X_np, mask, obs_coords, obs_tree,
    missing_positions, n_neighbors, rho,
    sample_size=500, rng=None,
    cp_inflation_power=1.0,
    krig_density_power=None,
    gamma_min_ratio=0.1,
    gamma_max_ratio=None,
    alpha_max=0.998,
):
    """
    Robust missing-rate-aware adaptive gamma (v3, Codex/gpt-5.4, 2026-04-04).

    Theory
    ------
    Z-update for missing positions (ignoring dual terms):
        Z_m ~= [rho * Y_m + (gamma + rho) * W_m] / (gamma + 2*rho)

    Precision matching to optimal linear fusion gives:
        gamma = rho * (sigma_y^2 / sigma_w_eff^2 - 1)

    Estimates
    ---------
    sigma_y^2      = Var(X_obs) * (obs_frac^(-p) - 1),   p=1  (linear)
    sigma_w_base^2 = robust LOO-IDW MSE on observed entries (20-80% trimmed)
    sigma_w_eff^2  = sigma_w_base^2 * obs_frac^(-q),     q=2/ndim=2/3
                     (corrects for lower obs density at high missing rate)

    gamma is capped by alpha_max: the max allowed kriging weight in Z-update.
    With alpha_max=0.998: gamma_max ~ 499 * rho (no manual gamma_scale needed).

    Returns
    -------
    gamma : float
    stats : dict  (obs_frac, sigma_y_sq, sigma_w_base_sq, sigma_w_eff_sq, ...)
    """
    N_total = X_np.size
    N_obs   = int(mask.sum())
    N_miss  = int(N_total - N_obs)

    empty_stats = {"obs_frac": 1.0, "sigma_y_sq": 0.0,
                   "sigma_w_base_sq": 0.0, "sigma_w_eff_sq": 0.0,
                   "gamma_cap": rho}
    if N_miss == 0 or N_obs < 3:
        return float(rho), empty_stats

    if rng is None:
        rng = np.random.default_rng(42)

    if krig_density_power is None:
        krig_density_power = 2.0 / max(X_np.ndim, 1)   # 3-D -> 2/3

    if gamma_max_ratio is None:
        gamma_max_ratio = _gamma_cap_from_alpha(alpha_max)   # ~499 for 0.998

    obs_frac = N_obs / N_total
    obs_var  = max(float(np.var(X_np[mask])), 1e-8)

    # ------------------------------------------------------------------
    # sigma_w_base^2: robust LOO-IDW on observed positions
    # ------------------------------------------------------------------
    n_sample = min(sample_size, N_obs)
    sample_obs_idx = rng.choice(N_obs, size=n_sample, replace=False)

    loo_err_sq = []
    for obs_idx in sample_obs_idx:
        ci, cj, ck = obs_coords[obs_idx]
        k_query = min(n_neighbors + 1, N_obs)
        dists, nbr_idx = obs_tree.query([ci, cj, ck], k=k_query)

        if np.isscalar(nbr_idx):
            continue

        keep = [t for t, idx in enumerate(nbr_idx) if int(idx) != int(obs_idx)]
        if len(keep) == 0:
            continue

        nbr_idx = np.asarray([int(nbr_idx[t]) for t in keep], dtype=int)
        dists   = np.asarray([float(dists[t])  for t in keep], dtype=float)

        X_l = np.asarray([X_np[int(obs_coords[idx][0]),
                               int(obs_coords[idx][1]),
                               int(obs_coords[idx][2])]
                          for idx in nbr_idx], dtype=float)

        w = 1.0 / np.maximum(dists, 1e-6)
        w_sum = float(w.sum())
        if w_sum < 1e-12:
            continue
        w /= w_sum

        x_hat  = float(np.dot(w, X_l))
        x_true = float(X_np[int(ci), int(cj), int(ck)])
        loo_err_sq.append((x_true - x_hat) ** 2)

    if not loo_err_sq:
        sigma_w_base_sq = max(obs_var * 0.1, 1e-8)
    else:
        errs = np.asarray(loo_err_sq, dtype=float)
        if errs.size >= 10:
            q20, q80 = np.quantile(errs, [0.2, 0.8])
            trimmed  = errs[(errs >= q20) & (errs <= q80)]
            if trimmed.size > 0:
                errs = trimmed
        sigma_w_base_sq = max(float(np.mean(errs)), 1e-8)

    # ------------------------------------------------------------------
    # sigma_y^2: CP error inflated by missing rate (p=1, linear)
    # sigma_w_eff^2: kriging error worsens as obs density falls (q=2/ndim)
    # ------------------------------------------------------------------
    sigma_y_sq     = max(obs_var * (obs_frac ** (-cp_inflation_power) - 1.0), 1e-8)
    sigma_w_eff_sq = max(sigma_w_base_sq * (obs_frac ** (-krig_density_power)), 1e-8)

    # Precision matching + principled cap (alpha_max dominance limit)
    gamma = rho * (sigma_y_sq / sigma_w_eff_sq - 1.0)
    gamma = float(np.clip(gamma, gamma_min_ratio * rho, gamma_max_ratio * rho))

    stats = {
        "obs_frac":          obs_frac,
        "sigma_y_sq":        sigma_y_sq,
        "sigma_w_base_sq":   sigma_w_base_sq,
        "sigma_w_eff_sq":    sigma_w_eff_sq,
        "cp_inflation_power": cp_inflation_power,
        "krig_density_power": krig_density_power,
        "gamma_cap":         gamma_max_ratio * rho,
    }
    return gamma, stats


# ---------------------------------------------------------------------------
# Adaptive gamma helpers (holdout-based, v7)
# ---------------------------------------------------------------------------

def _sample_held_out_obs(obs_coords, X_np, frac=0.10, rng=None):
    """
    Sample a fraction of observed positions as a held-out validation set.

    Returns
    -------
    active_coords : ndarray (N_active, 3)  -- observed coords minus held-out
    active_tree   : KDTree on active_coords
    held_coords   : ndarray (N_held, 3)
    held_vals     : ndarray (N_held,)      -- X_np values at held positions
    """
    obs_coords = np.asarray(obs_coords, dtype=int).reshape(-1, 3)
    if rng is None:
        rng = np.random.default_rng(42)

    n_obs = len(obs_coords)
    n_hold = max(1, int(np.floor(frac * n_obs))) if n_obs > 1 else 0
    n_hold = min(n_hold, n_obs - 1)  # keep at least one active point

    if n_hold == 0:
        return obs_coords.copy(), KDTree(obs_coords), \
               np.empty((0, 3), dtype=int), np.empty(0, dtype=float)

    held_idx   = np.sort(rng.choice(n_obs, size=n_hold, replace=False))
    active_mask = np.ones(n_obs, dtype=bool)
    active_mask[held_idx] = False

    held_coords = obs_coords[held_idx]
    held_vals   = X_np[
        held_coords[:, 0], held_coords[:, 1], held_coords[:, 2]
    ].astype(float)

    active_coords = obs_coords[active_mask]
    active_tree   = KDTree(active_coords)
    return active_coords, active_tree, held_coords, held_vals


def _estimate_gamma_from_holdout(
    Y, held_coords, held_vals, active_coords, active_tree, X_np,
    rho, n_neighbors=10, gamma_min=0.1, gamma_max=500.0,
):
    """
    Estimate gamma from holdout errors of IDW-kriging and CP reconstruction.

    Both estimators are computed on held-out *observed* positions using only
    the active observed set and the current CP factor Y -- neither depends on
    Z or the current gamma, eliminating the positive feedback loop.

    sigma_W^2 = mean((X_val - W_holdout)^2)   IDW on active_coords
    sigma_Y^2 = mean((X_val - Y_val)^2)        CP prediction
    gamma*    = rho * (sigma_Y^2 / sigma_W^2 - 1)

    Returns
    -------
    gamma : float  clipped to [gamma_min, gamma_max]
    """
    if held_coords.shape[0] == 0:
        return float(np.clip(gamma_min, gamma_min, gamma_max))

    k = min(max(int(n_neighbors), 1), len(active_coords))
    dists, nbr_idx = active_tree.query(held_coords, k=k)

    if k == 1:
        dists   = dists.reshape(-1, 1)
        nbr_idx = nbr_idx.reshape(-1, 1)

    active_vals = X_np[
        active_coords[:, 0], active_coords[:, 1], active_coords[:, 2]
    ].astype(float)
    X_neigh = active_vals[nbr_idx]          # (N_held, k)

    w = 1.0 / np.maximum(dists, 1e-6)
    w /= w.sum(axis=1, keepdims=True)
    W_holdout = (w * X_neigh).sum(axis=1)  # (N_held,)

    Y_vals = Y[
        held_coords[:, 0], held_coords[:, 1], held_coords[:, 2]
    ].astype(float)

    sigma_w_sq = max(float(np.mean((held_vals - W_holdout) ** 2)), 1e-12)
    sigma_y_sq = max(float(np.mean((held_vals - Y_vals)    ** 2)), 0.0)

    gamma = rho * (sigma_y_sq / sigma_w_sq - 1.0)
    return float(np.clip(gamma, gamma_min, gamma_max))


def _log_ema(prev, target, alpha, lo, hi):
    """Exponential moving average in log space (geometric mean blend)."""
    prev   = float(np.clip(prev,   lo, hi))
    target = float(np.clip(target, lo, hi))
    out = np.exp((1.0 - alpha) * np.log(prev) + alpha * np.log(target))
    return float(np.clip(out, lo, hi))


# ---------------------------------------------------------------------------
# Adaptive gamma helpers (EM closed-form, v8)
# ---------------------------------------------------------------------------

def _trimmed_mean_sq(vals, q=0.10):
    """Trimmed mean of squared values (remove top and bottom q fraction)."""
    vals = np.asarray(vals, dtype=float).ravel()
    if vals.size == 0:
        return 1e-8
    sq = vals ** 2
    if sq.size >= 10:
        lo, hi = np.quantile(sq, [q, 1.0 - q])
        trimmed = sq[(sq >= lo) & (sq <= hi)]
        if trimmed.size > 0:
            sq = trimmed
    return max(float(np.mean(sq)), 1e-12)


def _gamma_em_closed_form(residuals, rho, gamma_min, gamma_max):
    """
    EM closed-form gamma update from W-Z residuals at missing positions.

    Quadratic from EM M-step fixed point (see adaptive_gamma_v6.md Eq. A10):
        mse * gamma^2 + (mse*rho + 1 - rho) * gamma - rho^2 = 0

    Unique positive root:
        gamma_EM = (-b + sqrt(b^2 + 4*mse*rho^2)) / (2*mse)
        b = mse*rho + 1 - rho

    Returns gamma clipped to [gamma_min, gamma_max].
    """
    mse = _trimmed_mean_sq(residuals)
    b = mse * rho + 1.0 - rho
    discriminant = b * b + 4.0 * mse * (rho ** 2)
    gamma_em = (-b + np.sqrt(max(discriminant, 0.0))) / (2.0 * mse)
    return float(np.clip(gamma_em, gamma_min, gamma_max))


# ---------------------------------------------------------------------------
# Adaptive gamma helpers (same-sensor IDW + direct alpha, v8 Method B)
# ---------------------------------------------------------------------------

def _build_sensor_trees(active_coords, tau_j=3.0, tau_k=1.0):
    """
    Build per-sensor KDTrees in scaled (j, k) space for same-sensor IDW.

    Parameters
    ----------
    active_coords : ndarray (N, 3)  -- observed positions [i, j, k] (holdout excluded)
    tau_j         : float  -- day-axis scale (larger => days matter less in distance)
    tau_k         : float  -- slot-axis scale

    Returns
    -------
    sensor_trees : dict  {sensor_i: (KDTree on scaled (j,k), global_indices)}
    """
    active_coords = np.asarray(active_coords, dtype=int)
    sensor_trees = {}
    for si in np.unique(active_coords[:, 0]):
        si   = int(si)
        gidx = np.where(active_coords[:, 0] == si)[0]
        if len(gidx) == 0:
            continue
        jk     = active_coords[gidx, 1:3].astype(float)
        scaled = np.column_stack([jk[:, 0] / tau_j, jk[:, 1] / tau_k])
        sensor_trees[si] = (KDTree(scaled), gidx)
    return sensor_trees


def _same_sensor_idw_predict(held_coords, active_coords, active_vals,
                              sensor_trees, fallback_tree,
                              n_neighbors=10, tau_j=3.0, tau_k=1.0):
    """
    IDW prediction at holdout positions using same-sensor temporal neighbors.

    For each holdout point (i, j, k), finds n_neighbors nearest neighbors
    among active_coords with the *same* sensor i, using scaled (j, k) distance.
    Falls back to 3-D IDW when no same-sensor active observations exist.

    Parameters
    ----------
    held_coords   : ndarray (N_held, 3)  [i, j, k]
    active_coords : ndarray (N_act,  3)
    active_vals   : ndarray (N_act,)     X values at active_coords
    sensor_trees  : dict from _build_sensor_trees
    fallback_tree : KDTree on full active_coords (3-D)
    n_neighbors   : int
    tau_j, tau_k  : float  scale factors (must match _build_sensor_trees)

    Returns
    -------
    W_pred : ndarray (N_held,)
    """
    held_coords   = np.asarray(held_coords,   dtype=int)
    active_coords = np.asarray(active_coords, dtype=int)
    W_pred = np.empty(len(held_coords), dtype=float)

    for si in np.unique(held_coords[:, 0]):
        si    = int(si)
        idx_h = np.where(held_coords[:, 0] == si)[0]        # positions in held_coords
        jk_h  = held_coords[idx_h, 1:3].astype(float)
        scaled_h = np.column_stack([jk_h[:, 0] / tau_j, jk_h[:, 1] / tau_k])

        if si in sensor_trees:
            tree_si, gidx = sensor_trees[si]
            k_q = min(n_neighbors, len(gidx))
            dists, nbr = tree_si.query(scaled_h, k=k_q)
            if k_q == 1:
                dists = dists.reshape(-1, 1)
                nbr   = nbr.reshape(-1, 1)
            X_l = active_vals[gidx[nbr]]                     # (n_h, k_q)
            w   = 1.0 / np.maximum(dists, 1e-6)
            W_pred[idx_h] = (w * X_l).sum(axis=1) / w.sum(axis=1)
        else:
            # No same-sensor active obs → 3-D fallback
            query_pts = np.column_stack([
                np.full(len(idx_h), float(si)),
                jk_h[:, 0],
                jk_h[:, 1],
            ])
            k_q = min(n_neighbors, len(active_coords))
            dists_f, nbr_f = fallback_tree.query(query_pts, k=k_q)
            if k_q == 1:
                dists_f = dists_f.reshape(-1, 1)
                nbr_f   = nbr_f.reshape(-1, 1)
            X_l = active_vals[nbr_f]
            w   = 1.0 / np.maximum(dists_f, 1e-6)
            W_pred[idx_h] = (w * X_l).sum(axis=1) / w.sum(axis=1)

    return W_pred


def _estimate_gamma_from_alpha(held_vals, Y_V, W_V, rho, gamma_min, gamma_max):
    """
    Estimate gamma from direct alpha regression at holdout positions.

    Solves   min_alpha  ||X_V - alpha*W_V - (1-alpha)*Y_V||^2  (OLS no-intercept):
        alpha = sum((X_V - Y_V) * (W_V - Y_V)) / sum((W_V - Y_V)^2)

    Converts mixing weight to gamma (Z-update weight space):
        alpha_W = (gamma+rho) / (gamma+2*rho)   =>   gamma = rho*(2*alpha-1)/(1-alpha)

    alpha in (0.5, 1) corresponds to gamma > 0 (kriging helps).
    alpha <= 0.5 means CP is at least as good => return gamma_min.

    Parameters
    ----------
    held_vals : ndarray (N,)  true X values at holdout positions
    Y_V       : ndarray (N,)  current CP factor values at holdout positions
    W_V       : ndarray (N,)  same-sensor IDW predictions (precomputed, fixed)
    rho, gamma_min, gamma_max : float

    Returns
    -------
    gamma : float clipped to [gamma_min, gamma_max]
    """
    dW    = W_V - Y_V
    dX    = held_vals - Y_V
    denom = float(np.dot(dW, dW))

    if denom < 1e-12:
        return float(gamma_min)     # W_V ≈ Y_V: no kriging signal

    alpha = float(np.dot(dX, dW) / denom)
    if alpha <= 0.5:
        return float(gamma_min)     # CP at least as good

    alpha = float(np.clip(alpha, 0.5001, 0.9999))
    gamma = rho * (2.0 * alpha - 1.0) / (1.0 - alpha)
    return float(np.clip(gamma, gamma_min, gamma_max))


# ---------------------------------------------------------------------------
# Graph Laplacian and temporal smoothness helpers
# ---------------------------------------------------------------------------

def _build_sensor_graph_laplacian(sensor_coords, k=10, sigma=None):
    """
    Build normalized graph Laplacian from sensor spatial coordinates.

    Parameters
    ----------
    sensor_coords : ndarray (I, 2)
    k : int — number of nearest neighbors for graph edges
    sigma : float or None — Gaussian kernel bandwidth.
        If None, set to median pairwise distance among k-nearest neighbors.

    Returns
    -------
    L_s : sparse matrix (I, I) — normalized graph Laplacian L = D^{-1/2} (D - W) D^{-1/2}
    """
    I = len(sensor_coords)
    tree = KDTree(sensor_coords)
    k_use = min(k + 1, I)  # +1 because query includes self
    dists, idxs = tree.query(sensor_coords, k=k_use)

    rows, cols, weights = [], [], []
    for i in range(I):
        for j_idx in range(k_use):
            j = int(idxs[i, j_idx])
            if j == i:
                continue
            d = float(dists[i, j_idx])
            rows.append(i)
            cols.append(j)
            weights.append(d)

    dist_arr = np.array(weights)
    if sigma is None:
        sigma = float(np.median(dist_arr)) if len(dist_arr) > 0 else 1.0
    sigma = max(sigma, 1e-10)

    W_vals = np.exp(-dist_arr ** 2 / (2.0 * sigma ** 2))
    W = coo_matrix((W_vals, (rows, cols)), shape=(I, I)).tocsr()
    # Symmetrize
    W = (W + W.T) / 2.0

    d = np.array(W.sum(axis=1)).ravel()
    d_inv_sqrt = np.where(d > 1e-12, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = diags(d_inv_sqrt)
    # Normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
    L_s = speye(I) - D_inv_sqrt @ W @ D_inv_sqrt
    return L_s.tocsr()


def _build_circular_diff_matrix(K):
    """
    Build circular first-order finite difference operator D₁.

    D₁[k, k] = 2, D₁[k, (k-1)%K] = -1, D₁[k, (k+1)%K] = -1
    with circular boundary conditions (slot 0 and slot K-1 are adjacent).

    The smoothness penalty is ||D₁ C||²_F = tr(C^T D₁^T D₁ C) = tr(C^T D₁ C)
    since D₁ is symmetric and D₁ = D₁^T D₁ for this construction.

    Returns
    -------
    D1 : sparse matrix (K, K) — circular second-difference matrix (tridiagonal + corners)
    """
    diag_main = np.full(K, 2.0)
    diag_upper = np.full(K, -1.0)
    diag_lower = np.full(K, -1.0)
    D1 = diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], shape=(K, K),
               format="lil")
    # Circular entries
    D1[0, K - 1] = -1.0
    D1[K - 1, 0] = -1.0
    return D1.tocsc()


def _solve_circulant_tridiag(lower, main, upper, rhs):
    """
    Solve circulant tridiagonal system T*x = b using Sherman-Morrison.

    T has constant diagonals: main (d), upper (e), lower (c),
    plus corner entries T[0,K-1]=c and T[K-1,0]=e.

    Parameters
    ----------
    lower, main, upper : float — constant diagonal values
    rhs : ndarray (K,) — right-hand side

    Returns
    -------
    x : ndarray (K,)
    """
    K = len(rhs)
    if K < 4:
        # Small system: just use dense solve
        T = np.zeros((K, K))
        for i in range(K):
            T[i, i] = main
            if i > 0:
                T[i, i - 1] = lower
            if i < K - 1:
                T[i, i + 1] = upper
        T[0, K - 1] = lower
        T[K - 1, 0] = upper
        return np.linalg.solve(T, rhs)

    # Standard tridiagonal (remove circular entries)
    diag_l = np.full(K - 1, lower)
    diag_m = np.full(K, main)
    diag_u = np.full(K - 1, upper)

    # Thomas algorithm for T0 * y = rhs and T0 * z = u
    def thomas_solve(dl, dm, du, b):
        n = len(b)
        c = np.empty(n)
        d = np.empty(n)
        x = np.empty(n)
        c[0] = du[0] / dm[0]
        d[0] = b[0] / dm[0]
        for i in range(1, n):
            m = dm[i] - dl[i - 1] * c[i - 1]
            if i < n - 1:
                c[i] = du[i] / m
            d[i] = (b[i] - dl[i - 1] * d[i - 1]) / m
        x[n - 1] = d[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = d[i] - c[i] * x[i + 1]
        return x

    # T0 * y = rhs
    y = thomas_solve(diag_l, diag_m, diag_u, rhs)

    # u vector for rank-1 modification: T = T0 + u * v^T
    # Corner entries: T[0,K-1]=lower, T[K-1,0]=upper
    # u[0] = lower, u[K-1] = upper; v[K-1] = 1, v[0] = 1
    u = np.zeros(K)
    u[0] = lower
    u[K - 1] = upper

    # T0 * z = u
    z = thomas_solve(diag_l, diag_m, diag_u, u)

    # v^T y = y[K-1] + y[0] (v[K-1]=1, v[0]=1)
    # v^T z = z[K-1] + z[0]
    vty = y[K - 1] + y[0]
    vtz = z[K - 1] + z[0]

    # Sherman-Morrison: x = y - z * (v^T y) / (1 + v^T z)
    denom = 1.0 + vtz
    if abs(denom) < 1e-12:
        return y  # fallback
    x = y - z * vty / denom
    return x


# ---------------------------------------------------------------------------
# Main ADMM solver
# ---------------------------------------------------------------------------

def admm_joint(
    X,
    mask,
    locs,
    kriging_params,
    rank=5,
    gamma=1.0,
    beta=0.01,
    rho=1.0,
    max_iter=200,
    tol=1e-4,
    n_neighbors=50,
    seed=0,
    verbose=False,
    log_convergence=False,
    n_jobs=-1,
    adaptive_gamma=False,
    gamma_scale=1.0,
    adaptive_gamma_cfg=None,
    z_init="linear_time",
    sensor_coords=None,
    variogram_params=None,
    kriging_mode="variogram",
    independent_kriging=True,
    idw_power=1,
    idw_spatial_only=False,
    tau_j=3.0,
    tau_k=1.0,
):
    """
    ADMM: joint CP tensor decomposition + kriging.

    Parameters
    ----------
    X, mask        : tensor and observed mask (I, J, K)
    locs           : (I, 2) -- kept for API compat (use sensor_coords instead)
    kriging_params : dict   -- kept for API compat, not used
    rank           : CP rank R
    gamma          : kriging weight (used when adaptive_gamma=False)
    beta           : CP L2 regularization
    rho            : ADMM penalty
    max_iter, tol  : convergence settings
    n_neighbors    : KDTree neighbors per missing position
    seed           : random seed
    verbose        : print per-iteration residuals
    log_convergence: record primal/dual residuals and objective
    n_jobs         : (deprecated, kept for API compatibility)
    adaptive_gamma : if True, use EM per-iteration gamma update with RM-adaptive cap
    gamma_scale    : multiplicative scale on initial gamma (default 1.0)
    adaptive_gamma_cfg : dict with EM update settings
    z_init         : str = "linear_time"
        Warm-start strategy for Z at missing positions.
    sensor_coords  : ndarray (I, 2) or None
        Data-driven MDS-embedded sensor coordinates. If None, falls back to
        3D tensor-index space (legacy mode).
    variogram_params : dict or None
        Fitted variogram parameters {nugget, sill, range_, model}. If None and
        kriging_mode="variogram", variogram will be fitted from data.
    kriging_mode   : str = "variogram"
        "variogram" -- ordinary kriging with fitted variogram (recommended)
        "kkt_index" -- original 3D index-space KDTree + KKT weights (legacy)
        "idw"       -- inverse-distance weighting in data-driven coordinate space
    independent_kriging : bool = True
        If True, W is computed once from observed data before the ADMM loop and
        held fixed, eliminating positive feedback. If False, W is updated each
        iteration using z_tilde (legacy behavior).
    tau_j, tau_k   : float
        Temporal scale factors for day and time-slot axes in the KDTree.

    Returns
    -------
    Z_hat : ndarray (I, J, K)
    info  : dict with keys n_iter, gamma_used, gamma_hist, primal_res,
            dual_res, obj
    """
    I, J, K = X.shape
    R = rank
    rng = np.random.default_rng(seed)
    X_np = X

    # Build index arrays for observed and missing positions
    obs_indices = np.column_stack(np.where(mask)).astype(int)    # (N_obs, 3)
    miss_indices = np.column_stack(np.where(~mask)).astype(int)  # (N_miss, 3)
    missing_positions = list(zip(*np.where(~mask)))
    obs_mean = float(X[mask].mean()) if mask.any() else 0.0
    nm = ~mask

    # --- Build KDTree in appropriate coordinate space ---
    use_spatial = sensor_coords is not None and kriging_mode != "kkt_index"

    if use_spatial:
        # 4D meaningful coordinate space: [sensor_x, sensor_y, j/tau_j, k/tau_k]
        obs_spatial = np.column_stack([
            sensor_coords[obs_indices[:, 0], 0],
            sensor_coords[obs_indices[:, 0], 1],
            obs_indices[:, 1].astype(float) / tau_j,
            obs_indices[:, 2].astype(float) / tau_k,
        ])  # (N_obs, 4)
        obs_tree = KDTree(obs_spatial)
        miss_spatial = np.column_stack([
            sensor_coords[miss_indices[:, 0], 0],
            sensor_coords[miss_indices[:, 0], 1],
            miss_indices[:, 1].astype(float) / tau_j,
            miss_indices[:, 2].astype(float) / tau_k,
        ])  # (N_miss, 4)
    else:
        # Legacy: 3D tensor-index space
        obs_spatial = obs_indices.astype(float)
        obs_tree = KDTree(obs_indices)
        miss_spatial = miss_indices.astype(float)

    # --- Fit variogram if needed ---
    if kriging_mode == "variogram" and use_spatial:
        if variogram_params is None or "nugget" not in (variogram_params or {}):
            from src.kriging import fit_variogram_from_tensor
            variogram_params = fit_variogram_from_tensor(
                X_np, mask, sensor_coords, model="exponential",
            )
            if verbose:
                print("  [variogram fitted] nugget=%.4f  sill=%.4f  range=%.4f  rmse=%.4f" % (
                    variogram_params["nugget"], variogram_params["sill"],
                    variogram_params["range_"], variogram_params.get("rmse", 0.0)))

    # --- Pre-compute independent kriging estimate W_const ---
    W_const = None
    if independent_kriging and len(missing_positions) > 0:
        miss_i = miss_indices[:, 0]
        miss_j = miss_indices[:, 1]
        miss_k = miss_indices[:, 2]
        k_actual = min(n_neighbors, len(obs_indices))

        if kriging_mode == "variogram" and use_spatial and variogram_params is not None:
            # Ordinary kriging: one-time computation from observed data
            from src.kriging import solve_ordinary_kriging_batch, _variogram_func

            _, nbr_idx_all = obs_tree.query(miss_spatial, k=k_actual)
            M = len(miss_indices)
            K_nn = k_actual

            # Vectorized neighbor coordinate/value gathering
            nbr_flat = nbr_idx_all.ravel()
            nbr_coords = obs_spatial[nbr_flat].reshape(M, K_nn, -1)
            nbr_vals = X_np[obs_indices[nbr_flat, 0],
                            obs_indices[nbr_flat, 1],
                            obs_indices[nbr_flat, 2]].reshape(M, K_nn)

            w_vals, _ = solve_ordinary_kriging_batch(
                miss_spatial, nbr_coords, nbr_vals, variogram_params,
            )
            W_const = np.zeros_like(X_np)
            W_const[miss_i, miss_j, miss_k] = w_vals

            if verbose:
                print("  [independent kriging] W computed once: mean=%.4f  std=%.4f" % (
                    float(w_vals.mean()), float(w_vals.std())))

        elif use_spatial:
            # IDW in meaningful coordinate space (vectorized)
            dists_all, nbr_idx_all = obs_tree.query(miss_spatial, k=k_actual)
            if k_actual == 1:
                dists_all = dists_all.reshape(-1, 1)
                nbr_idx_all = nbr_idx_all.reshape(-1, 1)

            M_idw = len(miss_indices)
            K_idw = k_actual
            nbr_flat_idw = nbr_idx_all.ravel()
            nbr_vals_idw = X_np[obs_indices[nbr_flat_idw, 0],
                                 obs_indices[nbr_flat_idw, 1],
                                 obs_indices[nbr_flat_idw, 2]].reshape(M_idw, K_idw)
            d_safe = np.maximum(dists_all, 1e-12)
            w_idw = 1.0 / (d_safe ** idw_power)
            w_idw /= w_idw.sum(axis=1, keepdims=True)
            w_vals = np.einsum("mk,mk->m", w_idw, nbr_vals_idw)

            W_const = np.zeros_like(X_np)
            W_const[miss_i, miss_j, miss_k] = w_vals

            if verbose:
                print("  [independent IDW] W computed once: mean=%.4f  std=%.4f" % (
                    float(w_vals.mean()), float(w_vals.std())))

        else:
            # Legacy KKT in index space — also compute once if independent
            z_tilde_vals = np.full(len(miss_indices), obs_mean)
            _, nbr_idx_all = obs_tree.query(miss_indices, k=k_actual)
            nbr_flat = nbr_idx_all.ravel()
            X_neigh = X_np[obs_indices[nbr_flat, 0],
                           obs_indices[nbr_flat, 1],
                           obs_indices[nbr_flat, 2]].reshape(-1, k_actual)

            mu = X_neigh.mean(axis=1, keepdims=True)
            dX = X_neigh - mu
            denom = (dX ** 2).sum(axis=1)
            safe = denom > 1e-12
            lam = np.ones_like(X_neigh) / k_actual
            lam[safe] += (
                ((z_tilde_vals[safe] - mu[safe, 0]) / denom[safe])[:, None] * dX[safe]
            )
            w_vals = (lam * X_neigh).sum(axis=1)

            W_const = np.zeros_like(X_np)
            W_const[miss_i, miss_j, miss_k] = w_vals

            if verbose:
                print("  [independent KKT index] W computed once: mean=%.4f" % float(w_vals.mean()))

    # --- Parse adaptive_gamma_cfg ---
    cfg = adaptive_gamma_cfg or {}
    ag_warmup    = int(cfg.get("warmup_steps", 15))
    ag_freq      = int(cfg.get("update_freq",   5))
    ag_alpha     = float(cfg.get("ema_alpha",   0.25))
    ag_gmin      = float(cfg.get("gamma_min",   0.1 * rho))
    ag_freeze_f  = float(cfg.get("freeze_frac", 0.5))
    ag_em_factor = float(cfg.get("em_factor",   1.0))
    ag_freeze_t  = int(ag_freeze_f * max_iter)
    ag_gmax_cfg  = cfg.get("gamma_max", None)
    ag_use_b     = bool(cfg.get("use_method_b",  False))
    ag_holdout_f = float(cfg.get("holdout_frac", 0.05))
    ag_max_ho    = int(cfg.get("max_holdout",    1000))

    # --- Initial gamma + RM-adaptive bounds (v9) ---
    if adaptive_gamma:
        N_obs_n  = int(mask.sum())
        N_miss_n = int((~mask).sum())
        gamma_empirical = _empirical_gamma(N_obs_n, N_miss_n, rho=rho)
        gamma   = rho * gamma_scale
        ag_gmax = float(ag_gmax_cfg) if ag_gmax_cfg is not None \
                  else gamma_empirical * ag_em_factor
        ag_gmax = max(ag_gmax, ag_gmin * 2)

        if verbose:
            rm = N_miss_n / (N_obs_n + N_miss_n)
            print("  [adaptive gamma v9 init] "
                  "RM=%.0f%%  gamma_init=%.2f  gamma_max=%.2f  empirical=%.2f  (factor=%.1f)" % (
                      rm * 100, gamma, ag_gmax, gamma_empirical, ag_em_factor))

        # Method B: pre-sample holdout + build per-sensor KDTrees (done once)
        if ag_use_b and ag_freeze_t > 0 and N_obs_n >= 10:
            n_ho = min(ag_max_ho, max(5, int(ag_holdout_f * N_obs_n)))
            _frac_ho = n_ho / float(N_obs_n)
            rng_ho = np.random.default_rng(seed + 1)
            active_coords_b, _, held_coords_b, held_vals_b = _sample_held_out_obs(
                obs_indices, X_np, frac=_frac_ho, rng=rng_ho,
            )
            active_vals_b = X_np[
                active_coords_b[:, 0],
                active_coords_b[:, 1],
                active_coords_b[:, 2],
            ].astype(float)

            # Build sensor trees in meaningful coordinate space if available
            if use_spatial:
                active_spatial_b = np.column_stack([
                    sensor_coords[active_coords_b[:, 0], 0],
                    sensor_coords[active_coords_b[:, 0], 1],
                    active_coords_b[:, 1].astype(float) / tau_j,
                    active_coords_b[:, 2].astype(float) / tau_k,
                ])
                fallback_tree_b = KDTree(active_spatial_b)
            else:
                fallback_tree_b = KDTree(active_coords_b)
            sensor_trees_b  = _build_sensor_trees(
                active_coords_b, tau_j=tau_j, tau_k=tau_k,
            )
            W_V_b = _same_sensor_idw_predict(
                held_coords_b, active_coords_b, active_vals_b,
                sensor_trees_b, fallback_tree_b,
                n_neighbors=n_neighbors, tau_j=tau_j, tau_k=tau_k,
            )
            if verbose:
                print("  [Method B] holdout=%d  active=%d  sensors=%d" % (
                    len(held_coords_b), len(active_coords_b), len(sensor_trees_b)))
        else:
            held_coords_b = held_vals_b = W_V_b = None
    else:
        ag_gmax = float(ag_gmax_cfg) if ag_gmax_cfg is not None else 500.0 * rho
        held_coords_b = held_vals_b = W_V_b = None
    gamma_used = float(gamma)
    gamma_hist = [gamma_used]

    # --- Initialization ---
    Z = X.copy().astype(float)
    if z_init == "linear_time":
        Z_interp = _interp_linear_time(X.astype(float), mask)
        Z[~mask] = Z_interp[~mask]
        if verbose:
            print("  [z_init=linear_time]  missing mean=%.4f  (obs mean=%.4f)" % (
                float(Z[~mask].mean()), obs_mean))
    elif z_init == "daily_profile":
        Z_interp = _interp_daily_profile(X.astype(float), mask)
        Z[~mask] = Z_interp[~mask]
        if verbose:
            print("  [z_init=daily_profile]  missing mean=%.4f  (obs mean=%.4f)" % (
                float(Z[~mask].mean()), obs_mean))
    else:
        Z[~mask] = obs_mean

    A = rng.standard_normal((I, R))
    B = rng.standard_normal((J, R))
    C_fac = rng.standard_normal((K, R))
    Y = _cp_reconstruct(A, B, C_fac)

    # W initialization: use pre-computed W_const if independent, else copy Z
    if independent_kriging and W_const is not None:
        W = W_const.copy()
    else:
        W = Z.copy()
    lambda_ = {}

    Lambda1 = np.zeros_like(Z)
    Lambda2 = np.zeros_like(Z)

    primal_res_log = []
    dual_res_log   = []
    obj_log        = []

    # Pre-build Khatri-Rao indices for ALS
    js_A, ks_A = np.meshgrid(np.arange(J), np.arange(K), indexing="ij")
    js_A_flat, ks_A_flat = js_A.ravel(), ks_A.ravel()
    is_B, ks_B = np.meshgrid(np.arange(I), np.arange(K), indexing="ij")
    is_B_flat, ks_B_flat = is_B.ravel(), ks_B.ravel()
    is_C, js_C = np.meshgrid(np.arange(I), np.arange(J), indexing="ij")
    is_C_flat, js_C_flat = is_C.ravel(), js_C.ravel()

    for t in range(max_iter):
        Y_prev = Y.copy()
        W_prev = W.copy()

        # ---- Step 1: Update Y via ALS ----
        G = Z + Lambda1 / rho
        # Blend observed data into G to help ALS fit observed entries.
        # This accelerates convergence by letting the CP factorization directly
        # see the observed values, rather than relying solely on the Z=Y constraint.
        alpha_blend = 1.0 / (1.0 + rho)
        G[mask] = alpha_blend * X[mask] + (1.0 - alpha_blend) * G[mask]

        V_A = B[js_A_flat] * C_fac[ks_A_flat]
        MTM_A = V_A.T @ V_A + (beta / rho) * np.eye(R)
        for i in range(I):
            A[i] = np.linalg.solve(MTM_A, V_A.T @ G[i].ravel())

        for j in range(J):
            V_B_j = A[is_B_flat] * C_fac[ks_B_flat]
            B[j] = np.linalg.solve(
                V_B_j.T @ V_B_j + (beta / rho) * np.eye(R),
                V_B_j.T @ G[:, j, :].ravel()
            )

        for k in range(K):
            V_C_k = A[is_C_flat] * B[js_C_flat]
            C_fac[k] = np.linalg.solve(
                V_C_k.T @ V_C_k + (beta / rho) * np.eye(R),
                V_C_k.T @ G[:, :, k].ravel()
            )

        Y = _cp_reconstruct(A, B, C_fac)

        # ---- Step 2: Update W (kriging) ----
        if independent_kriging and W_const is not None:
            # W is fixed — independent kriging estimate from observed data
            W = W_const.copy()
        elif len(missing_positions) > 0:
            # Legacy per-iteration W update (z_tilde dependent)
            miss_i = miss_indices[:, 0]
            miss_j = miss_indices[:, 1]
            miss_k = miss_indices[:, 2]

            z_tilde_all = (Z[miss_i, miss_j, miss_k]
                           + Lambda2[miss_i, miss_j, miss_k] / (gamma_used + rho))

            k_actual = min(n_neighbors, len(obs_indices))

            if use_spatial:
                _, nbr_idx_all = obs_tree.query(miss_spatial, k=k_actual)
            else:
                _, nbr_idx_all = obs_tree.query(miss_indices, k=k_actual)

            nbr_flat = nbr_idx_all.ravel()
            X_neigh = X_np[obs_indices[nbr_flat, 0],
                           obs_indices[nbr_flat, 1],
                           obs_indices[nbr_flat, 2]].reshape(-1, k_actual)

            # KKT solver (legacy)
            mu   = X_neigh.mean(axis=1, keepdims=True)
            dX   = X_neigh - mu
            denom = (dX ** 2).sum(axis=1)
            safe  = denom > 1e-12

            lam = np.ones_like(X_neigh) / k_actual
            lam[safe] += (
                ((z_tilde_all[safe] - mu[safe, 0]) / denom[safe])[:, None] * dX[safe]
            )
            w_vals = (lam * X_neigh).sum(axis=1)

            W[miss_i, miss_j, miss_k] = w_vals
            for m_idx, (i, j, k) in enumerate(missing_positions):
                lambda_[(i, j, k)] = lam[m_idx]

        # ---- Step 3: Update Z ----
        Z_new = np.empty_like(Z)
        # Observed positions
        Z_new[mask] = (
            X[mask]
            + rho * (Y[mask] + W[mask])
            - Lambda1[mask]
            - Lambda2[mask]
        ) / (1.0 + 2.0 * rho)

        # Missing positions
        Z_new[nm] = (
            (gamma_used + rho) * W[nm] + rho * Y[nm]
            - Lambda1[nm] - Lambda2[nm]
        ) / (gamma_used + 2.0 * rho)

        Z = Z_new

        # ---- Step 4: Dual update ----
        Lambda1 = Lambda1 + rho * (Z - Y)
        Lambda2 = Lambda2 + rho * (Z - W)

        # ---- Step 5: Adaptive gamma update ----
        if adaptive_gamma and nm.any():
            do_update = (
                t >= ag_warmup and
                (t - ag_warmup) % ag_freq == 0 and
                t < ag_freeze_t
            )
            if do_update:
                if held_coords_b is not None and len(held_coords_b) > 0:
                    Y_V_b = Y[held_coords_b[:, 0],
                               held_coords_b[:, 1],
                               held_coords_b[:, 2]].astype(float)
                    gamma_target = _estimate_gamma_from_alpha(
                        held_vals_b, Y_V_b, W_V_b, rho, ag_gmin, ag_gmax,
                    )
                    if verbose:
                        print("    [B] gamma_target=%.4f" % gamma_target)
                else:
                    gamma_target = _gamma_em_closed_form(
                        W[nm] - Z[nm], rho, ag_gmin, ag_gmax,
                    )
                    if verbose:
                        print("    [A] gamma_em=%.4f" % gamma_target)
                gamma_used = _log_ema(
                    prev=gamma_used, target=gamma_target,
                    alpha=ag_alpha, lo=ag_gmin, hi=ag_gmax,
                )
                if verbose:
                    print("    gamma_ema=%.4f" % gamma_used)
            gamma_hist.append(gamma_used)

        # ---- Convergence check ----
        r_prim = float(
            np.linalg.norm(Z - Y) ** 2 + np.linalg.norm(Z - W) ** 2
        )
        r_dual = float(rho ** 2 * (
            np.linalg.norm(Y - Y_prev) ** 2 + np.linalg.norm(W - W_prev) ** 2
        ))

        if log_convergence:
            fit_obs  = 0.5 * float(np.sum((X[mask] - Z[mask]) ** 2))
            krig_err = (gamma_used / 2.0) * float(np.sum((Z[nm] - W[nm]) ** 2))
            reg      = 0.5 * beta * float(
                np.sum(A**2) + np.sum(B**2) + np.sum(C_fac**2)
            )
            primal_res_log.append(r_prim)
            dual_res_log.append(r_dual)
            obj_log.append(fit_obs + krig_err + reg)

        if verbose:
            print("  ADMM iter %3d: r_prim=%.4e  r_dual=%.4e  gamma=%.2f" % (
                t + 1, r_prim, r_dual, gamma_used))

        if r_prim < tol and r_dual < tol and t > 0:
            break

    info = {
        "n_iter":     t + 1,
        "gamma_used": gamma_used,
        "gamma_hist": gamma_hist,
        "primal_res": primal_res_log,
        "dual_res":   dual_res_log,
        "obj":        obj_log,
    }
    return Z, info


# ---------------------------------------------------------------------------
# Two-block ADMM: Z=Y constraint + kriging penalty (no Z=W constraint)
# ---------------------------------------------------------------------------

def admm_two_block(
    X,
    mask,
    locs,
    kriging_params,
    rank=5,
    gamma=1.0,
    beta=0.01,
    rho=1.0,
    max_iter=200,
    tol=1e-4,
    n_neighbors=50,
    seed=0,
    verbose=False,
    log_convergence=False,
    z_init="linear_time",
    sensor_coords=None,
    variogram_params=None,
    kriging_mode="variogram",
    idw_power=1,
    idw_spatial_only=False,
    tau_j=3.0,
    tau_k=1.0,
    adaptive_gamma=True,
    gamma_min=0.01,
    gamma_max=50.0,
    holdout_frac=0.05,
    max_holdout=2000,
    gamma_est_mode="em",
    lambda_s=0.0,
    lambda_t=0.0,
    graph_k=10,
    graph_sigma=None,
):
    """
    Two-block ADMM: single Z=Y constraint with kriging as quadratic penalty.

    Problem:
        min  (1/2)||P_Omega(X - Z)||_F^2
             + (gamma/2)||P_Omega_bar(Z - W)||_F^2
             + beta * R_Tensor(Theta)
        s.t. Z = Y = [[Theta]]

    Unlike admm_joint (three-block with Z=W constraint), this formulation
    treats kriging as a soft regularizer. When CP is already good (low MR),
    gamma stays small and the solution approaches pure CP-TD. When MR is
    high, gamma increases and kriging provides valuable spatial prior.

    Parameters
    ----------
    Most parameters same as admm_joint(). Key differences:
    adaptive_gamma : bool = True
        If True, estimate gamma each iteration.
    gamma_min, gamma_max : float
        Bounds for adaptive gamma.
    gamma_est_mode : str = "em"
        "em"    -- EM closed-form from W-Z residuals at missing positions (recommended).
                   Avoids the holdout flaw where CP always beats kriging at observed
                   positions, causing gamma to collapse to gamma_min.
        "holdout" -- legacy holdout-based estimation (broken for random missing).
    holdout_frac : float
        Fraction of observed entries to hold out (only used with gamma_est_mode="holdout").
    max_holdout : int
        Maximum holdout set size.
    lambda_s : float = 0.0
        Spatial graph Laplacian regularization strength on factor A.
        When > 0, enforces that nearby sensors have similar low-rank factor vectors,
        enabling information transfer from observed to unobserved sensors.
    lambda_t : float = 0.0
        Temporal smoothness regularization strength on factor C.
        When > 0, enforces smoothness across time slots (circular boundary for daily pattern).
    graph_k : int = 10
        Number of nearest neighbors for building sensor graph Laplacian.
    graph_sigma : float or None
        Gaussian kernel bandwidth for graph weights. If None, uses median k-NN distance.

    Returns
    -------
    Z_hat : ndarray (I, J, K)
    info  : dict
    """
    I, J, K = X.shape
    R = rank
    rng = np.random.default_rng(seed)
    X_np = X

    obs_indices = np.column_stack(np.where(mask)).astype(int)
    miss_indices = np.column_stack(np.where(~mask)).astype(int)
    obs_mean = float(X[mask].mean()) if mask.any() else 0.0
    nm = ~mask
    N_obs = int(mask.sum())
    N_miss = int(nm.sum())

    # --- Build KDTree ---
    use_spatial = sensor_coords is not None and kriging_mode != "kkt_index"

    if use_spatial:
        obs_spatial = np.column_stack([
            sensor_coords[obs_indices[:, 0], 0],
            sensor_coords[obs_indices[:, 0], 1],
            obs_indices[:, 1].astype(float) / tau_j,
            obs_indices[:, 2].astype(float) / tau_k,
        ])
        obs_tree = KDTree(obs_spatial)
        miss_spatial = np.column_stack([
            sensor_coords[miss_indices[:, 0], 0],
            sensor_coords[miss_indices[:, 0], 1],
            miss_indices[:, 1].astype(float) / tau_j,
            miss_indices[:, 2].astype(float) / tau_k,
        ])
    else:
        obs_spatial = obs_indices.astype(float)
        obs_tree = KDTree(obs_indices)
        miss_spatial = miss_indices.astype(float)

    # --- Fit variogram if needed ---
    if kriging_mode == "variogram" and use_spatial:
        if variogram_params is None or "nugget" not in (variogram_params or {}):
            from src.kriging import fit_variogram_from_tensor
            variogram_params = fit_variogram_from_tensor(
                X_np, mask, sensor_coords, model="exponential",
            )

    # --- Pre-compute kriging neighbor structure (reused for iterative W updates) ---
    W = np.zeros_like(X_np)
    # Store neighbor indices and weights for fast W recomputation
    _nbr_idx_cached = None
    _idw_weights_cached = None
    _kriging_weights_cached = None  # (M, K) Kriging weight matrix for variogram mode
    _obs_indices_cached = None
    _kriging_mode_cached = kriging_mode

    if N_miss > 0:
        miss_i = miss_indices[:, 0]
        miss_j = miss_indices[:, 1]
        miss_k = miss_indices[:, 2]
        k_actual = min(n_neighbors, len(obs_indices))

        if kriging_mode == "variogram" and use_spatial and variogram_params is not None:
            from src.kriging import solve_ordinary_kriging_batch

            _, nbr_idx_all = obs_tree.query(miss_spatial, k=k_actual)
            M = len(miss_indices)
            K_nn = k_actual

            # Vectorized neighbor coordinate/value gathering
            nbr_flat = nbr_idx_all.ravel()
            nbr_coords = obs_spatial[nbr_flat].reshape(M, K_nn, -1)
            nbr_vals = X_np[obs_indices[nbr_flat, 0],
                            obs_indices[nbr_flat, 1],
                            obs_indices[nbr_flat, 2]].reshape(M, K_nn)

            w_vals, _, lam_all = solve_ordinary_kriging_batch(
                miss_spatial, nbr_coords, nbr_vals, variogram_params,
                return_weights=True,
            )
            W[miss_i, miss_j, miss_k] = w_vals
            # Cache for iterative W updates
            _nbr_idx_cached = nbr_idx_all
            _kriging_weights_cached = lam_all  # (M, K_nn)
            _obs_indices_cached = obs_indices
            if verbose:
                print("  [kriging] variogram W computed: mean=%.4f std=%.4f" % (
                    float(w_vals.mean()), float(w_vals.std())))

        elif use_spatial:
            if idw_spatial_only:
                # IDW with spatial-only neighbor search (2D coords, like baseline IDW).
                # For each unobserved sensor, find K nearest observed sensors in 2D
                # space, then compute IDW for all time steps at once.
                # Fully vectorized for speed.
                obs_sensor_ids = np.unique(obs_indices[:, 0])
                obs_sensor_coords = sensor_coords[obs_sensor_ids]
                from scipy.spatial import KDTree as KDTree2D
                spatial_tree = KDTree2D(obs_sensor_coords)

                miss_sensor_ids_unique = np.unique(miss_indices[:, 0])
                n_miss_sensors = len(miss_sensor_ids_unique)
                k_spatial = min(k_actual, len(obs_sensor_ids))
                dists_spatial, spatial_nbr_idx = spatial_tree.query(
                    sensor_coords[miss_sensor_ids_unique], k=k_spatial)
                if k_spatial == 1:
                    dists_spatial = dists_spatial.reshape(-1, 1)
                    spatial_nbr_idx = spatial_nbr_idx.reshape(-1, 1)
                spatial_nbr_sids = obs_sensor_ids[spatial_nbr_idx]  # (n_miss, K)

                # IDW weights from spatial distances (same for all time steps)
                d_safe = np.maximum(dists_spatial, 1e-12)
                spatial_weights = 1.0 / (d_safe ** idw_power)  # (n_miss, K)
                spatial_weights /= spatial_weights.sum(axis=1, keepdims=True)

                # Compute W for each unobserved sensor using matrix multiplication
                # W[uid, :, :] = weights[si] @ X_obs[nbr_sids[si], :, :]
                # where weights is (K,) and X_obs[nbr_sids, :, :] is (K, J, K_dim)
                # Result is (J, K_dim) for each sensor
                for si in range(n_miss_sensors):
                    uid = miss_sensor_ids_unique[si]
                    nbrs = spatial_nbr_sids[si]  # (K,)
                    w = spatial_weights[si]       # (K,)
                    # Get all neighbor time series: (K, J, K_dim)
                    nbr_data = X_np[nbrs]
                    # Weighted sum: w (K,) @ nbr_data (K, J*K_dim) -> (J*K_dim,)
                    W[uid] = (w[:, None, None] * nbr_data).sum(axis=0)

                # For W-iter: we need to cache neighbor info in the format used
                # by _recompute_W_idw. Since spatial neighbors are the same for
                # all time steps of a sensor, we build the cache efficiently.
                M_idw = len(miss_indices)
                K_idw = k_spatial
                nbr_idx_all = np.zeros((M_idw, K_idw), dtype=int)
                w_idw_all = np.zeros((M_idw, K_idw))

                # Build obs_sid_starts: for each deployed sensor, the starting
                # row in obs_indices. Since obs_indices comes from np.where(mask),
                # entries are sorted by (sensor, j, k) and fully-observed deployed
                # sensors have J*K contiguous rows.
                obs_sid_starts = np.full(X_np.shape[0], -1, dtype=int)
                prev_sid = -1
                for oi in range(len(obs_indices)):
                    sid = int(obs_indices[oi, 0])
                    if sid != prev_sid:
                        obs_sid_starts[sid] = oi
                        prev_sid = sid

                # Map miss_sensor_id -> row index in miss_sensor_ids_unique
                miss_sid_to_row = np.full(X_np.shape[0], -1, dtype=int)
                for r, sid in enumerate(miss_sensor_ids_unique):
                    miss_sid_to_row[sid] = r

                # Vectorized cache building:
                # For each missing element (i,j,k), the neighbor obs_indices rows
                # are: obs_sid_starts[nbr_sid] + j * KK + k
                KK = X_np.shape[2]
                miss_rows = miss_sid_to_row[miss_i]  # (M,) -> row in miss_sensor_ids_unique
                # Set weights: same weights for all time steps of same sensor
                w_idw_all[:] = spatial_weights[miss_rows]  # (M, K)

                # For each neighbor ki, compute nbr_idx_all[:, ki]
                nbr_sids_expanded = spatial_nbr_sids[miss_rows]  # (M, K) -> neighbor sensor IDs
                nbr_starts = obs_sid_starts[nbr_sids_expanded]  # (M, K)
                # Offset: j * KK + k for each missing element
                jk_offsets = miss_j * KK + miss_k  # (M,)
                nbr_idx_all[:] = nbr_starts + jk_offsets[:, None]  # (M, K)

                _nbr_idx_cached = nbr_idx_all
                _idw_weights_cached = w_idw_all
                _obs_indices_cached = obs_indices
                if verbose:
                    print("  [kriging] IDW (spatial-only) W computed: mean=%.4f" % float(W[miss_i, miss_j, miss_k].mean()))
            else:
                # IDW (vectorized, 4D coords) — cache neighbor info for iterative updates
                dists_all, nbr_idx_all = obs_tree.query(miss_spatial, k=k_actual)
                if k_actual == 1:
                    dists_all = dists_all.reshape(-1, 1)
                    nbr_idx_all = nbr_idx_all.reshape(-1, 1)
                M_idw = len(miss_indices)
                K_idw = k_actual
                nbr_flat_idw = nbr_idx_all.ravel()
                nbr_vals_idw = X_np[obs_indices[nbr_flat_idw, 0],
                                     obs_indices[nbr_flat_idw, 1],
                                     obs_indices[nbr_flat_idw, 2]].reshape(M_idw, K_idw)
                d_safe = np.maximum(dists_all, 1e-12)
                w_idw = 1.0 / (d_safe ** idw_power)
                w_idw /= w_idw.sum(axis=1, keepdims=True)
                w_vals = np.einsum("mk,mk->m", w_idw, nbr_vals_idw)
                W[miss_i, miss_j, miss_k] = w_vals

                # Cache for iterative W updates
                _nbr_idx_cached = nbr_idx_all
                _idw_weights_cached = w_idw

                if verbose:
                    print("  [kriging] IDW W computed: mean=%.4f" % float(w_vals.mean()))

        else:
            # Legacy KKT in index space
            _, nbr_idx_all = obs_tree.query(miss_indices, k=k_actual)
            nbr_flat = nbr_idx_all.ravel()
            X_neigh = X_np[obs_indices[nbr_flat, 0],
                           obs_indices[nbr_flat, 1],
                           obs_indices[nbr_flat, 2]].reshape(-1, k_actual)

            z_tilde_init = np.full(len(miss_indices), obs_mean)
            mu = X_neigh.mean(axis=1, keepdims=True)
            dX = X_neigh - mu
            denom = (dX ** 2).sum(axis=1)
            safe = denom > 1e-12
            lam = np.ones_like(X_neigh) / k_actual
            lam[safe] += (
                ((z_tilde_init[safe] - mu[safe, 0]) / denom[safe])[:, None] * dX[safe]
            )
            w_vals = (lam * X_neigh).sum(axis=1)
            W[miss_i, miss_j, miss_k] = w_vals
            if verbose:
                print("  [kriging] KKT W computed: mean=%.4f" % float(w_vals.mean()))

    def _recompute_W_idw(Z_curr):
        """Recompute W using current Z values at observed neighbor positions.

        This is the key fix: by updating W each iteration with the current
        Z values (which include CP-refined estimates), kriging can benefit
        from CP information, creating true bidirectional synergy.

        Only works for IDW mode where we have cached neighbor indices and weights.
        """
        if _nbr_idx_cached is None or _idw_weights_cached is None:
            return W  # cannot recompute without cached info
        nbr_flat = _nbr_idx_cached.ravel()
        M_idw = len(miss_indices)
        K_idw = _nbr_idx_cached.shape[1]
        # Use the appropriate obs_indices source
        oi = _obs_indices_cached if _obs_indices_cached is not None else obs_indices
        # Clip indices to valid range (some may be padded with 0 for invalid neighbors)
        max_idx = len(oi) - 1
        nbr_flat_clipped = np.clip(nbr_flat, 0, max_idx)
        nbr_vals_new = Z_curr[oi[nbr_flat_clipped, 0],
                              oi[nbr_flat_clipped, 1],
                              oi[nbr_flat_clipped, 2]].reshape(M_idw, K_idw)
        w_vals_new = np.einsum("mk,mk->m", _idw_weights_cached, nbr_vals_new)
        W_new = np.zeros_like(Z_curr)
        W_new[miss_i, miss_j, miss_k] = w_vals_new
        return W_new

    def _recompute_W_variogram(Z_curr):
        """Recompute W for variogram kriging using cached weights.

        Kriging weights depend only on spatial configuration (not values),
        so we can reuse the pre-solved weight matrix and just apply it to
        the current Z values at neighbor positions.
        """
        if _nbr_idx_cached is None or _kriging_weights_cached is None:
            return W
        nbr_flat = _nbr_idx_cached.ravel()
        M_v = len(miss_indices)
        K_v = _nbr_idx_cached.shape[1]
        # Use CURRENT Z values at observed positions (includes CP refinement)
        nbr_vals_new = Z_curr[_obs_indices_cached[nbr_flat, 0],
                              _obs_indices_cached[nbr_flat, 1],
                              _obs_indices_cached[nbr_flat, 2]].reshape(M_v, K_v)
        w_vals_new = np.einsum("mk,mk->m", _kriging_weights_cached, nbr_vals_new)
        W_new = np.zeros_like(Z_curr)
        W_new[miss_i, miss_j, miss_k] = w_vals_new
        return W_new

    # --- Holdout setup for adaptive gamma (only for holdout mode) ---
    held_coords = held_vals = W_V_holdout = None
    if adaptive_gamma and gamma_est_mode == "holdout" and N_obs >= 20:
        n_ho = min(max_holdout, max(10, int(holdout_frac * N_obs)))
        ho_idx = rng.choice(N_obs, size=n_ho, replace=False)
        held_coords = obs_indices[ho_idx]
        held_vals = X_np[held_coords[:, 0], held_coords[:, 1],
                         held_coords[:, 2]].astype(float)

        # Build kriging prediction at holdout positions (LOO-style)
        active_mask_arr = np.ones(N_obs, dtype=bool)
        active_mask_arr[ho_idx] = False
        active_indices = obs_indices[active_mask_arr]

        if use_spatial:
            active_spatial = np.column_stack([
                sensor_coords[active_indices[:, 0], 0],
                sensor_coords[active_indices[:, 0], 1],
                active_indices[:, 1].astype(float) / tau_j,
                active_indices[:, 2].astype(float) / tau_k,
            ])
            active_tree = KDTree(active_spatial)
            held_spatial = np.column_stack([
                sensor_coords[held_coords[:, 0], 0],
                sensor_coords[held_coords[:, 0], 1],
                held_coords[:, 1].astype(float) / tau_j,
                held_coords[:, 2].astype(float) / tau_k,
            ])
        else:
            active_tree = KDTree(active_indices)
            held_spatial = held_coords.astype(float)

        k_ho = min(n_neighbors, len(active_indices))
        dists_ho, nbr_ho = active_tree.query(held_spatial, k=k_ho)
        if k_ho == 1:
            dists_ho = dists_ho.reshape(-1, 1)
            nbr_ho = nbr_ho.reshape(-1, 1)

        active_vals = X_np[active_indices[:, 0], active_indices[:, 1],
                           active_indices[:, 2]].astype(float)
        X_nbr_ho = active_vals[nbr_ho]
        w_ho = 1.0 / np.maximum(dists_ho, 1e-12)
        w_ho /= w_ho.sum(axis=1, keepdims=True)
        W_V_holdout = (w_ho * X_nbr_ho).sum(axis=1)

        if verbose:
            sigma_w = float(np.mean((held_vals - W_V_holdout) ** 2))
            print("  [holdout] n=%d  sigma_w^2=%.6f" % (n_ho, sigma_w))

    elif adaptive_gamma and gamma_est_mode == "em":
        # EM mode: set gamma_min/max from RM-adaptive empirical estimate
        if N_miss > 0:
            gamma_empirical = _empirical_gamma(N_obs, N_miss, rho=rho)
            # EM gamma_max: allow up to 2x the empirical estimate
            em_gamma_max = max(gamma_empirical * 2.0, 100.0 * rho)
            # Don't exceed the user-specified gamma_max
            em_gamma_max = min(em_gamma_max, gamma_max)
            em_gamma_min = max(gamma_min, 0.1 * rho)
        else:
            em_gamma_max = gamma_max
            em_gamma_min = gamma_min

        if verbose:
            rm = N_miss / (N_obs + N_miss) if (N_obs + N_miss) > 0 else 0
            print("  [EM gamma] RM=%.0f%%  empirical=%.1f  em_max=%.1f  em_min=%.4f" % (
                rm * 100, gamma_empirical if N_miss > 0 else 0, em_gamma_max, em_gamma_min))
    else:
        em_gamma_max = gamma_max
        em_gamma_min = gamma_min

    # --- Initialization ---
    Z = X.copy().astype(float)
    if z_init == "linear_time":
        Z_interp = _interp_linear_time(X.astype(float), mask)
        Z[~mask] = Z_interp[~mask]
    elif z_init == "daily_profile":
        Z_interp = _interp_daily_profile(X.astype(float), mask)
        Z[~mask] = Z_interp[~mask]
    else:
        Z[~mask] = obs_mean

    # Recompute W using the interpolated Z (not the raw X with zeros at fault positions).
    # This is critical when fault positions exist: X has zeros at fault locations,
    # but Z has been initialized with linear-time interpolation, giving much better
    # neighbor values for the IDW step.
    if N_miss > 0 and _nbr_idx_cached is not None:
        if _kriging_weights_cached is not None:
            W = _recompute_W_variogram(Z)
        else:
            W = _recompute_W_idw(Z)

    A = rng.standard_normal((I, R))
    B = rng.standard_normal((J, R))
    C_fac = rng.standard_normal((K, R))
    Y = _cp_reconstruct(A, B, C_fac)

    Lambda = np.zeros_like(Z)  # single dual variable

    primal_res_log = []
    dual_res_log = []
    obj_log = []
    gamma_hist = [float(gamma)]
    gamma_used = float(gamma)

    # --- Pre-compute graph Laplacian and temporal smoothness matrix ---
    L_s = None
    D1 = None
    if lambda_s > 0 and sensor_coords is not None:
        L_s = _build_sensor_graph_laplacian(sensor_coords, k=graph_k, sigma=graph_sigma)
        if verbose:
            print("  [spatial Laplacian] k=%d  sigma=%.4f  nnz=%d" % (
                graph_k, graph_sigma or 0.0, L_s.nnz))
    if lambda_t > 0:
        D1 = _build_circular_diff_matrix(K)
        if verbose:
            print("  [temporal smoothness] lambda_t=%.4f  K=%d" % (lambda_t, K))

    # Pre-build Khatri-Rao indices
    js_A, ks_A = np.meshgrid(np.arange(J), np.arange(K), indexing="ij")
    js_A_flat, ks_A_flat = js_A.ravel(), ks_A.ravel()
    is_B, ks_B = np.meshgrid(np.arange(I), np.arange(K), indexing="ij")
    is_B_flat, ks_B_flat = is_B.ravel(), ks_B.ravel()
    is_C, js_C = np.meshgrid(np.arange(I), np.arange(J), indexing="ij")
    is_C_flat, js_C_flat = is_C.ravel(), js_C.ravel()

    warmup_steps = 10
    W_update_freq = 5  # recompute W every N iterations for bidirectional info flow
    W_blend = 0.5      # blend factor for W update: W = blend*W_new + (1-blend)*W_old

    for t in range(max_iter):
        Y_prev = Y.copy()

        # ---- Step 0: Iterative W update (key fix for bidirectional synergy) ----
        # Every W_update_freq iterations, recompute W using current Z.
        # This lets kriging "see" CP-refined estimates at observed positions,
        # creating true information flow from CP back to kriging.
        # Use blending to prevent oscillation when fault positions exist.
        if t > 0 and t % W_update_freq == 0 and _nbr_idx_cached is not None:
            if _kriging_weights_cached is not None:
                W_new = _recompute_W_variogram(Z)
            else:
                W_new = _recompute_W_idw(Z)
            W = W_blend * W_new + (1.0 - W_blend) * W

        # ---- Step 1: Y-update (ALS) ----
        G = Z + Lambda / rho
        alpha_blend = 1.0 / (1.0 + rho)
        G[mask] = alpha_blend * X[mask] + (1.0 - alpha_blend) * G[mask]

        # KEY FIX: Inject kriging spatial information directly into G at missing
        # positions. This ensures the CP-ALS step sees the spatial structure,
        # rather than having the kriging penalty get erased by the Z=Y constraint
        # at convergence. The injection strength is controlled by gamma.
        if gamma_used > 0 and nm.any():
            krig_weight = gamma_used / (gamma_used + rho)
            G[nm] = krig_weight * W[nm] + (1.0 - krig_weight) * G[nm]

        V_A = B[js_A_flat] * C_fac[ks_A_flat]
        MTM_A = V_A.T @ V_A + (beta / rho) * np.eye(R)

        if L_s is not None and lambda_s > 0:
            # A-update with spatial graph Laplacian regularization
            eigvals_A, Q_A = np.linalg.eigh(MTM_A)
            rhs_A = V_A.T @ G.reshape(I, -1).T  # (R, I)
            rhs_trans = Q_A.T @ rhs_A  # (R, I)
            A_trans = np.empty_like(rhs_trans)
            L_scaled = (lambda_s / rho) * L_s
            for r in range(R):
                sys_mat = eigvals_A[r] * speye(I, format="csr") + L_scaled
                A_trans[r, :] = spsolve(sys_mat, rhs_trans[r, :])
            A = (Q_A @ A_trans).T  # (I, R)
        else:
            for i in range(I):
                A[i] = np.linalg.solve(MTM_A, V_A.T @ G[i].ravel())

        for j in range(J):
            V_B_j = A[is_B_flat] * C_fac[ks_B_flat]
            B[j] = np.linalg.solve(
                V_B_j.T @ V_B_j + (beta / rho) * np.eye(R),
                V_B_j.T @ G[:, j, :].ravel()
            )

        if D1 is not None and lambda_t > 0:
            # C-update with temporal smoothness regularization
            V_C = A[is_C_flat] * B[js_C_flat]  # (I*J, R)
            MTM_C = V_C.T @ V_C + (beta / rho) * np.eye(R)
            eigvals_C, Q_C = np.linalg.eigh(MTM_C)
            rhs_C = np.empty((K, R))
            for k in range(K):
                rhs_C[k, :] = V_C.T @ G[:, :, k].ravel()
            rhs_trans_C = rhs_C @ Q_C  # (K, R)
            C_trans = np.empty_like(rhs_trans_C)
            for r in range(R):
                d_main = eigvals_C[r] + (lambda_t / rho) * 2.0
                d_off = -(lambda_t / rho)
                C_trans[:, r] = _solve_circulant_tridiag(
                    d_off, d_main, d_off, rhs_trans_C[:, r]
                )
            C_fac = C_trans @ Q_C.T  # (K, R)
        else:
            for k in range(K):
                V_C_k = A[is_C_flat] * B[js_C_flat]
                C_fac[k] = np.linalg.solve(
                    V_C_k.T @ V_C_k + (beta / rho) * np.eye(R),
                    V_C_k.T @ G[:, :, k].ravel()
                )

        Y = _cp_reconstruct(A, B, C_fac)

        # ---- Step 2: Z-update ----
        Z_new = np.empty_like(Z)
        # Observed positions: fidelity + Z=Y penalty
        Z_new[mask] = (
            X[mask] + rho * Y[mask] - Lambda[mask]
        ) / (1.0 + rho)

        # Missing positions: kriging penalty + Z=Y penalty
        if gamma_used > 0:
            Z_new[nm] = (
                (gamma_used + rho) * W[nm] + rho * Y[nm] - Lambda[nm]
            ) / (gamma_used + 2.0 * rho)
        else:
            # gamma=0: pure CP-TD (no kriging contribution)
            Z_new[nm] = (rho * Y[nm] - Lambda[nm]) / (2.0 * rho)

        Z = Z_new

        # ---- Step 3: Dual update (single constraint Z=Y) ----
        Lambda = Lambda + rho * (Z - Y)

        # ---- Step 4: Adaptive gamma ----
        if adaptive_gamma and t >= warmup_steps:
            if gamma_est_mode == "em" and nm.any():
                # EM closed-form from W-Z residuals at missing positions
                # This avoids the holdout flaw where CP always beats kriging
                # at observed positions, causing gamma to collapse to gamma_min.
                residuals = W[nm] - Z[nm]
                gamma_target = _gamma_em_closed_form(
                    residuals, rho, em_gamma_min, em_gamma_max,
                )
                gamma_used = _log_ema(
                    prev=gamma_used, target=gamma_target,
                    alpha=0.15, lo=em_gamma_min, hi=em_gamma_max,
                )
            elif gamma_est_mode == "holdout" and held_coords is not None:
                Y_V = Y[held_coords[:, 0], held_coords[:, 1],
                         held_coords[:, 2]].astype(float)
                sigma_y_sq = max(float(np.mean((held_vals - Y_V) ** 2)), 1e-12)
                sigma_w_sq = max(float(np.mean((held_vals - W_V_holdout) ** 2)), 1e-12)

                gamma_target = rho * (sigma_y_sq / sigma_w_sq - 1.0)
                gamma_target = float(np.clip(gamma_target, gamma_min, gamma_max))

                # EMA in log space
                if gamma_used > 0 and gamma_target > 0:
                    alpha_ema = 0.2
                    gamma_used = float(np.exp(
                        (1.0 - alpha_ema) * np.log(max(gamma_used, 1e-8))
                        + alpha_ema * np.log(max(gamma_target, 1e-8))
                    ))
                else:
                    gamma_used = gamma_target
                gamma_used = float(np.clip(gamma_used, gamma_min, gamma_max))

        gamma_hist.append(gamma_used)

        # ---- Convergence check ----
        r_prim = float(np.linalg.norm(Z - Y) ** 2)
        r_dual = float(rho ** 2 * np.linalg.norm(Y - Y_prev) ** 2)

        if log_convergence:
            fit_obs = 0.5 * float(np.sum((X[mask] - Z[mask]) ** 2))
            krig_pen = (gamma_used / 2.0) * float(np.sum((Z[nm] - W[nm]) ** 2))
            reg = 0.5 * beta * float(np.sum(A**2) + np.sum(B**2) + np.sum(C_fac**2))
            primal_res_log.append(r_prim)
            dual_res_log.append(r_dual)
            obj_log.append(fit_obs + krig_pen + reg)

        if verbose and (t + 1) % 10 == 0:
            print("  ADMM-2B iter %3d: r_prim=%.4e  r_dual=%.4e  gamma=%.4f" % (
                t + 1, r_prim, r_dual, gamma_used))

        if r_prim < tol and r_dual < tol and t > 0:
            break

    info = {
        "n_iter": t + 1,
        "gamma_used": gamma_used,
        "gamma_hist": gamma_hist,
        "primal_res": primal_res_log,
        "dual_res": dual_res_log,
        "obj": obj_log,
    }
    return Z, info
