#!/usr/bin/env python3
"""
CLAIRE v5: Temporal causal analysis module

Adapted from MERIT Stage 3 (Granger causality + Transfer entropy + CCM).
Determines causal direction between GM3 contact distance (continuous)
and local lipid counts.

GM3_distance -> CHOL_count : active recruitment (binding drives remodeling)
CHOL_count -> GM3_distance : selective binding (CHOL attracts GM3)
Both directions            : feedback
Neither direction          : independent (passive)
"""

import numpy as np
import pandas as pd
import logging
from collections import Counter

logger = logging.getLogger(__name__)


# ================================================================== #
#  Data preparation
# ================================================================== #

def prepare_timeseries(df, distance_col='target_lipid_min_distance',
                       count_col='CHOL_count', protein=None,
                       bound_only=True, max_gap=5):
    """Extract and prepare paired time series for causal analysis.

    Uses only bound frames (where target lipid is in contact).
    Short gaps (up to max_gap frames) are linearly interpolated.
    The longest contiguous segment is returned.

    Parameters
    ----------
    df : DataFrame
        composition_data.csv with target_lipid_min_distance column
    distance_col : str
        Column for GM3 contact distance
    count_col : str
        Column for lipid count
    protein : str or None
        Protein copy name (if None, use all data)
    bound_only : bool
        If True, use only bound frames with gap interpolation
    max_gap : int
        Maximum gap (in frames) to interpolate

    Returns
    -------
    dist, count : 1D arrays (z-score normalized), empty if insufficient data
    """
    if protein is not None:
        sub = df[df['protein'] == protein].sort_values('frame').reset_index(drop=True)
    else:
        sub = df.sort_values('frame').reset_index(drop=True)

    # If requested columns don't exist, return empty (GM3-free systems)
    if distance_col not in sub.columns or count_col not in sub.columns:
        return np.array([]), np.array([])

    if bound_only and 'target_lipid_bound' in sub.columns:
        bound_mask = sub['target_lipid_bound'].values.astype(bool)
        occupancy = bound_mask.sum() / len(bound_mask) if len(bound_mask) > 0 else 0

        if occupancy < 0.3:
            return np.array([]), np.array([])

        dist_raw = sub[distance_col].values.astype(float)
        count_raw = sub[count_col].values.astype(float)
        dist_raw[~bound_mask] = np.nan
        count_raw[~bound_mask] = np.nan

        # Interpolate short gaps
        dist_raw = _interpolate_short_gaps(dist_raw, max_gap)
        count_raw = _interpolate_short_gaps(count_raw, max_gap)

        # Find longest contiguous non-NaN segment
        valid = np.isfinite(dist_raw) & np.isfinite(count_raw)
        best_start, best_len = _longest_contiguous_true(valid)

        if best_len < 100:
            return np.array([]), np.array([])

        dist = dist_raw[best_start:best_start + best_len]
        count = count_raw[best_start:best_start + best_len]
    else:
        dist = sub[distance_col].values.astype(float)
        count = sub[count_col].values.astype(float)
        mask = np.isfinite(dist) & np.isfinite(count)
        dist, count = dist[mask], count[mask]

    if len(dist) < 100:
        return np.array([]), np.array([])

    if dist.std() > 0:
        dist = (dist - dist.mean()) / dist.std()
    if count.std() > 0:
        count = (count - count.mean()) / count.std()

    return dist, count


def _interpolate_short_gaps(arr, max_gap):
    """Linearly interpolate NaN gaps of length <= max_gap."""
    result = arr.copy()
    n = len(result)
    i = 0
    while i < n:
        if np.isnan(result[i]):
            gap_start = i
            while i < n and np.isnan(result[i]):
                i += 1
            gap_len = i - gap_start
            if gap_len <= max_gap:
                left = result[gap_start - 1] if gap_start > 0 else np.nan
                right = result[i] if i < n else np.nan
                if np.isfinite(left) and np.isfinite(right):
                    for j in range(gap_len):
                        frac = (j + 1) / (gap_len + 1)
                        result[gap_start + j] = left + frac * (right - left)
        else:
            i += 1
    return result


def _longest_contiguous_true(mask):
    """Find start index and length of longest True segment."""
    best_start = 0
    best_len = 0
    cur_start = 0
    cur_len = 0
    for i, v in enumerate(mask):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
    return best_start, best_len


def make_stationary(series):
    """Difference if ADF test rejects stationarity."""
    from scipy.stats import pearsonr
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series, maxlag=20, autolag='AIC')
        if result[1] > 0.05:
            return np.diff(series), True
    except Exception:
        pass
    return series, False


# ================================================================== #
#  Granger Causality
# ================================================================== #

def granger_causality(x, y, max_lag=10):
    """Test if x Granger-causes y.

    Parameters
    ----------
    x : 1D array (cause candidate, stationary)
    y : 1D array (effect candidate, stationary)
    max_lag : int

    Returns
    -------
    dict with best_lag, f_statistic, p_value
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    n = min(len(x), len(y))
    data = np.column_stack([y[:n], x[:n]])

    if n < max_lag + 50:
        return {'best_lag': 0, 'f_statistic': 0.0, 'p_value': 1.0}

    try:
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        best_f = -1
        best_p = 1.0
        best_lag = 1

        for lag in range(1, max_lag + 1):
            f_val = results[lag][0]['ssr_ftest'][0]
            p_val = results[lag][0]['ssr_ftest'][1]
            bonf_p = min(p_val * max_lag, 1.0)
            if f_val > best_f:
                best_f = f_val
                best_p = bonf_p
                best_lag = lag

        return {'best_lag': best_lag, 'f_statistic': best_f, 'p_value': best_p}

    except Exception as e:
        logger.debug(f"GC failed: {e}")
        return {'best_lag': 0, 'f_statistic': 0.0, 'p_value': 1.0}


def conditional_granger_causality(x, y, z, max_lag=10):
    """Test if x Granger-causes y, conditioning on z.

    If x -> y is significant but x -> y | z is not, then the apparent
    causality was driven by z (a common driver such as domain dynamics).

    Method: compare two models:
        Model 1: y(t) = a*y(t-1:t-L) + b*z(t-1:t-L) + e
        Model 2: y(t) = a*y(t-1:t-L) + b*z(t-1:t-L) + c*x(t-1:t-L) + e
    F-test on whether adding x improves prediction beyond y and z.

    Parameters
    ----------
    x : 1D array (cause candidate)
    y : 1D array (effect candidate)
    z : 1D array (conditioning variable, e.g. DIPC count)
    max_lag : int

    Returns
    -------
    dict with best_lag, f_statistic, p_value
    """
    from scipy.stats import f as f_dist

    n = min(len(x), len(y), len(z))
    if n < max_lag + 50:
        return {'best_lag': 0, 'f_statistic': 0.0, 'p_value': 1.0}

    x, y, z = x[:n], y[:n], z[:n]

    best_f = -1
    best_p = 1.0
    best_lag = 1

    for lag in range(1, max_lag + 1):
        # Build lagged matrices
        start = lag
        y_target = y[start:]
        nn = len(y_target)

        # Restricted model: y_past + z_past
        y_lags = np.column_stack([y[start - i - 1:n - i - 1] for i in range(lag)])
        z_lags = np.column_stack([z[start - i - 1:n - i - 1] for i in range(lag)])
        X_restricted = np.column_stack([np.ones(nn), y_lags, z_lags])

        # Full model: y_past + z_past + x_past
        x_lags = np.column_stack([x[start - i - 1:n - i - 1] for i in range(lag)])
        X_full = np.column_stack([X_restricted, x_lags])

        try:
            # OLS for restricted model
            beta_r = np.linalg.lstsq(X_restricted, y_target, rcond=None)[0]
            resid_r = y_target - X_restricted @ beta_r
            ssr_r = np.sum(resid_r**2)

            # OLS for full model
            beta_f = np.linalg.lstsq(X_full, y_target, rcond=None)[0]
            resid_f = y_target - X_full @ beta_f
            ssr_f = np.sum(resid_f**2)

            # F-test
            df_diff = lag  # additional parameters from x_lags
            df_resid = nn - X_full.shape[1]
            if df_resid <= 0 or ssr_f <= 0:
                continue

            f_val = ((ssr_r - ssr_f) / df_diff) / (ssr_f / df_resid)
            p_val = 1.0 - f_dist.cdf(f_val, df_diff, df_resid)
            bonf_p = min(p_val * max_lag, 1.0)

            if f_val > best_f:
                best_f = f_val
                best_p = bonf_p
                best_lag = lag

        except Exception:
            continue

    return {'best_lag': best_lag, 'f_statistic': best_f, 'p_value': best_p}


# ================================================================== #
#  Transfer Entropy
# ================================================================== #

def _discretize(series, n_bins=8):
    """Equal-frequency binning."""
    bins = np.percentile(series, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 3:
        bins = np.linspace(series.min() - 1, series.max() + 1, n_bins + 1)
    bins[0] -= 1
    bins[-1] += 1
    return np.digitize(series, bins[1:])


def _joint_entropy(arr):
    """Joint entropy H(X1, X2, ...)."""
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    tuples = [tuple(row) for row in arr]
    counts = np.array(list(Counter(tuples).values()))
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-12))


def transfer_entropy(source, target, lags, k=3, n_bins=8):
    """Compute TE(source -> target) at multiple lags.

    Parameters
    ----------
    source, target : 1D arrays (stationary)
    lags : list of int
    k : int, embedding dimension
    n_bins : int

    Returns
    -------
    dict with best_lag, te_value, te_by_lag
    """
    src = _discretize(source, n_bins)
    tgt = _discretize(target, n_bins)
    n = len(tgt)

    te_by_lag = {}
    for lag in lags:
        max_offset = lag + k - 1
        if n <= max_offset + 1:
            te_by_lag[lag] = 0.0
            continue

        indices = np.arange(max_offset, n)
        y_future = tgt[indices]
        y_past = np.column_stack([tgt[indices - i - 1] for i in range(k)])
        x_past = np.column_stack([src[indices - lag - i] for i in range(k)])

        h_yf_yp = _joint_entropy(np.column_stack([y_future, y_past]))
        h_yp_xp = _joint_entropy(np.column_stack([y_past, x_past]))
        h_yf_yp_xp = _joint_entropy(np.column_stack([y_future, y_past, x_past]))
        h_yp = _joint_entropy(y_past)

        te = max(h_yf_yp + h_yp_xp - h_yf_yp_xp - h_yp, 0.0)
        te_by_lag[lag] = te

    best_lag = max(te_by_lag, key=te_by_lag.get)
    return {'best_lag': best_lag, 'te_value': te_by_lag[best_lag],
            'te_by_lag': te_by_lag}


# ================================================================== #
#  Surrogate test
# ================================================================== #

def phase_randomize(x):
    """Phase randomization surrogate (preserves power spectrum)."""
    n = len(x)
    ft = np.fft.rfft(x)
    phases = np.random.uniform(0, 2 * np.pi, len(ft))
    phases[0] = 0
    if n % 2 == 0:
        phases[-1] = 0
    ft_surr = np.abs(ft) * np.exp(1j * phases)
    return np.fft.irfft(ft_surr, n=n)


def surrogate_test(source, target, method='gc', max_lag=10,
                   te_lags=None, n_surrogates=100, **kwargs):
    """Surrogate significance test.

    Parameters
    ----------
    source, target : 1D arrays
    method : 'gc' or 'te'
    max_lag : int (for GC)
    te_lags : list of int (for TE)
    n_surrogates : int

    Returns
    -------
    dict with observed, surrogate_p, surrogate_mean, surrogate_std
    """
    if method == 'gc':
        observed = granger_causality(source, target, max_lag)
        obs_stat = observed['f_statistic']

        surr_stats = []
        for _ in range(n_surrogates):
            src_surr = phase_randomize(source)
            r = granger_causality(src_surr, target, max_lag)
            surr_stats.append(r['f_statistic'])

    elif method == 'te':
        if te_lags is None:
            te_lags = [1, 2, 5, 10]
        k = kwargs.get('k', 3)
        n_bins = kwargs.get('n_bins', 8)

        observed = transfer_entropy(source, target, te_lags, k, n_bins)
        obs_stat = observed['te_value']

        surr_stats = []
        for _ in range(n_surrogates):
            src_surr = phase_randomize(source)
            r = transfer_entropy(src_surr, target, te_lags, k, n_bins)
            surr_stats.append(r['te_value'])

    surr_stats = np.array(surr_stats)
    p_value = np.mean(surr_stats >= obs_stat)

    return {
        'observed': observed,
        'observed_statistic': obs_stat,
        'surrogate_p': p_value,
        'surrogate_mean': surr_stats.mean(),
        'surrogate_std': surr_stats.std(),
    }


# ================================================================== #
#  CCM (Convergent Cross Mapping)
# ================================================================== #

def _embed(series, E, tau=1):
    """Takens time-delay embedding."""
    n = len(series)
    max_offset = (E - 1) * tau
    t = np.arange(max_offset, n)
    M = np.column_stack([series[t - i * tau] for i in range(E)])
    return M, t


def ccm(x, y, E=4, tau=1, lib_sizes=None, n_boot=30):
    """Convergent Cross Mapping: test if X causes Y.

    If X causes Y, then the shadow manifold of Y contains information
    about X. So M_Y can predict X, and prediction improves with
    library size (convergence).

    Parameters
    ----------
    x, y : 1D arrays
    E : int, embedding dimension
    tau : int, time delay
    lib_sizes : list of int
    n_boot : int

    Returns
    -------
    dict with rho_by_L, converged, convergence_slope
    """
    if lib_sizes is None:
        n = len(x)
        lib_sizes = [max(E+2, n//20), n//10, n//5, n//2, n]
        lib_sizes = sorted(set(min(l, n) for l in lib_sizes))

    M_y, t_y = _embed(y, E, tau)
    x_t = x[t_y]
    n_emb = len(t_y)
    all_idx = np.arange(n_emb)

    rho_by_L = []
    for L in lib_sizes:
        if L > n_emb or L < E + 2:
            rho_by_L.append(np.nan)
            continue

        rhos = []
        for seed in range(n_boot):
            rng = np.random.default_rng(seed)
            lib_idx = rng.choice(all_idx, size=L, replace=False)

            n_neighbors = E + 1
            x_pred = np.full(n_emb, np.nan)

            M_lib = M_y[lib_idx]
            x_lib = x_t[lib_idx]

            for pi in range(n_emb):
                cands = lib_idx[lib_idx != pi] if pi in set(lib_idx) else lib_idx
                if len(cands) < n_neighbors:
                    continue

                dists = np.sqrt(np.sum((M_y[cands] - M_y[pi])**2, axis=1))
                nn_idx = np.argpartition(dists, n_neighbors)[:n_neighbors]
                nn_dists = dists[nn_idx]

                d_min = nn_dists.min()
                if d_min == 0:
                    w = (nn_dists == 0).astype(float)
                else:
                    w = np.exp(-nn_dists / d_min)
                w /= w.sum()

                x_pred[pi] = np.dot(w, x_t[cands][nn_idx])

            valid = ~np.isnan(x_pred)
            if valid.sum() < 10:
                continue

            rho = np.corrcoef(x_pred[valid], x_t[valid])[0, 1]
            rhos.append(rho)

        rho_by_L.append(np.mean(rhos) if rhos else np.nan)

    # Convergence test
    from scipy.stats import spearmanr
    valid = [(l, r) for l, r in zip(lib_sizes, rho_by_L) if not np.isnan(r)]
    if len(valid) >= 3:
        ls, rs = zip(*valid)
        sr, _ = spearmanr(ls, rs)
        slope = rs[-1] - rs[0]
        converged = (sr > 0.8) and (slope > 0.05)
    else:
        sr, slope, converged = np.nan, np.nan, False

    return {
        'rho_by_L': rho_by_L,
        'lib_sizes': lib_sizes,
        'rho_final': rho_by_L[-1] if rho_by_L else np.nan,
        'convergence_slope': slope,
        'convergence_spearman': sr,
        'converged': converged,
    }


# ================================================================== #
#  Integrated causal direction test
# ================================================================== #

def test_causal_direction(dist, count, max_lag=10,
                          te_lags=None, n_surrogates=100,
                          ccm_E=4, run_ccm=True,
                          condition=None):
    """Test causal direction between GM3 distance and lipid count.

    Runs GC + conditional GC + TE (with surrogates) + CCM in both directions.

    Parameters
    ----------
    dist : 1D array (z-score normalized GM3 distance)
    count : 1D array (z-score normalized lipid count)
    max_lag : int
    te_lags : list of int
    n_surrogates : int
    ccm_E : int
    run_ccm : bool
    condition : 1D array or None
        Conditioning variable (e.g., DIPC count, z-score normalized).
        If provided, conditional GC is run to check for domain dynamics.

    Returns
    -------
    dict with results for both directions and overall classification
    """
    if te_lags is None:
        te_lags = [1, 2, 3, 5, 10]

    # Stationarity
    dist_s, dist_diffed = make_stationary(dist)
    count_s, count_diffed = make_stationary(count)

    # Align after differencing
    n = min(len(dist_s), len(count_s))
    dist_s, count_s = dist_s[:n], count_s[:n]

    # Condition variable: apply same differencing
    cond_s = None
    if condition is not None:
        cond_s, _ = make_stationary(condition)
        cond_s = cond_s[:n]

    results = {}

    # Direction 1: distance -> count (active recruitment)
    print("    Testing: GM3_distance -> lipid_count (active recruitment)")
    gc_fwd = surrogate_test(dist_s, count_s, 'gc', max_lag, n_surrogates=n_surrogates)
    te_fwd = surrogate_test(dist_s, count_s, 'te', te_lags=te_lags, n_surrogates=n_surrogates)
    results['forward'] = {
        'direction': 'distance->count',
        'interpretation': 'active_recruitment',
        'gc': gc_fwd,
        'te': te_fwd,
        'gc_significant': gc_fwd['surrogate_p'] < 0.05,
        'te_significant': te_fwd['surrogate_p'] < 0.05,
    }
    print(f"      GC: F={gc_fwd['observed_statistic']:.2f}, "
          f"p={gc_fwd['surrogate_p']:.3f} "
          f"{'***' if gc_fwd['surrogate_p'] < 0.05 else 'n.s.'}")
    print(f"      TE: {te_fwd['observed_statistic']:.4f}, "
          f"p={te_fwd['surrogate_p']:.3f} "
          f"{'***' if te_fwd['surrogate_p'] < 0.05 else 'n.s.'}")

    # Conditional GC (forward)
    if cond_s is not None:
        cgc_fwd = conditional_granger_causality(dist_s, count_s, cond_s, max_lag)
        results['forward']['cgc'] = cgc_fwd
        results['forward']['cgc_significant'] = cgc_fwd['p_value'] < 0.05
        print(f"      Conditional GC: F={cgc_fwd['f_statistic']:.2f}, "
              f"p={cgc_fwd['p_value']:.3f} "
              f"{'***' if cgc_fwd['p_value'] < 0.05 else 'n.s.'}")

    # Direction 2: count -> distance (selective binding)
    print("    Testing: lipid_count -> GM3_distance (selective binding)")
    gc_rev = surrogate_test(count_s, dist_s, 'gc', max_lag, n_surrogates=n_surrogates)
    te_rev = surrogate_test(count_s, dist_s, 'te', te_lags=te_lags, n_surrogates=n_surrogates)
    results['reverse'] = {
        'direction': 'count->distance',
        'interpretation': 'selective_binding',
        'gc': gc_rev,
        'te': te_rev,
        'gc_significant': gc_rev['surrogate_p'] < 0.05,
        'te_significant': te_rev['surrogate_p'] < 0.05,
    }
    print(f"      GC: F={gc_rev['observed_statistic']:.2f}, "
          f"p={gc_rev['surrogate_p']:.3f} "
          f"{'***' if gc_rev['surrogate_p'] < 0.05 else 'n.s.'}")
    print(f"      TE: {te_rev['observed_statistic']:.4f}, "
          f"p={te_rev['surrogate_p']:.3f} "
          f"{'***' if te_rev['surrogate_p'] < 0.05 else 'n.s.'}")

    # Conditional GC (reverse)
    if cond_s is not None:
        cgc_rev = conditional_granger_causality(count_s, dist_s, cond_s, max_lag)
        results['reverse']['cgc'] = cgc_rev
        results['reverse']['cgc_significant'] = cgc_rev['p_value'] < 0.05
        print(f"      Conditional GC: F={cgc_rev['f_statistic']:.2f}, "
              f"p={cgc_rev['p_value']:.3f} "
              f"{'***' if cgc_rev['p_value'] < 0.05 else 'n.s.'}")

    # CCM (on raw, not differenced)
    if run_ccm and len(dist) > 200:
        print("    Testing: CCM convergence")
        ccm_fwd = ccm(dist, count, E=ccm_E)
        ccm_rev = ccm(count, dist, E=ccm_E)
        results['forward']['ccm'] = ccm_fwd
        results['reverse']['ccm'] = ccm_rev
        results['forward']['ccm_converged'] = ccm_fwd['converged']
        results['reverse']['ccm_converged'] = ccm_rev['converged']
        print(f"      CCM dist->count: rho={ccm_fwd['rho_final']:.3f}, "
              f"converged={'Yes' if ccm_fwd['converged'] else 'No'}")
        print(f"      CCM count->dist: rho={ccm_rev['rho_final']:.3f}, "
              f"converged={'Yes' if ccm_rev['converged'] else 'No'}")

    # Overall classification
    fwd_gc = results['forward']['gc_significant']
    rev_gc = results['reverse']['gc_significant']
    fwd_te = results['forward']['te_significant']
    rev_te = results['reverse']['te_significant']
    fwd_sig = fwd_gc or fwd_te
    rev_sig = rev_gc or rev_te

    # Check if conditional GC changes the conclusion
    if cond_s is not None:
        fwd_cgc = results['forward'].get('cgc_significant', False)
        rev_cgc = results['reverse'].get('cgc_significant', False)

        if fwd_sig and not fwd_cgc and not rev_sig:
            classification = "DOMAIN DYNAMICS"
            print(f"    Note: GC/TE significant but conditional GC not significant")
            print(f"          => apparent causality driven by domain dynamics")
        elif fwd_sig and fwd_cgc and not rev_sig:
            classification = "ACTIVE RECRUITMENT"
        elif rev_sig and rev_cgc and not fwd_sig:
            classification = "SELECTIVE BINDING"
        elif rev_sig and not rev_cgc and not fwd_sig:
            classification = "DOMAIN DYNAMICS"
        elif fwd_sig and rev_sig:
            classification = "FEEDBACK"
        else:
            classification = "INDEPENDENT"
    else:
        if fwd_sig and not rev_sig:
            classification = "ACTIVE RECRUITMENT"
        elif rev_sig and not fwd_sig:
            classification = "SELECTIVE BINDING"
        elif fwd_sig and rev_sig:
            classification = "FEEDBACK"
        else:
            classification = "INDEPENDENT"

    results['classification'] = classification
    print(f"    => {classification}")

    return results


def run_temporal_causal(df, comp_lipids, distance_col='target_lipid_min_distance',
                        max_lag=10, te_lags=None, n_surrogates=100,
                        ccm_E=4, run_ccm=True, bound_only=True):
    """Run temporal causal analysis on all lipids and copies.

    Parameters
    ----------
    df : DataFrame
        composition_data.csv
    comp_lipids : list of str
        Lipid types to test (e.g., ['CHOL', 'DPSM', 'DIPC'])
    distance_col : str
    max_lag : int
    te_lags : list of int
    n_surrogates : int
    ccm_E : int
    run_ccm : bool
    bound_only : bool
        If True, analyze only bound frames (default). If False, use all
        frames (for GM3-free control systems).

    Returns
    -------
    dict of results per lipid, per copy
    """
    if bound_only and distance_col not in df.columns:
        print(f"ERROR: {distance_col} not found in DataFrame.")
        print(f"  Available columns: {list(df.columns)}")
        return {}

    if not bound_only:
        print("\n  *** ALL-FRAMES MODE (no bound filtering) ***\n")

    if te_lags is None:
        te_lags = [1, 2, 3, 5, 10]

    print("\n" + "=" * 70)
    print("CLAIRE v5: MERIT TEMPORAL CAUSAL ANALYSIS")
    print(f"  Methods: Granger Causality + Transfer Entropy + CCM")
    print(f"  Surrogates: {n_surrogates}")
    print(f"  Max lag: {max_lag}, TE lags: {te_lags}")
    print("=" * 70)

    all_results = {}

    # Check if distance column available (not present in GM3-free systems)
    has_distance = distance_col in df.columns

    # --- Pair set 1: GM3 distance vs lipid counts ---
    for lt in comp_lipids:
        count_col = f'{lt}_count'
        if count_col not in df.columns:
            continue

        print(f"\n{'─'*70}")
        print(f"  {lt}: GM3_distance vs {lt}_count")
        print(f"{'─'*70}")

        copy_results = []

        for prot in sorted(df['protein'].unique()):
            print(f"\n  {prot}:")
            dist, count = prepare_timeseries(df, distance_col, count_col, prot, bound_only=bound_only)

            if len(dist) < 200:
                print(f"    Too few data points ({len(dist)}), skipping")
                continue

            # Conditioning variable: use the other major lipid
            # For CHOL, condition on DIPC (both respond to domain dynamics)
            # For DIPC, condition on CHOL
            condition = None
            cond_name = None
            if lt == 'CHOL' and 'DIPC_count' in df.columns:
                cond_name = 'DIPC'
                _, condition = prepare_timeseries(df, distance_col, 'DIPC_count', prot, bound_only=bound_only)
            elif lt == 'DIPC' and 'CHOL_count' in df.columns:
                cond_name = 'CHOL'
                _, condition = prepare_timeseries(df, distance_col, 'CHOL_count', prot, bound_only=bound_only)
            elif lt == 'DPSM' and 'DIPC_count' in df.columns:
                cond_name = 'DIPC'
                _, condition = prepare_timeseries(df, distance_col, 'DIPC_count', prot, bound_only=bound_only)

            if condition is not None and len(condition) == len(count):
                print(f"    Conditioning on: {cond_name}")
            else:
                condition = None

            r = test_causal_direction(
                dist, count, max_lag=max_lag, te_lags=te_lags,
                n_surrogates=n_surrogates, ccm_E=ccm_E, run_ccm=run_ccm,
                condition=condition,
            )
            r['protein'] = prot
            r['n_frames'] = len(dist)
            r['pair'] = f'distance_vs_{lt}'
            copy_results.append(r)

        if copy_results:
            classifications = [cr['classification'] for cr in copy_results]
            from collections import Counter as Cnt
            class_counts = Cnt(classifications)
            dominant = class_counts.most_common(1)[0]
            print(f"\n  distance_vs_{lt} Population: {dict(class_counts)}")
            print(f"  Dominant: {dominant[0]} ({dominant[1]}/{len(copy_results)} copies)")
            all_results[f'distance_vs_{lt}'] = {
                'per_copy': copy_results,
                'classifications': classifications,
                'dominant': dominant[0],
            }

    # --- Pair set 2: TM tilt vs lipid counts (if tilt data available) ---
    if 'tm_tilt' in df.columns and df['tm_tilt'].notna().sum() > 100:
        for lt in comp_lipids:
            count_col = f'{lt}_count'
            if count_col not in df.columns:
                continue

            print(f"\n{'─'*70}")
            print(f"  {lt}: TM_tilt vs {lt}_count")
            print(f"{'─'*70}")

            copy_results = []
            for prot in sorted(df['protein'].unique()):
                print(f"\n  {prot}:")
                tilt, count = prepare_timeseries(df, 'tm_tilt', count_col, prot, bound_only=bound_only)

                if len(tilt) < 200:
                    print(f"    Too few data points ({len(tilt)}), skipping")
                    continue

                condition = None
                cond_name = None
                if lt == 'CHOL' and 'DIPC_count' in df.columns:
                    cond_name = 'DIPC'
                    _, condition = prepare_timeseries(df, 'tm_tilt', 'DIPC_count', prot, bound_only=bound_only)
                elif lt == 'DIPC' and 'CHOL_count' in df.columns:
                    cond_name = 'CHOL'
                    _, condition = prepare_timeseries(df, 'tm_tilt', 'CHOL_count', prot, bound_only=bound_only)

                if condition is not None and len(condition) == len(count):
                    print(f"    Conditioning on: {cond_name}")
                else:
                    condition = None

                r = test_causal_direction(
                    tilt, count, max_lag=max_lag, te_lags=te_lags,
                    n_surrogates=n_surrogates, ccm_E=ccm_E, run_ccm=run_ccm,
                    condition=condition,
                )
                r['protein'] = prot
                r['n_frames'] = len(tilt)
                r['pair'] = f'tilt_vs_{lt}'
                copy_results.append(r)

            if copy_results:
                classifications = [cr['classification'] for cr in copy_results]
                from collections import Counter as Cnt
                class_counts = Cnt(classifications)
                dominant = class_counts.most_common(1)[0]
                print(f"\n  tilt_vs_{lt} Population: {dict(class_counts)}")
                print(f"  Dominant: {dominant[0]} ({dominant[1]}/{len(copy_results)} copies)")
                all_results[f'tilt_vs_{lt}'] = {
                    'per_copy': copy_results,
                    'classifications': classifications,
                    'dominant': dominant[0],
                }

        # --- Pair set 3: GM3 distance vs TM tilt ---
        print(f"\n{'─'*70}")
        print(f"  GM3_distance vs TM_tilt")
        print(f"{'─'*70}")

        copy_results = []
        for prot in sorted(df['protein'].unique()):
            print(f"\n  {prot}:")
            dist, tilt = prepare_timeseries(df, distance_col, 'tm_tilt', prot, bound_only=bound_only)

            if len(dist) < 200:
                print(f"    Too few data points ({len(dist)}), skipping")
                continue

            r = test_causal_direction(
                dist, tilt, max_lag=max_lag, te_lags=te_lags,
                n_surrogates=n_surrogates, ccm_E=ccm_E, run_ccm=run_ccm,
                condition=None,
            )
            r['protein'] = prot
            r['n_frames'] = len(dist)
            r['pair'] = 'distance_vs_tilt'
            copy_results.append(r)

        if copy_results:
            classifications = [cr['classification'] for cr in copy_results]
            from collections import Counter as Cnt
            class_counts = Cnt(classifications)
            dominant = class_counts.most_common(1)[0]
            print(f"\n  distance_vs_tilt Population: {dict(class_counts)}")
            print(f"  Dominant: {dominant[0]} ({dominant[1]}/{len(copy_results)} copies)")
            all_results['distance_vs_tilt'] = {
                'per_copy': copy_results,
                'classifications': classifications,
                'dominant': dominant[0],
            }

    # --- Pair set 3b: Lipid count vs lipid count (15A cross-lipid) ---
    # Tests temporal precedence between lipid species (e.g., CHOL vs DPSM)
    count_cols = [f'{lt}_count' for lt in comp_lipids if f'{lt}_count' in df.columns]
    if len(count_cols) > 1:
        for i, col_x in enumerate(count_cols):
            lt_x = col_x.replace('_count', '')
            for col_y in count_cols[i+1:]:
                lt_y = col_y.replace('_count', '')
                pair_name = f'{lt_x}_vs_{lt_y}'

                print(f"\n{'─'*70}")
                print(f"  {lt_x}_count vs {lt_y}_count")
                print(f"{'─'*70}")

                copy_results = []
                for prot in sorted(df['protein'].unique()):
                    print(f"\n  {prot}:")
                    x, y = prepare_timeseries(df, col_x, col_y, prot, bound_only=bound_only)

                    if len(x) < 200:
                        print(f"    Too few data points ({len(x)}), skipping")
                        continue

                    # Condition on a third lipid if available
                    condition = None
                    remaining = [c for c in count_cols if c != col_x and c != col_y]
                    if remaining:
                        _, condition = prepare_timeseries(df, col_x, remaining[0], prot, bound_only=bound_only)
                        if condition is not None and len(condition) != len(y):
                            condition = None

                    r = test_causal_direction(
                        x, y, max_lag=max_lag, te_lags=te_lags,
                        n_surrogates=n_surrogates, ccm_E=ccm_E, run_ccm=run_ccm,
                        condition=condition,
                    )
                    r['protein'] = prot
                    r['n_frames'] = len(x)
                    r['pair'] = pair_name
                    copy_results.append(r)

                if copy_results:
                    classifications = [cr['classification'] for cr in copy_results]
                    from collections import Counter as Cnt
                    class_counts = Cnt(classifications)
                    dominant = class_counts.most_common(1)[0]
                    print(f"\n  {pair_name} Population: {dict(class_counts)}")
                    print(f"  Dominant: {dominant[0]} ({dominant[1]}/{len(copy_results)} copies)")
                    all_results[pair_name] = {
                        'per_copy': copy_results,
                        'classifications': classifications,
                        'dominant': dominant[0],
                    }

    # --- Pair set 4: local S_CD vs lipid counts (if S_CD data available) ---
    scd_columns = [c for c in ['local_scd', 'local_scd_excl_fs']
                   if c in df.columns and df[c].notna().sum() > 100]

    for scd_col in scd_columns:
        scd_label = 'scd' if scd_col == 'local_scd' else 'scd_xfs'
        for lt in comp_lipids:
            count_col = f'{lt}_count'
            if count_col not in df.columns:
                continue

            print(f"\n{'─'*70}")
            print(f"  {lt}: {scd_col} vs {lt}_count")
            print(f"{'─'*70}")

            copy_results = []
            for prot in sorted(df['protein'].unique()):
                print(f"\n  {prot}:")
                scd, count = prepare_timeseries(df, scd_col, count_col, prot, bound_only=bound_only)

                if len(scd) < 200:
                    print(f"    Too few data points ({len(scd)}), skipping")
                    continue

                condition = None
                cond_name = None
                if lt == 'CHOL' and 'DIPC_count' in df.columns:
                    cond_name = 'DIPC'
                    _, condition = prepare_timeseries(df, scd_col, 'DIPC_count', prot, bound_only=bound_only)
                elif lt == 'DIPC' and 'CHOL_count' in df.columns:
                    cond_name = 'CHOL'
                    _, condition = prepare_timeseries(df, scd_col, 'CHOL_count', prot, bound_only=bound_only)

                if condition is not None and len(condition) == len(count):
                    print(f"    Conditioning on: {cond_name}")
                else:
                    condition = None

                r = test_causal_direction(
                    scd, count, max_lag=max_lag, te_lags=te_lags,
                    n_surrogates=n_surrogates, ccm_E=ccm_E, run_ccm=run_ccm,
                    condition=condition,
                )
                r['protein'] = prot
                r['n_frames'] = len(scd)
                r['pair'] = f'{scd_label}_vs_{lt}'
                copy_results.append(r)

            if copy_results:
                classifications = [cr['classification'] for cr in copy_results]
                from collections import Counter as Cnt
                class_counts = Cnt(classifications)
                dominant = class_counts.most_common(1)[0]
                print(f"\n  {scd_label}_vs_{lt} Population: {dict(class_counts)}")
                print(f"  Dominant: {dominant[0]} ({dominant[1]}/{len(copy_results)} copies)")
                all_results[f'{scd_label}_vs_{lt}'] = {
                    'per_copy': copy_results,
                    'classifications': classifications,
                    'dominant': dominant[0],
                }

        # --- Pair set 5: GM3 distance vs S_CD ---
        print(f"\n{'─'*70}")
        print(f"  GM3_distance vs {scd_col}")
        print(f"{'─'*70}")

        copy_results = []
        for prot in sorted(df['protein'].unique()):
            print(f"\n  {prot}:")
            dist, scd = prepare_timeseries(df, distance_col, scd_col, prot, bound_only=bound_only)

            if len(dist) < 200:
                print(f"    Too few data points ({len(dist)}), skipping")
                continue

            r = test_causal_direction(
                dist, scd, max_lag=max_lag, te_lags=te_lags,
                n_surrogates=n_surrogates, ccm_E=ccm_E, run_ccm=run_ccm,
                condition=None,
            )
            r['protein'] = prot
            r['n_frames'] = len(dist)
            r['pair'] = f'distance_vs_{scd_label}'
            copy_results.append(r)

        if copy_results:
            classifications = [cr['classification'] for cr in copy_results]
            from collections import Counter as Cnt
            class_counts = Cnt(classifications)
            dominant = class_counts.most_common(1)[0]
            print(f"\n  distance_vs_{scd_label} Population: {dict(class_counts)}")
            print(f"  Dominant: {dominant[0]} ({dominant[1]}/{len(copy_results)} copies)")
            all_results[f'distance_vs_{scd_label}'] = {
                'per_copy': copy_results,
                'classifications': classifications,
                'dominant': dominant[0],
            }

    # --- Pair sets 6+: first_shell vs all variables ---
    # Automatically detect all first_shell columns and test against
    # local_scd, all lipid counts (15A), and other first_shell columns
    fs_cols = [c for c in df.columns if c.endswith('_first_shell') and df[c].notna().sum() > 100]

    if fs_cols:
        # 6a: Each first_shell vs each S_CD column
        scd_cols_for_fs = [c for c in ['local_scd', 'local_scd_excl_fs']
                           if c in df.columns and df[c].notna().sum() > 100]

        for scd_col in scd_cols_for_fs:
            scd_lbl = 'scd' if scd_col == 'local_scd' else 'scd_xfs'
            for fs_col in fs_cols:
                fs_lipid = fs_col.replace('_first_shell', '')
                pair_name = f'fs_{fs_lipid}_vs_{scd_lbl}'

                print(f"\n{'─'*70}")
                print(f"  {fs_col} vs {scd_col}")
                print(f"{'─'*70}")

                copy_results = []
                for prot in sorted(df['protein'].unique()):
                    print(f"\n  {prot}:")
                    fs_ts, scd_ts = prepare_timeseries(df, fs_col, scd_col, prot, bound_only=bound_only)
                    if len(fs_ts) < 200:
                        print(f"    Too few data points ({len(fs_ts)}), skipping")
                        continue
                    r = test_causal_direction(
                        fs_ts, scd_ts, max_lag=max_lag, te_lags=te_lags,
                        n_surrogates=n_surrogates, ccm_E=ccm_E, run_ccm=run_ccm,
                        condition=None,
                    )
                    r['protein'] = prot
                    r['n_frames'] = len(fs_ts)
                    r['pair'] = pair_name
                    copy_results.append(r)

                if copy_results:
                    classifications = [cr['classification'] for cr in copy_results]
                    from collections import Counter as Cnt
                    class_counts = Cnt(classifications)
                    dominant = class_counts.most_common(1)[0]
                    print(f"\n  {pair_name} Population: {dict(class_counts)}")
                    print(f"  Dominant: {dominant[0]} ({dominant[1]}/{len(copy_results)} copies)")
                    all_results[pair_name] = {
                        'per_copy': copy_results,
                        'classifications': classifications,
                        'dominant': dominant[0],
                    }

        # 6b: Each first_shell vs each lipid count (15A)
        for fs_col in fs_cols:
            fs_lipid = fs_col.replace('_first_shell', '')
            for lt in comp_lipids:
                count_col = f'{lt}_count'
                if count_col not in df.columns:
                    continue
                pair_name = f'fs_{fs_lipid}_vs_{lt}'

                print(f"\n{'─'*70}")
                print(f"  {fs_col} vs {count_col}")
                print(f"{'─'*70}")

                copy_results = []
                for prot in sorted(df['protein'].unique()):
                    print(f"\n  {prot}:")
                    fs_ts, count_ts = prepare_timeseries(df, fs_col, count_col, prot, bound_only=bound_only)
                    if len(fs_ts) < 200:
                        print(f"    Too few data points ({len(fs_ts)}), skipping")
                        continue

                    # Condition on another lipid count
                    condition = None
                    if lt == 'CHOL' and 'DIPC_count' in df.columns:
                        _, condition = prepare_timeseries(df, fs_col, 'DIPC_count', prot, bound_only=bound_only)
                    elif lt == 'DIPC' and 'CHOL_count' in df.columns:
                        _, condition = prepare_timeseries(df, fs_col, 'CHOL_count', prot, bound_only=bound_only)
                    elif lt == 'DPSM' and 'DIPC_count' in df.columns:
                        _, condition = prepare_timeseries(df, fs_col, 'DIPC_count', prot, bound_only=bound_only)
                    if condition is not None and len(condition) != len(count_ts):
                        condition = None

                    r = test_causal_direction(
                        fs_ts, count_ts, max_lag=max_lag, te_lags=te_lags,
                        n_surrogates=n_surrogates, ccm_E=ccm_E, run_ccm=run_ccm,
                        condition=condition,
                    )
                    r['protein'] = prot
                    r['n_frames'] = len(fs_ts)
                    r['pair'] = pair_name
                    copy_results.append(r)

                if copy_results:
                    classifications = [cr['classification'] for cr in copy_results]
                    from collections import Counter as Cnt
                    class_counts = Cnt(classifications)
                    dominant = class_counts.most_common(1)[0]
                    print(f"\n  {pair_name} Population: {dict(class_counts)}")
                    print(f"  Dominant: {dominant[0]} ({dominant[1]}/{len(copy_results)} copies)")
                    all_results[pair_name] = {
                        'per_copy': copy_results,
                        'classifications': classifications,
                        'dominant': dominant[0],
                    }

        # 6c: Cross first_shell pairs (e.g., fs_DIPC vs fs_DPSM)
        for i, fs_col_x in enumerate(fs_cols):
            fs_lipid_x = fs_col_x.replace('_first_shell', '')
            for fs_col_y in fs_cols[i+1:]:
                fs_lipid_y = fs_col_y.replace('_first_shell', '')
                pair_name = f'fs_{fs_lipid_x}_vs_fs_{fs_lipid_y}'

                print(f"\n{'─'*70}")
                print(f"  {fs_col_x} vs {fs_col_y}")
                print(f"{'─'*70}")

                copy_results = []
                for prot in sorted(df['protein'].unique()):
                    print(f"\n  {prot}:")
                    fs_x, fs_y = prepare_timeseries(df, fs_col_x, fs_col_y, prot, bound_only=bound_only)
                    if len(fs_x) < 200:
                        print(f"    Too few data points ({len(fs_x)}), skipping")
                        continue
                    r = test_causal_direction(
                        fs_x, fs_y, max_lag=max_lag, te_lags=te_lags,
                        n_surrogates=n_surrogates, ccm_E=ccm_E, run_ccm=run_ccm,
                        condition=None,
                    )
                    r['protein'] = prot
                    r['n_frames'] = len(fs_x)
                    r['pair'] = pair_name
                    copy_results.append(r)

                if copy_results:
                    classifications = [cr['classification'] for cr in copy_results]
                    from collections import Counter as Cnt
                    class_counts = Cnt(classifications)
                    dominant = class_counts.most_common(1)[0]
                    print(f"\n  {pair_name} Population: {dict(class_counts)}")
                    print(f"  Dominant: {dominant[0]} ({dominant[1]}/{len(copy_results)} copies)")
                    all_results[pair_name] = {
                        'per_copy': copy_results,
                        'classifications': classifications,
                        'dominant': dominant[0],
                    }

        # 6d: GM3 distance vs each first_shell
        for fs_col in fs_cols:
            fs_lipid = fs_col.replace('_first_shell', '')
            pair_name = f'distance_vs_fs_{fs_lipid}'

            print(f"\n{'─'*70}")
            print(f"  GM3_distance vs {fs_col}")
            print(f"{'─'*70}")

            copy_results = []
            for prot in sorted(df['protein'].unique()):
                print(f"\n  {prot}:")
                dist_ts, fs_ts = prepare_timeseries(df, distance_col, fs_col, prot, bound_only=bound_only)
                if len(dist_ts) < 200:
                    print(f"    Too few data points ({len(dist_ts)}), skipping")
                    continue
                r = test_causal_direction(
                    dist_ts, fs_ts, max_lag=max_lag, te_lags=te_lags,
                    n_surrogates=n_surrogates, ccm_E=ccm_E, run_ccm=run_ccm,
                    condition=None,
                )
                r['protein'] = prot
                r['n_frames'] = len(dist_ts)
                r['pair'] = pair_name
                copy_results.append(r)

            if copy_results:
                classifications = [cr['classification'] for cr in copy_results]
                from collections import Counter as Cnt
                class_counts = Cnt(classifications)
                dominant = class_counts.most_common(1)[0]
                print(f"\n  {pair_name} Population: {dict(class_counts)}")
                print(f"  Dominant: {dominant[0]} ({dominant[1]}/{len(copy_results)} copies)")
                all_results[pair_name] = {
                    'per_copy': copy_results,
                    'classifications': classifications,
                    'dominant': dominant[0],
                }

        # 6e: Each first_shell vs TM tilt
        if 'tm_tilt' in df.columns and df['tm_tilt'].notna().sum() > 100:
            for fs_col in fs_cols:
                fs_lipid = fs_col.replace('_first_shell', '')
                pair_name = f'fs_{fs_lipid}_vs_tilt'

                print(f"\n{'─'*70}")
                print(f"  {fs_col} vs tm_tilt")
                print(f"{'─'*70}")

                copy_results = []
                for prot in sorted(df['protein'].unique()):
                    print(f"\n  {prot}:")
                    fs_ts, tilt_ts = prepare_timeseries(df, fs_col, 'tm_tilt', prot, bound_only=bound_only)
                    if len(fs_ts) < 200:
                        print(f"    Too few data points ({len(fs_ts)}), skipping")
                        continue
                    r = test_causal_direction(
                        fs_ts, tilt_ts, max_lag=max_lag, te_lags=te_lags,
                        n_surrogates=n_surrogates, ccm_E=ccm_E, run_ccm=run_ccm,
                        condition=None,
                    )
                    r['protein'] = prot
                    r['n_frames'] = len(fs_ts)
                    r['pair'] = pair_name
                    copy_results.append(r)

                if copy_results:
                    classifications = [cr['classification'] for cr in copy_results]
                    from collections import Counter as Cnt
                    class_counts = Cnt(classifications)
                    dominant = class_counts.most_common(1)[0]
                    print(f"\n  {pair_name} Population: {dict(class_counts)}")
                    print(f"  Dominant: {dominant[0]} ({dominant[1]}/{len(copy_results)} copies)")
                    all_results[pair_name] = {
                        'per_copy': copy_results,
                        'classifications': classifications,
                        'dominant': dominant[0],
                    }

    # --- Pair set 7: TM tilt vs S_CD ---
    if 'tm_tilt' in df.columns and df['tm_tilt'].notna().sum() > 100:
        for scd_col in scd_columns:
            scd_lbl = 'scd' if scd_col == 'local_scd' else 'scd_xfs'
            pair_name = f'tilt_vs_{scd_lbl}'

            print(f"\n{'─'*70}")
            print(f"  tm_tilt vs {scd_col}")
            print(f"{'─'*70}")

            copy_results = []
            for prot in sorted(df['protein'].unique()):
                print(f"\n  {prot}:")
                tilt_ts, scd_ts = prepare_timeseries(df, 'tm_tilt', scd_col, prot, bound_only=bound_only)
                if len(tilt_ts) < 200:
                    print(f"    Too few data points ({len(tilt_ts)}), skipping")
                    continue
                r = test_causal_direction(
                    tilt_ts, scd_ts, max_lag=max_lag, te_lags=te_lags,
                    n_surrogates=n_surrogates, ccm_E=ccm_E, run_ccm=run_ccm,
                    condition=None,
                )
                r['protein'] = prot
                r['n_frames'] = len(tilt_ts)
                r['pair'] = pair_name
                copy_results.append(r)

            if copy_results:
                classifications = [cr['classification'] for cr in copy_results]
                from collections import Counter as Cnt
                class_counts = Cnt(classifications)
                dominant = class_counts.most_common(1)[0]
                print(f"\n  {pair_name} Population: {dict(class_counts)}")
                print(f"  Dominant: {dominant[0]} ({dominant[1]}/{len(copy_results)} copies)")
                all_results[pair_name] = {
                    'per_copy': copy_results,
                    'classifications': classifications,
                    'dominant': dominant[0],
                }

    # Final summary
    print("\n" + "=" * 70)
    print("TEMPORAL CAUSAL SUMMARY")
    print("=" * 70)
    for lt, r in all_results.items():
        print(f"  {lt}: {r['dominant']}  ({dict(Counter(r['classifications']))})")
    print("=" * 70)

    return all_results


def get_temporal_causal_table(results):
    """Summary table from temporal causal analysis."""
    rows = []
    for lt, r in results.items():
        for cr in r['per_copy']:
            fwd = cr['forward']
            rev = cr['reverse']
            rows.append({
                'pair': cr.get('pair', lt),
                'lipid': lt,
                'protein': cr['protein'],
                'n_frames': cr['n_frames'],
                'classification': cr['classification'],
                'gc_fwd_F': fwd['gc']['observed_statistic'],
                'gc_fwd_p': fwd['gc']['surrogate_p'],
                'gc_rev_F': rev['gc']['observed_statistic'],
                'gc_rev_p': rev['gc']['surrogate_p'],
                'cgc_fwd_F': fwd.get('cgc', {}).get('f_statistic', np.nan),
                'cgc_fwd_p': fwd.get('cgc', {}).get('p_value', np.nan),
                'cgc_rev_F': rev.get('cgc', {}).get('f_statistic', np.nan),
                'cgc_rev_p': rev.get('cgc', {}).get('p_value', np.nan),
                'te_fwd': fwd['te']['observed_statistic'],
                'te_fwd_p': fwd['te']['surrogate_p'],
                'te_rev': rev['te']['observed_statistic'],
                'te_rev_p': rev['te']['surrogate_p'],
                'ccm_fwd_rho': fwd.get('ccm', {}).get('rho_final', np.nan),
                'ccm_fwd_conv': fwd.get('ccm_converged', False),
                'ccm_rev_rho': rev.get('ccm', {}).get('rho_final', np.nan),
                'ccm_rev_conv': rev.get('ccm_converged', False),
            })
    return pd.DataFrame(rows)
