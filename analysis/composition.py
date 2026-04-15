#!/usr/bin/env python3
"""
CLAIRE v5: Hierarchical Bayesian composition analysis with count-based
           causal mediation decomposition and MERIT temporal causal analysis

Stage 2 analysis engine. No MDAnalysis dependency.

Key features:
    1. Hierarchical Bayesian composition analysis (fractions, partial pooling)
    2. Count-based causal mediation (no compositional constraint)
    3. Data-driven mediator selection (counts)
    4. MERIT temporal causal analysis: Granger causality + Transfer entropy
       + CCM on continuous GM3 contact distance vs lipid counts to determine
       causal direction (active recruitment vs selective binding vs passive)
       See analysis/temporal_causal.py for the temporal module.
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    warnings.warn("PyMC not available. Hierarchical Bayesian analysis disabled.")


class CompositionAnalyzer:
    """CLAIRE v3 analysis engine"""

    def __init__(self, lipid_types, target_lipid=None):
        self.lipid_types = lipid_types
        self.target_lipid = target_lipid
        self.comp_lipids = [lt for lt in lipid_types if lt != target_lipid]

    # ================================================================== #
    #  DataFrame construction (from Stage 1)
    # ================================================================== #

    def frames_to_dataframe(self, frame_data_list):
        rows = []
        for fd in frame_data_list:
            if fd is None:
                continue
            # Handle nested structure from calculate_frame_composition
            if 'proteins' in fd:
                for prot_name, pdata in fd['proteins'].items():
                    row = {
                        'frame': fd['frame'],
                        'time': fd['time'],
                        'protein': prot_name,
                        'total_lipids': pdata.get('total_lipids', 0),
                        'target_lipid_bound': pdata.get('target_lipid_bound', False),
                        'target_lipid_min_distance': pdata.get('target_lipid_min_distance', np.nan),
                        'tm_tilt': pdata.get('tm_tilt', np.nan),
                        'local_scd': pdata.get('local_scd', np.nan),
                        'local_scd_excl_fs': pdata.get('local_scd_excl_fs', np.nan),
                    }
                    # First shell counts
                    fs = pdata.get('first_shell_counts', {})
                    for lt, c in fs.items():
                        row[f'{lt}_first_shell'] = c
                    counts = pdata.get('lipid_counts', {})
                    total_comp = sum(counts.get(lt, 0) for lt in self.comp_lipids)
                    row['total_for_ratio'] = total_comp
                    for lt in self.comp_lipids:
                        c = counts.get(lt, 0)
                        row[f'{lt}_count'] = c
                        row[f'{lt}_fraction'] = c / total_comp if total_comp > 0 else 0.0
                    rows.append(row)
            else:
                # Already flattened format
                row = {
                    'frame': fd['frame'],
                    'time': fd['time'],
                    'protein': fd['protein'],
                    'total_lipids': fd['total_lipids'],
                    'target_lipid_bound': fd['target_lipid_bound'],
                    'target_lipid_min_distance': fd.get('target_lipid_min_distance', np.nan),
                    'tm_tilt': fd.get('tm_tilt', np.nan),
                    'local_scd': fd.get('local_scd', np.nan),
                    'local_scd_excl_fs': fd.get('local_scd_excl_fs', np.nan),
                }
                # First shell counts
                fs = fd.get('first_shell_counts', {})
                for lt, c in fs.items():
                    row[f'{lt}_first_shell'] = c
                counts = fd.get('counts', {})
                total_comp = sum(counts.get(lt, 0) for lt in self.comp_lipids)
                row['total_for_ratio'] = total_comp
                for lt in self.comp_lipids:
                    c = counts.get(lt, 0)
                    row[f'{lt}_count'] = c
                    row[f'{lt}_fraction'] = c / total_comp if total_comp > 0 else 0.0
                rows.append(row)
        return pd.DataFrame(rows)

    # ================================================================== #
    #  Utilities
    # ================================================================== #

    def calculate_conservation_ratios(self, df):
        """Add conservation ratio column (sum of composition fractions).
        Should be ~1.0 if all composition lipids are accounted for."""
        frac_cols = [f'{lt}_fraction' for lt in self.comp_lipids
                     if f'{lt}_fraction' in df.columns]
        if frac_cols:
            df = df.copy()
            for lt in self.comp_lipids:
                count_col = f'{lt}_count'
                frac_col = f'{lt}_fraction'
                if count_col in df.columns and frac_col in df.columns:
                    ratio_col = f'{lt}_ratio'
                    total = df['total_for_ratio']
                    df[ratio_col] = df[count_col] / total.where(total > 0, 1)
        return df

    @staticmethod
    def _effective_sample_size(x):
        n = len(x)
        if n < 10:
            return max(n, 1)
        x = x - x.mean()
        var = np.var(x, ddof=1)
        if var < 1e-15:
            return max(n, 1)
        acf = np.correlate(x, x, mode='full')[n-1:] / (var * n)
        tau_int = 0.0
        for k in range(1, min(n // 2, len(acf))):
            if acf[k] <= 0:
                break
            tau_int += acf[k]
        return max(n / (1 + 2 * tau_int), 1)

    def _ensure_counts(self, df):
        """Ensure count columns exist, reconstruct from fraction × total if needed"""
        for lt in self.comp_lipids:
            if f'{lt}_count' not in df.columns:
                if f'{lt}_fraction' in df.columns and 'total_for_ratio' in df.columns:
                    df[f'{lt}_count'] = df[f'{lt}_fraction'] * df['total_for_ratio']
                else:
                    raise ValueError(f"Cannot compute {lt}_count: need {lt}_fraction and total_for_ratio")
        return df

    # ================================================================== #
    #  Hierarchical Bayesian model
    # ================================================================== #

    def _run_hierarchical_model(self, deltas, se_deltas,
                                prior_scale=0.5, prior_scale_between=0.3,
                                n_chains=4, n_tune=2000, n_samples=2000,
                                random_seed=42):
        if not PYMC_AVAILABLE:
            raise RuntimeError("PyMC required")
        with pm.Model():
            mu = pm.Normal('mu', mu=0, sigma=prior_scale)
            sigma_between = pm.HalfNormal('sigma_between', sigma=prior_scale_between)
            total_sd = pm.math.sqrt(sigma_between**2 + se_deltas**2)
            pm.Normal('obs', mu=mu, sigma=total_sd, observed=deltas)
            trace = pm.sample(n_samples, tune=n_tune, chains=n_chains,
                              random_seed=random_seed, progressbar=False,
                              target_accept=0.95)
        mu_s = trace.posterior['mu'].values.flatten()
        hdi = [float(np.percentile(mu_s, 2.5)), float(np.percentile(mu_s, 97.5))]
        return {
            'mu_mean': float(mu_s.mean()), 'mu_hdi': hdi,
            'sigma_between': float(trace.posterior['sigma_between'].values.flatten().mean()),
            'P_positive': float((mu_s > 0).mean()),
            'P_negative': float((mu_s < 0).mean()),
            'rhat': float(az.rhat(trace)['mu'].values),
            'ess': float(az.ess(trace)['mu'].values),
            'trace': trace,
        }

    # ================================================================== #
    #  Composition analysis (FRACTIONS, for reporting)
    # ================================================================== #

    def analyze_composition_hierarchical(self, df, mediator_column='target_lipid_bound',
                                         prior_scale=0.5, prior_scale_between=0.3,
                                         n_chains=4, n_tune=2000, n_samples=2000,
                                         random_seed=42):
        """Hierarchical Bayesian composition analysis (FRACTION-based)"""
        if not PYMC_AVAILABLE:
            raise RuntimeError("PyMC required")

        print("\n" + "=" * 70)
        print("CLAIRE v3: HIERARCHICAL COMPOSITION ANALYSIS (fractions)")
        print("=" * 70)

        results = {}
        for lt in self.comp_lipids:
            frac_col = f'{lt}_fraction'
            if frac_col not in df.columns:
                continue

            deltas, ses, prots = [], [], []
            for prot in sorted(df['protein'].unique()):
                pdf = df[df['protein'] == prot]
                b = pdf[pdf[mediator_column] == True]
                u = pdf[pdf[mediator_column] == False]
                if len(b) < 10 or len(u) < 5:
                    continue
                delta = b[frac_col].mean() - u[frac_col].mean()
                neb = self._effective_sample_size(b[frac_col].values)
                neu = self._effective_sample_size(u[frac_col].values)
                se = np.sqrt(np.std(b[frac_col].values, ddof=1)**2/neb +
                             np.std(u[frac_col].values, ddof=1)**2/neu)
                deltas.append(delta); ses.append(se); prots.append(prot)

            if len(deltas) < 2:
                continue

            deltas_a, ses_a = np.array(deltas), np.array(ses)
            np_ = int((deltas_a > 0).sum()); nn_ = int((deltas_a < 0).sum())

            print(f"\n{'─'*70}\n  {lt}: {len(deltas)} copies  (sign: {np_}+/{nn_}−)")
            for i, p in enumerate(prots):
                print(f"    {p}: Δf = {deltas[i]*100:+.1f}pp")

            b = self._run_hierarchical_model(deltas_a, ses_a, prior_scale,
                                              prior_scale_between, n_chains,
                                              n_tune, n_samples, random_seed)
            print(f"    μ = {b['mu_mean']*100:+.2f}pp  "
                  f"HDI [{b['mu_hdi'][0]*100:.2f}, {b['mu_hdi'][1]*100:.2f}]  "
                  f"P(+) = {b['P_positive']:.3f}")

            results[lt] = {
                'mu_mean': b['mu_mean'], 'mu_hdi': b['mu_hdi'],
                'sigma_between': b['sigma_between'],
                'P_positive': b['P_positive'], 'P_negative': b['P_negative'],
                'rhat': b['rhat'], 'ess': b['ess'],
                'sign_consistency': (np_, nn_), 'per_copy_frac': deltas,
                'proteins': prots, 'trace': b['trace'],
            }

        mu_sum = sum(r['mu_mean'] for r in results.values())
        print(f"\n{'─'*70}\n  Σ μ_Δf = {mu_sum*100:+.4f}pp")
        print("=" * 70)
        return results

    # ================================================================== #
    #  Mediator selection (COUNT-based)
    # ================================================================== #

    def select_mediator(self, df, mediator_column='target_lipid_bound'):
        """Data-driven mediator selection using COUNT-based coupling

        Using raw counts removes Σf=1: coupling reflects physical
        co-occurrence (e.g., CHOL-DPSM positive in Lo domains).

        score = |coupling| × coup_consistency × |Δn_med| × med_consistency
        """
        df = self._ensure_counts(df)

        print("\n" + "=" * 70)
        print("CLAIRE v3: MEDIATOR SELECTION (count-based)")
        print("=" * 70)

        candidate_results = {}
        for candidate in self.comp_lipids:
            responses = [lt for lt in self.comp_lipids if lt != candidate]
            med_col = f'{candidate}_count'
            best_score, best_resp, best_info = 0.0, None, {}

            for resp in responses:
                resp_col = f'{resp}_count'
                coups, med_ds = [], []

                for prot in sorted(df['protein'].unique()):
                    pdf = df[df['protein'] == prot]
                    b = pdf[pdf[mediator_column] == True]
                    u = pdf[pdf[mediator_column] == False]
                    if len(b) < 10 or len(u) < 5:
                        continue
                    T = pdf[mediator_column].astype(float).values
                    M = pdf[med_col].values
                    X = np.column_stack([T, M, np.ones(len(pdf))])
                    y = pdf[resp_col].values
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    coups.append(beta[1])
                    med_ds.append(b[med_col].mean() - u[med_col].mean())

                if len(coups) < 2:
                    continue

                ca, ma = np.array(coups), np.array(med_ds)
                n_neg, n_pos = int((ca < 0).sum()), int((ca > 0).sum())
                coup_cons = max(n_neg, n_pos) / len(ca)
                n_md_p, n_md_n = int((ma > 0).sum()), int((ma < 0).sum())
                med_cons = max(n_md_p, n_md_n) / len(ma)
                score = abs(ca.mean()) * coup_cons * abs(ma.mean()) * med_cons

                print(f"  {candidate} → {resp}:  "
                      f"coupling={ca.mean():+.3f} ({n_pos}+/{n_neg}−)  "
                      f"Δn={ma.mean():+.2f} ({n_md_p}+/{n_md_n}−)  "
                      f"score={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_resp = resp
                    best_info = {'coupling': abs(ca.mean()),
                                 'coupling_consistency': coup_cons,
                                 'mediator_response': abs(ma.mean()),
                                 'mediator_consistency': med_cons}

            if best_resp:
                candidate_results[candidate] = {'best_score': best_score,
                                                 'best_response': best_resp,
                                                 **best_info}

        if not candidate_results:
            return {'selected_mediator': None, 'all_results': {}}

        selected = max(candidate_results, key=lambda k: candidate_results[k]['best_score'])
        r = candidate_results[selected]
        print(f"\n{'─' * 70}")
        print(f"  Selected: {selected}  (score={r['best_score']:.4f})")
        print("=" * 70)
        return {'selected_mediator': selected, 'all_results': candidate_results}

    # ================================================================== #
    #  Causal mediation (COUNT-based)
    # ================================================================== #

    def analyze_mediation(self, df, mediator_lipid='auto',
                          mediator_column='target_lipid_bound',
                          prior_scale=0.5, prior_scale_between=0.3,
                          n_chains=4, n_tune=2000, n_samples=2000,
                          random_seed=42):
        """Count-based causal mediation decomposition

        n_Y = a + β_direct·T + β_coupling·n_M + ε

        Avoids Σf=1 compositional constraint. Coupling reflects
        physical co-occurrence, not mathematical artifact.
        """
        if not PYMC_AVAILABLE:
            raise RuntimeError("PyMC required")

        df = self._ensure_counts(df)

        if mediator_lipid == 'auto':
            sel = self.select_mediator(df, mediator_column)
            mediator_lipid = sel['selected_mediator']
            if not mediator_lipid:
                return {}

        response_lipids = [lt for lt in self.comp_lipids if lt != mediator_lipid]

        print("\n" + "=" * 70)
        print("CLAIRE v3: COUNT-BASED CAUSAL MEDIATION")
        print(f"n_Y = a + β_direct·T + β_coupling·n_{mediator_lipid} + ε")
        print("=" * 70)

        results = {}
        for resp in response_lipids:
            resp_c = f'{resp}_count'
            med_c = f'{mediator_lipid}_count'
            resp_f = f'{resp}_fraction'
            med_f = f'{mediator_lipid}_fraction'

            print(f"\n{'─'*70}\n  {resp}")

            tot_n, dir_n, coup_n, se_t, se_d, prots = [], [], [], [], [], []
            tot_f, dir_f, coup_f = [], [], []  # fraction comparison

            for prot in sorted(df['protein'].unique()):
                pdf = df[df['protein'] == prot]
                b = pdf[pdf[mediator_column] == True]
                u = pdf[pdf[mediator_column] == False]
                if len(b) < 10 or len(u) < 5:
                    continue
                prots.append(prot)

                # COUNT regression
                total = b[resp_c].mean() - u[resp_c].mean()
                T = pdf[mediator_column].astype(float).values
                M = pdf[med_c].values
                X = np.column_stack([T, M, np.ones(len(pdf))])
                y = pdf[resp_c].values
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                direct, coupling = beta[0], beta[1]
                indirect = total - direct

                resid = y - X @ beta
                mse = np.sum(resid**2) / (len(y) - 3)
                se_dir = np.sqrt(mse * np.linalg.inv(X.T @ X)[0, 0])
                neb = self._effective_sample_size(b[resp_c].values)
                neu = self._effective_sample_size(u[resp_c].values)
                se_tot = np.sqrt(np.std(b[resp_c].values, ddof=1)**2/neb +
                                 np.std(u[resp_c].values, ddof=1)**2/neu)

                tot_n.append(total); dir_n.append(direct); coup_n.append(coupling)
                se_t.append(se_tot); se_d.append(se_dir)

                # FRACTION regression (for comparison)
                if resp_f in df.columns and med_f in df.columns:
                    tf = b[resp_f].mean() - u[resp_f].mean()
                    Mf = pdf[med_f].values
                    Xf = np.column_stack([T, Mf, np.ones(len(pdf))])
                    yf = pdf[resp_f].values
                    bf = np.linalg.lstsq(Xf, yf, rcond=None)[0]
                    tot_f.append(tf); dir_f.append(bf[0]); coup_f.append(bf[1])

                med_pct = indirect / total * 100 if abs(total) > 1e-6 else float('nan')
                print(f"    {prot}: Δn={total:+.2f}  direct={direct:+.2f}  "
                      f"indirect={indirect:+.2f}  coupling={coupling:+.3f}")

            if len(tot_n) < 2:
                continue

            ta, da = np.array(tot_n), np.array(dir_n)
            st, sd = np.array(se_t), np.array(se_d)

            print(f"\n    Hierarchical models...")
            tb = self._run_hierarchical_model(ta, st, prior_scale,
                                              prior_scale_between, n_chains,
                                              n_tune, n_samples, random_seed)
            db = self._run_hierarchical_model(da, sd, prior_scale,
                                              prior_scale_between, n_chains,
                                              n_tune, n_samples, random_seed+1)

            mt = tb['trace'].posterior['mu'].values.flatten()
            md = db['trace'].posterior['mu'].values.flatten()
            n_min = min(len(mt), len(md))
            mi = mt[:n_min] - md[:n_min]
            ind_mean = float(mi.mean())
            ind_hdi = [float(np.percentile(mi, 2.5)), float(np.percentile(mi, 97.5))]
            med_post = np.where(np.abs(mt[:n_min]) > 1e-6, mi / mt[:n_min], 0.0)
            med_frac = float(np.median(med_post))

            npt, nnt = int((ta > 0).sum()), int((ta < 0).sum())
            npd, nnd = int((da > 0).sum()), int((da < 0).sum())

            print(f"\n    COUNTS:  total μ={tb['mu_mean']:+.3f}  direct μ={db['mu_mean']:+.3f}  "
                  f"indirect μ={ind_mean:+.3f}  mediation={med_frac*100:.0f}%  "
                  f"coupling={np.mean(coup_n):+.3f}")

            if tot_f:
                fi = np.mean(tot_f) - np.mean(dir_f)
                fm = fi / np.mean(tot_f) * 100 if abs(np.mean(tot_f)) > 1e-6 else float('nan')
                print(f"    FRACS:   total={np.mean(tot_f)*100:+.1f}pp  "
                      f"direct={np.mean(dir_f)*100:+.1f}pp  "
                      f"indirect={fi*100:+.1f}pp  mediation={fm:.0f}%  "
                      f"coupling={np.mean(coup_f):+.3f}")

            results[resp] = {
                'total': {'mu_mean': tb['mu_mean'], 'mu_hdi': tb['mu_hdi'],
                          'sigma_between': tb['sigma_between'],
                          'P_positive': tb['P_positive'], 'P_negative': tb['P_negative'],
                          'sign_consistency': (npt, nnt), 'per_copy': ta.tolist()},
                'direct': {'mu_mean': db['mu_mean'], 'mu_hdi': db['mu_hdi'],
                           'sigma_between': db['sigma_between'],
                           'P_positive': db['P_positive'], 'P_negative': db['P_negative'],
                           'sign_consistency': (npd, nnd), 'per_copy': da.tolist()},
                'indirect': {'mu_mean': ind_mean, 'mu_hdi': ind_hdi},
                'mediation_fraction': med_frac,
                'coupling_count': float(np.mean(coup_n)),
                'coupling_count_per_copy': coup_n,
                'coupling_fraction': float(np.mean(coup_f)) if coup_f else None,
                'fraction_comparison': {
                    'total': float(np.mean(tot_f)),
                    'direct': float(np.mean(dir_f)),
                    'coupling': float(np.mean(coup_f)),
                } if tot_f else None,
                'proteins': prots,
            }

        print("\n" + "=" * 70)
        return results

    # ================================================================== #
    #  Summary tables
    # ================================================================== #

    def get_summary_table(self, results):
        rows = []
        for lt, r in results.items():
            np_, nn_ = r['sign_consistency']
            rows.append({
                'lipid_type': lt,
                'mu_delta_pp': r['mu_mean']*100,
                'hdi_low_pp': r['mu_hdi'][0]*100, 'hdi_high_pp': r['mu_hdi'][1]*100,
                'sigma_between_pp': r['sigma_between']*100,
                'P_positive': r['P_positive'], 'P_negative': r['P_negative'],
                'sign_pos': np_, 'sign_neg': nn_,
                'rhat': r['rhat'], 'ess': r['ess'],
            })
        return pd.DataFrame(rows)

    def get_mediation_table(self, mediation_results, mediator_lipid):
        rows = []
        for resp, r in mediation_results.items():
            np_t, nn_t = r['total']['sign_consistency']
            np_d, nn_d = r['direct']['sign_consistency']
            rows.append({
                'response': resp, 'mediator': mediator_lipid,
                'total_mu': r['total']['mu_mean'],
                'total_hdi': f"[{r['total']['mu_hdi'][0]:+.3f}, {r['total']['mu_hdi'][1]:+.3f}]",
                'direct_mu': r['direct']['mu_mean'],
                'direct_hdi': f"[{r['direct']['mu_hdi'][0]:+.3f}, {r['direct']['mu_hdi'][1]:+.3f}]",
                'indirect_mu': r['indirect']['mu_mean'],
                'mediation_pct': r['mediation_fraction'] * 100,
                'coupling_count': r['coupling_count'],
                'coupling_fraction': r.get('coupling_fraction'),
                'total_sign': f"{np_t}+/{nn_t}−",
                'direct_sign': f"{np_d}+/{nn_d}−",
            })
        return pd.DataFrame(rows)

    # ================================================================== #
    #  Legacy
    # ================================================================== #

    def analyze_composition_changes(self, df, mediator_column='target_lipid_bound',
                                     method='binary'):
        results = {}
        for lt in self.comp_lipids:
            frac_col = f'{lt}_fraction'
            if frac_col not in df.columns:
                continue
            bound = df[df[mediator_column] == True][frac_col]
            unbound = df[df[mediator_column] == False][frac_col]
            if len(bound) < 10 or len(unbound) < 10:
                continue
            t_stat, p_val = stats.ttest_ind(bound, unbound, equal_var=False)
            delta = bound.mean() - unbound.mean()
            results[lt] = {
                'delta': delta, 'delta_pp': delta * 100,
                't_stat': t_stat, 'p_value': p_val,
                'bound_mean': bound.mean(), 'unbound_mean': unbound.mean(),
            }
        return results


    # ================================================================== #
    #  Temporal causal analysis (v5): see analysis/temporal_causal.py
    #  Uses MERIT (GC + TE + CCM) on continuous GM3 distance vs lipid counts
    # ================================================================== #
