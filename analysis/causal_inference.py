#!/usr/bin/env python3
"""
Causal inference for target lipid binding effects on composition

Implements:
1. Granger causality test
2. Transfer entropy
3. Structural causal models (optional)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
import warnings


class CausalInference:
    """Test causal relationships between target lipid binding and composition"""

    def __init__(self, max_lag=10, significance=0.05):
        """Initialize causal inference analyzer

        Parameters
        ----------
        max_lag : int, default 10
            Maximum time lag to test
        significance : float, default 0.05
            Significance level for statistical tests
        """
        self.max_lag = max_lag
        self.significance = significance

    def granger_causality_test(self, x, y, max_lag=None):
        """Granger causality test: Does X Granger-cause Y?

        Tests whether past values of X help predict Y beyond what Y's own past predicts.

        Parameters
        ----------
        x : array-like
            Potential cause variable (e.g., GM3 binding state)
        y : array-like
            Effect variable (e.g., lipid composition)
        max_lag : int, optional
            Maximum lag to test. If None, uses self.max_lag

        Returns
        -------
        dict
            Test results including F-statistic, p-value, and optimal lag
        """
        if max_lag is None:
            max_lag = self.max_lag

        x = np.asarray(x)
        y = np.asarray(y)

        if len(x) != len(y):
            raise ValueError("x and y must have same length")

        # Remove NaN values
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if len(x) < 2 * max_lag:
            return {
                'f_statistic': np.nan,
                'p_value': np.nan,
                'optimal_lag': 0,
                'causes': False,
                'error': 'Insufficient data'
            }

        best_lag = 1
        best_p = 1.0
        best_f = 0.0

        for lag in range(1, max_lag + 1):
            # Build lagged matrices
            n = len(y) - lag

            # Restricted model: Y ~ Y_lag
            Y_restricted = y[lag:]
            X_restricted = np.column_stack([y[lag-i:-i if i > 0 else None] for i in range(1, lag + 1)])

            # Unrestricted model: Y ~ Y_lag + X_lag
            X_unrestricted = np.column_stack([
                X_restricted,
                np.column_stack([x[lag-i:-i if i > 0 else None] for i in range(1, lag + 1)])
            ])

            # Add intercept
            X_restricted = np.column_stack([np.ones(n), X_restricted])
            X_unrestricted = np.column_stack([np.ones(n), X_unrestricted])

            try:
                # Fit restricted model
                beta_restricted = np.linalg.lstsq(X_restricted, Y_restricted, rcond=None)[0]
                residuals_restricted = Y_restricted - X_restricted @ beta_restricted
                rss_restricted = np.sum(residuals_restricted ** 2)

                # Fit unrestricted model
                beta_unrestricted = np.linalg.lstsq(X_unrestricted, Y_restricted, rcond=None)[0]
                residuals_unrestricted = Y_restricted - X_unrestricted @ beta_unrestricted
                rss_unrestricted = np.sum(residuals_unrestricted ** 2)

                # F-test
                df1 = lag  # Number of restrictions (X lags added)
                df2 = n - X_unrestricted.shape[1]

                if df2 <= 0 or rss_unrestricted <= 0:
                    continue

                f_stat = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2)
                p_value = 1 - stats.f.cdf(f_stat, df1, df2)

                if p_value < best_p:
                    best_p = p_value
                    best_f = f_stat
                    best_lag = lag

            except (np.linalg.LinAlgError, ValueError):
                continue

        return {
            'f_statistic': best_f,
            'p_value': best_p,
            'optimal_lag': best_lag,
            'causes': best_p < self.significance,
            'max_lag_tested': max_lag
        }

    def transfer_entropy(self, source, target, k=1, bins=10):
        """Calculate transfer entropy from source to target

        Transfer entropy quantifies information flow: how much knowing the past
        of source reduces uncertainty about the future of target.

        Parameters
        ----------
        source : array-like
            Source time series (e.g., GM3 binding)
        target : array-like
            Target time series (e.g., lipid composition)
        k : int, default 1
            Time lag
        bins : int, default 10
            Number of bins for discretization

        Returns
        -------
        dict
            Transfer entropy value and significance
        """
        source = np.asarray(source)
        target = np.asarray(target)

        # Remove NaN
        mask = np.isfinite(source) & np.isfinite(target)
        source = source[mask]
        target = target[mask]

        if len(source) < k + 10:
            return {
                'transfer_entropy': np.nan,
                'normalized_te': np.nan,
                'error': 'Insufficient data'
            }

        # Discretize continuous variables
        source_binned = self._discretize(source, bins)
        target_binned = self._discretize(target, bins)

        # Build time-lagged variables
        n = len(target) - k
        target_future = target_binned[k:]
        target_past = target_binned[:n]
        source_past = source_binned[:n]

        # Calculate transfer entropy using histogram method
        te = self._calculate_te_histogram(source_past, target_past, target_future, bins)

        # Normalize by target entropy
        h_target = self._entropy_histogram(target_future, bins)
        normalized_te = te / h_target if h_target > 0 else 0

        return {
            'transfer_entropy': te,
            'normalized_te': normalized_te,
            'k': k,
            'bins': bins
        }

    def _discretize(self, x, bins):
        """Discretize continuous variable into bins"""
        if len(np.unique(x)) <= bins:
            # Already discrete
            return x.astype(int)

        # Use quantile-based binning
        return pd.qcut(x, bins, labels=False, duplicates='drop')

    def _calculate_te_histogram(self, source_past, target_past, target_future, bins):
        """Calculate transfer entropy using histogram method"""
        # Joint probabilities
        p_target_future = self._get_probabilities(target_future, bins)
        p_target_past = self._get_probabilities(target_past, bins)
        p_source_past = self._get_probabilities(source_past, bins)

        # Conditional probabilities
        te = 0.0
        n = len(target_future)

        for i in range(n):
            tf = target_future[i]
            tp = target_past[i]
            sp = source_past[i]

            # p(target_future | target_past, source_past)
            mask1 = (target_past == tp) & (source_past == sp)
            if np.sum(mask1) > 0:
                p_tf_given_tp_sp = np.mean(target_future[mask1] == tf)
            else:
                continue

            # p(target_future | target_past)
            mask2 = (target_past == tp)
            if np.sum(mask2) > 0:
                p_tf_given_tp = np.mean(target_future[mask2] == tf)
            else:
                continue

            if p_tf_given_tp_sp > 0 and p_tf_given_tp > 0:
                te += (1/n) * np.log2(p_tf_given_tp_sp / p_tf_given_tp)

        return te

    def _get_probabilities(self, x, bins):
        """Get probability distribution"""
        counts = np.bincount(x.astype(int), minlength=bins)
        return counts / len(x)

    def _entropy_histogram(self, x, bins):
        """Calculate entropy using histogram"""
        p = self._get_probabilities(x, bins)
        p = p[p > 0]  # Remove zero probabilities
        return -np.sum(p * np.log2(p))

    def analyze_causality(self, df, target_lipid='DPG3', composition_lipids=None):
        """Comprehensive causal analysis

        Parameters
        ----------
        df : pandas.DataFrame
            Data with 'target_lipid_bound' and composition columns
        target_lipid : str
            Target lipid name
        composition_lipids : list of str, optional
            List of lipid types to analyze

        Returns
        -------
        dict
            Causal inference results for each protein and lipid type
        """
        print("\n" + "="*70)
        print("CAUSAL INFERENCE ANALYSIS")
        print("="*70)
        print(f"Testing: {target_lipid} binding → composition changes")
        print(f"Max lag: {self.max_lag} frames")
        print(f"Significance level: {self.significance}")

        if composition_lipids is None:
            composition_lipids = ['CHOL', 'DPSM', 'DIPC']

        if 'target_lipid_bound' not in df.columns:
            print("ERROR: 'target_lipid_bound' column not found")
            return {}

        if 'protein' not in df.columns:
            print("ERROR: 'protein' column not found")
            return {}

        results = {}

        for protein_name in sorted(df['protein'].unique()):
            protein_df = df[df['protein'] == protein_name].sort_values('frame')

            # Convert boolean to numeric for causality tests
            x = protein_df['target_lipid_bound'].astype(float).values

            print(f"\n{protein_name}:")
            print("-" * 70)

            results[protein_name] = {}

            for lipid_type in composition_lipids:
                col_name = f'{lipid_type}_fraction'
                if col_name not in protein_df.columns:
                    continue

                y = protein_df[col_name].values

                # Granger causality test
                granger_result = self.granger_causality_test(x, y)

                # Transfer entropy
                te_result = self.transfer_entropy(x, y, k=1, bins=5)

                results[protein_name][lipid_type] = {
                    'granger': granger_result,
                    'transfer_entropy': te_result
                }

                # Print results
                print(f"\n  {lipid_type}:")
                print(f"    Granger causality:")
                print(f"      F-statistic: {granger_result['f_statistic']:.4f}")
                print(f"      p-value: {granger_result['p_value']:.4f}")
                print(f"      Optimal lag: {granger_result['optimal_lag']} frames")
                print(f"      Causes: {'YES ***' if granger_result['causes'] else 'NO'}")

                print(f"    Transfer entropy:")
                print(f"      TE: {te_result['transfer_entropy']:.4f} bits")
                print(f"      Normalized TE: {te_result['normalized_te']:.4f}")

        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        # Count significant causal relationships
        total_tests = 0
        significant_granger = 0
        significant_te = 0

        for protein_name, lipid_results in results.items():
            for lipid_type, test_results in lipid_results.items():
                total_tests += 1
                if test_results['granger']['causes']:
                    significant_granger += 1
                    print(f"✓ {protein_name} - {lipid_type}: Granger causality detected")
                if test_results['transfer_entropy']['normalized_te'] > 0.1:
                    significant_te += 1

        print(f"\nGranger causality: {significant_granger}/{total_tests} significant")
        print(f"Transfer entropy > 0.1: {significant_te}/{total_tests}")
        print("="*70)

        return results

    def get_summary_table(self, causal_results):
        """Create summary table of causal inference results

        Parameters
        ----------
        causal_results : dict
            Results from analyze_causality()

        Returns
        -------
        pandas.DataFrame
            Summary table
        """
        rows = []

        for protein_name, lipid_results in causal_results.items():
            for lipid_type, test_results in lipid_results.items():
                granger = test_results['granger']
                te = test_results['transfer_entropy']

                rows.append({
                    'protein': protein_name,
                    'lipid_type': lipid_type,
                    'granger_f': granger['f_statistic'],
                    'granger_p': granger['p_value'],
                    'granger_lag': granger['optimal_lag'],
                    'granger_significant': granger['causes'],
                    'transfer_entropy': te['transfer_entropy'],
                    'normalized_te': te['normalized_te']
                })

        return pd.DataFrame(rows)
