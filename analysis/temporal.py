#!/usr/bin/env python3
"""
Temporal composition analysis
"""

import numpy as np
import pandas as pd
from scipy import stats, signal


class TemporalAnalyzer:
    """Analyze temporal patterns in lipid composition"""

    def __init__(self, window_size=100, step_size=10):
        """Initialize analyzer

        Parameters
        ----------
        window_size : int, default 100
            Window size for sliding window analysis (frames)
        step_size : int, default 10
            Step size between windows (frames)
        """
        self.window_size = window_size
        self.step_size = step_size

    def sliding_window_composition_per_protein(self, df, lipid_types):
        """Calculate sliding window composition for each protein separately

        Parameters
        ----------
        df : pandas.DataFrame
            Input data with 'protein' column
        lipid_types : list of str
            Lipid types to analyze

        Returns
        -------
        dict
            Dictionary with protein names as keys, each containing a DataFrame of windowed composition
        """
        print("\n" + "="*70)
        print("PER-PROTEIN SLIDING WINDOW COMPOSITION")
        print("="*70)

        if 'protein' not in df.columns:
            print("ERROR: 'protein' column not found")
            return {}

        protein_windows = {}

        for protein_name in sorted(df['protein'].unique()):
            print(f"\nProcessing {protein_name}...")
            protein_df = df[df['protein'] == protein_name]

            # Calculate windows for this protein
            window_df = self.sliding_window_composition(protein_df, lipid_types)
            protein_windows[protein_name] = window_df

            print(f"  ✓ {len(window_df)} windows for {protein_name}")

        print("="*70)
        return protein_windows

    def sliding_window_composition(self, df, lipid_types):
        """Calculate composition in sliding windows

        Parameters
        ----------
        df : pandas.DataFrame
            Input data sorted by frame
        lipid_types : list of str
            Lipid types to analyze

        Returns
        -------
        pandas.DataFrame
            Windowed composition data
        """
        print("\n" + "="*70)
        print("SLIDING WINDOW COMPOSITION ANALYSIS")
        print("="*70)
        print(f"Window size: {self.window_size} frames")
        print(f"Step size: {self.step_size} frames")

        # Ensure sorted by frame
        df = df.sort_values('frame')

        frames = df['frame'].unique()
        n_frames = len(frames)

        if n_frames < self.window_size:
            print(f"WARNING: Only {n_frames} frames, less than window size {self.window_size}")
            print("Adjusting window size...")
            self.window_size = max(10, n_frames // 2)

        window_results = []

        # Slide window
        for start_idx in range(0, n_frames - self.window_size + 1, self.step_size):
            end_idx = start_idx + self.window_size

            window_frames = frames[start_idx:end_idx]
            window_data = df[df['frame'].isin(window_frames)]

            center_frame = window_frames[len(window_frames) // 2]
            center_time = window_data[window_data['frame'] == center_frame]['time'].iloc[0]

            # Calculate composition in this window
            window_comp = {'center_frame': center_frame, 'center_time': center_time}

            # Calculate conserved ratios
            total = sum(window_data[f'{lt}_count'] for lt in lipid_types)
            total = total.sum()

            for lipid_type in lipid_types:
                count_sum = window_data[f'{lipid_type}_count'].sum()
                ratio = count_sum / total if total > 0 else 0
                window_comp[f'{lipid_type}_ratio'] = ratio

            window_results.append(window_comp)

        result_df = pd.DataFrame(window_results)

        print(f"✓ Calculated {len(result_df)} windows")
        return result_df

    def detect_binding_events_per_protein(self, df, target_col='target_lipid_bound'):
        """Detect target lipid binding and unbinding events for each protein separately

        Parameters
        ----------
        df : pandas.DataFrame
            Input data with target lipid binding information and 'protein' column
        target_col : str, default 'target_lipid_bound'
            Column indicating target lipid binding state

        Returns
        -------
        dict
            Dictionary with protein names as keys, each containing 'binding' and 'unbinding' lists
        """
        print("\n" + "="*70)
        print("PER-PROTEIN BINDING EVENT DETECTION")
        print("="*70)

        if target_col not in df.columns:
            print(f"ERROR: {target_col} not found")
            return {}

        if 'protein' not in df.columns:
            print(f"ERROR: 'protein' column not found")
            return {}

        protein_events = {}

        for protein_name in sorted(df['protein'].unique()):
            protein_df = df[df['protein'] == protein_name].sort_values('frame')
            frames = protein_df['frame'].values
            bound_state = protein_df[target_col].values

            binding_events = []
            unbinding_events = []

            # Detect state changes for this protein
            for i in range(1, len(bound_state)):
                prev_state = bound_state[i-1]
                curr_state = bound_state[i]

                # Binding event: False → True
                if not prev_state and curr_state:
                    binding_events.append({
                        'frame': frames[i],
                        'protein': protein_name,
                        'event_type': 'binding'
                    })

                # Unbinding event: True → False
                elif prev_state and not curr_state:
                    unbinding_events.append({
                        'frame': frames[i],
                        'protein': protein_name,
                        'event_type': 'unbinding'
                    })

            protein_events[protein_name] = {
                'binding': binding_events,
                'unbinding': unbinding_events
            }

            print(f"\n{protein_name}:")
            print(f"  Binding events: {len(binding_events)}")
            print(f"  Unbinding events: {len(unbinding_events)}")

            # Show first few events
            if len(binding_events) > 0:
                print(f"  First binding: Frame {binding_events[0]['frame']:.0f}")
                if len(binding_events) > 1:
                    print(f"  Last binding:  Frame {binding_events[-1]['frame']:.0f}")

        print("="*70)

        return protein_events

    def detect_binding_events(self, df, target_col='target_lipid_bound'):
        """Detect target lipid binding and unbinding events (legacy method, aggregated)

        Parameters
        ----------
        df : pandas.DataFrame
            Input data with target lipid binding information
        target_col : str, default 'target_lipid_bound'
            Column indicating target lipid binding state

        Returns
        -------
        dict
            Dictionary with 'binding' and 'unbinding' lists of frames (aggregated across all proteins)
        """
        # Use per-protein detection and aggregate
        protein_events = self.detect_binding_events_per_protein(df, target_col)

        if not protein_events:
            return {'binding': [], 'unbinding': []}

        # Aggregate all events
        all_binding = []
        all_unbinding = []

        for protein_name, events in protein_events.items():
            all_binding.extend(events['binding'])
            all_unbinding.extend(events['unbinding'])

        # Sort by frame
        all_binding = sorted(all_binding, key=lambda x: x['frame'])
        all_unbinding = sorted(all_unbinding, key=lambda x: x['frame'])

        return {
            'binding': all_binding,
            'unbinding': all_unbinding
        }

    def detect_composition_transitions(self, window_df, lipid_type, threshold=0.05):
        """Detect significant composition transitions

        Parameters
        ----------
        window_df : pandas.DataFrame
            Windowed composition data
        lipid_type : str
            Lipid type to analyze
        threshold : float, default 0.05
            Minimum change to consider as transition

        Returns
        -------
        list of dict
            Detected transitions
        """
        ratio_col = f'{lipid_type}_ratio'

        if ratio_col not in window_df.columns:
            print(f"ERROR: {ratio_col} not found")
            return []

        ratios = window_df[ratio_col].values
        frames = window_df['center_frame'].values

        transitions = []

        # Find significant changes
        for i in range(1, len(ratios)):
            change = ratios[i] - ratios[i-1]

            if abs(change) >= threshold:
                transitions.append({
                    'frame_before': frames[i-1],
                    'frame_after': frames[i],
                    'ratio_before': ratios[i-1],
                    'ratio_after': ratios[i],
                    'change': change
                })

        print(f"\nDetected {len(transitions)} transitions for {lipid_type} (threshold={threshold})")

        return transitions

    def calculate_autocorrelation(self, df, lipid_type, max_lag=50):
        """Calculate autocorrelation for composition time series

        Parameters
        ----------
        df : pandas.DataFrame
            Input data sorted by frame
        lipid_type : str
            Lipid type to analyze
        max_lag : int, default 50
            Maximum lag (frames)

        Returns
        -------
        tuple
            (lags, autocorrelations)
        """
        ratio_col = f'{lipid_type}_ratio'

        if ratio_col not in df.columns:
            print(f"ERROR: {ratio_col} not found")
            return np.array([]), np.array([])

        # Get time series
        df_sorted = df.sort_values('frame')
        ratios = df_sorted[ratio_col].dropna().values

        if len(ratios) < max_lag * 2:
            max_lag = len(ratios) // 2

        autocorr = []
        lags = range(max_lag + 1)

        for lag in lags:
            if lag == 0:
                autocorr.append(1.0)
            else:
                c = np.corrcoef(ratios[:-lag], ratios[lag:])[0, 1]
                autocorr.append(c if not np.isnan(c) else 0)

        return np.array(lags), np.array(autocorr)

    def calculate_lag_correlation_per_protein(self, df, mediator_col, lipid_types, max_lag=50):
        """Calculate time-lagged correlation for each protein separately

        Parameters
        ----------
        df : pandas.DataFrame
            Input data with 'protein' column
        mediator_col : str
            Mediator column (e.g., 'target_lipid_bound')
        lipid_types : list of str
            Lipid types to analyze
        max_lag : int, default 50
            Maximum lag (frames)

        Returns
        -------
        dict
            Dictionary with {protein_name: {lipid_type: {'lags': [...], 'correlations': [...], 'p_values': [...]}}}
        """
        print("\n" + "="*70)
        print("PER-PROTEIN LAG CORRELATION ANALYSIS")
        print("="*70)

        if 'protein' not in df.columns:
            print("ERROR: 'protein' column not found")
            return {}

        protein_lag_results = {}

        for protein_name in sorted(df['protein'].unique()):
            print(f"\n{protein_name}:")
            protein_df = df[df['protein'] == protein_name]

            protein_lag_results[protein_name] = {}

            for lipid_type in lipid_types:
                lags, corrs, p_vals = self.calculate_lag_correlation(
                    protein_df, mediator_col, lipid_type, max_lag
                )

                if len(lags) > 0:
                    protein_lag_results[protein_name][lipid_type] = {
                        'lags': lags,
                        'correlations': corrs,
                        'p_values': p_vals
                    }

                    # Find maximum correlation
                    finite_mask = np.isfinite(corrs)
                    if np.any(finite_mask):
                        max_idx = np.argmax(np.abs(corrs[finite_mask]))
                        finite_lags = lags[finite_mask]
                        finite_corrs = corrs[finite_mask]
                        finite_pvals = p_vals[finite_mask]

                        max_lag_val = finite_lags[max_idx]
                        max_corr = finite_corrs[max_idx]
                        max_p = finite_pvals[max_idx]

                        sig_str = '***' if max_p < 0.001 else '**' if max_p < 0.01 else '*' if max_p < 0.05 else 'ns'

                        print(f"  {lipid_type}: lag={max_lag_val:.0f}, r={max_corr:.3f} ({sig_str})")

        print("="*70)
        return protein_lag_results

    def calculate_lag_correlation(self, df, mediator_col, lipid_type, max_lag=50):
        """Calculate time-lagged correlation between mediator and composition

        Parameters
        ----------
        df : pandas.DataFrame
            Input data
        mediator_col : str
            Mediator column (e.g., 'target_lipid_bound' or continuous measure)
        lipid_type : str
            Lipid type to analyze
        max_lag : int, default 50
            Maximum lag (frames)

        Returns
        -------
        tuple
            (lags, correlations, p_values)
        """
        ratio_col = f'{lipid_type}_ratio'

        # Sort and aggregate by frame if needed
        df_sorted = df.sort_values('frame')

        if mediator_col not in df_sorted.columns or ratio_col not in df_sorted.columns:
            print(f"ERROR: Required columns not found")
            return np.array([]), np.array([]), np.array([])

        # Get time series
        mediator = df_sorted[mediator_col].values
        composition = df_sorted[ratio_col].values

        min_len = min(len(mediator), len(composition))
        mediator = mediator[:min_len]
        composition = composition[:min_len]

        if len(mediator) < max_lag * 2:
            max_lag = len(mediator) // 2

        lags = []
        correlations = []
        p_values = []

        # Calculate correlation at each lag
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # Mediator leads composition
                m = mediator[:lag]
                c = composition[-lag:]
            elif lag > 0:
                # Composition leads mediator
                m = mediator[lag:]
                c = composition[:-lag]
            else:
                # No lag
                m = mediator
                c = composition

            if len(m) > 10:
                corr, pval = stats.pearsonr(m, c)
                lags.append(lag)
                correlations.append(corr)
                p_values.append(pval)

        return np.array(lags), np.array(correlations), np.array(p_values)

    def apply_smoothing(self, df, columns, method='savgol', window=21, poly_order=3):
        """Apply smoothing to time series

        Parameters
        ----------
        df : pandas.DataFrame
            Input data
        columns : list of str
            Columns to smooth
        method : str, default 'savgol'
            Smoothing method: 'savgol', 'rolling', or 'gaussian'
        window : int, default 21
            Window size (must be odd for savgol)
        poly_order : int, default 3
            Polynomial order for savgol

        Returns
        -------
        pandas.DataFrame
            DataFrame with smoothed columns
        """
        df_smooth = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            if method == 'savgol':
                if len(df) > window:
                    # Ensure odd window
                    if window % 2 == 0:
                        window += 1
                    df_smooth[col] = signal.savgol_filter(df[col].values, window, poly_order)

            elif method == 'rolling':
                df_smooth[col] = df[col].rolling(window=window, center=True, min_periods=1).mean()

            elif method == 'gaussian':
                df_smooth[col] = signal.gaussian_filter1d(df[col].values, sigma=window/4)

        return df_smooth
