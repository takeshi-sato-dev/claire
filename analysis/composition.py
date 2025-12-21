#!/usr/bin/env python3
"""
Composition analysis with mass conservation
"""

import numpy as np
import pandas as pd
from scipy import stats


class CompositionAnalyzer:
    """Analyze lipid composition changes with mass conservation"""

    def __init__(self, lipid_types, target_lipid=None):
        """Initialize analyzer

        Parameters
        ----------
        lipid_types : list of str
            Lipid types to analyze
        target_lipid : str, optional
            Target lipid name (excluded from composition ratios)
        """
        self.lipid_types = lipid_types
        self.target_lipid = target_lipid

        # Lipids for composition ratio (excluding target)
        self.comp_lipids = [lt for lt in lipid_types
                           if target_lipid is None or lt != target_lipid]

    def frames_to_dataframe(self, frame_data_list):
        """Convert frame data to pandas DataFrame

        Parameters
        ----------
        frame_data_list : list of dict
            List of frame composition data

        Returns
        -------
        pandas.DataFrame
            Flattened data with one row per (frame, protein) combination
        """
        rows = []

        for frame_data in frame_data_list:
            frame_idx = frame_data['frame']
            time = frame_data['time']

            for protein_name, protein_data in frame_data['proteins'].items():
                row = {
                    'frame': frame_idx,
                    'time': time,
                    'protein': protein_name,
                    'total_lipids': protein_data['total_lipids'],
                    'total_for_ratio': protein_data['total_for_ratio'],
                    'target_lipid_bound': protein_data.get('target_lipid_bound', False)
                }

                # Add counts
                for lipid_type in self.lipid_types:
                    count = protein_data['lipid_counts'].get(lipid_type, 0)
                    row[f'{lipid_type}_count'] = count

                # Add fractions
                for lipid_type in self.comp_lipids:
                    frac = protein_data['lipid_fractions'].get(lipid_type, 0.0)
                    row[f'{lipid_type}_fraction'] = frac

                rows.append(row)

        return pd.DataFrame(rows)

    def calculate_conservation_ratios(self, df):
        """Calculate composition ratios ensuring conservation (sum = 1)

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe with lipid counts

        Returns
        -------
        pandas.DataFrame
            DataFrame with added ratio columns
        """
        print("\nCalculating composition ratios (with conservation)...")

        # Calculate total of composition lipids
        total_comp = sum(df[f'{lt}_count'] for lt in self.comp_lipids)
        total_comp = total_comp.replace(0, 1)  # Avoid division by zero

        # Calculate ratios
        for lipid_type in self.comp_lipids:
            ratio_col = f'{lipid_type}_ratio'
            df[ratio_col] = df[f'{lipid_type}_count'] / total_comp

        # Verify conservation
        total_ratio = sum(df[f'{lt}_ratio'] for lt in self.comp_lipids)
        mean_total = total_ratio.mean()

        print(f"  Conservation check: mean total ratio = {mean_total:.6f} (should be 1.0)")

        if abs(mean_total - 1.0) > 0.001:
            print(f"  WARNING: Conservation violated by {abs(mean_total - 1.0):.6f}")

        return df

    def analyze_composition_changes(self, df, mediator_column='target_lipid_bound',
                                   method='quartile'):
        """Analyze composition changes relative to mediator

        Parameters
        ----------
        df : pandas.DataFrame
            Input data
        mediator_column : str
            Column indicating mediator presence
        method : str, default 'quartile'
            Comparison method: 'quartile', 'median', or 'binary'

        Returns
        -------
        dict
            Results for each lipid type
        """
        print("\n" + "="*70)
        print("COMPOSITION CHANGE ANALYSIS")
        print("="*70)

        results = {}

        # Determine high/low groups
        if method == 'binary':
            # For binary mediator (e.g., target_lipid_bound)
            high_group = df[df[mediator_column] == True]
            low_group = df[df[mediator_column] == False]
            print(f"\nBinary split on {mediator_column}:")

        elif method == 'quartile':
            # For continuous mediator
            if mediator_column in df.columns and pd.api.types.is_numeric_dtype(df[mediator_column]):
                q75 = df[mediator_column].quantile(0.75)
                q25 = df[mediator_column].quantile(0.25)
                high_group = df[df[mediator_column] > q75]
                low_group = df[df[mediator_column] < q25]
                print(f"\nQuartile split on {mediator_column}:")
                print(f"  Q75 threshold: {q75:.4f}")
                print(f"  Q25 threshold: {q25:.4f}")
            else:
                print(f"ERROR: {mediator_column} not suitable for quartile split")
                return results

        elif method == 'median':
            median = df[mediator_column].median()
            high_group = df[df[mediator_column] > median]
            low_group = df[df[mediator_column] <= median]
            print(f"\nMedian split on {mediator_column} (median = {median:.4f}):")

        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"  High group: {len(high_group)} frames")
        print(f"  Low group: {len(low_group)} frames")

        if len(high_group) < 10 or len(low_group) < 10:
            print("WARNING: Small sample sizes may affect statistical power")

        # Calculate separately for each group to ensure conservation
        print("\nCalculating group-specific composition ratios...")

        # High group ratios
        high_total = sum(high_group[f'{lt}_count'] for lt in self.comp_lipids)
        high_total = high_total.replace(0, 1)

        # Low group ratios
        low_total = sum(low_group[f'{lt}_count'] for lt in self.comp_lipids)
        low_total = low_total.replace(0, 1)

        # Analyze each lipid
        print("\nLipid composition changes:")
        print("-" * 70)

        for lipid_type in self.comp_lipids:
            # Calculate group-specific ratios
            high_ratio = (high_group[f'{lipid_type}_count'] / high_total).mean()
            low_ratio = (low_group[f'{lipid_type}_count'] / low_total).mean()

            # Calculate changes
            absolute_change = high_ratio - low_ratio
            percent_change = (absolute_change / low_ratio * 100) if low_ratio > 0 else 0

            # Statistical test
            if len(high_group) >= 2 and len(low_group) >= 2:
                high_vals = high_group[f'{lipid_type}_count'] / high_total
                low_vals = low_group[f'{lipid_type}_count'] / low_total

                t_stat, p_value = stats.ttest_ind(
                    high_vals.dropna(),
                    low_vals.dropna(),
                    equal_var=False  # Welch's t-test
                )
            else:
                p_value = 1.0

            # Significance
            if p_value < 0.001:
                sig = '***'
            elif p_value < 0.01:
                sig = '**'
            elif p_value < 0.05:
                sig = '*'
            else:
                sig = 'ns'

            results[lipid_type] = {
                'low_ratio': low_ratio,
                'high_ratio': high_ratio,
                'absolute_change': absolute_change,
                'percent_change': percent_change,
                'p_value': p_value,
                'significance': sig,
                'n_high': len(high_group),
                'n_low': len(low_group)
            }

            print(f"{lipid_type:6s}: {low_ratio:.4f} → {high_ratio:.4f} "
                  f"(Δ{absolute_change:+.4f}, {percent_change:+.1f}%) {sig}")

        # Verify conservation of changes
        total_change = sum(res['absolute_change'] for res in results.values())
        print("-" * 70)
        print(f"Conservation check: Σ(changes) = {total_change:+.6f} (should be ~0)")

        if abs(total_change) > 0.001:
            print(f"  WARNING: Total change deviates from zero")

        print("="*70)

        return results

    def analyze_per_protein(self, df, mediator_column='target_lipid_bound'):
        """Analyze composition changes for each protein separately

        Parameters
        ----------
        df : pandas.DataFrame
            Input data
        mediator_column : str
            Column indicating mediator presence

        Returns
        -------
        dict
            Results for each protein
        """
        print("\n" + "="*70)
        print("PER-PROTEIN COMPOSITION ANALYSIS")
        print("="*70)

        protein_results = {}

        for protein_name in df['protein'].unique():
            print(f"\n{protein_name}:")
            protein_df = df[df['protein'] == protein_name]

            # Calculate composition changes for this protein
            results = self.analyze_composition_changes(
                protein_df, mediator_column, method='binary'
            )

            protein_results[protein_name] = results

        return protein_results
