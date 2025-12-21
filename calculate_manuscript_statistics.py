#!/usr/bin/env python3
"""
Calculate composition statistics for the FEBS Journal manuscript
Analyzes EGFR, Notch, and EphA2 data to generate accurate numerical values
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

def cohen_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt((var1 + var2) / 2)
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def analyze_receptor(receptor_name, csv_path):
    """Analyze composition data for one receptor"""
    print(f"\n{'='*60}")
    print(f"ANALYZING {receptor_name}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)

    # Split by GM3 binding status
    bound = df[df['target_lipid_bound'] == True]
    unbound = df[df['target_lipid_bound'] == False]

    # Count frames
    total_frames = len(df)
    bound_frames = len(bound)
    unbound_frames = len(unbound)
    bound_percent = (bound_frames / total_frames) * 100

    print(f"\nGM3 Binding Frequency:")
    print(f"  Total frames: {total_frames}")
    print(f"  GM3-bound frames: {bound_frames} ({bound_percent:.1f}%)")
    print(f"  GM3-unbound frames: {unbound_frames} ({100-bound_percent:.1f}%)")

    # Composition statistics
    lipids = ['CHOL', 'DPSM', 'DIPC']

    print(f"\nLipid Composition (mean ± SEM):")
    print(f"{'Lipid':<8} {'Unbound':<20} {'Bound':<20} {'Δ (abs)':<15} {'Δ (%)':<15} {'p-value':<12} {'Cohen d':<10}")
    print("-" * 110)

    results = {}
    for lipid in lipids:
        col = f'{lipid}_fraction'

        # Calculate means and SEMs
        unbound_mean = unbound[col].mean() * 100  # Convert to percentage
        unbound_sem = unbound[col].sem() * 100
        bound_mean = bound[col].mean() * 100
        bound_sem = bound[col].sem() * 100

        # Calculate changes
        abs_change = bound_mean - unbound_mean
        rel_change = (abs_change / unbound_mean) * 100 if unbound_mean > 0 else 0

        # Statistical test (Welch's t-test)
        if len(bound) > 0 and len(unbound) > 0:
            t_stat, p_value = stats.ttest_ind(bound[col], unbound[col], equal_var=False)
            d = cohen_d(bound[col].values, unbound[col].values)
        else:
            p_value = np.nan
            d = np.nan

        # Significance stars
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "ns"

        print(f"{lipid:<8} {unbound_mean:5.1f}±{unbound_sem:4.1f}     "
              f"{bound_mean:5.1f}±{bound_sem:4.1f}     "
              f"{abs_change:+6.1f} {sig:<4}  "
              f"{rel_change:+6.1f}%       "
              f"{p_value:.2e}   "
              f"{d:6.2f}")

        results[lipid] = {
            'unbound_mean': unbound_mean,
            'unbound_sem': unbound_sem,
            'bound_mean': bound_mean,
            'bound_sem': bound_sem,
            'abs_change': abs_change,
            'rel_change': rel_change,
            'p_value': p_value,
            'cohen_d': d
        }

    return {
        'receptor': receptor_name,
        'total_frames': total_frames,
        'bound_frames': bound_frames,
        'unbound_frames': unbound_frames,
        'bound_percent': bound_percent,
        'lipid_results': results
    }

def main():
    # Define paths
    base_dir = "/Users/takeshi/Library/CloudStorage/OneDrive-学校法人京都薬科大学/manuscript/JOSS/conservedxx5done/claire_new"

    receptors = {
        'EGFR': f'{base_dir}/claire_output_EGFR/composition_data.csv',
        'Notch': f'{base_dir}/claire_output_Notch/composition_data.csv',
        'EphA2': f'{base_dir}/claire_output_EphA2/composition_data.csv'
    }

    all_results = {}

    for receptor, csv_path in receptors.items():
        if os.path.exists(csv_path):
            all_results[receptor] = analyze_receptor(receptor, csv_path)
        else:
            print(f"WARNING: {csv_path} not found")

    # Print summary table for manuscript
    print(f"\n{'='*80}")
    print("MANUSCRIPT TABLE 1: Summary of Lipid Composition Changes")
    print(f"{'='*80}\n")

    print("| Receptor | State | CHOL (%) | DPSM (%) | DIPC (%) | n frames |")
    print("|----------|-------|----------|----------|----------|----------|")

    for receptor in ['EGFR', 'Notch', 'EphA2']:
        if receptor in all_results:
            r = all_results[receptor]
            lr = r['lipid_results']

            # Unbound row
            print(f"| **{receptor}** | Unbound | "
                  f"{lr['CHOL']['unbound_mean']:.1f} ± {lr['CHOL']['unbound_sem']:.1f} | "
                  f"{lr['DPSM']['unbound_mean']:.1f} ± {lr['DPSM']['unbound_sem']:.1f} | "
                  f"{lr['DIPC']['unbound_mean']:.1f} ± {lr['DIPC']['unbound_sem']:.1f} | "
                  f"{r['unbound_frames']} |")

            # Bound row
            print(f"| | Bound | "
                  f"{lr['CHOL']['bound_mean']:.1f} ± {lr['CHOL']['bound_sem']:.1f} | "
                  f"{lr['DPSM']['bound_mean']:.1f} ± {lr['DPSM']['bound_sem']:.1f} | "
                  f"{lr['DIPC']['bound_mean']:.1f} ± {lr['DIPC']['bound_sem']:.1f} | "
                  f"{r['bound_frames']} |")

            # Delta row
            sig_chol = "***" if lr['CHOL']['p_value'] < 0.001 else "**" if lr['CHOL']['p_value'] < 0.01 else "*" if lr['CHOL']['p_value'] < 0.05 else ""
            sig_dpsm = "***" if lr['DPSM']['p_value'] < 0.001 else "**" if lr['DPSM']['p_value'] < 0.01 else "*" if lr['DPSM']['p_value'] < 0.05 else ""
            sig_dipc = "***" if lr['DIPC']['p_value'] < 0.001 else "**" if lr['DIPC']['p_value'] < 0.01 else "*" if lr['DIPC']['p_value'] < 0.05 else ""

            print(f"| | Δ (bound - unbound) | "
                  f"{lr['CHOL']['abs_change']:+.1f}{sig_chol} (d={lr['CHOL']['cohen_d']:.1f}) | "
                  f"{lr['DPSM']['abs_change']:+.1f}{sig_dpsm} (d={lr['DPSM']['cohen_d']:.1f}) | "
                  f"{lr['DIPC']['abs_change']:+.1f}{sig_dipc} (d={lr['DIPC']['cohen_d']:.1f}) | |")

    print("\nValues are mean ± SEM. Statistical significance: *** p<0.001, ** p<0.01, * p<0.05.")
    print("Cohen's d in parentheses.\n")

if __name__ == '__main__':
    main()
