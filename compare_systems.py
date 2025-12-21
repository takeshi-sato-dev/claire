#!/usr/bin/env python3
"""
Compare composition across different systems
For example: EGFR (no GM3) vs EGFR (GM3 unbound) vs EGFR (GM3 bound)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy import stats


def load_composition_data(csv_file):
    """Load composition data from CSV file

    Parameters
    ----------
    csv_file : str
        Path to composition_data.csv

    Returns
    -------
    pandas.DataFrame
        Composition data
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")

    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} frames from {csv_file}")

    return df


def compare_three_way(nogm3_csv, withgm3_csv, lipid_types, output_dir='comparison_output'):
    """Compare 3 conditions: No GM3, GM3 unbound, GM3 bound

    Parameters
    ----------
    nogm3_csv : str
        Path to composition_data.csv from GM3-free system
    withgm3_csv : str
        Path to composition_data.csv from GM3-containing system
    lipid_types : list of str
        Lipid types to compare
    output_dir : str
        Output directory

    Returns
    -------
    dict
        Comparison results
    """
    print("\n" + "="*70)
    print("3-WAY COMPOSITION COMPARISON")
    print("="*70)

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\nLoading data...")
    df_nogm3 = load_composition_data(nogm3_csv)
    df_withgm3 = load_composition_data(withgm3_csv)

    # Check if target_lipid_bound column exists
    if 'target_lipid_bound' not in df_withgm3.columns:
        raise ValueError(f"{withgm3_csv} must contain 'target_lipid_bound' column")

    # Calculate mean compositions
    print("\nCalculating compositions...")

    # Condition 1: No GM3 system (all frames)
    nogm3_mean = {}
    nogm3_std = {}
    for lt in lipid_types:
        col = f'{lt}_fraction'
        if col in df_nogm3.columns:
            nogm3_mean[lt] = df_nogm3[col].mean()
            nogm3_std[lt] = df_nogm3[col].std()
        else:
            nogm3_mean[lt] = 0.0
            nogm3_std[lt] = 0.0

    # Condition 2: GM3 present, unbound
    unbound_df = df_withgm3[df_withgm3['target_lipid_bound'] == False]
    unbound_mean = {}
    unbound_std = {}
    for lt in lipid_types:
        col = f'{lt}_fraction'
        if col in unbound_df.columns:
            unbound_mean[lt] = unbound_df[col].mean()
            unbound_std[lt] = unbound_df[col].std()
        else:
            unbound_mean[lt] = 0.0
            unbound_std[lt] = 0.0

    # Condition 3: GM3 bound
    bound_df = df_withgm3[df_withgm3['target_lipid_bound'] == True]
    bound_mean = {}
    bound_std = {}
    for lt in lipid_types:
        col = f'{lt}_fraction'
        if col in bound_df.columns:
            bound_mean[lt] = bound_df[col].mean()
            bound_std[lt] = bound_df[col].std()
        else:
            bound_mean[lt] = 0.0
            bound_std[lt] = 0.0

    # Print results
    print("\n" + "-"*70)
    print(f"{'Lipid':<8} {'No GM3':<15} {'GM3 (unbound)':<15} {'GM3 (bound)':<15}")
    print("-"*70)

    for lt in lipid_types:
        print(f"{lt:<8} {nogm3_mean[lt]:.3f}±{nogm3_std[lt]:.3f}   "
              f"{unbound_mean[lt]:.3f}±{unbound_std[lt]:.3f}   "
              f"{bound_mean[lt]:.3f}±{bound_std[lt]:.3f}")

    print("-"*70)

    # Statistical tests
    print("\nStatistical comparisons:")
    print("-"*70)

    results = {}

    for lt in lipid_types:
        col = f'{lt}_fraction'

        # No GM3 vs GM3 unbound
        if col in df_nogm3.columns and col in unbound_df.columns:
            t1, p1 = stats.ttest_ind(df_nogm3[col].dropna(), unbound_df[col].dropna())
            sig1 = '***' if p1 < 0.001 else '**' if p1 < 0.01 else '*' if p1 < 0.05 else 'ns'
        else:
            p1, sig1 = np.nan, 'N/A'

        # No GM3 vs GM3 bound
        if col in df_nogm3.columns and col in bound_df.columns:
            t2, p2 = stats.ttest_ind(df_nogm3[col].dropna(), bound_df[col].dropna())
            sig2 = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else 'ns'
        else:
            p2, sig2 = np.nan, 'N/A'

        # GM3 unbound vs bound
        if col in unbound_df.columns and col in bound_df.columns:
            t3, p3 = stats.ttest_ind(unbound_df[col].dropna(), bound_df[col].dropna())
            sig3 = '***' if p3 < 0.001 else '**' if p3 < 0.01 else '*' if p3 < 0.05 else 'ns'
        else:
            p3, sig3 = np.nan, 'N/A'

        print(f"\n{lt}:")
        print(f"  No GM3 vs GM3 unbound: p={p1:.4f} {sig1}")
        print(f"  No GM3 vs GM3 bound:   p={p2:.4f} {sig2}")
        print(f"  GM3 unbound vs bound:  p={p3:.4f} {sig3}")

        results[lt] = {
            'nogm3_mean': nogm3_mean[lt],
            'unbound_mean': unbound_mean[lt],
            'bound_mean': bound_mean[lt],
            'nogm3_std': nogm3_std[lt],
            'unbound_std': unbound_std[lt],
            'bound_std': bound_std[lt],
            'p_nogm3_vs_unbound': p1,
            'p_nogm3_vs_bound': p2,
            'p_unbound_vs_bound': p3
        }

    print("-"*70)
    print("="*70)

    # Plot
    plot_three_way_comparison(results, lipid_types, output_dir)

    return results


def plot_three_way_comparison(results, lipid_types, output_dir):
    """Plot 3-way comparison

    Parameters
    ----------
    results : dict
        Comparison results
    lipid_types : list
        Lipid types
    output_dir : str
        Output directory
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Color scheme
    colors = {'CHOL': '#3498db', 'DPSM': '#2ecc71', 'DIPC': '#e74c3c',
              'POPC': '#f39c12', 'POPE': '#9b59b6', 'POPS': '#1abc9c'}

    # Panel 1: Bar plot
    x = np.arange(len(lipid_types))
    width = 0.25

    nogm3_vals = [results[lt]['nogm3_mean'] for lt in lipid_types]
    nogm3_errs = [results[lt]['nogm3_std'] for lt in lipid_types]

    unbound_vals = [results[lt]['unbound_mean'] for lt in lipid_types]
    unbound_errs = [results[lt]['unbound_std'] for lt in lipid_types]

    bound_vals = [results[lt]['bound_mean'] for lt in lipid_types]
    bound_errs = [results[lt]['bound_std'] for lt in lipid_types]

    ax1.bar(x - width, nogm3_vals, width, yerr=nogm3_errs,
           label='No GM3', color='gray', alpha=0.6, capsize=5)
    ax1.bar(x, unbound_vals, width, yerr=unbound_errs,
           label='GM3 present (unbound)', color='lightblue', alpha=0.7, capsize=5)
    ax1.bar(x + width, bound_vals, width, yerr=bound_errs,
           label='GM3 bound', alpha=0.9, capsize=5,
           color=[colors.get(lt, 'blue') for lt in lipid_types])

    ax1.set_xlabel('Lipid Type', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Fraction (protein vicinity)', fontweight='bold', fontsize=13)
    ax1.set_title('3-Way Composition Comparison', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lipid_types, fontsize=12)
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(axis='y', alpha=0.3)

    # Panel 2: Line plot (trajectories)
    for lt in lipid_types:
        vals = [results[lt]['nogm3_mean'],
                results[lt]['unbound_mean'],
                results[lt]['bound_mean']]
        ax2.plot(['No GM3', 'GM3\n(unbound)', 'GM3\n(bound)'], vals,
                marker='o', linewidth=2.5, markersize=10,
                label=lt, color=colors.get(lt, 'blue'))

    ax2.set_ylabel('Fraction', fontweight='bold', fontsize=13)
    ax2.set_title('Compositional Trajectory', fontweight='bold', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.suptitle('EGFR: No GM3 vs GM3 Unbound vs GM3 Bound',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'three_way_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot: {output_path}")

    output_path_svg = os.path.join(output_dir, 'three_way_comparison.svg')
    plt.savefig(output_path_svg, bbox_inches='tight')
    print(f"✓ Saved plot: {output_path_svg}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare compositions across systems')
    parser.add_argument('--nogm3', required=True,
                       help='Path to composition_data.csv from GM3-free system')
    parser.add_argument('--withgm3', required=True,
                       help='Path to composition_data.csv from GM3-containing system')
    parser.add_argument('--lipids', nargs='+', default=['CHOL', 'DPSM', 'DIPC'],
                       help='Lipid types to compare')
    parser.add_argument('--output', default='comparison_output',
                       help='Output directory')

    args = parser.parse_args()

    results = compare_three_way(
        args.nogm3,
        args.withgm3,
        args.lipids,
        args.output
    )

    print("\n✓ 3-way comparison complete!")


if __name__ == '__main__':
    main()
