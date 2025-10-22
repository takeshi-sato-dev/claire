#!/usr/bin/env python3
"""
Visualization for bulk vs protein vicinity composition
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


def plot_bulk_vs_protein_comparison(results, output_path=None, figsize=(14, 10)):
    """Plot bulk vs protein vicinity composition comparison

    Parameters
    ----------
    results : dict
        Results from calculate_bulk_vs_protein_composition
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    lipid_types = results['lipid_types']
    bulk_mean = results['bulk_mean']
    bulk_std = results['bulk_std']
    protein_mean = results['protein_mean']
    protein_std = results['protein_std']
    enrichment = results['enrichment']
    p_values = results['p_values']

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Color scheme
    colors = {'CHOL': '#3498db', 'DPSM': '#2ecc71', 'DIPC': '#e74c3c',
              'POPC': '#f39c12', 'POPE': '#9b59b6', 'POPS': '#1abc9c'}

    # --- Panel 1: Bar plot comparison ---
    ax1 = fig.add_subplot(gs[0, 0])

    x = np.arange(len(lipid_types))
    width = 0.35

    bulk_values = [bulk_mean[lt] for lt in lipid_types]
    bulk_errors = [bulk_std[lt] for lt in lipid_types]
    protein_values = [protein_mean[lt] for lt in lipid_types]
    protein_errors = [protein_std[lt] for lt in lipid_types]

    ax1.bar(x - width/2, bulk_values, width, yerr=bulk_errors,
            label='Bulk membrane', color='gray', alpha=0.6, capsize=5)
    ax1.bar(x + width/2, protein_values, width, yerr=protein_errors,
            label='Protein vicinity (15Å)', alpha=0.8, capsize=5,
            color=[colors.get(lt, 'blue') for lt in lipid_types])

    # Add significance stars
    for i, lt in enumerate(lipid_types):
        p = p_values.get(lt, 1.0)
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = ''

        if sig:
            y_max = max(bulk_values[i] + bulk_errors[i],
                       protein_values[i] + protein_errors[i])
            ax1.text(i, y_max + 0.02, sig, ha='center', fontsize=12, fontweight='bold')

    ax1.set_xlabel('Lipid Type', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Fraction', fontweight='bold', fontsize=12)
    ax1.set_title('Bulk vs Protein Vicinity Composition', fontweight='bold', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lipid_types)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # --- Panel 2: Enrichment factors ---
    ax2 = fig.add_subplot(gs[0, 1])

    enrich_values = [enrichment.get(lt, 1.0) for lt in lipid_types]
    bar_colors = ['green' if e > 1 else 'red' for e in enrich_values]

    ax2.barh(lipid_types, enrich_values, color=bar_colors, alpha=0.7)
    ax2.axvline(1.0, color='black', linestyle='--', linewidth=2, label='No enrichment')

    # Add value labels
    for i, (lt, e) in enumerate(zip(lipid_types, enrich_values)):
        label = f'{e:.2f}x'
        x_pos = e + 0.05 if e > 1 else e - 0.05
        ha = 'left' if e > 1 else 'right'
        ax2.text(x_pos, i, label, va='center', ha=ha, fontweight='bold')

    ax2.set_xlabel('Enrichment Factor', fontweight='bold', fontsize=12)
    ax2.set_title('Protein Vicinity Enrichment\n(vs Bulk)', fontweight='bold', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(axis='x', alpha=0.3)

    # --- Panel 3: Delta composition ---
    ax3 = fig.add_subplot(gs[1, 0])

    deltas = [protein_mean[lt] - bulk_mean[lt] for lt in lipid_types]
    delta_colors = [colors.get(lt, 'blue') for lt in lipid_types]

    bars = ax3.bar(lipid_types, deltas, color=delta_colors, alpha=0.7)
    ax3.axhline(0, color='black', linestyle='-', linewidth=1)

    # Add value labels
    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        label = f'{delta:+.3f}'
        y_pos = height + 0.01 if height > 0 else height - 0.01
        va = 'bottom' if height > 0 else 'top'
        ax3.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                ha='center', va=va, fontsize=10, fontweight='bold')

    ax3.set_ylabel('Δ Fraction (Protein - Bulk)', fontweight='bold', fontsize=12)
    ax3.set_title('Composition Change at Protein Surface', fontweight='bold', fontsize=13)
    ax3.grid(axis='y', alpha=0.3)

    # --- Panel 4: Scatter plot ---
    ax4 = fig.add_subplot(gs[1, 1])

    for lt in lipid_types:
        ax4.scatter(bulk_mean[lt], protein_mean[lt],
                   s=200, alpha=0.7, label=lt,
                   color=colors.get(lt, 'blue'))

    # Perfect correlation line
    all_values = list(bulk_mean.values()) + list(protein_mean.values())
    min_val, max_val = min(all_values), max(all_values)
    ax4.plot([min_val, max_val], [min_val, max_val],
            'k--', linewidth=2, alpha=0.5, label='No change')

    ax4.set_xlabel('Bulk Fraction', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Protein Vicinity Fraction', fontweight='bold', fontsize=12)
    ax4.set_title('Bulk vs Protein Correlation', fontweight='bold', fontsize=13)
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    ax4.set_aspect('equal', adjustable='box')

    plt.suptitle('Bulk Membrane vs Protein Vicinity Composition',
                fontsize=15, fontweight='bold', y=0.98)

    if output_path:
        # Save both PNG and SVG
        base_path = os.path.splitext(output_path)[0]
        for fmt in ['png', 'svg']:
            out_file = f"{base_path}.{fmt}"
            fig.savefig(out_file, dpi=300, bbox_inches='tight', format=fmt)
            print(f"  ✓ Saved {out_file}")

    return fig


def plot_bulk_protein_gm3_comparison(bulk_results, gm3_results, lipid_types,
                                     output_path=None, figsize=(14, 8)):
    """Plot 3-way comparison: Bulk, Protein (unbound), Protein (bound)

    Parameters
    ----------
    bulk_results : dict
        Results from calculate_bulk_vs_protein_composition
    gm3_results : dict
        Results from calculate_bulk_vs_protein_gm3_dependent
    lipid_types : list
        Lipid types
    output_path : str, optional
        Output path
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Color scheme
    colors = {'CHOL': '#3498db', 'DPSM': '#2ecc71', 'DIPC': '#e74c3c',
              'POPC': '#f39c12', 'POPE': '#9b59b6', 'POPS': '#1abc9c'}

    # --- Panel 1: Bar plot ---
    x = np.arange(len(lipid_types))
    width = 0.25

    bulk_values = [bulk_results['bulk_mean'][lt] for lt in lipid_types]
    unbound_values = [gm3_results.get('unbound_protein_mean', {}).get(lt, 0) for lt in lipid_types]
    bound_values = [gm3_results.get('bound_protein_mean', {}).get(lt, 0) for lt in lipid_types]

    ax1.bar(x - width, bulk_values, width, label='Bulk membrane',
           color='gray', alpha=0.6)
    ax1.bar(x, unbound_values, width, label='Protein (GM3 unbound)',
           color='lightblue', alpha=0.8)
    ax1.bar(x + width, bound_values, width, label='Protein (GM3 bound)',
           color=[colors.get(lt, 'blue') for lt in lipid_types], alpha=0.9)

    ax1.set_xlabel('Lipid Type', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Fraction', fontweight='bold', fontsize=12)
    ax1.set_title('3-Way Composition Comparison', fontweight='bold', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lipid_types)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # --- Panel 2: Heatmap ---
    # Create matrix: rows = conditions, cols = lipids
    conditions = ['Bulk', 'Protein\n(unbound)', 'Protein\n(bound)']
    matrix = np.array([
        bulk_values,
        unbound_values,
        bound_values
    ])

    im = ax2.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=matrix.max())

    # Set ticks
    ax2.set_xticks(np.arange(len(lipid_types)))
    ax2.set_yticks(np.arange(len(conditions)))
    ax2.set_xticklabels(lipid_types)
    ax2.set_yticklabels(conditions)

    # Add text annotations
    for i in range(len(conditions)):
        for j in range(len(lipid_types)):
            text = ax2.text(j, i, f'{matrix[i, j]:.2f}',
                          ha='center', va='center', color='black', fontweight='bold')

    ax2.set_title('Composition Heatmap', fontweight='bold', fontsize=13)
    plt.colorbar(im, ax=ax2, label='Fraction')

    plt.suptitle('Bulk vs Protein Vicinity - GM3 Dependent',
                fontsize=15, fontweight='bold')
    plt.tight_layout()

    if output_path:
        base_path = os.path.splitext(output_path)[0]
        for fmt in ['png', 'svg']:
            out_file = f"{base_path}.{fmt}"
            fig.savefig(out_file, dpi=300, bbox_inches='tight', format=fmt)
            print(f"  ✓ Saved {out_file}")

    return fig
