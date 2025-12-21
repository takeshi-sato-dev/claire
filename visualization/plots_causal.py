#!/usr/bin/env python3
"""
Visualization for causal inference results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns


def plot_causal_inference_results(causal_results, output_path=None):
    """Plot comprehensive causal inference results

    Parameters
    ----------
    causal_results : dict
        Results from CausalInference.analyze_causality()
    output_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Extract data
    proteins = sorted(causal_results.keys())
    lipid_types = []
    for protein_results in causal_results.values():
        lipid_types.extend(protein_results.keys())
    lipid_types = sorted(set(lipid_types))

    n_proteins = len(proteins)
    n_lipids = len(lipid_types)

    # Create matrices for heatmaps
    granger_p_matrix = np.full((n_proteins, n_lipids), np.nan)
    granger_f_matrix = np.full((n_proteins, n_lipids), np.nan)
    te_matrix = np.full((n_proteins, n_lipids), np.nan)
    lag_matrix = np.full((n_proteins, n_lipids), np.nan)

    for i, protein in enumerate(proteins):
        for j, lipid in enumerate(lipid_types):
            if lipid in causal_results[protein]:
                granger = causal_results[protein][lipid]['granger']
                te = causal_results[protein][lipid]['transfer_entropy']

                granger_p_matrix[i, j] = granger['p_value']
                granger_f_matrix[i, j] = granger['f_statistic']
                te_matrix[i, j] = te['normalized_te']
                lag_matrix[i, j] = granger['optimal_lag']

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Color schemes
    cmap_pvalue = 'RdYlGn_r'  # Red = significant, Green = not significant
    cmap_te = 'viridis'
    cmap_f = 'plasma'

    # 1. Granger causality p-values
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(granger_p_matrix, cmap=cmap_pvalue, aspect='auto', vmin=0, vmax=0.2)
    ax1.set_xticks(range(n_lipids))
    ax1.set_yticks(range(n_proteins))
    ax1.set_xticklabels(lipid_types, rotation=45, ha='right')
    ax1.set_yticklabels(proteins)
    ax1.set_title('Granger Causality (p-value)\nRed = Significant', fontweight='bold')

    # Add significance markers
    for i in range(n_proteins):
        for j in range(n_lipids):
            if not np.isnan(granger_p_matrix[i, j]):
                p_val = granger_p_matrix[i, j]
                if p_val < 0.001:
                    marker = '***'
                elif p_val < 0.01:
                    marker = '**'
                elif p_val < 0.05:
                    marker = '*'
                else:
                    marker = ''

                if marker:
                    ax1.text(j, i, marker, ha='center', va='center',
                            color='white', fontweight='bold', fontsize=12)

    plt.colorbar(im1, ax=ax1, label='p-value')

    # 2. Granger F-statistic
    ax2 = fig.add_subplot(gs[0, 1])
    # Use log scale for F-statistic for better visualization
    f_matrix_log = np.log10(granger_f_matrix + 1)
    im2 = ax2.imshow(f_matrix_log, cmap=cmap_f, aspect='auto')
    ax2.set_xticks(range(n_lipids))
    ax2.set_yticks(range(n_proteins))
    ax2.set_xticklabels(lipid_types, rotation=45, ha='right')
    ax2.set_yticklabels(proteins)
    ax2.set_title('Granger F-statistic (log10)', fontweight='bold')

    # Add values
    for i in range(n_proteins):
        for j in range(n_lipids):
            if not np.isnan(granger_f_matrix[i, j]):
                val = granger_f_matrix[i, j]
                ax2.text(j, i, f'{val:.1f}', ha='center', va='center',
                        color='white' if f_matrix_log[i, j] > np.nanmedian(f_matrix_log) else 'black',
                        fontsize=9)

    plt.colorbar(im2, ax=ax2, label='log10(F + 1)')

    # 3. Transfer Entropy
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(te_matrix, cmap=cmap_te, aspect='auto', vmin=0, vmax=0.5)
    ax3.set_xticks(range(n_lipids))
    ax3.set_yticks(range(n_proteins))
    ax3.set_xticklabels(lipid_types, rotation=45, ha='right')
    ax3.set_yticklabels(proteins)
    ax3.set_title('Transfer Entropy (normalized)\nHigher = More Information Flow', fontweight='bold')

    # Add values
    for i in range(n_proteins):
        for j in range(n_lipids):
            if not np.isnan(te_matrix[i, j]):
                val = te_matrix[i, j]
                ax3.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color='white' if val > 0.15 else 'black',
                        fontsize=9)

    plt.colorbar(im3, ax=ax3, label='Normalized TE')

    # 4. Optimal lag
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(lag_matrix, cmap='coolwarm', aspect='auto')
    ax4.set_xticks(range(n_lipids))
    ax4.set_yticks(range(n_proteins))
    ax4.set_xticklabels(lipid_types, rotation=45, ha='right')
    ax4.set_yticklabels(proteins)
    ax4.set_title('Optimal Time Lag (frames)', fontweight='bold')

    # Add values
    for i in range(n_proteins):
        for j in range(n_lipids):
            if not np.isnan(lag_matrix[i, j]):
                val = int(lag_matrix[i, j])
                ax4.text(j, i, str(val), ha='center', va='center',
                        color='black', fontsize=10)

    plt.colorbar(im4, ax=ax4, label='Lag (frames)')

    # 5. Summary bar plot - Significant causality by lipid
    ax5 = fig.add_subplot(gs[1, 1])
    sig_counts = []
    for j, lipid in enumerate(lipid_types):
        count = np.sum(granger_p_matrix[:, j] < 0.05)
        sig_counts.append(count)

    bars = ax5.bar(range(n_lipids), sig_counts, color=['#e74c3c', '#2ecc71', '#3498db'][:n_lipids])
    ax5.set_xticks(range(n_lipids))
    ax5.set_xticklabels(lipid_types, rotation=45, ha='right')
    ax5.set_ylabel('Number of Proteins', fontweight='bold')
    ax5.set_title('Significant Causal Effects (p < 0.05)', fontweight='bold')
    ax5.set_ylim(0, n_proteins + 0.5)

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, sig_counts)):
        if count > 0:
            ax5.text(i, count + 0.1, str(int(count)), ha='center', va='bottom',
                    fontweight='bold', fontsize=11)

    # 6. Summary bar plot - Transfer entropy by lipid
    ax6 = fig.add_subplot(gs[1, 2])
    te_means = np.nanmean(te_matrix, axis=0)
    te_stds = np.nanstd(te_matrix, axis=0)

    bars = ax6.bar(range(n_lipids), te_means, yerr=te_stds, capsize=5,
                   color=['#e74c3c', '#2ecc71', '#3498db'][:n_lipids],
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax6.set_xticks(range(n_lipids))
    ax6.set_xticklabels(lipid_types, rotation=45, ha='right')
    ax6.set_ylabel('Normalized Transfer Entropy', fontweight='bold')
    ax6.set_title('Average Information Flow', fontweight='bold')
    ax6.axhline(0.1, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Threshold (0.1)')
    ax6.legend()

    # Overall title
    fig.suptitle('Causal Inference: Target Lipid Binding → Composition Changes',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved causal inference plot to {output_path}")

    return fig


def plot_causal_network(causal_results, significance=0.05, output_path=None):
    """Plot causal network diagram

    Parameters
    ----------
    causal_results : dict
        Results from CausalInference.analyze_causality()
    significance : float
        Significance threshold for edges
    output_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Position nodes
    proteins = sorted(causal_results.keys())
    lipid_types = []
    for protein_results in causal_results.values():
        lipid_types.extend(protein_results.keys())
    lipid_types = sorted(set(lipid_types))

    n_proteins = len(proteins)
    n_lipids = len(lipid_types)

    # Target lipid in center
    target_pos = (0, 0)

    # Proteins in a circle
    protein_positions = {}
    radius = 3
    for i, protein in enumerate(proteins):
        angle = 2 * np.pi * i / n_proteins
        protein_positions[protein] = (radius * np.cos(angle), radius * np.sin(angle))

    # Lipids in outer circle
    lipid_positions = {}
    radius_lipid = 6
    for i, lipid in enumerate(lipid_types):
        angle = 2 * np.pi * i / n_lipids
        lipid_positions[lipid] = (radius_lipid * np.cos(angle), radius_lipid * np.sin(angle))

    # Draw edges (Target → Protein → Lipid)
    for protein in proteins:
        px, py = protein_positions[protein]

        # Target to Protein (always draw)
        ax.arrow(target_pos[0], target_pos[1], px * 0.8, py * 0.8,
                head_width=0.15, head_length=0.2, fc='gray', ec='gray',
                alpha=0.3, linewidth=1.5)

        for lipid in lipid_types:
            if lipid in causal_results[protein]:
                granger = causal_results[protein][lipid]['granger']
                te = causal_results[protein][lipid]['transfer_entropy']

                if granger['causes']:
                    lx, ly = lipid_positions[lipid]

                    # Line width proportional to TE
                    linewidth = 1 + te['normalized_te'] * 10

                    # Color based on p-value
                    if granger['p_value'] < 0.001:
                        color = '#e74c3c'  # Red
                        alpha = 0.9
                    elif granger['p_value'] < 0.01:
                        color = '#f39c12'  # Orange
                        alpha = 0.7
                    else:
                        color = '#3498db'  # Blue
                        alpha = 0.5

                    # Draw arrow from protein to lipid
                    dx = lx - px
                    dy = ly - py
                    length = np.sqrt(dx**2 + dy**2)
                    dx_norm = dx / length * 0.9
                    dy_norm = dy / length * 0.9

                    ax.arrow(px, py, dx_norm, dy_norm,
                            head_width=0.15, head_length=0.2,
                            fc=color, ec=color, alpha=alpha,
                            linewidth=linewidth)

    # Draw nodes
    # Target lipid
    ax.scatter(*target_pos, s=1000, c='red', marker='*', edgecolors='black',
              linewidths=2, zorder=10, label='Target Lipid (GM3)')
    ax.text(target_pos[0], target_pos[1] - 0.5, 'GM3\nBinding',
           ha='center', va='top', fontsize=12, fontweight='bold')

    # Proteins
    for protein, pos in protein_positions.items():
        ax.scatter(*pos, s=500, c='lightblue', edgecolors='black',
                  linewidths=2, zorder=10)
        ax.text(pos[0], pos[1], protein.replace('Protein_', 'P'),
               ha='center', va='center', fontsize=10, fontweight='bold')

    # Lipids
    colors = {'CHOL': '#3498db', 'DPSM': '#2ecc71', 'DIPC': '#e74c3c'}
    for lipid, pos in lipid_positions.items():
        color = colors.get(lipid, 'gray')
        ax.scatter(*pos, s=600, c=color, edgecolors='black',
                  linewidths=2, zorder=10, alpha=0.7)
        ax.text(pos[0], pos[1], lipid,
               ha='center', va='center', fontsize=11, fontweight='bold',
               color='white')

    # Legend
    legend_elements = [
        mpatches.Patch(color='#e74c3c', label='p < 0.001 ***'),
        mpatches.Patch(color='#f39c12', label='p < 0.01 **'),
        mpatches.Patch(color='#3498db', label='p < 0.05 *'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Causal Network: GM3 Binding → Protein → Composition Changes',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved causal network to {output_path}")

    return fig
