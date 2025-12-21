#!/usr/bin/env python3
"""
Improved target position effect visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def plot_target_position_effect_improved(position_df, lipid_types, radii, output_path=None, figsize=(18, 14)):
    """Plot improved target position effect with all shells

    Parameters
    ----------
    position_df : pandas.DataFrame
        DataFrame from analyze_target_position_effect()
    lipid_types : list of str
        Lipid types in config order
    radii : list of float
        Shell radii
    output_path : str, optional
        Path to save figure
    figsize : tuple, default (18, 14)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    from visualization.plots import setup_publication_style, save_figure

    setup_publication_style()

    if len(position_df) == 0:
        print("No position data to plot")
        return None

    n_lipids = len(lipid_types)
    n_shells = len(radii)

    # Create figure with 3 rows:
    # Row 1: Distance binning effect (bar plots for all shells)
    # Row 2: Binding state comparison (bound vs unbound)
    # Row 3: Target count effect (0, 1, 2, 3+ GM3)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, n_lipids, hspace=0.3, wspace=0.3)

    # Color map for lipids
    color_map = {
        'CHOL': '#3498db',  # Blue
        'DIPC': '#e74c3c',  # Red
        'DPSM': '#2ecc71',  # Green
        'POPC': '#f39c12',  # Orange
        'POPE': '#9b59b6',  # Purple
        'POPS': '#1abc9c',  # Turquoise
    }

    # Row 1: Distance binning effect
    for j, lipid_type in enumerate(lipid_types):
        ax = fig.add_subplot(gs[0, j])

        # Bin target-protein distance
        bins = [0, 4, 6, 10, 100]
        bin_labels = ['0-4Å\n(tight)', '4-6Å\n(loose)', '6-10Å\n(near)', '>10Å\n(far)']
        position_df['distance_bin'] = pd.cut(position_df['target_distance'],
                                             bins=bins, labels=bin_labels)

        # Plot all shells for this lipid
        shell_data = []
        shell_labels = []
        shell_counts = []  # Track number of samples per bin

        for i, radius in enumerate(radii):
            inner_radius = radii[i-1] if i > 0 else 0.0
            col_name = f'{lipid_type}_ratio_shell_{inner_radius:.0f}_{radius:.0f}'

            if col_name in position_df.columns:
                shell_label = f'{inner_radius:.0f}-{radius:.0f}'
                shell_labels.append(shell_label)

                # Calculate mean for each distance bin
                bin_means = []
                bin_ns = []
                for bin_label in bin_labels:
                    bin_data = position_df[position_df['distance_bin'] == bin_label][col_name]
                    bin_ns.append(len(bin_data))
                    if len(bin_data) > 0:
                        bin_means.append(bin_data.mean())
                    else:
                        bin_means.append(0)  # Show 0 for empty bins

                shell_data.append(bin_means)
                shell_counts.append(bin_ns)

        # Plot grouped bar chart
        x = np.arange(len(bin_labels))
        width = 0.2

        for i, (shell_means, shell_label, shell_ns) in enumerate(zip(shell_data, shell_labels, shell_counts)):
            offset = (i - len(shell_data)/2 + 0.5) * width
            bars = ax.bar(x + offset, shell_means, width, label=shell_label, alpha=0.8)

            # Add "n=0" annotation for empty bins
            for bar_idx, (bar, n) in enumerate(zip(bars, shell_ns)):
                if n == 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           'n=0', ha='center', va='bottom', fontsize=7, color='red')

        ax.set_xlabel('Target-Protein Distance', fontweight='bold', fontsize=10)
        ax.set_ylabel(f'{lipid_type} Fraction', fontweight='bold', fontsize=10)
        ax.set_title(f'{lipid_type}: Distance Effect', fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=9)
        ax.legend(title='Shell (Å)', fontsize=8, title_fontsize=9, loc='best')
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)  # Slightly larger to accommodate n=0 labels

    # Row 2: Binding state comparison (bound vs unbound)
    for j, lipid_type in enumerate(lipid_types):
        ax = fig.add_subplot(gs[1, j])

        bound_data = position_df[position_df['target_bound'] == True]
        unbound_data = position_df[position_df['target_bound'] == False]

        shell_labels = []
        bound_means = []
        unbound_means = []

        for i, radius in enumerate(radii):
            inner_radius = radii[i-1] if i > 0 else 0.0
            col_name = f'{lipid_type}_ratio_shell_{inner_radius:.0f}_{radius:.0f}'

            if col_name in position_df.columns:
                shell_label = f'{inner_radius:.0f}-{radius:.0f}'
                shell_labels.append(shell_label)

                bound_mean = bound_data[col_name].mean() if len(bound_data) > 0 else 0
                unbound_mean = unbound_data[col_name].mean() if len(unbound_data) > 0 else 0

                bound_means.append(bound_mean)
                unbound_means.append(unbound_mean)

        x = np.arange(len(shell_labels))
        width = 0.35

        ax.bar(x - width/2, bound_means, width, label=f'Bound (n={len(bound_data)})',
               color=color_map.get(lipid_type, '#7f8c8d'), alpha=0.8)
        ax.bar(x + width/2, unbound_means, width, label=f'Unbound (n={len(unbound_data)})',
               color=color_map.get(lipid_type, '#7f8c8d'), alpha=0.4)

        ax.set_xlabel('Shell (Å)', fontweight='bold', fontsize=10)
        ax.set_ylabel(f'{lipid_type} Fraction', fontweight='bold', fontsize=10)
        ax.set_title(f'{lipid_type}: Bound vs Unbound', fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(shell_labels, fontsize=9)
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylim(0, 1)

    # Row 3: Target bound count effect (number of GM3 in contact)
    for j, lipid_type in enumerate(lipid_types):
        ax = fig.add_subplot(gs[2, j])

        # Bin by target bound count (number of GM3 molecules in contact with protein)
        position_df['target_count_bin'] = position_df['target_bound_count'].apply(
            lambda x: '0' if x == 0 else '1' if x == 1 else '2' if x == 2 else '3+'
        )
        count_bins = ['0', '1', '2', '3+']

        shell_data = []
        shell_labels = []
        shell_counts = []

        for i, radius in enumerate(radii):
            inner_radius = radii[i-1] if i > 0 else 0.0
            col_name = f'{lipid_type}_ratio_shell_{inner_radius:.0f}_{radius:.0f}'

            if col_name in position_df.columns:
                shell_label = f'{inner_radius:.0f}-{radius:.0f}'
                shell_labels.append(shell_label)

                # Calculate mean for each count bin
                bin_means = []
                bin_ns = []
                for count_bin in count_bins:
                    bin_data = position_df[position_df['target_count_bin'] == count_bin][col_name]
                    bin_ns.append(len(bin_data))
                    if len(bin_data) > 0:
                        bin_means.append(bin_data.mean())
                    else:
                        bin_means.append(0)

                shell_data.append(bin_means)
                shell_counts.append(bin_ns)

        # Plot grouped bar chart
        x = np.arange(len(count_bins))
        width = 0.2

        for i, (shell_means, shell_label, shell_ns) in enumerate(zip(shell_data, shell_labels, shell_counts)):
            offset = (i - len(shell_data)/2 + 0.5) * width
            bars = ax.bar(x + offset, shell_means, width, label=shell_label, alpha=0.8)

            # Add "n=0" annotation for empty bins
            for bar_idx, (bar, n) in enumerate(zip(bars, shell_ns)):
                if n == 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           'n=0', ha='center', va='bottom', fontsize=7, color='red')

        ax.set_xlabel('Number of Bound GM3', fontweight='bold', fontsize=10)
        ax.set_ylabel(f'{lipid_type} Fraction', fontweight='bold', fontsize=10)
        ax.set_title(f'{lipid_type}: Bound GM3 Count Effect', fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(count_bins, fontsize=9)
        ax.legend(title='Shell (Å)', fontsize=8, title_fontsize=9, loc='best')
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)  # Slightly larger to accommodate n=0 labels

    if output_path:
        save_figure(fig, output_path)

    return fig
