#!/usr/bin/env python3
"""
New radial profile plotting function with both shell types
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_radial_profiles_both(radial_profiles, lipid_types, output_path=None, figsize=(18, 10)):
    """Plot radial composition profiles (supports both individual and cumulative shells)

    Parameters
    ----------
    radial_profiles : dict
        Radial profile data
    lipid_types : list of str
        Lipid types (in the order specified in config)
    output_path : str, optional
        Path to save figure
    figsize : tuple, default (18, 10)
        Figure size
    """
    from visualization.plots import setup_publication_style, save_figure

    setup_publication_style()

    n_proteins = len(radial_profiles)
    n_lipids = len(lipid_types)

    # Detect shell type
    first_protein = list(radial_profiles.values())[0]
    has_individual = any(k.startswith('shell_') for k in first_protein['shells'].keys())
    has_cumulative = any(k.startswith('cumulative_') for k in first_protein['shells'].keys())

    # Determine number of subplot columns
    n_cols = 0
    if has_individual:
        n_cols += n_lipids
    if has_cumulative:
        n_cols += n_lipids

    fig, axes = plt.subplots(n_proteins, n_cols,
                            figsize=(figsize[0] * n_cols / n_lipids, figsize[1]),
                            sharex='col', sharey='row')

    if n_proteins == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Use lipid-specific colors based on config order
    color_map = {
        'CHOL': '#3498db',  # Blue
        'DIPC': '#e74c3c',  # Red
        'DPSM': '#2ecc71',  # Green
        'POPC': '#f39c12',  # Orange
        'POPE': '#9b59b6',  # Purple
        'POPS': '#1abc9c',  # Turquoise
    }
    colors = [color_map.get(lt, '#7f8c8d') for lt in lipid_types]

    for i, (protein_name, profile) in enumerate(radial_profiles.items()):
        radii = profile['radii']
        col_offset = 0

        # Plot individual shells
        if has_individual:
            # Sort shell keys by numerical value (inner radius), not alphabetically
            shell_list = [k for k in profile['shells'].keys() if k.startswith('shell_')]
            shell_keys = sorted(shell_list, key=lambda x: float(x.replace('shell_', '').split('_')[0]))

            # Get shell centers
            shell_centers = []
            for shell_key in shell_keys:
                parts = shell_key.replace('shell_', '').split('_')
                inner = float(parts[0])
                outer = float(parts[1])
                center = (inner + outer) / 2
                shell_centers.append(center)

            for j, lipid_type in enumerate(lipid_types):
                ax = axes[i, col_offset + j]

                # Get means and stds
                means = [profile['shells'][sk].get(f'{lipid_type}_mean', 0)
                        for sk in shell_keys]
                stds = [profile['shells'][sk].get(f'{lipid_type}_std', 0)
                       for sk in shell_keys]

                ax.plot(shell_centers, means, 'o-', color=colors[j],
                       linewidth=2, markersize=8, label=lipid_type)
                ax.fill_between(shell_centers,
                               np.array(means) - np.array(stds),
                               np.array(means) + np.array(stds),
                               alpha=0.3, color=colors[j])

                if i == 0:
                    ax.set_title(f'{lipid_type}\n(Individual shells)', fontweight='bold', fontsize=12)
                if j == 0:
                    ax.set_ylabel(f'{protein_name}\nFraction', fontweight='bold', fontsize=11)
                if i == n_proteins - 1:
                    ax.set_xlabel('Distance from protein (Å)', fontweight='bold', fontsize=11)

                ax.grid(alpha=0.3)
                ax.set_ylim(0, 1)

            col_offset += n_lipids

        # Plot cumulative shells
        if has_cumulative:
            # Sort cumulative keys by numerical value (outer radius), not alphabetically
            cumul_list = [k for k in profile['shells'].keys() if k.startswith('cumulative_')]
            cumul_keys = sorted(cumul_list, key=lambda x: float(x.replace('cumulative_', '').split('_')[1]))

            # Get radii for cumulative shells
            cumul_radii = []
            for cumul_key in cumul_keys:
                parts = cumul_key.replace('cumulative_', '').split('_')
                radius = float(parts[1])
                cumul_radii.append(radius)

            for j, lipid_type in enumerate(lipid_types):
                ax = axes[i, col_offset + j]

                # Get means and stds
                means = [profile['shells'][ck].get(f'{lipid_type}_mean', 0)
                        for ck in cumul_keys]
                stds = [profile['shells'][ck].get(f'{lipid_type}_std', 0)
                       for ck in cumul_keys]

                ax.plot(cumul_radii, means, 's-', color=colors[j],
                       linewidth=2, markersize=8, label=lipid_type)
                ax.fill_between(cumul_radii,
                               np.array(means) - np.array(stds),
                               np.array(means) + np.array(stds),
                               alpha=0.3, color=colors[j])

                if i == 0:
                    ax.set_title(f'{lipid_type}\n(Cumulative)', fontweight='bold', fontsize=12)
                if has_individual and j == 0:
                    # No y-label for second set of columns
                    pass
                if i == n_proteins - 1:
                    ax.set_xlabel('Distance from protein (Å)', fontweight='bold', fontsize=11)

                ax.grid(alpha=0.3)
                ax.set_ylim(0, 1)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig
