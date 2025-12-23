#!/usr/bin/env python3
"""
Visualization functions for CLAIRE
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


def setup_publication_style():
    """Set up publication-quality plot style"""
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5
    })


def save_figure(fig, output_path, formats=['png', 'svg'], dpi=300):
    """Save figure in multiple formats

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    output_path : str
        Base output path (without extension)
    formats : list of str, default ['png', 'svg']
        Output formats
    dpi : int, default 300
        Resolution for raster formats
    """
    if output_path is None:
        return

    # Remove extension if present
    base_path = os.path.splitext(output_path)[0]

    for fmt in formats:
        out_file = f"{base_path}.{fmt}"
        try:
            fig.savefig(out_file, dpi=dpi, bbox_inches='tight', format=fmt)
            print(f"  ✓ Saved {out_file}")
        except Exception as e:
            print(f"  ERROR saving {out_file}: {str(e)}")
            import traceback
            traceback.print_exc()

    return base_path


def plot_composition_changes(results, lipid_types, output_path=None, figsize=(12, 6)):
    """Plot composition changes as bar chart

    Parameters
    ----------
    results : dict
        Composition analysis results
    lipid_types : list of str
        Lipid types
    output_path : str, optional
        Path to save figure
    figsize : tuple, default (18, 6)
        Figure size
    """
    setup_publication_style()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel A: Absolute changes
    ax = axes[0]
    changes = [results[lt]['absolute_change'] for lt in lipid_types]
    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in changes]

    bars = ax.bar(range(len(lipid_types)), changes, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5)

    # Add significance stars
    for i, lt in enumerate(lipid_types):
        sig = results[lt]['significance']
        if sig != 'ns':
            y_pos = changes[i] + (0.02 if changes[i] > 0 else -0.02)
            ax.text(i, y_pos, sig, ha='center', va='bottom' if changes[i] > 0 else 'top',
                   fontsize=14, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Composition Change (fraction)', fontweight='bold')
    ax.set_xlabel('Lipid Type', fontweight='bold')
    ax.set_xticks(range(len(lipid_types)))
    ax.set_xticklabels(lipid_types)
    ax.set_title('Absolute Composition Changes', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Panel B: Percent changes
    ax = axes[1]
    pct_changes = [results[lt]['percent_change'] for lt in lipid_types]
    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in pct_changes]

    ax.bar(range(len(lipid_types)), pct_changes, color=colors, alpha=0.8,
           edgecolor='black', linewidth=1.5)

    for i, lt in enumerate(lipid_types):
        sig = results[lt]['significance']
        if sig != 'ns':
            y_pos = pct_changes[i] + (2 if pct_changes[i] > 0 else -2)
            ax.text(i, y_pos, sig, ha='center', va='bottom' if pct_changes[i] > 0 else 'top',
                   fontsize=14, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Composition Change (%)', fontweight='bold')
    ax.set_xlabel('Lipid Type', fontweight='bold')
    ax.set_xticks(range(len(lipid_types)))
    ax.set_xticklabels(lipid_types)
    ax.set_title('Percent Composition Changes', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Panel C: Composition percentages (Low vs High)
    ax = axes[2]
    x = np.arange(len(lipid_types))
    width = 0.35

    low_ratios = [results[lt]['low_ratio'] * 100 for lt in lipid_types]
    high_ratios = [results[lt]['high_ratio'] * 100 for lt in lipid_types]

    ax.bar(x - width/2, low_ratios, width, label='Low (Unbound)',
           color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.bar(x + width/2, high_ratios, width, label='High (Bound)',
           color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Composition (%)', fontweight='bold')
    ax.set_xlabel('Lipid Type', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(lipid_types)
    ax.set_title('Lipid Composition (Low vs High)', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_temporal_composition_per_protein(protein_windows, protein_events, lipid_types, output_path=None, figsize=(16, 12)):
    """Plot temporal composition evolution for each protein separately

    Parameters
    ----------
    protein_windows : dict
        Dictionary with {protein_name: window_df}
    protein_events : dict
        Dictionary with {protein_name: {'binding': [...], 'unbinding': [...]}}
    lipid_types : list of str
        Lipid types
    output_path : str, optional
        Path to save figure
    figsize : tuple, default (16, 12)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    setup_publication_style()

    n_proteins = len(protein_windows)
    n_lipids = len(lipid_types)

    # Create grid: n_proteins rows × n_lipids columns
    fig, axes = plt.subplots(n_proteins, n_lipids, figsize=figsize, sharex=True)

    if n_proteins == 1:
        axes = axes.reshape(1, -1)
    if n_lipids == 1:
        axes = axes.reshape(-1, 1)

    colors = plt.cm.tab10(np.linspace(0, 1, n_lipids))

    for i, (protein_name, window_df) in enumerate(sorted(protein_windows.items())):
        for j, lipid_type in enumerate(lipid_types):
            ax = axes[i, j]
            ratio_col = f'{lipid_type}_ratio'

            if ratio_col in window_df.columns:
                # Plot composition
                ax.plot(window_df['center_frame'], window_df[ratio_col],
                       color=colors[j], linewidth=2, label=lipid_type)
                ax.fill_between(window_df['center_frame'], 0, window_df[ratio_col],
                               alpha=0.3, color=colors[j])

                # Plot binding events for this protein
                if protein_events and protein_name in protein_events:
                    events = protein_events[protein_name]

                    # Binding events (green upward arrows)
                    if 'binding' in events and len(events['binding']) > 0:
                        for event in events['binding']:
                            frame = event['frame']
                            if window_df['center_frame'].min() <= frame <= window_df['center_frame'].max():
                                closest_idx = np.argmin(np.abs(window_df['center_frame'] - frame))
                                y_pos = window_df.iloc[closest_idx][ratio_col]
                                ax.annotate('', xy=(frame, y_pos), xytext=(frame, max(0, y_pos - 0.08)),
                                          arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.7),
                                          zorder=5)

                    # Unbinding events (red downward arrows)
                    if 'unbinding' in events and len(events['unbinding']) > 0:
                        for event in events['unbinding']:
                            frame = event['frame']
                            if window_df['center_frame'].min() <= frame <= window_df['center_frame'].max():
                                closest_idx = np.argmin(np.abs(window_df['center_frame'] - frame))
                                y_pos = window_df.iloc[closest_idx][ratio_col]
                                ax.annotate('', xy=(frame, y_pos), xytext=(frame, min(1, y_pos + 0.08)),
                                          arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.7),
                                          zorder=5)

            # Labels
            if j == 0:
                ax.set_ylabel(f'{protein_name}\nFraction', fontweight='bold', fontsize=10)
            if i == 0:
                ax.set_title(lipid_type, fontweight='bold', fontsize=12)
            if i == n_proteins - 1:
                ax.set_xlabel('Frame', fontweight='bold', fontsize=10)

            ax.grid(alpha=0.3)
            ax.set_ylim([0, 1])

    # Add overall legend
    if protein_events:
        n_binding_total = sum(len(e.get('binding', [])) for e in protein_events.values())
        n_unbinding_total = sum(len(e.get('unbinding', [])) for e in protein_events.values())
        fig.text(0.99, 0.99, f'▲ GM3 binding (n={n_binding_total})\n▼ GM3 unbinding (n={n_unbinding_total})',
                ha='right', va='top', fontsize=10, color='black',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Per-Protein Temporal Evolution of Lipid Composition',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_temporal_composition(window_df, lipid_types, binding_events=None, output_path=None, figsize=(14, 8)):
    """Plot temporal composition evolution with GM3 binding/unbinding events

    Parameters
    ----------
    window_df : pandas.DataFrame
        Windowed composition data
    lipid_types : list of str
        Lipid types
    binding_events : dict, optional
        Dictionary with 'binding' and 'unbinding' lists from detect_binding_events()
    output_path : str, optional
        Path to save figure
    figsize : tuple, default (14, 8)
        Figure size
    """
    setup_publication_style()

    n_lipids = len(lipid_types)
    fig, axes = plt.subplots(n_lipids, 1, figsize=figsize, sharex=True)

    if n_lipids == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_lipids))

    for i, lipid_type in enumerate(lipid_types):
        ax = axes[i]
        ratio_col = f'{lipid_type}_ratio'

        if ratio_col in window_df.columns:
            ax.plot(window_df['center_frame'], window_df[ratio_col],
                   color=colors[i], linewidth=2, label=lipid_type)
            ax.fill_between(window_df['center_frame'], 0, window_df[ratio_col],
                           alpha=0.3, color=colors[i])

            # Plot GM3 binding/unbinding events if provided
            if binding_events:
                # Binding events (green upward arrows)
                if 'binding' in binding_events and len(binding_events['binding']) > 0:
                    for event in binding_events['binding']:
                        frame = event['frame']
                        # Only plot if within data range
                        if window_df['center_frame'].min() <= frame <= window_df['center_frame'].max():
                            # Find composition at this frame
                            closest_idx = np.argmin(np.abs(window_df['center_frame'] - frame))
                            y_pos = window_df.iloc[closest_idx][ratio_col]

                            # Green upward arrow
                            ax.annotate('', xy=(frame, y_pos), xytext=(frame, y_pos - 0.08),
                                      arrowprops=dict(arrowstyle='->', color='green', lw=2.5, alpha=0.7),
                                      zorder=5)

                    # Add legend entry
                    ax.plot([], [], '^', color='green', markersize=10, alpha=0.7,
                           label=f'GM3 binding (n={len(binding_events["binding"])})')

                # Unbinding events (red downward arrows)
                if 'unbinding' in binding_events and len(binding_events['unbinding']) > 0:
                    for event in binding_events['unbinding']:
                        frame = event['frame']
                        # Only plot if within data range
                        if window_df['center_frame'].min() <= frame <= window_df['center_frame'].max():
                            # Find composition at this frame
                            closest_idx = np.argmin(np.abs(window_df['center_frame'] - frame))
                            y_pos = window_df.iloc[closest_idx][ratio_col]

                            # Red downward arrow
                            ax.annotate('', xy=(frame, y_pos), xytext=(frame, y_pos + 0.08),
                                      arrowprops=dict(arrowstyle='->', color='red', lw=2.5, alpha=0.7),
                                      zorder=5)

                    # Add legend entry
                    ax.plot([], [], 'v', color='red', markersize=10, alpha=0.7,
                           label=f'GM3 unbinding (n={len(binding_events["unbinding"])})')

        ax.set_ylabel(f'{lipid_type}\nFraction', fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel('Frame', fontweight='bold')
    axes[0].set_title('Temporal Evolution of Lipid Composition with GM3-Protein Binding Events',
                     fontweight='bold', fontsize=13)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_radial_profiles(radial_profiles, lipid_types, output_path=None, figsize=(14, 10)):
    """Plot radial composition profiles

    Parameters
    ----------
    radial_profiles : dict
        Radial profile data
    lipid_types : list of str
        Lipid types
    output_path : str, optional
        Path to save figure
    figsize : tuple, default (14, 10)
        Figure size
    """
    setup_publication_style()

    n_proteins = len(radial_profiles)
    n_lipids = len(lipid_types)

    fig, axes = plt.subplots(n_proteins, n_lipids, figsize=figsize,
                            sharex=True, sharey='col')

    if n_proteins == 1:
        axes = axes.reshape(1, -1)
    if n_lipids == 1:
        axes = axes.reshape(-1, 1)

    colors = plt.cm.tab10(np.linspace(0, 1, n_lipids))

    for i, (protein_name, profile) in enumerate(radial_profiles.items()):
        radii = profile['radii']
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
            ax = axes[i, j]

            # Get means and stds
            means = [profile['shells'][sk].get(f'{lipid_type}_mean', 0)
                    for sk in shell_keys]
            stds = [profile['shells'][sk].get(f'{lipid_type}_std', 0)
                   for sk in shell_keys]

            ax.plot(shell_centers, means, 'o-', color=colors[j],
                   linewidth=2, markersize=8, label=lipid_type)
            ax.fill_between(shell_centers,
                           np.array(means) - np.array(stds),
                           alpha=0.3, color=colors[j])

            if i == 0:
                ax.set_title(lipid_type, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{protein_name}\nFraction', fontweight='bold')
            if i == n_proteins - 1:
                ax.set_xlabel('Distance from protein (Å)', fontweight='bold')

            ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_ml_predictions(results, lipid_types, output_path=None, figsize=(15, 10)):
    """Plot ML prediction results

    Parameters
    ----------
    results : dict
        ML prediction results
    lipid_types : list of str
        Lipid types
    output_path : str, optional
        Path to save figure
    figsize : tuple, default (15, 10)
        Figure size
    """
    setup_publication_style()

    n_lipids = len([lt for lt in lipid_types if lt in results])
    if n_lipids == 0:
        print("No results to plot")
        return None

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, n_lipids, figure=fig, hspace=0.3, wspace=0.3)

    col = 0
    for lipid_type in lipid_types:
        if lipid_type not in results:
            continue

        # Panel A: Model comparison
        ax1 = fig.add_subplot(gs[0, col])

        model_names = list(results[lipid_type].keys())
        test_r2s = [results[lipid_type][name]['test_r2'] for name in model_names]

        colors = ['#3498db' if r2 > 0 else '#e74c3c' for r2 in test_r2s]
        ax1.barh(model_names, test_r2s, color=colors, alpha=0.8, edgecolor='black')

        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax1.set_xlabel('Test R²', fontweight='bold')
        ax1.set_title(f'{lipid_type} - Model Performance', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # Panel B: Best model predictions vs actual
        ax2 = fig.add_subplot(gs[1, col])

        best_model = max(model_names, key=lambda k: results[lipid_type][k]['test_r2'])
        y_test_pred = results[lipid_type][best_model]['predictions']

        # We need y_test - this should be stored in results
        # For now, plot predictions
        ax2.scatter(range(len(y_test_pred)), y_test_pred, alpha=0.6, s=30)
        ax2.set_xlabel('Sample Index', fontweight='bold')
        ax2.set_ylabel('Predicted Fraction', fontweight='bold')
        ax2.set_title(f'Best: {best_model} (R²={results[lipid_type][best_model]["test_r2"]:.3f})',
                     fontweight='bold')
        ax2.grid(alpha=0.3)

        col += 1

    plt.suptitle('Machine Learning Prediction Results', fontsize=16, fontweight='bold')

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_comprehensive_summary(comp_results, temporal_df, radial_profiles,
                               lipid_types, output_path=None, figsize=(20, 12)):
    """Create comprehensive summary figure

    Parameters
    ----------
    comp_results : dict
        Composition change results
    temporal_df : pandas.DataFrame
        Temporal composition data
    radial_profiles : dict
        Radial profile data
    lipid_types : list of str
        Lipid types
    output_path : str, optional
        Path to save figure
    figsize : tuple, default (20, 12)
        Figure size
    """
    setup_publication_style()

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: Composition changes
    ax_comp = fig.add_subplot(gs[0, :2])
    changes = [comp_results[lt]['absolute_change'] for lt in lipid_types]
    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in changes]
    ax_comp.bar(lipid_types, changes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax_comp.axhline(y=0, color='black', linestyle='-')
    ax_comp.set_ylabel('Composition Change', fontweight='bold')
    ax_comp.set_title('A. Composition Changes', fontweight='bold', loc='left')
    ax_comp.grid(axis='y', alpha=0.3)

    # Panel B: Statistical summary
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_stats.axis('tight')
    ax_stats.axis('off')

    table_data = []
    for lt in lipid_types:
        table_data.append([
            lt,
            f"{comp_results[lt]['percent_change']:+.1f}%",
            comp_results[lt]['significance']
        ])

    table = ax_stats.table(cellText=table_data,
                          colLabels=['Lipid', 'Change (%)', 'Sig.'],
                          cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax_stats.set_title('B. Statistical Summary', fontweight='bold', loc='left')

    # Panel C: Temporal evolution
    ax_temp = fig.add_subplot(gs[1, :])
    colors_temp = plt.cm.tab10(np.linspace(0, 1, len(lipid_types)))

    for i, lt in enumerate(lipid_types):
        ratio_col = f'{lt}_ratio'
        if ratio_col in temporal_df.columns:
            ax_temp.plot(temporal_df['center_frame'], temporal_df[ratio_col],
                        label=lt, color=colors_temp[i], linewidth=2)

    ax_temp.set_xlabel('Frame', fontweight='bold')
    ax_temp.set_ylabel('Composition Fraction', fontweight='bold')
    ax_temp.set_title('C. Temporal Evolution', fontweight='bold', loc='left')
    ax_temp.legend(loc='best')
    ax_temp.grid(alpha=0.3)

    # Panel D: Radial profiles
    if radial_profiles:
        first_protein = list(radial_profiles.keys())[0]
        profile = radial_profiles[first_protein]

        # Sort shell keys by numerical value (inner radius), not alphabetically
        shell_list = [k for k in profile['shells'].keys() if k.startswith('shell_')]
        shell_keys = sorted(shell_list, key=lambda x: float(x.replace('shell_', '').split('_')[0]))
        shell_centers = []
        for sk in shell_keys:
            parts = sk.replace('shell_', '').split('_')
            center = (float(parts[0]) + float(parts[1])) / 2
            shell_centers.append(center)

        for i, lt in enumerate(lipid_types):
            ax_rad = fig.add_subplot(gs[2, i])

            means = [profile['shells'][sk].get(f'{lt}_mean', 0) for sk in shell_keys]

            ax_rad.plot(shell_centers, means, 'o-', color=colors_temp[i],
                       linewidth=2, markersize=8)
            ax_rad.set_xlabel('Distance (Å)', fontweight='bold')
            ax_rad.set_ylabel('Fraction', fontweight='bold')
            ax_rad.set_title(f'D{i+1}. {lt} Radial Profile', fontweight='bold', loc='left')
            ax_rad.grid(alpha=0.3)

    plt.suptitle('CLAIRE Comprehensive Analysis', fontsize=18, fontweight='bold')

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_target_dependent_radial_profiles(target_results, lipid_types, output_path=None, figsize=(16, 12)):
    """Plot radial profiles comparing bound vs unbound states

    Parameters
    ----------
    target_results : dict
        Results from calculate_target_dependent_profiles()
    lipid_types : list of str
        Lipid types
    output_path : str, optional
        Path to save figure
    figsize : tuple, default (16, 12)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    setup_publication_style()

    if target_results is None:
        print("No target-dependent results to plot")
        return None

    bound_profiles = target_results['bound']
    unbound_profiles = target_results['unbound']
    diff_profiles = target_results['difference']

    n_proteins = len(bound_profiles)
    n_lipids = len(lipid_types)

    # Create figure with 3 rows: bound, unbound, difference
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, n_lipids, figure=fig, hspace=0.3, wspace=0.3)

    colors = plt.cm.tab10(np.linspace(0, 1, n_lipids))

    protein_name = list(bound_profiles.keys())[0]  # Use first protein
    bound_profile = bound_profiles[protein_name]
    unbound_profile = unbound_profiles[protein_name]
    diff_profile = diff_profiles[protein_name]

    # Sort shell keys by numerical value (inner radius), not alphabetically
    shell_list = [k for k in bound_profile['shells'].keys() if k.startswith('shell_')]
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
        # Panel A: Bound state
        ax_bound = fig.add_subplot(gs[0, j])

        bound_means = [bound_profile['shells'][sk].get(f'{lipid_type}_mean', 0)
                      for sk in shell_keys]
        bound_stds = [bound_profile['shells'][sk].get(f'{lipid_type}_std', 0)
                     for sk in shell_keys]

        ax_bound.plot(shell_centers, bound_means, 'o-', color=colors[j],
                     linewidth=2, markersize=8, label='Bound')
        ax_bound.fill_between(shell_centers,
                             np.array(bound_means) - np.array(bound_stds),
                             np.array(bound_means) + np.array(bound_stds),
                             alpha=0.3, color=colors[j])

        ax_bound.set_title(f'{lipid_type} - BOUND', fontweight='bold')
        if j == 0:
            ax_bound.set_ylabel('Fraction', fontweight='bold')
        ax_bound.grid(alpha=0.3)
        ax_bound.set_ylim([0, 1])

        # Panel B: Unbound state
        ax_unbound = fig.add_subplot(gs[1, j])

        unbound_means = [unbound_profile['shells'][sk].get(f'{lipid_type}_mean', 0)
                        for sk in shell_keys]
        unbound_stds = [unbound_profile['shells'][sk].get(f'{lipid_type}_std', 0)
                       for sk in shell_keys]

        ax_unbound.plot(shell_centers, unbound_means, 'o-', color=colors[j],
                       linewidth=2, markersize=8, label='Unbound')
        ax_unbound.fill_between(shell_centers,
                               np.array(unbound_means) - np.array(unbound_stds),
                               np.array(unbound_means) + np.array(unbound_stds),
                               alpha=0.3, color=colors[j])

        ax_unbound.set_title(f'{lipid_type} - UNBOUND', fontweight='bold')
        if j == 0:
            ax_unbound.set_ylabel('Fraction', fontweight='bold')
        ax_unbound.grid(alpha=0.3)
        ax_unbound.set_ylim([0, 1])

        # Panel C: Difference (bound - unbound)
        ax_diff = fig.add_subplot(gs[2, j])

        diffs = []
        p_values = []
        for sk in shell_keys:
            if lipid_type in diff_profile['shells'][sk]:
                diff_data = diff_profile['shells'][sk][lipid_type]
                diffs.append(diff_data['difference'])
                p_values.append(diff_data['p_value'])
            else:
                diffs.append(0)
                p_values.append(1.0)

        # Color bars by significance
        bar_colors = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in p_values]

        ax_diff.bar(shell_centers, diffs, width=2.5, color=bar_colors,
                   alpha=0.8, edgecolor='black', linewidth=1)
        ax_diff.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

        ax_diff.set_title(f'{lipid_type} - Δ (Bound-Unbound)', fontweight='bold')
        ax_diff.set_xlabel('Distance from protein (Å)', fontweight='bold')
        if j == 0:
            ax_diff.set_ylabel('Δ Fraction', fontweight='bold')
        ax_diff.grid(alpha=0.3, axis='y')

    plt.suptitle(f'Target Lipid-Dependent Radial Profiles (n_bound={target_results["n_bound"]}, n_unbound={target_results["n_unbound"]})',
                fontsize=16, fontweight='bold')

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_lag_correlation(lag_results, lipid_types, output_path=None, figsize=(14, 10)):
    """Plot time-lagged correlation between mediator and lipid composition

    Parameters
    ----------
    lag_results : dict
        Dictionary with {lipid_type: {'lags': array, 'correlations': array, 'p_values': array}}
    lipid_types : list of str
        Lipid types
    output_path : str, optional
        Path to save figure
    figsize : tuple, default (14, 10)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    setup_publication_style()

    if not lag_results:
        print("No lag correlation results to plot")
        return None

    n_lipids = len(lipid_types)
    fig, axes = plt.subplots(n_lipids, 1, figsize=figsize, sharex=True)

    if n_lipids == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_lipids))

    for i, lipid_type in enumerate(lipid_types):
        if lipid_type not in lag_results:
            continue

        ax = axes[i]
        lags = lag_results[lipid_type]['lags']
        corrs = lag_results[lipid_type]['correlations']
        p_vals = lag_results[lipid_type]['p_values']

        # Plot all correlations
        ax.plot(lags, corrs, 'o-', color=colors[i], linewidth=2, markersize=4,
               alpha=0.7, label=lipid_type)

        # Highlight significant correlations
        sig_mask = p_vals < 0.05
        if np.any(sig_mask):
            ax.scatter(lags[sig_mask], corrs[sig_mask],
                      color='red', s=150, marker='*', zorder=5,
                      edgecolors='black', linewidths=1,
                      label='p < 0.05')

        # Zero lines
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

        # Find and annotate maximum correlation (handle NaN/Inf)
        finite_mask = np.isfinite(corrs)
        if np.any(finite_mask):
            finite_corrs = np.abs(corrs[finite_mask])
            max_idx_finite = np.argmax(finite_corrs)
            # Map back to original index
            finite_indices = np.where(finite_mask)[0]
            max_idx = finite_indices[max_idx_finite]

            max_lag = lags[max_idx]
            max_corr = corrs[max_idx]

            # Determine if max correlation is significant
            max_sig = '***' if p_vals[max_idx] < 0.001 else '**' if p_vals[max_idx] < 0.01 else '*' if p_vals[max_idx] < 0.05 else 'ns'

            # Annotation
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7, edgecolor='black', linewidth=1.5)
            annotation_text = f'Max at lag={max_lag:.0f}\nr={max_corr:.3f} {max_sig}'

            ax.annotate(annotation_text,
                       xy=(max_lag, max_corr),
                       xytext=(max_lag + len(lags)*0.15, max_corr + 0.15*np.sign(max_corr)),
                       arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                       fontsize=11, fontweight='bold',
                       bbox=bbox_props)
        else:
            # No valid correlations
            max_lag = 0
            max_corr = 0

        # Labels and formatting
        ax.set_ylabel(f'{lipid_type}\nCorrelation', fontweight='bold', fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

        # Set y-axis limits (handle NaN/Inf)
        valid_corrs = corrs[np.isfinite(corrs)]
        if len(valid_corrs) > 0:
            ax.set_ylim([min(valid_corrs) - 0.1, max(valid_corrs) + 0.2])
        else:
            ax.set_ylim([-1, 1])

        # Add interpretation text
        if max_lag < 0:
            interpretation = f'GM3 binding → {abs(max_lag):.0f} frames → Composition change'
        elif max_lag > 0:
            interpretation = f'Composition change → {max_lag:.0f} frames → GM3 binding'
        else:
            interpretation = 'Synchronous: GM3 binding ⇄ Composition change'

        ax.text(0.02, 0.02, interpretation,
               transform=ax.transAxes, fontsize=10, style='italic',
               verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # X-axis label only on bottom plot
    axes[-1].set_xlabel('Lag (frames)\n← Negative: GM3 binding precedes composition change | Positive: Composition change precedes GM3 binding →',
                       fontweight='bold', fontsize=11)

    plt.suptitle('Time-Lagged Correlation: GM3-Protein Binding ↔ Lipid Composition',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_target_position_correlation(position_df, lipid_types, output_path=None, figsize=(16, 10)):
    """Plot correlation between target lipid position and composition

    Parameters
    ----------
    position_df : pandas.DataFrame
        DataFrame from analyze_target_position_effect()
    lipid_types : list of str
        Lipid types
    output_path : str, optional
        Path to save figure
    figsize : tuple, default (16, 10)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    setup_publication_style()

    if len(position_df) == 0:
        print("No position data to plot")
        return None

    n_lipids = len(lipid_types)
    fig, axes = plt.subplots(2, n_lipids, figsize=figsize)

    if n_lipids == 1:
        axes = axes.reshape(-1, 1)

    colors = plt.cm.tab10(np.linspace(0, 1, n_lipids))

    for j, lipid_type in enumerate(lipid_types):
        ratio_col = f'{lipid_type}_ratio'

        if ratio_col not in position_df.columns:
            continue

        # Panel A: Scatter plot with regression
        ax1 = axes[0, j]

        x = position_df['target_distance']
        y = position_df[ratio_col]

        ax1.scatter(x, y, alpha=0.5, color=colors[j], s=30, edgecolors='black', linewidths=0.5)

        # Fit line
        from scipy.stats import pearsonr
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax1.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.7)

        # Add correlation
        corr, p_val = pearsonr(x, y)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax1.text(0.05, 0.95, f'r = {corr:.3f} {sig}',
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax1.set_xlabel('Target-Protein Distance (Å)', fontweight='bold')
        ax1.set_ylabel(f'{lipid_type} Fraction', fontweight='bold')
        ax1.set_title(f'{lipid_type} Enrichment', fontweight='bold')
        ax1.grid(alpha=0.3)

        # Panel B: Binned analysis
        ax2 = axes[1, j]

        # Bin by distance
        bins = [0, 5, 10, 15, 20, 100]
        bin_labels = ['0-5', '5-10', '10-15', '15-20', '>20']
        position_df['distance_bin'] = pd.cut(position_df['target_distance'],
                                             bins=bins, labels=bin_labels)

        bin_means = []
        bin_stds = []
        bin_ns = []

        for label in bin_labels:
            bin_data = position_df[position_df['distance_bin'] == label][ratio_col]
            if len(bin_data) > 0:
                bin_means.append(bin_data.mean())
                bin_stds.append(bin_data.std())
                bin_ns.append(len(bin_data))
            else:
                bin_means.append(0)
                bin_stds.append(0)
                bin_ns.append(0)

        x_pos = np.arange(len(bin_labels))
        ax2.bar(x_pos, bin_means, yerr=bin_stds, color=colors[j],
               alpha=0.8, edgecolor='black', linewidth=1.5, capsize=5)

        # Add sample sizes
        for i, (n, mean) in enumerate(zip(bin_ns, bin_means)):
            if n > 0:
                ax2.text(i, mean + bin_stds[i] + 0.02, f'n={n}',
                        ha='center', fontsize=9, fontweight='bold')

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(bin_labels)
        ax2.set_xlabel('Target-Protein Distance (Å)', fontweight='bold')
        ax2.set_ylabel(f'{lipid_type} Fraction', fontweight='bold')
        ax2.set_title(f'{lipid_type} by Distance Bin', fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')

    plt.suptitle('Target Lipid Position vs Lipid Composition', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig
