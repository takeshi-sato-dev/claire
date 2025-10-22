#!/usr/bin/env python3
"""
Visualization for microscopy predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_microscopy_predictions(microscopy_results, lipid_types=None, output_path=None):
    """Plot predicted microscopy images at multiple scales

    Parameters
    ----------
    microscopy_results : dict
        Results from MicroscopyAnalyzer.analyze_and_simulate()
    lipid_types : list of str, optional
        Lipid types to plot
    output_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    target_radii = sorted(microscopy_results.keys())
    n_scales = len(target_radii)

    if lipid_types is None:
        # Get lipid types from first scale
        first_scale = target_radii[0]
        lipid_types = list(microscopy_results[first_scale]['microscopy_images'].keys())

    n_lipids = len(lipid_types)

    # Create figure
    fig = plt.figure(figsize=(6 * n_scales, 5 * n_lipids))
    gs = GridSpec(n_lipids, n_scales, figure=fig, hspace=0.3, wspace=0.3)

    # Color maps for each lipid
    cmaps = {
        'CHOL': 'Blues',
        'DPSM': 'Greens',
        'DIPC': 'Reds',
        'DPG3': 'Purples'
    }

    for i, lipid in enumerate(lipid_types):
        for j, radius in enumerate(target_radii):
            ax = fig.add_subplot(gs[i, j])

            # Get image
            if radius in microscopy_results and lipid in microscopy_results[radius]['microscopy_images']:
                img_data = microscopy_results[radius]['microscopy_images'][lipid]
                image = img_data['continuous']

                # Plot
                cmap = cmaps.get(lipid, 'viridis')
                im = ax.imshow(image, cmap=cmap, origin='lower', interpolation='bilinear')

                # Colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, label='Intensity (a.u.)')

                # Title
                radius_nm = radius / 10.0
                if i == 0:
                    ax.set_title(f'{radius_nm:.0f} nm scale\n{lipid}', fontweight='bold', fontsize=12)
                else:
                    ax.set_title(lipid, fontweight='bold', fontsize=11)

                # Scale bar
                box_size_nm = 40  # Assuming 40 nm box
                pixels_per_nm = image.shape[0] / box_size_nm
                scale_bar_nm = 10  # 10 nm scale bar
                scale_bar_pixels = scale_bar_nm * pixels_per_nm

                ax.plot([10, 10 + scale_bar_pixels], [image.shape[0] - 10, image.shape[0] - 10],
                       'w-', linewidth=3)
                ax.text(10 + scale_bar_pixels / 2, image.shape[0] - 20, f'{scale_bar_nm} nm',
                       ha='center', va='top', color='white', fontweight='bold', fontsize=9)

            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

            ax.axis('off')

    fig.suptitle('Predicted Fluorescence Microscopy Images\n(Continuous Distribution Model)',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved microscopy predictions to {output_path}")

    return fig


def plot_microscopy_beads(microscopy_results, lipid_types=None, output_path=None):
    """Plot simulated microscopy with individual lipid beads

    Parameters
    ----------
    microscopy_results : dict
        Results from MicroscopyAnalyzer.analyze_and_simulate()
    lipid_types : list of str, optional
        Lipid types to plot
    output_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    target_radii = sorted(microscopy_results.keys())
    n_scales = len(target_radii)

    if lipid_types is None:
        first_scale = target_radii[0]
        lipid_types = list(microscopy_results[first_scale]['microscopy_images'].keys())

    n_lipids = len(lipid_types)

    # Create figure
    fig = plt.figure(figsize=(6 * n_scales, 5 * n_lipids))
    gs = GridSpec(n_lipids, n_scales, figure=fig, hspace=0.3, wspace=0.3)

    # Color maps
    cmaps = {
        'CHOL': 'Blues',
        'DPSM': 'Greens',
        'DIPC': 'Reds',
        'DPG3': 'Purples'
    }

    for i, lipid in enumerate(lipid_types):
        for j, radius in enumerate(target_radii):
            ax = fig.add_subplot(gs[i, j])

            if radius in microscopy_results and lipid in microscopy_results[radius]['microscopy_images']:
                img_data = microscopy_results[radius]['microscopy_images'][lipid]
                image = img_data['beads']

                # Plot
                cmap = cmaps.get(lipid, 'viridis')
                im = ax.imshow(image, cmap=cmap, origin='lower', interpolation='bilinear')

                # Colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, label='Intensity (a.u.)')

                # Title
                radius_nm = radius / 10.0
                n_beads = len(img_data['bead_positions'])
                if i == 0:
                    ax.set_title(f'{radius_nm:.0f} nm scale\n{lipid} ({n_beads} molecules)',
                                fontweight='bold', fontsize=11)
                else:
                    ax.set_title(f'{lipid} ({n_beads} molecules)',
                                fontweight='bold', fontsize=10)

                # Scale bar
                box_size_nm = 40
                pixels_per_nm = image.shape[0] / box_size_nm
                scale_bar_nm = 10
                scale_bar_pixels = scale_bar_nm * pixels_per_nm

                ax.plot([10, 10 + scale_bar_pixels], [image.shape[0] - 10, image.shape[0] - 10],
                       'w-', linewidth=3)
                ax.text(10 + scale_bar_pixels / 2, image.shape[0] - 20, f'{scale_bar_nm} nm',
                       ha='center', va='top', color='white', fontweight='bold', fontsize=9)

            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

            ax.axis('off')

    fig.suptitle('Simulated Single-Molecule Fluorescence Microscopy\n(Individual Lipid Beads)',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved bead microscopy to {output_path}")

    return fig


def plot_microscopy_comparison(microscopy_results, lipid_type='DPSM', output_path=None):
    """Compare continuous vs bead models for a single lipid

    Parameters
    ----------
    microscopy_results : dict
        Results from MicroscopyAnalyzer.analyze_and_simulate()
    lipid_type : str
        Lipid type to compare
    output_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    target_radii = sorted(microscopy_results.keys())
    n_scales = len(target_radii)

    fig, axes = plt.subplots(2, n_scales, figsize=(6 * n_scales, 10))

    cmap = {'CHOL': 'Blues', 'DPSM': 'Greens', 'DIPC': 'Reds', 'DPG3': 'Purples'}.get(lipid_type, 'viridis')

    for j, radius in enumerate(target_radii):
        radius_nm = radius / 10.0

        if radius in microscopy_results and lipid_type in microscopy_results[radius]['microscopy_images']:
            img_data = microscopy_results[radius]['microscopy_images'][lipid_type]

            # Continuous model
            ax_cont = axes[0, j]
            im_cont = ax_cont.imshow(img_data['continuous'], cmap=cmap, origin='lower', interpolation='bilinear')
            divider = make_axes_locatable(ax_cont)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im_cont, cax=cax)
            ax_cont.set_title(f'{radius_nm:.0f} nm - Continuous', fontweight='bold')
            ax_cont.axis('off')

            # Bead model
            ax_bead = axes[1, j]
            im_bead = ax_bead.imshow(img_data['beads'], cmap=cmap, origin='lower', interpolation='bilinear')
            divider = make_axes_locatable(ax_bead)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im_bead, cax=cax)
            n_beads = len(img_data['bead_positions'])
            ax_bead.set_title(f'{radius_nm:.0f} nm - Single Molecules ({n_beads} beads)', fontweight='bold')
            ax_bead.axis('off')

            # Scale bars
            box_size_nm = 40
            pixels_per_nm = img_data['continuous'].shape[0] / box_size_nm
            scale_bar_nm = 10
            scale_bar_pixels = scale_bar_nm * pixels_per_nm

            for ax in [ax_cont, ax_bead]:
                ax.plot([10, 10 + scale_bar_pixels], [img_data['continuous'].shape[0] - 10] * 2,
                       'w-', linewidth=3)
                ax.text(10 + scale_bar_pixels / 2, img_data['continuous'].shape[0] - 20,
                       f'{scale_bar_nm} nm', ha='center', va='top', color='white',
                       fontweight='bold', fontsize=10)

    fig.suptitle(f'Microscopy Simulation Comparison: {lipid_type}\nContinuous vs Single-Molecule Detection',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved microscopy comparison to {output_path}")

    return fig


def plot_multicolor_overlay(microscopy_results, radius=100.0, mode='beads', output_path=None):
    """Create multi-color overlay image (like real multi-channel microscopy)

    Parameters
    ----------
    microscopy_results : dict
        Results from MicroscopyAnalyzer.analyze_and_simulate()
    radius : float
        Which scale to visualize (Å)
    mode : str
        'beads' or 'continuous'
    output_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    if radius not in microscopy_results:
        print(f"ERROR: Radius {radius} not in results")
        return None

    lipid_types = list(microscopy_results[radius]['microscopy_images'].keys())

    # Create RGB image
    color_map = {
        'CHOL': [0, 0, 1],    # Blue
        'DPSM': [0, 1, 0],    # Green
        'DIPC': [1, 0, 0],    # Red
        'DPG3': [1, 0, 1]     # Magenta
    }

    # Get image size from first lipid
    first_lipid = lipid_types[0]
    img_data = microscopy_results[radius]['microscopy_images'][first_lipid]
    image = img_data[mode]
    shape = image.shape

    # Initialize RGB image
    rgb_image = np.zeros((*shape, 3))

    # Combine channels
    for lipid in lipid_types:
        if lipid in microscopy_results[radius]['microscopy_images']:
            img_data = microscopy_results[radius]['microscopy_images'][lipid]
            channel = img_data[mode]

            # Normalize
            channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-10)

            # Add to RGB
            color = color_map.get(lipid, [1, 1, 1])
            for c in range(3):
                rgb_image[:, :, c] += channel_norm * color[c]

    # Clip to [0, 1]
    rgb_image = np.clip(rgb_image, 0, 1)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Individual channels
    ax1 = axes[0]
    n_lipids = len(lipid_types)
    for i, lipid in enumerate(lipid_types):
        if lipid in microscopy_results[radius]['microscopy_images']:
            img_data = microscopy_results[radius]['microscopy_images'][lipid]
            channel = img_data[mode]
            channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-10)

            # Show as colored overlay
            color_img = np.zeros((*shape, 3))
            color = color_map.get(lipid, [1, 1, 1])
            for c in range(3):
                color_img[:, :, c] = channel_norm * color[c]

            ax1.imshow(color_img, alpha=0.7, origin='lower')

    ax1.set_title('Individual Channels (Overlaid)', fontweight='bold', fontsize=12)
    ax1.axis('off')

    # Legend for channels
    legend_elements = []
    for lipid in lipid_types:
        color = color_map.get(lipid, [1, 1, 1])
        legend_elements.append(mpatches.Patch(color=color, label=lipid))
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Merged image
    ax2 = axes[1]
    ax2.imshow(rgb_image, origin='lower')
    ax2.set_title('Merged Multi-Channel Image', fontweight='bold', fontsize=12)
    ax2.axis('off')

    # Scale bar
    box_size_nm = 40
    pixels_per_nm = shape[0] / box_size_nm
    scale_bar_nm = 10
    scale_bar_pixels = scale_bar_nm * pixels_per_nm

    for ax in axes:
        ax.plot([10, 10 + scale_bar_pixels], [shape[0] - 10] * 2, 'w-', linewidth=3)
        ax.text(10 + scale_bar_pixels / 2, shape[0] - 20, f'{scale_bar_nm} nm',
               ha='center', va='top', color='white', fontweight='bold', fontsize=10)

    radius_nm = radius / 10.0
    mode_text = 'Single-Molecule' if mode == 'beads' else 'Continuous'
    fig.suptitle(f'Multi-Color Fluorescence Microscopy Simulation\n{radius_nm:.0f} nm scale - {mode_text} Mode',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved multi-color overlay to {output_path}")

    return fig
