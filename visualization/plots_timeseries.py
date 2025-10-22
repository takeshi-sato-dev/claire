#!/usr/bin/env python3
"""
Visualization for time-series ML predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


def plot_training_history(history, output_path):
    """Plot training and validation loss curves

    Parameters
    ----------
    history : dict
        Training history with 'train_loss' and 'val_loss'
    output_path : str
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if history['val_loss'] and history['val_loss'][0] != 0:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved training history plot: {output_path}")


def plot_prediction_vs_actual(y_true, y_pred, lipid_types, output_path):
    """Plot predicted vs actual compositions

    Parameters
    ----------
    y_true : numpy.ndarray
        True compositions (n_samples, n_lipids)
    y_pred : numpy.ndarray
        Predicted compositions (n_samples, n_lipids)
    lipid_types : list of str
        Lipid type names
    output_path : str
        Path to save figure
    """
    n_lipids = len(lipid_types)
    fig, axes = plt.subplots(1, n_lipids, figsize=(6*n_lipids, 5))

    if n_lipids == 1:
        axes = [axes]

    for i, (ax, lipid) in enumerate(zip(axes, lipid_types)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=20)

        # Perfect prediction line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true[:, i], y_pred[:, i])

        ax.set_xlabel('True Fraction', fontsize=12)
        ax.set_ylabel('Predicted Fraction', fontsize=12)
        ax.set_title(f'{lipid}\nR² = {r2:.3f}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved prediction vs actual plot: {output_path}")


def plot_time_series_example(true_sequence, pred_sequence, lipid_types,
                             binding_states, output_path, n_lookback=10):
    """Plot example time-series prediction

    Parameters
    ----------
    true_sequence : numpy.ndarray
        True composition sequence (n_frames, n_lipids)
    pred_sequence : numpy.ndarray
        Predicted composition sequence (n_frames, n_lipids)
    lipid_types : list of str
        Lipid type names
    binding_states : numpy.ndarray
        GM3 binding states (n_frames,)
    output_path : str
        Path to save figure
    n_lookback : int
        Number of frames used for lookback
    """
    n_frames = len(true_sequence)
    time = np.arange(n_frames)

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(len(lipid_types) + 1, 1, height_ratios=[3]*len(lipid_types) + [1])

    # Plot each lipid type
    for i, lipid in enumerate(lipid_types):
        ax = fig.add_subplot(gs[i])

        # Lookback region
        ax.axvspan(0, n_lookback-1, alpha=0.2, color='gray', label='Lookback')

        # True composition
        ax.plot(time, true_sequence[:, i], 'b-', linewidth=2, label='True', alpha=0.7)

        # Predicted composition
        ax.plot(time[n_lookback:], pred_sequence[:, i], 'r--', linewidth=2, label='Predicted')

        ax.set_ylabel(f'{lipid}\nFraction', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_title('Time-Series Prediction Example', fontsize=14, fontweight='bold')
        if i < len(lipid_types) - 1:
            ax.set_xticklabels([])

    # GM3 binding state
    ax_binding = fig.add_subplot(gs[-1])
    ax_binding.fill_between(time, 0, binding_states,
                           where=(binding_states > 0),
                           color='purple', alpha=0.6, label='GM3 Bound')
    ax_binding.set_ylabel('GM3\nBinding', fontsize=11)
    ax_binding.set_xlabel('Frame', fontsize=12)
    ax_binding.set_ylim(-0.1, 1.1)
    ax_binding.set_yticks([0, 1])
    ax_binding.legend(loc='upper right', fontsize=9)
    ax_binding.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved time-series example plot: {output_path}")


def plot_long_term_prediction(predictions, lipid_types, output_path,
                              initial_state=None, frame_step=1):
    """Plot long-term autoregressive predictions

    Parameters
    ----------
    predictions : numpy.ndarray
        Predicted compositions (n_steps, n_lipids)
    lipid_types : list of str
        Lipid type names
    output_path : str
        Path to save figure
    initial_state : numpy.ndarray, optional
        Initial composition state
    frame_step : int
        Frame step for time axis
    """
    n_steps = len(predictions)
    time = np.arange(n_steps) * frame_step

    fig, axes = plt.subplots(len(lipid_types), 1, figsize=(12, 3*len(lipid_types)), sharex=True)

    if len(lipid_types) == 1:
        axes = [axes]

    for i, (ax, lipid) in enumerate(zip(axes, lipid_types)):
        ax.plot(time, predictions[:, i], linewidth=2, label=f'{lipid} Predicted')

        # Mark initial state if provided
        if initial_state is not None:
            ax.axhline(initial_state[i], color='gray', linestyle='--',
                      alpha=0.5, label=f'{lipid} Initial')

        ax.set_ylabel('Fraction', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_title('Long-Term Composition Prediction (Autoregressive)',
                        fontsize=14, fontweight='bold')

    axes[-1].set_xlabel('Frame', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved long-term prediction plot: {output_path}")


def create_nanodomain_animation(predictions, lipid_types, output_dir,
                                frame_step=1, fps=10):
    """Create animation of predicted nanodomain dynamics

    Parameters
    ----------
    predictions : numpy.ndarray
        Predicted compositions over time (n_steps, n_lipids)
    lipid_types : list of str
        Lipid type names
    output_dir : str
        Directory to save frames and animation
    frame_step : int
        Frame step for time axis
    fps : int
        Frames per second for animation
    """
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("  WARNING: Animation requires matplotlib with Pillow. Skipping animation.")
        return

    n_steps = len(predictions)
    time = np.arange(n_steps) * frame_step

    fig, axes = plt.subplots(len(lipid_types), 1, figsize=(10, 3*len(lipid_types)), sharex=True)

    if len(lipid_types) == 1:
        axes = [axes]

    # Initialize plot elements
    lines = []
    for i, (ax, lipid) in enumerate(zip(axes, lipid_types)):
        line, = ax.plot([], [], linewidth=2, label=lipid)
        lines.append(line)

        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Fraction', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_title('Nanodomain Dynamics Prediction', fontsize=14, fontweight='bold')

    axes[-1].set_xlabel('Frame', fontsize=12)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        for i, line in enumerate(lines):
            line.set_data(time[:frame+1], predictions[:frame+1, i])
        return lines

    anim = FuncAnimation(fig, update, init_func=init, frames=n_steps,
                        interval=1000//fps, blit=True, repeat=True)

    output_path = os.path.join(output_dir, 'nanodomain_dynamics.gif')
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)

    plt.close()

    print(f"  Saved animation: {output_path}")


def plot_error_over_time(errors, lipid_types, output_path):
    """Plot prediction error as a function of prediction horizon

    Parameters
    ----------
    errors : numpy.ndarray
        Prediction errors (n_horizons, n_lipids)
    lipid_types : list of str
        Lipid type names
    output_path : str
        Path to save figure
    """
    horizons = np.arange(1, len(errors) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, lipid in enumerate(lipid_types):
        ax.plot(horizons, errors[:, i], marker='o', linewidth=2,
               markersize=6, label=lipid)

    ax.set_xlabel('Prediction Horizon (frames)', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Prediction Error vs Horizon', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved error over time plot: {output_path}")
