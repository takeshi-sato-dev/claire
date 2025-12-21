#!/usr/bin/env python3
"""
Configuration file for CLAIRE analysis
"""

import os
import multiprocessing as mp

# ===== Input Files =====
# Set paths to your topology and trajectory files here
# These are used as defaults if --topology and --trajectory are not specified
#TOPOLOGY_FILE = None  # e.g., "/path/to/system.psf"
#TRAJECTORY_FILE = None  # e.g., "/path/to/trajectory.xtc"

# Example configurations:

# For EGFR:
TOPOLOGY_FILE = '/Users/takeshi/Desktop/EGFR_MD_Exp/CG/241227L658Qpspscholsmgm3/gromacs/step5_assembly.psf'
TRAJECTORY_FILE = '/Users/takeshi/Desktop/EGFR_MD_Exp/CG/241227L658Qpspscholsmgm3/gromacs/step7_production.xtc'
# For EphA2":
#TOPOLOGY_FILE = "/Users/takeshi/Desktop/EphA2_MD/240225EphA2monomerized_DIPCDOPSCHOLSMGM3/gromacs/step5_assembly.psf"
#TRAJECTORY_FILE = "//Users/takeshi/Desktop/EphA2_MD/240225EphA2monomerized_DIPCDOPSCHOLSMGM3/gromacs/step7_production.xtc"

# For Notch:
#TOPOLOGY_FILE = "/Users/takeshi/Desktop/Notch_results/250717NotchTMJMDIPCDIPSCHOLSMGM3/gromacs/step5_assembly.psf"
#TRAJECTORY_FILE = "/Users/takeshi/Desktop/Notch_results/250717NotchTMJMDIPCDIPSCHOLSMGM3/gromacs/step7_production.xtc"

# ===== Frame Selection =====
# Set default frame range for analysis
# These are used if --start, --stop, --step are not specified
FRAME_START = 20000          # Start frame (default: 0) - include early frames for unbound states
FRAME_STOP = 80000        # Stop frame (default: None = all frames)
FRAME_STEP = 20           # Frame step (default: 1 = every frame) - larger step for speed

# Example: Analyze every 10th frame from 1000 to 5000
# FRAME_START = 1000
# FRAME_STOP = 5000
# FRAME_STEP = 10

# ===== File paths =====
TEMP_FILES_DIR = "temp_files"

# ===== Physical parameters =====
CONTACT_CUTOFF = 6.0  # Å for GM3 binding detection (residue-level, LIPAC method)
COMPOSITION_CUTOFF = 15.0  # Å for lipid composition analysis (TM domain vicinity, XY plane)
LEAFLET_CUTOFF = 10.0  # Å for leaflet detection

# ===== Lipid types =====
# Default lipid types - can be overridden (composition lipids, excluding target)
DEFAULT_LIPID_TYPES = ['CHOL', 'DPSM', 'DIPC']
TARGET_LIPID = 'DPG3'  # Set to specific lipid name if analyzing mediator effects

# ===== Protein structure =====
# TM domain residue ranges for each protein chain (for composition calculation)
# Format: {segid: (start_resid, end_resid)}
# If None, uses entire protein
TM_RESIDUES = {
    'PROA': (621, 644),  # EGFR TM domain
    'PROB': (621, 644),
    'PROC': (621, 644),
    'PROD': (621, 644)
}
# Set to None to use entire protein instead of TM domain only
# TM_RESIDUES = None

# ===== Analysis parameters =====
# Composition analysis
COMPOSITION_RATIO_THRESHOLD = 0.75  # Quartile threshold for high/low groups
STATISTICAL_SIGNIFICANCE = 0.05  # P-value threshold

# Temporal analysis
TEMPORAL_WINDOW_SIZE = 100  # frames for sliding window
TEMPORAL_STEP_SIZE = 10  # frames between windows

# Spatial analysis
SPATIAL_RADII = [5.0, 10.0, 15.0, 20.0]  # Å for radial composition profiles
SPATIAL_SHELL_TYPE = 'both'  # 'individual', 'cumulative', or 'both'
SPATIAL_N_FRAMES = 1000  # Number of frames for spatial analysis (None = all frames)

# ML analysis
ML_TEST_SIZE = 0.3  # Fraction for test set
ML_RANDOM_SEED = 42
ML_N_BOOTSTRAP = 1000

# ===== Parallel processing =====
# Use parallel processing by default
USE_PARALLEL = True
# Automatically detect CPU cores
N_CORES = max(2, int(mp.cpu_count() * 0.75))
BATCH_SIZE = 50  # Frames per batch
MIN_CORES = 2  # Minimum cores for parallel processing

# ===== Visualization =====
# Figure settings
FIGURE_DPI = 300
FIGURE_FORMATS = ['png', 'svg']  # Save all plots in these formats
FONT_SIZE = 10
FONT_FAMILY = 'Arial'

# Color schemes
COLOR_PALETTE = {
    'CHOL': '#3498db',  # Blue
    'DIPC': '#e74c3c',  # Red
    'DPSM': '#2ecc71',  # Green
    'POPC': '#f39c12',  # Orange
    'POPE': '#9b59b6',  # Purple
    'POPS': '#1abc9c',  # Turquoise
}

# ===== Output settings =====
OUTPUT_DIR = "claire_output"
SAVE_INTERMEDIATE = True
CHECKPOINT_INTERVAL = 100  # frames

# ===== Debug settings =====
VERBOSE = True
DEBUG = False
