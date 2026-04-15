#!/usr/bin/env python3
"""
Configuration for CLAIRE v21 analysis

v21: Full causal chain analysis
     Static mediation + Temporal causal (GC/CGC/TE)
     Variables: distance, tilt, local_scd, lipid counts, first_shell contacts
"""

import multiprocessing as mp

# ===== Input Files =====
TOPOLOGY_FILE = '/Users/takeshi/Desktop/EGFR_MD_Exp/CG/241204EGFRTMJM4peptides_DIPCDOPSCHOLSMGM3/gromacs/step5_assembly.psf'
TRAJECTORY_FILE ='/Users/takeshi/Desktop/EGFR_MD_Exp/CG/241204EGFRTMJM4peptides_DIPCDOPSCHOLSMGM3/gromacs/step7_production.xtc'

# ===== Frame Selection =====
FRAME_START = 20000    # Start frame (2 us at 100 ps/frame)
FRAME_STOP = 80000     # Stop frame (8 us)
FRAME_STEP = 20        # Every 20th frame -> 3000 frames per copy

# ===== Physical Parameters =====
CONTACT_CUTOFF = 6.0       # A - target lipid binding detection (LIPAC-consistent)
COMPOSITION_CUTOFF = 15.0  # A - local composition cylinder radius (xy-plane)
LEAFLET_CUTOFF = 10.0      # A - leaflet assignment

# ===== Lipid Definitions =====
DEFAULT_LIPID_TYPES = ['CHOL', 'DPSM', 'DIPC']
TARGET_LIPID = 'DPG3'

# ===== Protein Structure =====
# TM domain residue range (REQUIRED for v5 min distance calculation)
# Used for:
#   1. Composition cylinder center (COM of TM domain)
#   2. Target lipid min distance (v5: bead-to-bead distance to TM only)
# Format: {segid: (start_resid, end_resid)}
# Example for EGFR TM-JM (residues 620-650 for TM helix):
#   TM_RESIDUES = {'PROA': (620, 650), 'PROB': (620, 650),
#                  'PROC': (620, 650), 'PROD': (620, 650)}
TM_RESIDUES = {
    'PROA': (65, 88),
    'PROB': (65, 88),
    'PROC': (65, 88),
    'PROD': (65, 88),
}  # SET THIS for accurate v5 temporal analysis

# ===== S_CD Chain Patterns (v11) =====
# MARTINI CG tail bead selections for order parameter calculation
# Covers both saturated (C) and unsaturated (D) beads
# Set to None to skip local S_CD calculation
CHAIN_A_PATTERN = "name C1A C2A C3A C4A D2A D3A D4A"  # sn-1 tail
CHAIN_B_PATTERN = "name C1B C2B C3B C4B D2B D3B D4B"  # sn-2 tail

# ===== Parallel Processing =====
USE_PARALLEL = True
N_CORES = max(2, int(mp.cpu_count() * 0.75))
BATCH_SIZE = 50

# ===== Output =====
OUTPUT_DIR = "claire_output"
