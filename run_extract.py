#!/usr/bin/env python3
"""
CLAIRE v21 Stage 1: Trajectory extraction -> CSV

Extracts per-frame, per-copy lipid composition and structural variables
from a coarse-grained MD trajectory. Produces composition_data.csv.

Variables extracted:
  - target_lipid_min_distance: GM3-TM minimum bead distance (A)
  - tm_tilt: TM helix tilt angle (degrees)
  - local_scd: mean acyl chain order parameter in 15A cylinder
  - {LIPID}_count: lipid counts in 15A cylinder
  - {LIPID}_first_shell: lipid molecule contacts within 6A of TM

Usage:
    python run_extract.py --topology step5.psf --trajectory step7.xtc
    python run_extract.py  # uses config.py defaults
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.trajectory_loader import (load_universe, identify_lipid_leaflets,
                                    select_proteins, select_lipids)
from utils.parallel import test_multiprocessing, process_frames_parallel, process_frames_serial
from analysis.composition import CompositionAnalyzer
from config import *


def main():
    parser = argparse.ArgumentParser(
        description='CLAIRE v2 Stage 1 — Trajectory extraction')
    parser.add_argument('--topology', default=None)
    parser.add_argument('--trajectory', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--lipids', nargs='+', default=None)
    parser.add_argument('--target-lipid', default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--stop', type=int, default=None)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--cutoff', type=float, default=None)
    parser.add_argument('--no-parallel', action='store_true')
    parser.add_argument('--n-workers', type=int, default=None)
    parser.add_argument('--leaflet', choices=['upper', 'lower'], default='upper')

    args = parser.parse_args()

    topology = args.topology or TOPOLOGY_FILE
    trajectory = args.trajectory or TRAJECTORY_FILE
    lipid_types = args.lipids or DEFAULT_LIPID_TYPES
    target_lipid = args.target_lipid or TARGET_LIPID
    frame_start = args.start if args.start is not None else FRAME_START
    frame_stop = args.stop if args.stop is not None else FRAME_STOP
    frame_step = args.step if args.step is not None else FRAME_STEP
    cutoff = args.cutoff if args.cutoff is not None else COMPOSITION_CUTOFF
    output_dir = args.output or OUTPUT_DIR
    use_parallel = not args.no_parallel and USE_PARALLEL

    if topology is None or trajectory is None:
        print("ERROR: Specify --topology and --trajectory (or set in config.py)")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('temp_files', exist_ok=True)

    print("\n" + "=" * 70)
    print("CLAIRE v21 Stage 1: Trajectory Extraction")
    print("=" * 70)
    print(f"Topology:   {topology}")
    print(f"Trajectory: {trajectory}")
    print(f"Frames:     {frame_start}–{frame_stop or 'end'} (step {frame_step})")
    print(f"Cutoff:     {cutoff} Å")
    print("=" * 70)

    # Load
    u = load_universe(topology, trajectory)
    if u is None:
        return

    stop = frame_stop if frame_stop is not None else len(u.trajectory)
    frames = list(range(frame_start, stop, frame_step))
    print(f"Processing {len(frames)} frames")

    # Leaflets
    lipid_types_all = lipid_types.copy()
    if target_lipid and target_lipid not in lipid_types_all:
        lipid_types_all.append(target_lipid)

    upper, lower = identify_lipid_leaflets(u, lipid_types_all, leaflet_cutoff=LEAFLET_CUTOFF)
    leaflet = upper if args.leaflet == 'upper' else lower

    # Proteins & lipids
    proteins = select_proteins(u)
    lipid_selections = select_lipids(u, leaflet, lipid_types_all)
    protein_segids = ['PROA', 'PROB', 'PROC', 'PROD'][:len(proteins)]
    leaflet_resids = [res.resid for res in leaflet.residues]

    # Process frames
    try:
        chain_a = CHAIN_A_PATTERN
        chain_b = CHAIN_B_PATTERN
    except NameError:
        chain_a = None
        chain_b = None
    if chain_a and chain_b:
        print(f"S_CD:       enabled (sn-1: {chain_a}, sn-2: {chain_b})")
    else:
        print(f"S_CD:       disabled")

    if use_parallel:
        success, ctx = test_multiprocessing()
        if success:
            frame_data_list = process_frames_parallel(
                frames, topology, trajectory, protein_segids,
                leaflet_resids, lipid_types_all, cutoff, target_lipid,
                TM_RESIDUES, args.n_workers, ctx,
                chain_a, chain_b)
        else:
            frame_data_list = process_frames_serial(
                frames, u, proteins, lipid_selections, leaflet,
                cutoff, target_lipid, TM_RESIDUES,
                chain_a, chain_b)
    else:
        frame_data_list = process_frames_serial(
            frames, u, proteins, lipid_selections, leaflet,
            cutoff, target_lipid, TM_RESIDUES,
            chain_a, chain_b)

    if not frame_data_list:
        print("ERROR: No frames processed")
        return

    # Convert to DataFrame and save
    analyzer = CompositionAnalyzer(lipid_types_all, target_lipid)
    df = analyzer.frames_to_dataframe(frame_data_list)
    df = analyzer.calculate_conservation_ratios(df)

    csv_path = os.path.join(output_dir, 'composition_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*70}")
    print(f"Stage 1 complete. Saved: {csv_path}")
    print(f"Run Stage 2: python run_analyze.py {csv_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
