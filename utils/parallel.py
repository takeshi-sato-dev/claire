#!/usr/bin/env python3
"""
Parallel processing utilities
Adapted from LIPAC stage1_contact_analysis
"""

import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import platform
import time


def test_multiprocessing():
    """Test multiprocessing setup

    Returns
    -------
    tuple
        (success, context_method)
    """
    print("\n" + "="*70)
    print("TESTING MULTIPROCESSING")
    print("="*70)

    try:
        n_cores = cpu_count()
        print(f"System has {n_cores} CPU cores")
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"Machine: {platform.machine()}")

        # Detect Apple Silicon
        is_apple_silicon = (platform.system() == 'Darwin' and platform.machine() == 'arm64')
        if is_apple_silicon:
            print("✓ Detected Apple Silicon (M1/M2/M3)")

        # Select context
        if platform.system() == 'Darwin':
            ctx_method = 'fork'
            print("Using 'fork' context for macOS")
        else:
            ctx_method = 'spawn'
            print("Using 'spawn' context")

        print("\nRunning parallel test...")

        # Use simple test that doesn't require pickling local functions
        ctx = mp.get_context(ctx_method)
        with ctx.Pool(processes=min(4, n_cores)) as pool:
            # Use built-in function instead of local function
            results = pool.map(abs, [-1, -2, -3, -4])
            print(f"✓ Test successful: {results}")

        print("="*70)
        return True, ctx_method

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 'fork'


def process_frames_parallel(frames, top_file, traj_file, protein_segids,
                           leaflet_resids, lipid_types, contact_cutoff=15.0,
                           target_lipid=None, tm_residues=None, n_workers=None, ctx_method='fork',
                           chain_a_pattern=None, chain_b_pattern=None):
    """Process frames in parallel

    Parameters
    ----------
    frames : list of int
        Frame indices to process
    top_file : str
        Topology file path
    traj_file : str
        Trajectory file path
    protein_segids : list of str
        Protein segment IDs
    leaflet_resids : list of int
        Leaflet residue IDs
    lipid_types : list of str
        Lipid residue names
    contact_cutoff : float, default 15.0
        Contact cutoff distance (A)
    target_lipid : str, optional
        Target lipid name
    tm_residues : dict, optional
        TM domain residue ranges {segid: (start, end)}
    n_workers : int, optional
        Number of parallel workers. If None, auto-detect.
    ctx_method : str, default 'fork'
        Multiprocessing context ('fork' or 'spawn')
    chain_a_pattern : str, optional
        MDAnalysis selection for sn-1 tail beads (for S_CD)
    chain_b_pattern : str, optional
        MDAnalysis selection for sn-2 tail beads (for S_CD)

    Returns
    -------
    list
        List of frame composition data dictionaries
    """
    import sys
    import os
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.frame_processor import process_frame_wrapper

    if n_workers is None:
        n_workers = max(2, int(cpu_count() * 0.75))

    n_workers = min(n_workers, len(frames))

    print(f"\nProcessing {len(frames)} frames with {n_workers} workers...")

    # Prepare arguments for each frame
    worker_args = [
        (frame_idx, top_file, traj_file, protein_segids, leaflet_resids,
         lipid_types, contact_cutoff, target_lipid, tm_residues,
         chain_a_pattern, chain_b_pattern)
        for frame_idx in frames
    ]

    # Process in parallel
    try:
        ctx = mp.get_context(ctx_method)
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(process_frame_wrapper, worker_args)

        # Filter out None results
        successful = [r for r in results if r is not None]
        print(f"✓ Successfully processed {len(successful)}/{len(frames)} frames")

        return successful

    except Exception as e:
        print(f"ERROR in parallel processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def process_frames_serial(frames, universe, proteins, lipid_selections, leaflet,
                          contact_cutoff=15.0, target_lipid=None, tm_residues=None,
                          chain_a_pattern=None, chain_b_pattern=None):
    """Process frames serially (fallback)

    Parameters
    ----------
    frames : list of int
        Frame indices to process
    universe : MDAnalysis.Universe
        Universe object
    proteins : dict
        Protein selections
    lipid_selections : dict
        Lipid selections
    leaflet : MDAnalysis.AtomGroup
        Leaflet atoms
    contact_cutoff : float, default 15.0
        Contact cutoff distance (A)
    target_lipid : str, optional
        Target lipid name
    tm_residues : dict, optional
        TM domain residue ranges {segid: (start, end)}
    chain_a_pattern : str, optional
        MDAnalysis selection for sn-1 tail beads (for S_CD)
    chain_b_pattern : str, optional
        MDAnalysis selection for sn-2 tail beads (for S_CD)

    Returns
    -------
    list
        List of frame composition data dictionaries
    """
    import sys
    import os
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.frame_processor import calculate_frame_composition

    print(f"\nProcessing {len(frames)} frames serially...")

    results = []
    for i, frame_idx in enumerate(frames):
        if i % 10 == 0:
            print(f"  Frame {i+1}/{len(frames)}")

        frame_data = calculate_frame_composition(
            universe, frame_idx, proteins, lipid_selections, leaflet,
            contact_cutoff, target_lipid, tm_residues,
            chain_a_pattern, chain_b_pattern
        )

        if frame_data is not None:
            results.append(frame_data)

    print(f"✓ Processed {len(results)}/{len(frames)} frames")
    return results
