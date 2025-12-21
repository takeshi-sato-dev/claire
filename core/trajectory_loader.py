#!/usr/bin/env python3
"""
Trajectory loading and leaflet identification
Adapted from LIPAC stage1_contact_analysis
"""

import os
import pickle
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder


def load_universe(top_file, traj_file):
    """Load MD trajectory

    Parameters
    ----------
    top_file : str
        Path to topology file (PSF, PDB, GRO, etc.)
    traj_file : str
        Path to trajectory file (XTC, DCD, TRR, etc.)

    Returns
    -------
    MDAnalysis.Universe or None
        Loaded universe object, or None if loading failed
    """
    print(f"Loading topology: {top_file}")
    print(f"Loading trajectory: {traj_file}")

    if not os.path.exists(top_file):
        print(f"ERROR: Topology file not found: {top_file}")
        return None

    if not os.path.exists(traj_file):
        print(f"ERROR: Trajectory file not found: {traj_file}")
        return None

    try:
        universe = mda.Universe(top_file, traj_file)
        print(f"✓ Trajectory loaded: {len(universe.atoms)} atoms, {len(universe.trajectory)} frames")
        return universe
    except Exception as e:
        print(f"ERROR loading universe: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def identify_lipid_leaflets(universe, lipid_types=None, leaflet_cutoff=10.0, temp_dir="temp_files"):
    """Identify and cache lipid leaflets

    Uses LeafletFinder for initial detection, then caches results for reuse.
    Determines upper/lower leaflets based on sphingomyelin (DPSM) content.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        The universe object
    lipid_types : list of str, optional
        Lipid residue names to consider. If None, auto-detect.
    leaflet_cutoff : float, default 10.0
        Cutoff distance (Å) for leaflet detection
    temp_dir : str, default "temp_files"
        Directory for caching leaflet information

    Returns
    -------
    tuple
        (upper_leaflet, lower_leaflet) as MDAnalysis.AtomGroup objects
    """
    print("\n" + "="*70)
    print("LEAFLET IDENTIFICATION")
    print("="*70)

    os.makedirs(temp_dir, exist_ok=True)

    # Create system-specific cache filename based on topology filename
    import hashlib
    topology_name = os.path.basename(universe.filename)
    system_hash = hashlib.md5(topology_name.encode()).hexdigest()[:8]
    leaflet_cache = os.path.join(temp_dir, f"leaflet_info_{system_hash}.pickle")
    print(f"Cache file: {leaflet_cache}")

    # Try to load cached leaflet information
    if os.path.exists(leaflet_cache):
        print(f"Loading cached leaflet information from {leaflet_cache}")
        try:
            with open(leaflet_cache, 'rb') as f:
                leaflet_info = pickle.load(f)

            upper_resids = leaflet_info['upper_leaflet_resids']
            lower_resids = leaflet_info.get('lower_leaflet_resids', [])

            # Reconstruct leaflets from cached resids
            if lipid_types is None:
                lipid_types = ['CHOL', 'DIPC', 'DPSM', 'POPC', 'POPE', 'POPS', 'DPG3']

            lipid_resnames = " ".join(lipid_types)
            lipid_atoms = universe.select_atoms(f"resname {lipid_resnames}")

            upper_leaflet = mda.AtomGroup([], universe)
            lower_leaflet = mda.AtomGroup([], universe)

            # Batch selection to avoid recursion errors
            batch_size = 100
            print(f"Reconstructing upper leaflet ({len(upper_resids)} residues)...")
            for i in range(0, len(upper_resids), batch_size):
                batch = upper_resids[i:i+batch_size]
                sel_str = " or ".join([f"resid {r}" for r in batch])
                upper_leaflet = upper_leaflet.union(lipid_atoms.select_atoms(sel_str))

            if lower_resids:
                print(f"Reconstructing lower leaflet ({len(lower_resids)} residues)...")
                for i in range(0, len(lower_resids), batch_size):
                    batch = lower_resids[i:i+batch_size]
                    sel_str = " or ".join([f"resid {r}" for r in batch])
                    lower_leaflet = lower_leaflet.union(lipid_atoms.select_atoms(sel_str))

            print(f"✓ Upper leaflet: {len(upper_leaflet.residues)} molecules")
            print(f"✓ Lower leaflet: {len(lower_leaflet.residues)} molecules")

            return upper_leaflet, lower_leaflet

        except Exception as e:
            print(f"Error loading cached leaflet: {str(e)}")
            print("Proceeding with fresh leaflet detection...")

    # Perform fresh leaflet detection
    print("Running LeafletFinder for fresh leaflet detection...")

    # Select lipid headgroup beads for Martini CG
    # PO4: phosphate beads, ROH: cholesterol, GL1/GL2: glycerol, AM1/AM2: amide, GM1/GM2: ganglioside
    L = LeafletFinder(universe, "name PO4 ROH GL1 GL2 AM1 AM2 GM1 GM2")
    cutoff = L.update(leaflet_cutoff)

    # Handle case where update returns None
    if cutoff is None:
        cutoff = leaflet_cutoff
        print(f"Using specified cutoff: {cutoff:.2f} Å")
    else:
        print(f"Optimized cutoff distance: {cutoff:.2f} Å")

    print(f"Number of leaflets found: {len(L.components)}")

    if len(L.components) < 2:
        print("WARNING: Only 1 leaflet found. Using same leaflet for both.")
        upper_leaflet = L.groups(0)
        lower_leaflet = upper_leaflet
        upper_resids = [res.resid for res in upper_leaflet.residues]
        lower_resids = []
    else:
        leaflet0 = L.groups(0)
        leaflet1 = L.groups(1)

        # Determine upper leaflet based on DPSM content (sphingomyelin in upper leaflet)
        dpsm_count0 = len(leaflet0.select_atoms("resname DPSM").residues)
        dpsm_count1 = len(leaflet1.select_atoms("resname DPSM").residues)

        print(f"DPSM content: Leaflet 0 = {dpsm_count0}, Leaflet 1 = {dpsm_count1}")

        if dpsm_count0 >= dpsm_count1:
            upper_leaflet = leaflet0
            lower_leaflet = leaflet1
        else:
            upper_leaflet = leaflet1
            lower_leaflet = leaflet0

        upper_resids = [res.resid for res in upper_leaflet.residues]
        lower_resids = [res.resid for res in lower_leaflet.residues]

    # Cache leaflet information
    leaflet_info = {
        'upper_leaflet_resids': upper_resids,
        'lower_leaflet_resids': lower_resids,
    }

    try:
        with open(leaflet_cache, 'wb') as f:
            pickle.dump(leaflet_info, f)
        print(f"✓ Leaflet information cached to {leaflet_cache}")
    except Exception as e:
        print(f"Warning: Could not cache leaflet information: {str(e)}")

    print("="*70)
    return upper_leaflet, lower_leaflet


def select_lipids(universe, leaflet, lipid_types=None, verbose=True):
    """Select lipids from specified leaflet

    Parameters
    ----------
    universe : MDAnalysis.Universe
        The universe object
    leaflet : MDAnalysis.AtomGroup
        Leaflet atoms to select from
    lipid_types : list of str, optional
        Lipid residue names. If None, use default set.
    verbose : bool, default True
        Whether to print selection information

    Returns
    -------
    dict
        Dictionary mapping lipid_type -> AtomGroup
    """
    if lipid_types is None:
        lipid_types = ['CHOL', 'DIPC', 'DPSM', 'POPC', 'POPE', 'POPS']

    lipid_selections = {}
    total_molecules = len(leaflet.residues)

    if verbose:
        print(f"\nSelecting lipids from leaflet ({total_molecules} total molecules):")

    for lipid_type in lipid_types:
        try:
            lipid_sel = leaflet.select_atoms(f"resname {lipid_type}")
            n_molecules = len(lipid_sel.residues)
            percentage = (n_molecules / total_molecules * 100) if total_molecules > 0 else 0

            # LIPAC format: {'sel': [lipid_sel]}
            lipid_selections[lipid_type] = {'sel': [lipid_sel]}
            if verbose:
                print(f"  {lipid_type}: {n_molecules} molecules ({percentage:.1f}%)")

        except Exception as e:
            if verbose:
                print(f"  {lipid_type}: Error - {str(e)}")
            lipid_selections[lipid_type] = {'sel': [mda.AtomGroup([], universe)]}

    return lipid_selections


def select_proteins(universe, n_proteins=None, segids=None, selection="protein"):
    """Select proteins from universe

    Parameters
    ----------
    universe : MDAnalysis.Universe
        The universe object
    n_proteins : int, optional
        Number of proteins to select. If None, auto-detect.
    segids : list of str, optional
        Specific segment IDs to select. If None, use standard PROA/PROB/PROC/PROD
    selection : str, default "protein"
        MDAnalysis selection string for protein atoms

    Returns
    -------
    dict
        Dictionary mapping protein_name -> AtomGroup
    """
    proteins = {}

    # Determine segment IDs
    if segids is None:
        if n_proteins is None:
            # Auto-detect available segments
            test_segids = ['PROA', 'PROB', 'PROC', 'PROD']
            segids = []
            for seg in test_segids:
                try:
                    test_sel = universe.select_atoms(f"segid {seg}")
                    if len(test_sel) > 0:
                        segids.append(seg)
                except:
                    pass
        else:
            segids = ['PROA', 'PROB', 'PROC', 'PROD'][:n_proteins]

    if len(segids) == 0:
        print("WARNING: No proteins found")
        return proteins

    print(f"\nSelecting proteins:")
    for i, segid in enumerate(segids, 1):
        protein_name = f"Protein_{i}"
        try:
            protein_sel = universe.select_atoms(f"segid {segid} and {selection}")

            if len(protein_sel) > 0:
                proteins[protein_name] = protein_sel
                n_residues = len(protein_sel.residues)
                resid_range = f"{protein_sel.residues.resids.min()}:{protein_sel.residues.resids.max()}"
                print(f"  {protein_name}: {len(protein_sel)} atoms, {n_residues} residues (resid {resid_range})")
            else:
                print(f"  {protein_name}: No atoms found in segid {segid}")

        except Exception as e:
            print(f"  {protein_name}: Error - {str(e)}")
            proteins[protein_name] = mda.AtomGroup([], universe)

    return proteins
