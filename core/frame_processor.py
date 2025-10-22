#!/usr/bin/env python3
"""
Frame processing for composition analysis
"""

import numpy as np
import traceback


def calculate_lipid_protein_residue_contacts_lipac(protein, lipid_sel, box, cutoff=6.0):
    """Calculate residue-level lipid-protein contacts using LIPAC method

    Exact implementation matching LIPAC including optimizations

    Parameters
    ----------
    protein : MDAnalysis.AtomGroup
        Protein selection
    lipid_sel : MDAnalysis.AtomGroup
        Lipid selection
    box : numpy.ndarray
        Box dimensions
    cutoff : float, default 6.0
        Contact cutoff in Angstroms

    Returns
    -------
    numpy.ndarray
        Contact array (length = number of protein residues)
    """
    contacts = np.zeros(len(protein.residues))

    if len(lipid_sel) == 0:
        return contacts

    # Calculate average Z coordinate of leaflet (for optimization)
    leaflet_z_avg = np.mean([atom.position[2] for atom in lipid_sel.atoms])

    # Loop through each protein residue (LIPAC method)
    for i, res in enumerate(protein.residues):
        # Calculate average Z coordinate (height) of residue
        res_z_avg = np.mean([atom.position[2] for atom in res.atoms])

        # Skip if Z distance is too large (LIPAC optimization)
        z_diff = abs(res_z_avg - leaflet_z_avg)
        if z_diff > 15.0:  # Skip if separated by more than 15Å
            continue

        # Loop through each lipid molecule
        for lipid_res in lipid_sel.residues:
            # Optimization: first check residue COM distance (LIPAC method)
            res_com = res.atoms.center_of_mass()
            lipid_com = lipid_res.atoms.center_of_mass()

            # COM distance calculation with PBC correction
            com_diff = res_com - lipid_com
            for dim in range(3):
                if com_diff[dim] > box[dim] * 0.5:
                    com_diff[dim] -= box[dim]
                elif com_diff[dim] < -box[dim] * 0.5:
                    com_diff[dim] += box[dim]

            com_dist = np.sqrt(np.sum(com_diff * com_diff))

            # Skip if COMs are too far apart (LIPAC optimization)
            max_atom_dist = 8.0  # Estimated maximum distance between atoms in residue
            if com_dist > (cutoff + max_atom_dist):
                continue

            # Calculate minimum atom-atom distance between residue and lipid
            min_dist = float('inf')

            for atom in res.atoms:
                for lipid_atom in lipid_res.atoms:
                    # 3D distance calculation with PBC correction (LIPAC method)
                    diff = atom.position - lipid_atom.position
                    # Apply PBC for each dimension
                    for dim in range(3):
                        if diff[dim] > box[dim] * 0.5:
                            diff[dim] -= box[dim]
                        elif diff[dim] < -box[dim] * 0.5:
                            diff[dim] += box[dim]

                    dist = np.sqrt(np.sum(diff * diff))
                    min_dist = min(min_dist, dist)

                    # Early termination if distance below cutoff found
                    if min_dist <= cutoff:
                        break

                if min_dist <= cutoff:
                    break

            # Count as contact if within cutoff
            if min_dist <= cutoff:
                contacts[i] += 1

    return contacts


def calculate_lipid_protein_distances(protein_com, lipid_positions, box, cutoff=15.0):
    """Calculate distances between protein COM and lipids with PBC

    For composition analysis: Uses XY-plane distance only (membrane lateral distance)
    Z-direction is ignored for membrane protein systems

    Parameters
    ----------
    protein_com : numpy.ndarray
        Protein TM domain center of mass (3D, but only XY used)
    lipid_positions : numpy.ndarray
        Lipid positions (N x 3)
    box : numpy.ndarray
        Box dimensions (3D)
    cutoff : float
        Distance cutoff in Angstroms

    Returns
    -------
    tuple
        (distances, within_cutoff_mask)
    """
    if len(lipid_positions) == 0:
        return np.array([]), np.array([], dtype=bool)

    # Calculate displacement vectors
    dr = lipid_positions - protein_com

    # Apply periodic boundary conditions
    dr = dr - box * np.round(dr / box)

    # Calculate XY-plane distances only (ignore Z direction for membrane proteins)
    distances_xy = np.sqrt(dr[:, 0]**2 + dr[:, 1]**2)

    # Mask for lipids within cutoff
    within_cutoff = distances_xy <= cutoff

    return distances_xy, within_cutoff


def calculate_frame_composition(universe, frame_idx, proteins, lipid_selections,
                                leaflet, contact_cutoff=15.0, target_lipid=None,
                                tm_residues=None):
    """Calculate lipid composition around proteins for one frame

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe object
    frame_idx : int
        Frame index to process
    proteins : dict
        Dictionary of protein selections
    lipid_selections : dict
        Dictionary of lipid selections
    leaflet : MDAnalysis.AtomGroup
        Leaflet to analyze
    contact_cutoff : float
        Distance cutoff for contacts (Å)
    target_lipid : str, optional
        Name of target lipid (e.g., 'DPG3' for GM3)
    tm_residues : dict, optional
        TM domain residue ranges {segid: (start, end)}

    Returns
    -------
    dict
        Frame composition data
    """
    try:
        universe.trajectory[frame_idx]
        box = universe.dimensions[:3]

        frame_data = {
            'frame': frame_idx,
            'time': universe.trajectory.time,
            'proteins': {}
        }

        for protein_name, protein in proteins.items():
            if len(protein) == 0:
                continue

            # Get TM domain COM if TM residues are specified
            if tm_residues is not None:
                # Extract segid from protein
                segid = protein.segids[0]
                if segid in tm_residues:
                    start_res, end_res = tm_residues[segid]
                    tm_domain = protein.select_atoms(f"resid {start_res}:{end_res}")
                    if len(tm_domain) > 0:
                        protein_com = tm_domain.center_of_mass()
                    else:
                        protein_com = protein.center_of_mass()
                else:
                    protein_com = protein.center_of_mass()
            else:
                protein_com = protein.center_of_mass()
            protein_data = {
                'com': protein_com,
                'lipid_counts': {},
                'lipid_fractions': {},
                'target_lipid_bound': False
            }

            total_lipids_in_contact = 0

            # Count each lipid type
            for lipid_type, sel_info in lipid_selections.items():
                # LIPAC format: sel_info = {'sel': [lipid_sel]}
                lipid_sel = sel_info['sel'][0]

                if len(lipid_sel) == 0:
                    protein_data['lipid_counts'][lipid_type] = 0
                    continue

                # Get lipid COM positions for composition (15Å cutoff)
                lipid_positions = np.array([res.atoms.center_of_mass() for res in lipid_sel.residues])

                # Calculate distances
                distances, in_contact = calculate_lipid_protein_distances(
                    protein_com, lipid_positions, box, contact_cutoff
                )

                count = np.sum(in_contact)
                protein_data['lipid_counts'][lipid_type] = count
                total_lipids_in_contact += count

                # Check target lipid binding using LIPAC method (residue-level contacts)
                if target_lipid and lipid_type == target_lipid:
                    # Calculate residue-level contacts (same as LIPAC)
                    residue_contacts = calculate_lipid_protein_residue_contacts_lipac(
                        protein, lipid_sel, box, cutoff=6.0
                    )
                    contact_sum = np.sum(residue_contacts)

                    # Bound if any residue has contact (LIPAC logic)
                    protein_data['target_lipid_bound'] = (contact_sum > 0)
                    protein_data['target_bound_count'] = int(contact_sum)

            # Calculate composition ratios (excluding target lipid if specified)
            lipid_types_for_ratio = [lt for lt in lipid_selections.keys()
                                     if target_lipid is None or lt != target_lipid]

            total_for_ratio = sum(protein_data['lipid_counts'].get(lt, 0)
                                 for lt in lipid_types_for_ratio)

            if total_for_ratio > 0:
                for lipid_type in lipid_types_for_ratio:
                    count = protein_data['lipid_counts'].get(lipid_type, 0)
                    protein_data['lipid_fractions'][lipid_type] = count / total_for_ratio
            else:
                for lipid_type in lipid_types_for_ratio:
                    protein_data['lipid_fractions'][lipid_type] = 0.0

            protein_data['total_lipids'] = total_lipids_in_contact
            protein_data['total_for_ratio'] = total_for_ratio

            frame_data['proteins'][protein_name] = protein_data

        return frame_data

    except Exception as e:
        print(f"Error processing frame {frame_idx}: {str(e)}")
        traceback.print_exc()
        return None


def process_frame_wrapper(args):
    """Wrapper for parallel processing

    Parameters
    ----------
    args : tuple
        (frame_idx, top_file, traj_file, protein_names, leaflet_resids,
         lipid_types, contact_cutoff, target_lipid, tm_residues)

    Returns
    -------
    dict
        Frame composition data
    """
    import MDAnalysis as mda
    from .trajectory_loader import select_proteins, select_lipids

    (frame_idx, top_file, traj_file, protein_segids, leaflet_resids,
     lipid_types, contact_cutoff, target_lipid, tm_residues) = args

    try:
        # Load universe in worker process
        u = mda.Universe(top_file, traj_file)
        u.trajectory[frame_idx]

        # Default lipid types if not provided
        if lipid_types is None or len(lipid_types) == 0:
            lipid_types = ['CHOL', 'DPSM', 'DIPC', 'POPC', 'POPE', 'POPS', 'DPG3']

        # Reconstruct leaflet
        lipid_resnames = " ".join(lipid_types)
        lipid_atoms = u.select_atoms(f"resname {lipid_resnames}")

        leaflet = mda.AtomGroup([], u)
        batch_size = 100
        for i in range(0, len(leaflet_resids), batch_size):
            batch = leaflet_resids[i:i+batch_size]
            sel_str = " or ".join([f"resid {r}" for r in batch])
            leaflet = leaflet.union(lipid_atoms.select_atoms(sel_str))

        # Select proteins
        proteins = {}
        for i, segid in enumerate(protein_segids, 1):
            protein_name = f"Protein_{i}"
            proteins[protein_name] = u.select_atoms(f"segid {segid} and protein")

        # Select lipids (verbose=False to avoid spam in parallel workers)
        lipid_selections = select_lipids(u, leaflet, lipid_types, verbose=False)

        # Calculate composition
        return calculate_frame_composition(
            u, frame_idx, proteins, lipid_selections, leaflet,
            contact_cutoff, target_lipid, tm_residues
        )

    except Exception as e:
        print(f"Worker error on frame {frame_idx}: {str(e)}")
        traceback.print_exc()
        return None
