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


def _count_first_shell_molecules(protein, lipid_sel, box, cutoff=6.0):
    """Count lipid molecules with any bead within cutoff of any protein bead.

    Parameters
    ----------
    protein : MDAnalysis.AtomGroup
        Protein (TM domain) atoms
    lipid_sel : MDAnalysis.AtomGroup
        Lipid selection (all molecules of one type in leaflet)
    box : numpy.ndarray
        Box dimensions [x, y, z]
    cutoff : float
        First shell cutoff in Angstroms

    Returns
    -------
    int
        Number of lipid molecules in first shell contact
    """
    if len(lipid_sel) == 0 or len(protein) == 0:
        return 0

    prot_pos = protein.positions
    prot_com = prot_pos.mean(axis=0)
    count = 0

    for res in lipid_sel.residues:
        lip_pos = res.atoms.positions
        lip_com = lip_pos.mean(axis=0)

        # XY prescreen on COM (15A - within composition cylinder)
        dx = lip_com[0] - prot_com[0]
        dy = lip_com[1] - prot_com[1]
        dx -= box[0] * round(dx / box[0])
        dy -= box[1] * round(dy / box[1])
        if abs(dx) > 15.0 or abs(dy) > 15.0:
            continue

        # Full bead-to-bead distance check
        diff = lip_pos[np.newaxis, :, :] - prot_pos[:, np.newaxis, :]
        diff -= box * np.round(diff / box)
        dists = np.sqrt(np.sum(diff * diff, axis=2))

        if dists.min() < cutoff:
            count += 1

    return count


def _compute_lipid_scd(residue, chain_a_pattern, chain_b_pattern):
    """Compute order parameter S_CD for a single lipid residue.

    Parameters
    ----------
    residue : MDAnalysis.Residue
        Lipid residue
    chain_a_pattern : str
        MDAnalysis selection for sn-1 tail beads
    chain_b_pattern : str
        MDAnalysis selection for sn-2 tail beads

    Returns
    -------
    float
        S_CD value, or NaN if calculation fails
    """
    chain_a = residue.atoms.select_atoms(chain_a_pattern)
    chain_b = residue.atoms.select_atoms(chain_b_pattern)

    if len(chain_a) < 2 and len(chain_b) < 2:
        return np.nan

    cos2_values = []

    for chain_atoms in [chain_a, chain_b]:
        if len(chain_atoms) < 2:
            continue
        vectors = np.diff(chain_atoms.positions, axis=0)
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0] = np.finfo(float).eps
        cos_theta = vectors[:, 2] / norms  # z-component
        cos2_values.extend(cos_theta ** 2)

    if len(cos2_values) == 0:
        return np.nan

    scd = (3.0 * np.mean(cos2_values) - 1.0) / 2.0
    return scd if np.isfinite(scd) else np.nan


def _compute_local_mean_scd(lipid_selections, protein_com, box, cutoff,
                            chain_a_pattern, chain_b_pattern,
                            exclude_types=None,
                            protein_for_fs=None, fs_cutoff=6.0):
    """Compute mean S_CD of lipids within the composition cylinder.

    Optionally excludes lipids in the first shell (within fs_cutoff of
    the protein) to avoid measurement overlap between first_shell counts
    and local S_CD.

    Parameters
    ----------
    lipid_selections : dict
        Lipid selections from LIPAC format
    protein_com : numpy.ndarray
        Protein center of mass
    box : numpy.ndarray
        Box dimensions [x, y, z]
    cutoff : float
        Cylinder radius in Angstroms (15A)
    chain_a_pattern : str
        MDAnalysis selection for sn-1 tail
    chain_b_pattern : str
        MDAnalysis selection for sn-2 tail
    exclude_types : set or None
        Lipid types to exclude (e.g., CHOL, target lipid)
    protein_for_fs : MDAnalysis.AtomGroup or None
        If provided, exclude lipids within fs_cutoff of this atom group
        from the S_CD calculation (first shell exclusion)
    fs_cutoff : float
        First shell cutoff in Angstroms (default 6.0)

    Returns
    -------
    float
        Mean S_CD of lipids in cylinder (excluding first shell if requested),
        NaN if none
    """
    if exclude_types is None:
        exclude_types = {'CHOL', 'DPG3'}

    # Precompute protein positions for first shell check
    prot_pos = None
    if protein_for_fs is not None and len(protein_for_fs) > 0:
        prot_pos = protein_for_fs.positions

    scd_values = []

    for lipid_type, sel_info in lipid_selections.items():
        if lipid_type in exclude_types:
            continue

        lipid_sel = sel_info['sel'][0]
        if len(lipid_sel) == 0:
            continue

        for res in lipid_sel.residues:
            com = res.atoms.center_of_mass()
            # PBC-corrected XY distance to protein COM
            dx = com[0] - protein_com[0]
            dy = com[1] - protein_com[1]
            dx -= box[0] * round(dx / box[0])
            dy -= box[1] * round(dy / box[1])
            dist_xy = np.sqrt(dx*dx + dy*dy)

            if dist_xy > cutoff:
                continue

            # First shell exclusion: skip if any bead is within fs_cutoff
            if prot_pos is not None:
                lip_pos = res.atoms.positions
                diff = lip_pos[np.newaxis, :, :] - prot_pos[:, np.newaxis, :]
                diff -= box * np.round(diff / box)
                min_dist = np.sqrt(np.sum(diff * diff, axis=2)).min()
                if min_dist < fs_cutoff:
                    continue

            scd = _compute_lipid_scd(res, chain_a_pattern, chain_b_pattern)
            if np.isfinite(scd):
                scd_values.append(scd)

    return np.mean(scd_values) if len(scd_values) > 0 else np.nan


def _compute_tm_tilt(tm_domain):
    """Compute TM helix tilt angle relative to membrane normal (z-axis).

    Fits a principal axis through TM backbone bead positions and computes
    the angle with the z-axis. Returns angle in degrees (0 = parallel to z,
    90 = perpendicular).

    Parameters
    ----------
    tm_domain : MDAnalysis.AtomGroup
        TM domain atoms (BB beads)

    Returns
    -------
    float
        Tilt angle in degrees
    """
    positions = tm_domain.positions
    if len(positions) < 3:
        return np.nan

    # Principal axis via SVD
    centroid = positions.mean(axis=0)
    centered = positions - centroid
    _, _, vh = np.linalg.svd(centered)
    principal_axis = vh[0]  # first principal component

    # Ensure consistent direction (pointing toward +z)
    if principal_axis[2] < 0:
        principal_axis = -principal_axis

    # Angle with z-axis
    cos_angle = np.abs(principal_axis[2]) / np.linalg.norm(principal_axis)
    tilt_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    return tilt_deg


def _min_distance_to_target(protein, lipid_sel, box):
    """Minimum bead-to-bead distance between protein and nearest target lipid molecule.

    Computes the global minimum distance across all protein beads and all
    target lipid beads, with PBC correction. Uses COM pre-screening to
    skip distant molecules (>20 Angstrom XY-plane distance).

    Parameters
    ----------
    protein : MDAnalysis.AtomGroup
        Protein selection (TM domain)
    lipid_sel : MDAnalysis.AtomGroup
        Target lipid selection (all molecules of that type in leaflet)
    box : numpy.ndarray
        Box dimensions [x, y, z]

    Returns
    -------
    float
        Minimum distance in Angstroms (np.inf if no lipids nearby)
    """
    if len(lipid_sel) == 0:
        return np.inf

    protein_com = protein.center_of_mass()
    protein_positions = protein.positions
    global_min = np.inf

    for lipid_res in lipid_sel.residues:
        # COM pre-screening: skip molecules far in XY plane
        lipid_com = lipid_res.atoms.center_of_mass()
        xy_diff = lipid_com[:2] - protein_com[:2]
        xy_diff = xy_diff - box[:2] * np.round(xy_diff / box[:2])
        xy_dist = np.sqrt(np.sum(xy_diff * xy_diff))

        if xy_dist > 20.0:
            continue

        # Vectorized pairwise distance with PBC
        lipid_positions = lipid_res.atoms.positions
        diff = lipid_positions[:, np.newaxis, :] - protein_positions[np.newaxis, :, :]
        diff = diff - box * np.round(diff / box)
        dists = np.sqrt(np.sum(diff * diff, axis=2))

        mol_min = dists.min()
        if mol_min < global_min:
            global_min = mol_min

        if global_min < 3.0:
            return global_min

    return global_min


def calculate_frame_composition(universe, frame_idx, proteins, lipid_selections,
                                leaflet, contact_cutoff=15.0, target_lipid=None,
                                tm_residues=None,
                                chain_a_pattern=None, chain_b_pattern=None):
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

            # Get TM domain if TM residues are specified
            tm_domain = protein  # fallback: full protein
            tm_tilt = np.nan
            if tm_residues is not None:
                segid = protein.segids[0]
                if segid in tm_residues:
                    start_res, end_res = tm_residues[segid]
                    tm_sel = protein.select_atoms(f"resid {start_res}:{end_res}")
                    if len(tm_sel) > 0:
                        tm_domain = tm_sel
                        tm_tilt = _compute_tm_tilt(tm_domain)

            protein_com = tm_domain.center_of_mass()

            protein_data = {
                'com': protein_com,
                'lipid_counts': {},
                'lipid_fractions': {},
                'target_lipid_bound': False,
                'tm_tilt': tm_tilt,
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

                    # v5: Minimum bead-to-bead distance to nearest target lipid molecule
                    # Uses TM domain only for accurate pre-screening and distance
                    min_dist_all = _min_distance_to_target(tm_domain, lipid_sel, box)
                    protein_data['target_lipid_min_distance'] = min_dist_all

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

            # v21: First shell (6A) contact count for each composition lipid
            # Uses TM domain bead-to-bead distances (same as LIPAC)
            protein_data['first_shell_counts'] = {}
            for lipid_type, sel_info in lipid_selections.items():
                if target_lipid and lipid_type == target_lipid:
                    continue  # skip target lipid
                lipid_sel = sel_info['sel'][0]
                fs_count = _count_first_shell_molecules(tm_domain, lipid_sel, box, cutoff=6.0)
                protein_data['first_shell_counts'][lipid_type] = fs_count

            # v21: Local mean S_CD of lipids in the composition cylinder
            if chain_a_pattern is not None and chain_b_pattern is not None:
                exclude = {'CHOL'}
                if target_lipid:
                    exclude.add(target_lipid)
                # All lipids in 15A cylinder
                local_scd = _compute_local_mean_scd(
                    lipid_selections, protein_com, box, contact_cutoff,
                    chain_a_pattern, chain_b_pattern,
                    exclude_types=exclude
                )
                # Excluding first shell lipids (6A) to avoid measurement overlap
                local_scd_excl_fs = _compute_local_mean_scd(
                    lipid_selections, protein_com, box, contact_cutoff,
                    chain_a_pattern, chain_b_pattern,
                    exclude_types=exclude,
                    protein_for_fs=tm_domain, fs_cutoff=6.0
                )
                protein_data['local_scd'] = local_scd
                protein_data['local_scd_excl_fs'] = local_scd_excl_fs
            else:
                protein_data['local_scd'] = np.nan
                protein_data['local_scd_excl_fs'] = np.nan

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
         lipid_types, contact_cutoff, target_lipid, tm_residues,
         chain_a_pattern, chain_b_pattern)

    Returns
    -------
    dict
        Frame composition data
    """
    import MDAnalysis as mda
    from .trajectory_loader import select_proteins, select_lipids

    # Support both old (9-tuple) and new (11-tuple) format
    if len(args) == 11:
        (frame_idx, top_file, traj_file, protein_segids, leaflet_resids,
         lipid_types, contact_cutoff, target_lipid, tm_residues,
         chain_a_pattern, chain_b_pattern) = args
    else:
        (frame_idx, top_file, traj_file, protein_segids, leaflet_resids,
         lipid_types, contact_cutoff, target_lipid, tm_residues) = args
        chain_a_pattern = None
        chain_b_pattern = None

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
            contact_cutoff, target_lipid, tm_residues,
            chain_a_pattern, chain_b_pattern
        )

    except Exception as e:
        print(f"Worker error on frame {frame_idx}: {str(e)}")
        traceback.print_exc()
        return None
