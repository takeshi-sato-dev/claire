#!/usr/bin/env python3
"""
Spatial composition analysis with target lipid dependence
"""

import numpy as np
import pandas as pd
from scipy import stats


class SpatialAnalyzer:
    """Analyze spatial patterns in lipid composition with target lipid dependence"""

    def __init__(self, radii=[5.0, 10.0, 15.0, 20.0], shell_type='both'):
        """Initialize analyzer

        Parameters
        ----------
        radii : list of float, default [5.0, 10.0, 15.0, 20.0]
            Radii for spatial shells (Angstroms)
        shell_type : str, default 'both'
            Type of shells: 'individual', 'cumulative', or 'both'
        """
        self.radii = sorted(radii)
        self.shell_type = shell_type

    def calculate_radial_composition(self, universe, frame_idx, proteins,
                                    lipid_selections, leaflet, lipid_types,
                                    target_lipid=None):
        """Calculate radial composition profiles with optional target lipid tracking

        Parameters
        ----------
        universe : MDAnalysis.Universe
            Universe object
        frame_idx : int
            Frame to analyze
        proteins : dict
            Protein selections
        lipid_selections : dict
            Lipid selections
        leaflet : MDAnalysis.AtomGroup
            Leaflet to analyze
        lipid_types : list of str
            Lipid types
        target_lipid : str, optional
            Target lipid name (e.g., DPG3)

        Returns
        -------
        dict
            Radial composition data with target lipid information
        """
        import sys
        import os
        # Add parent directory to path for imports
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from core.frame_processor import calculate_lipid_protein_distances

        universe.trajectory[frame_idx]
        box = universe.dimensions[:3]

        results = {
            'frame': frame_idx,
            'proteins': {}
        }

        for protein_name, protein in proteins.items():
            if len(protein) == 0:
                continue

            protein_com = protein.center_of_mass()
            protein_results = {'radii': self.radii}

            # Track target lipid position if specified
            target_lipid_info = None
            if target_lipid and target_lipid in lipid_selections:
                # LIPAC format: {'sel': [lipid_sel]}
                target_sel = lipid_selections[target_lipid]['sel'][0]
                if len(target_sel.residues) > 0:
                    # Use residue-level contact calculation (LIPAC method) for binding state
                    from core.frame_processor import calculate_lipid_protein_residue_contacts_lipac

                    residue_contacts = calculate_lipid_protein_residue_contacts_lipac(
                        protein, target_sel, box, cutoff=6.0
                    )

                    # Calculate minimum atom-atom distance and count bound GM3 MOLECULES
                    min_atom_dist = float('inf')
                    bound_gm3_molecules = 0

                    for lipid_res in target_sel.residues:
                        lipid_min_dist = float('inf')

                        for lipid_atom in lipid_res.atoms:
                            for protein_atom in protein.atoms:
                                diff = lipid_atom.position - protein_atom.position
                                diff = diff - box * np.round(diff / box)
                                dist = np.sqrt(np.sum(diff * diff))
                                lipid_min_dist = min(lipid_min_dist, dist)
                                min_atom_dist = min(min_atom_dist, dist)

                        # If this GM3 molecule has any atom within 6Å, count it
                        if lipid_min_dist <= 6.0:
                            bound_gm3_molecules += 1

                    target_lipid_info = {
                        'present': True,
                        'count': len(target_sel.residues),  # Total GM3 in system
                        'bound_count': bound_gm3_molecules,  # Number of GM3 molecules in contact
                        'min_distance': min_atom_dist,
                        'bound': residue_contacts.sum() > 0  # LIPAC criterion (residue-level)
                    }

            if target_lipid_info is None:
                target_lipid_info = {
                    'present': False,
                    'count': 0,
                    'min_distance': np.inf,
                    'bound': False
                }

            protein_results['target_lipid_info'] = target_lipid_info

            # Calculate both individual and cumulative shells based on shell_type
            # For each radius shell
            for i, radius in enumerate(self.radii):
                # Individual shells (donut-shaped)
                if self.shell_type in ['individual', 'both']:
                    inner_radius = self.radii[i-1] if i > 0 else 0.0

                    shell_counts = {}
                    total_in_shell = 0

                    # Count lipids in shell
                    for lipid_type in lipid_types:
                        if lipid_type not in lipid_selections or len(lipid_selections[lipid_type]['sel'][0]) == 0:
                            shell_counts[lipid_type] = 0
                            continue

                        # LIPAC format: {'sel': [lipid_sel]}
                        lipid_sel = lipid_selections[lipid_type]['sel'][0]
                        lipid_positions = np.array([res.atoms.center_of_mass()
                                                   for res in lipid_sel.residues])

                        distances, _ = calculate_lipid_protein_distances(
                            protein_com, lipid_positions, box, cutoff=radius
                        )

                        # Count in shell (between inner_radius and radius)
                        in_shell = np.sum((distances > inner_radius) & (distances <= radius))
                        shell_counts[lipid_type] = in_shell
                        total_in_shell += in_shell

                    # Calculate composition ratios for this shell
                    shell_ratios = {}
                    if total_in_shell > 0:
                        for lipid_type in lipid_types:
                            shell_ratios[lipid_type] = shell_counts[lipid_type] / total_in_shell
                    else:
                        for lipid_type in lipid_types:
                            shell_ratios[lipid_type] = 0.0

                    shell_key = f"shell_{inner_radius:.1f}_{radius:.1f}"
                    protein_results[shell_key] = {
                        'counts': shell_counts,
                        'ratios': shell_ratios,
                        'total': total_in_shell
                    }

                # Cumulative shells (disk-shaped, 0 to radius)
                if self.shell_type in ['cumulative', 'both']:
                    cumul_counts = {}
                    total_cumul = 0

                    # Count lipids in cumulative shell (0 to radius)
                    for lipid_type in lipid_types:
                        if lipid_type not in lipid_selections or len(lipid_selections[lipid_type]['sel'][0]) == 0:
                            cumul_counts[lipid_type] = 0
                            continue

                        # LIPAC format: {'sel': [lipid_sel]}
                        lipid_sel = lipid_selections[lipid_type]['sel'][0]
                        lipid_positions = np.array([res.atoms.center_of_mass()
                                                   for res in lipid_sel.residues])

                        distances, _ = calculate_lipid_protein_distances(
                            protein_com, lipid_positions, box, cutoff=radius
                        )

                        # Count in cumulative shell (0 to radius)
                        in_cumul = np.sum(distances <= radius)
                        cumul_counts[lipid_type] = in_cumul
                        total_cumul += in_cumul

                    # Calculate composition ratios for cumulative shell
                    cumul_ratios = {}
                    if total_cumul > 0:
                        for lipid_type in lipid_types:
                            cumul_ratios[lipid_type] = cumul_counts[lipid_type] / total_cumul
                    else:
                        for lipid_type in lipid_types:
                            cumul_ratios[lipid_type] = 0.0

                    cumul_key = f"cumulative_0.0_{radius:.1f}"
                    protein_results[cumul_key] = {
                        'counts': cumul_counts,
                        'ratios': cumul_ratios,
                        'total': total_cumul
                    }

            results['proteins'][protein_name] = protein_results

        return results

    def calculate_target_dependent_profiles(self, radial_data_list, lipid_types, target_lipid):
        """Calculate radial profiles separately for target lipid bound/unbound states

        Parameters
        ----------
        radial_data_list : list of dict
            List of radial composition data from multiple frames
        lipid_types : list of str
            Lipid types
        target_lipid : str
            Target lipid name

        Returns
        -------
        dict
            Profiles for bound, unbound, and difference
        """
        print("\n" + "="*70)
        print(f"TARGET LIPID ({target_lipid}) DEPENDENT RADIAL PROFILES")
        print("="*70)

        # Separate frames by target lipid binding state
        bound_data = []
        unbound_data = []

        for radial_data in radial_data_list:
            for protein_name, protein_data in radial_data['proteins'].items():
                target_info = protein_data.get('target_lipid_info', {})
                if target_info.get('bound', False):
                    bound_data.append(radial_data)
                else:
                    unbound_data.append(radial_data)
                break  # Only check first protein

        print(f"Bound frames: {len(bound_data)}")
        print(f"Unbound frames: {len(unbound_data)}")

        if len(bound_data) == 0 or len(unbound_data) == 0:
            print("WARNING: Insufficient data for bound/unbound comparison")
            return None

        # Calculate profiles for each state
        bound_profiles = self._calculate_profiles_internal(bound_data, lipid_types, "BOUND")
        unbound_profiles = self._calculate_profiles_internal(unbound_data, lipid_types, "UNBOUND")

        # Calculate differences
        print("\n" + "="*70)
        print("COMPOSITION DIFFERENCES (BOUND - UNBOUND)")
        print("="*70)

        diff_profiles = {}
        for protein_name in bound_profiles:
            if protein_name not in unbound_profiles:
                continue

            diff_profiles[protein_name] = {
                'radii': bound_profiles[protein_name]['radii'],
                'shells': {}
            }

            print(f"\n{protein_name}:")
            print(f"{'Shell (Å)':<15} " + " ".join([f"{lt:>8s}" for lt in lipid_types]) + "  Sig")
            print("-" * (15 + 10 * len(lipid_types) + 5))

            for shell_key in sorted(bound_profiles[protein_name]['shells'].keys()):
                if shell_key not in unbound_profiles[protein_name]['shells']:
                    continue

                # Parse shell range (handle both shell_ and cumulative_ prefixes)
                if shell_key.startswith('cumulative_'):
                    parts = shell_key.replace('cumulative_', '').split('_')
                    inner = float(parts[0])
                    outer = float(parts[1])
                    shell_str = f"0.0-{outer:.1f} (cum)"
                else:
                    parts = shell_key.replace('shell_', '').split('_')
                    inner = float(parts[0])
                    outer = float(parts[1])
                    shell_str = f"{inner:.1f}-{outer:.1f}"

                diff_profiles[protein_name]['shells'][shell_key] = {}

                # Calculate differences and test significance
                diffs = []
                sig_markers = []

                for lipid_type in lipid_types:
                    bound_vals = bound_profiles[protein_name]['shells'][shell_key][lipid_type]
                    unbound_vals = unbound_profiles[protein_name]['shells'][shell_key][lipid_type]

                    bound_mean = np.mean(bound_vals)
                    unbound_mean = np.mean(unbound_vals)
                    diff = bound_mean - unbound_mean

                    # Statistical test
                    if len(bound_vals) >= 2 and len(unbound_vals) >= 2:
                        t_stat, p_value = stats.ttest_ind(bound_vals, unbound_vals, equal_var=False)
                        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                    else:
                        p_value = 1.0
                        sig = ''

                    diff_profiles[protein_name]['shells'][shell_key][lipid_type] = {
                        'bound_mean': bound_mean,
                        'unbound_mean': unbound_mean,
                        'difference': diff,
                        'p_value': p_value,
                        'bound_values': bound_vals,
                        'unbound_values': unbound_vals
                    }

                    diffs.append(diff)
                    sig_markers.append(sig)

                # Print with significance markers
                diff_str = " ".join([f"{d:>8.4f}" for d in diffs])
                sig_str = " ".join([f"{s:>2s}" for s in sig_markers])
                print(f"{shell_str:<15} {diff_str}  {sig_str}")

        print("="*70)

        return {
            'bound': bound_profiles,
            'unbound': unbound_profiles,
            'difference': diff_profiles,
            'n_bound': len(bound_data),
            'n_unbound': len(unbound_data)
        }

    def _calculate_profiles_internal(self, radial_data_list, lipid_types, label=""):
        """Internal method to calculate average profiles"""

        protein_profiles = {}

        for radial_data in radial_data_list:
            for protein_name, protein_data in radial_data['proteins'].items():
                if protein_name not in protein_profiles:
                    protein_profiles[protein_name] = {
                        'radii': protein_data['radii'],
                        'shells': {}
                    }

                # For each shell (both individual and cumulative)
                for shell_key in protein_data.keys():
                    if not (shell_key.startswith('shell_') or shell_key.startswith('cumulative_')):
                        continue

                    if shell_key not in protein_profiles[protein_name]['shells']:
                        protein_profiles[protein_name]['shells'][shell_key] = {
                            lt: [] for lt in lipid_types
                        }

                    # Collect ratios
                    for lipid_type in lipid_types:
                        ratio = protein_data[shell_key]['ratios'].get(lipid_type, 0.0)
                        protein_profiles[protein_name]['shells'][shell_key][lipid_type].append(ratio)

        # Calculate averages
        if label:
            print(f"\n{label} STATE:")
        for protein_name in protein_profiles:
            if label:
                print(f"  {protein_name}:")
                print(f"  {'Shell (Å)':<15} " + " ".join([f"{lt:>8s}" for lt in lipid_types]))
                print("  " + "-" * (15 + 10 * len(lipid_types)))

            for shell_key in sorted(protein_profiles[protein_name]['shells'].keys()):
                # Parse shell range (handle both shell_ and cumulative_ prefixes)
                if shell_key.startswith('cumulative_'):
                    parts = shell_key.replace('cumulative_', '').split('_')
                    inner = float(parts[0])
                    outer = float(parts[1])
                    shell_str = f"0.0-{outer:.1f} (cum)"
                else:
                    parts = shell_key.replace('shell_', '').split('_')
                    inner = float(parts[0])
                    outer = float(parts[1])
                    shell_str = f"{inner:.1f}-{outer:.1f}"

                # Calculate means
                means = []
                for lipid_type in lipid_types:
                    values = protein_profiles[protein_name]['shells'][shell_key][lipid_type]
                    mean = np.mean(values) if len(values) > 0 else 0.0
                    means.append(mean)

                    # Store for later use
                    protein_profiles[protein_name]['shells'][shell_key][f'{lipid_type}_mean'] = mean
                    protein_profiles[protein_name]['shells'][shell_key][f'{lipid_type}_std'] = np.std(values) if len(values) > 0 else 0.0

                if label:
                    print(f"  {shell_str:<15} " + " ".join([f"{m:>8.4f}" for m in means]))

        return protein_profiles

    def analyze_target_position_effect(self, radial_data_list, lipid_types, target_lipid):
        """Analyze how target lipid position affects surrounding lipid composition

        Parameters
        ----------
        radial_data_list : list of dict
            Radial composition data
        lipid_types : list of str
            Lipid types
        target_lipid : str
            Target lipid name

        Returns
        -------
        pandas.DataFrame
            Target position vs composition data
        """
        print("\n" + "="*70)
        print(f"TARGET LIPID POSITION vs COMPOSITION ANALYSIS")
        print("="*70)

        results = []

        for radial_data in radial_data_list:
            frame_idx = radial_data['frame']

            for protein_name, protein_data in radial_data['proteins'].items():
                target_info = protein_data.get('target_lipid_info', {})

                if not target_info.get('present', False):
                    continue

                # Collect composition for ALL shells
                row = {
                    'frame': frame_idx,
                    'protein': protein_name,
                    'target_distance': target_info['min_distance'],
                    'target_bound_count': target_info['bound_count'],  # Number of GM3 in contact
                    'target_total_count': target_info['count'],  # Total GM3 in system (for reference)
                    'target_bound': target_info['bound']
                }

                # Add composition for each shell
                for i, radius in enumerate(self.radii):
                    inner_radius = self.radii[i-1] if i > 0 else 0.0
                    shell_key = f"shell_{inner_radius:.1f}_{radius:.1f}"

                    if shell_key in protein_data:
                        shell_ratios = protein_data[shell_key]['ratios']
                        for lipid_type in lipid_types:
                            # Use shell-specific column names
                            col_name = f'{lipid_type}_ratio_shell_{inner_radius:.0f}_{radius:.0f}'
                            row[col_name] = shell_ratios.get(lipid_type, 0.0)

                results.append(row)

        df = pd.DataFrame(results)

        if len(df) > 0:
            print(f"\nCollected {len(df)} frames with target lipid present")

            # Correlation analysis for ALL shells
            print("\nCorrelation: Target distance vs lipid composition (all shells)")

            for i, radius in enumerate(self.radii):
                inner_radius = self.radii[i-1] if i > 0 else 0.0
                shell_str = f"{inner_radius:.0f}-{radius:.0f}Å"

                print(f"\n  Shell {shell_str}:")
                print(f"  {'Lipid':<10} {'Correlation':>12} {'P-value':>10} {'Trend':>10}")
                print("  " + "-" * 45)

                for lipid_type in lipid_types:
                    col = f'{lipid_type}_ratio_shell_{inner_radius:.0f}_{radius:.0f}'
                    if col in df.columns:
                        # Remove NaN values
                        valid_data = df[['target_distance', col]].dropna()
                        if len(valid_data) > 2 and valid_data[col].std() > 0:
                            corr, p_val = stats.pearsonr(valid_data['target_distance'], valid_data[col])
                            trend = 'enriched' if corr < 0 else 'depleted' if corr > 0 else 'no trend'
                            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                            print(f"  {lipid_type:<10} {corr:>12.4f} {p_val:>10.4f} {trend:>10s} {sig}")
                        else:
                            print(f"  {lipid_type:<10} {'N/A':>12s} {'N/A':>10s} {'no var':>10s}")
        else:
            print("WARNING: No frames with target lipid present")

        print("="*70)

        return df

    def calculate_average_radial_profiles(self, radial_data_list, lipid_types):
        """Calculate average radial profiles from multiple frames (legacy method)

        Parameters
        ----------
        radial_data_list : list of dict
            List of radial composition data from multiple frames
        lipid_types : list of str
            Lipid types

        Returns
        -------
        dict
            Averaged profiles per protein
        """
        print("\n" + "="*70)
        print("OVERALL RADIAL COMPOSITION PROFILES")
        print("="*70)

        return self._calculate_profiles_internal(radial_data_list, lipid_types, label="")
