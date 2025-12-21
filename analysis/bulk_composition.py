#!/usr/bin/env python3
"""
Bulk membrane composition analysis
Compare protein vicinity vs bulk membrane composition
"""

import numpy as np
import pandas as pd
from scipy import stats


def calculate_bulk_composition(universe, frame_idx, lipid_selections, leaflet, lipid_types):
    """Calculate bulk membrane composition (entire leaflet)

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe object
    frame_idx : int
        Frame index
    lipid_selections : dict
        Lipid selections in LIPAC format
    leaflet : MDAnalysis.AtomGroup
        Leaflet to analyze
    lipid_types : list of str
        Lipid types to analyze

    Returns
    -------
    dict
        Bulk composition fractions
    """
    universe.trajectory[frame_idx]

    bulk_counts = {}

    for lipid_type in lipid_types:
        # LIPAC format: {'sel': [lipid_sel]}
        lipid_sel = lipid_selections[lipid_type]['sel'][0]

        # Count molecules in leaflet
        bulk_counts[lipid_type] = len(lipid_sel.residues)

    # Calculate fractions
    total = sum(bulk_counts.values())
    bulk_fractions = {lt: count/total for lt, count in bulk_counts.items()}

    return bulk_fractions


def calculate_bulk_vs_protein_composition(df, universe, lipid_selections, leaflet,
                                         lipid_types, proteins, n_frames=100):
    """Calculate bulk composition and compare with protein vicinity

    Parameters
    ----------
    df : pandas.DataFrame
        Protein vicinity composition data
    universe : MDAnalysis.Universe
        Universe object
    lipid_selections : dict
        Lipid selections
    leaflet : MDAnalysis.AtomGroup
        Leaflet
    lipid_types : list of str
        Lipid types
    proteins : dict
        Protein selections
    n_frames : int
        Number of frames to sample for bulk calculation

    Returns
    -------
    dict
        Results with bulk composition, protein composition, and statistics
    """
    print("\n" + "="*70)
    print("BULK vs PROTEIN VICINITY COMPOSITION ANALYSIS")
    print("="*70)

    # Get frame indices
    all_frames = sorted(df['frame'].unique())

    if n_frames is None or n_frames >= len(all_frames):
        sample_frames = all_frames
    else:
        # Sample evenly
        indices = np.linspace(0, len(all_frames)-1, n_frames, dtype=int)
        sample_frames = [all_frames[i] for i in indices]

    print(f"\nCalculating bulk composition for {len(sample_frames)} frames...")

    # Calculate bulk composition for each frame
    bulk_data = []

    for i, frame_idx in enumerate(sample_frames):
        if i % 20 == 0:
            print(f"  Frame {i+1}/{len(sample_frames)}")

        bulk_comp = calculate_bulk_composition(
            universe, frame_idx, lipid_selections, leaflet, lipid_types
        )
        bulk_comp['frame'] = frame_idx
        bulk_data.append(bulk_comp)

    bulk_df = pd.DataFrame(bulk_data)

    # Calculate mean bulk composition
    bulk_mean = {lt: bulk_df[lt].mean() for lt in lipid_types}
    bulk_std = {lt: bulk_df[lt].std() for lt in lipid_types}

    print("\nBulk membrane composition (average across leaflet):")
    for lt in lipid_types:
        print(f"  {lt:6s}: {bulk_mean[lt]:.3f} ± {bulk_std[lt]:.3f}")

    # Calculate protein vicinity composition (average across all proteins and frames)
    protein_mean = {}
    protein_std = {}

    for lt in lipid_types:
        frac_col = f'{lt}_fraction'
        if frac_col in df.columns:
            protein_mean[lt] = df[frac_col].mean()
            protein_std[lt] = df[frac_col].std()
        else:
            protein_mean[lt] = 0.0
            protein_std[lt] = 0.0

    print("\nProtein vicinity composition (15Å radius, all proteins):")
    for lt in lipid_types:
        print(f"  {lt:6s}: {protein_mean[lt]:.3f} ± {protein_std[lt]:.3f}")

    # Calculate enrichment/depletion
    print("\nEnrichment in protein vicinity (vs bulk):")
    print("-" * 70)

    enrichment = {}
    t_stats = {}
    p_values = {}

    for lt in lipid_types:
        # Enrichment factor
        if bulk_mean[lt] > 0:
            enrich = protein_mean[lt] / bulk_mean[lt]
            delta = protein_mean[lt] - bulk_mean[lt]
            pct_change = (delta / bulk_mean[lt]) * 100
        else:
            enrich = np.nan
            delta = np.nan
            pct_change = np.nan

        enrichment[lt] = enrich

        # Statistical test: protein vicinity vs bulk
        frac_col = f'{lt}_fraction'
        if frac_col in df.columns:
            protein_values = df[frac_col].dropna()
            bulk_values = bulk_df[lt].dropna()

            if len(protein_values) > 0 and len(bulk_values) > 0:
                t_stat, p_val = stats.ttest_ind(protein_values, bulk_values)
                t_stats[lt] = t_stat
                p_values[lt] = p_val

                # Significance stars
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    sig = 'ns'
            else:
                t_stats[lt] = np.nan
                p_values[lt] = np.nan
                sig = 'N/A'
        else:
            t_stats[lt] = np.nan
            p_values[lt] = np.nan
            sig = 'N/A'

        print(f"{lt:6s}: {bulk_mean[lt]:.3f} → {protein_mean[lt]:.3f} "
              f"(Δ{delta:+.3f}, {pct_change:+.1f}%, {enrich:.2f}x) {sig}")

    print("-" * 70)

    # Check conservation
    total_bulk = sum(bulk_mean.values())
    total_protein = sum(protein_mean.values())
    print(f"\nConservation check:")
    print(f"  Bulk total: {total_bulk:.3f} (should be 1.0)")
    print(f"  Protein vicinity total: {total_protein:.3f} (should be 1.0)")

    print("="*70)

    return {
        'bulk_mean': bulk_mean,
        'bulk_std': bulk_std,
        'protein_mean': protein_mean,
        'protein_std': protein_std,
        'enrichment': enrichment,
        't_stats': t_stats,
        'p_values': p_values,
        'bulk_df': bulk_df,
        'lipid_types': lipid_types
    }


def calculate_bulk_vs_protein_gm3_dependent(df, universe, lipid_selections, leaflet,
                                            lipid_types, proteins, target_lipid,
                                            n_frames=100):
    """Calculate bulk vs protein composition, split by GM3 binding state

    Parameters
    ----------
    df : pandas.DataFrame
        Composition data with target_lipid_bound column
    universe : MDAnalysis.Universe
        Universe
    lipid_selections : dict
        Lipid selections
    leaflet : MDAnalysis.AtomGroup
        Leaflet
    lipid_types : list of str
        Lipid types
    proteins : dict
        Proteins
    target_lipid : str
        Target lipid name
    n_frames : int
        Frames to sample

    Returns
    -------
    dict
        Results for bound and unbound states
    """
    print("\n" + "="*70)
    print("BULK vs PROTEIN VICINITY - GM3 DEPENDENT")
    print("="*70)

    # Split by GM3 binding
    bound_df = df[df['target_lipid_bound'] == True]
    unbound_df = df[df['target_lipid_bound'] == False]

    print(f"\nGM3 bound frames: {len(bound_df)}")
    print(f"GM3 unbound frames: {len(unbound_df)}")

    results = {}

    # Analyze bound state
    if len(bound_df) > 0:
        print("\n--- GM3 BOUND STATE ---")

        bound_protein_mean = {}
        for lt in lipid_types:
            frac_col = f'{lt}_fraction'
            if frac_col in bound_df.columns:
                bound_protein_mean[lt] = bound_df[frac_col].mean()
            else:
                bound_protein_mean[lt] = 0.0

        print("Protein vicinity composition (GM3 bound):")
        for lt in lipid_types:
            print(f"  {lt:6s}: {bound_protein_mean[lt]:.3f}")

        results['bound_protein_mean'] = bound_protein_mean

    # Analyze unbound state
    if len(unbound_df) > 0:
        print("\n--- GM3 UNBOUND STATE ---")

        unbound_protein_mean = {}
        for lt in lipid_types:
            frac_col = f'{lt}_fraction'
            if frac_col in unbound_df.columns:
                unbound_protein_mean[lt] = unbound_df[frac_col].mean()
            else:
                unbound_protein_mean[lt] = 0.0

        print("Protein vicinity composition (GM3 unbound):")
        for lt in lipid_types:
            print(f"  {lt:6s}: {unbound_protein_mean[lt]:.3f}")

        results['unbound_protein_mean'] = unbound_protein_mean

    print("="*70)

    return results
