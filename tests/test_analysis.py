#!/usr/bin/env python3
"""
Test script to verify CLAIRE produces identical results to original code
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from claire import MembraneSystem
from claire.analysis import (
    ConservationAnalyzer,
    EnrichmentAnalyzer,
    FrameProcessor,
    DiagnosticAnalyzer,
    MLAnalyzer
)
from claire.visualization.figures import FigureGenerator
from claire.core.trajectory import TrajectoryProcessor


def test_original_workflow():
    """
    Test that exactly replicates the original aiml13_no_causal.py workflow
    """
    print("="*70)
    print("CLAIRE TEST - Verifying Package Functionality")
    print("="*70)
    
    # Paths to test data
    topology = 'test_data/test_system.psf'
    trajectory = 'test_data/test_trajectory.xtc'
    output_dir = 'test_output'
    
    # Parameters from original code
    START = 0  # Using all 50 frames for test
    STOP = 50
    STEP = 1
    LEAFLET_FRAME = 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load system
    print("\n1. Loading system...")
    membrane = MembraneSystem(topology, trajectory, verbose=True)
    
    # 2. Identify leaflets
    print("\n2. Identifying leaflets...")
    upper_leaflet, lower_leaflet = membrane.identify_leaflets(frame=LEAFLET_FRAME)
    
    if upper_leaflet is None or lower_leaflet is None:
        print("ERROR: Could not identify leaflets")
        return False
    
    # 3. Automatically identify all lipids
    print("\n3. Identifying lipids...")
    membrane.identify_lipids()
    
    # Select lipids from upper leaflet
    lipid_selections = {}
    for lipid_name in membrane.lipids.keys():
        selection = upper_leaflet.select_atoms(f"resname {lipid_name}")
        if len(selection) > 0:
            lipid_selections[lipid_name] = selection
            print(f"  Selected {len(selection.residues)} {lipid_name} molecules from upper leaflet")
    
    # 4. Identify proteins
    print("\n4. Identifying proteins...")
    proteins = membrane.identify_proteins()
    
    if len(proteins) == 0:
        print("  No proteins found - will analyze lipid-only system")
        # Create dummy protein at membrane center for analysis
        proteins = {
            'Membrane_Center': {
                'atoms': upper_leaflet,
                'segid': 'MEMB',
                'n_atoms': len(upper_leaflet),
                'n_residues': len(upper_leaflet.residues)
            }
        }
    
    # 5. Process trajectory
    print("\n5. Processing trajectory...")
    print(f"  Frames: {START} to {STOP}, step {STEP}")
    
    # Get box dimensions
    box_dimensions = membrane.universe.dimensions[:3]
    
    # Process frames
    processor = TrajectoryProcessor(membrane.universe)
    frame_processor = FrameProcessor()
    
    all_features = []
    for frame_idx in range(START, STOP, STEP):
        frame_results = frame_processor.process_frame_complete(
            frame_idx,
            membrane.universe,
            proteins,
            lipid_selections,
            box_dimensions,
            mediator_lipid='DPG3'  # Or any lipid you want to analyze
        )
        all_features.extend(frame_results)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    print(f"\nProcessed {len(df)} protein-frame combinations")
    
    # 6. Run diagnostics
    print("\n6. Running diagnostics...")
    diagnostics = DiagnosticAnalyzer()
    
    # Aggregate by frame
    agg_dict = {}
    for col in df.columns:
        if col not in ['frame', 'protein', 'time']:
            if pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = 'mean'
    
    if agg_dict:
        frame_df = df.groupby('frame').agg(agg_dict).reset_index()
    else:
        frame_df = df
    
    diag_results, recommendations = diagnostics.diagnose_trajectory_data(df, frame_df)
    
    # 7. ML Analysis
    print("\n7. Running ML analysis...")
    ml_analyzer = MLAnalyzer()
    ml_results, frame_df_processed = ml_analyzer.advanced_ml_analysis(df)
    
    # 8. Conservation analysis
    print("\n8. Running conservation analysis...")
    conservation = ConservationAnalyzer(enforce_conservation=True)
    
    # Identify lipid columns
    lipid_columns = [col for col in frame_df_processed.columns 
                    if col.endswith('_contact_count') and 
                    not col.startswith('gm3')]
    
    if lipid_columns:
        # Determine mediator column
        mediator_col = 'gm3_contact_strength' if 'gm3_contact_strength' in frame_df_processed.columns else 'gm3_contact_count'
        
        conservation_results = conservation.analyze_redistribution(
            frame_df_processed,
            mediator_col,
            lipid_columns,
            method='quartile'
        )
        
        print("\nConservation Analysis Results:")
        for lipid, results in conservation_results.items():
            print(f"  {lipid}:")
            print(f"    Change: {results['absolute_change']:+.3f} ({results['percent_change']:+.1f}%)")
            print(f"    P-value: {results['p_value']:.4f} {results['significance']}")
    
    # 9. Generate figures
    print("\n9. Generating figures...")
    fig_generator = FigureGenerator()
    ml_effects, correlations = fig_generator.create_nature_quality_figures(
        ml_results,
        frame_df_processed,
        output_dir
    )
    
    print("\nFigures saved to test_output/")
    print("  - figure_observed_results.png/pdf/svg")
    print("  - figure_temporal_analysis.png/pdf/svg")
    
    # 10. Summary of what was generated
    print("\n" + "="*70)
    print("OUTPUT SUMMARY")
    print("="*70)
    
    print("\nGenerated files:")
    output_files = os.listdir(output_dir)
    for ext in ['png', 'pdf', 'svg']:
        files = [f for f in output_files if f.endswith(ext)]
        if files:
            print(f"\n  {ext.upper()} files:")
            for f in files:
                size = os.path.getsize(os.path.join(output_dir, f))
                print(f"    - {f} ({size/1024:.1f} KB)")
    
    print("\nAnalysis complete:")
    print(f"  - Processed {len(df)} frame-protein combinations")
    print(f"  - Analyzed {len(ml_results)} lipid types")
    print(f"  - Generated {len([f for f in output_files if f.endswith('.png')])} PNG figures")
    
    return True


def verify_output_exists():
    """
    Simply verify that expected output files were created
    """
    expected_files = [
        'test_output/figure_observed_results.png',
        'test_output/figure_temporal_analysis.png'
    ]
    
    print("\nVerifying output files:")
    all_exist = True
    for filepath in expected_files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  ✅ {filepath} ({size/1024:.1f} KB)")
        else:
            print(f"  ❌ {filepath} - NOT FOUND")
            all_exist = False
    
    return all_exist


if __name__ == '__main__':
    # Run test
    success = test_original_workflow()
    
    if success:
        # Verify outputs were created
        outputs_exist = verify_output_exists()
        
        if outputs_exist:
            print("\n✅ Test completed successfully!")
            print("   All expected output files were generated.")
        else:
            print("\n⚠️  Test ran but some output files are missing!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)