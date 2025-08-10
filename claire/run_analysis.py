#!/usr/bin/env python3
"""
Run CLAIRE analysis - interactive or command-line mode
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List

# Add parent directory to path if running from claire directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all modular components
from claire.core.membrane import MembraneSystem
from claire.core.trajectory import TrajectoryProcessor
from claire.core.topology import TopologyReader

from claire.analysis.conservation import ConservationAnalyzer
from claire.analysis.enrichment import EnrichmentAnalyzer
from claire.analysis.frame_processor import FrameProcessor
from claire.analysis.diagnostics import DiagnosticAnalyzer
from claire.analysis.temporal import TemporalAnalyzer
from claire.analysis.statistical import StatisticalAnalyzer
from claire.analysis.ml_analysis import MLAnalyzer

from claire.physics.calculations import PhysicsCalculator
from claire.physics.rdf import RDFCalculator

from claire.visualization.plots import Visualizer
from claire.visualization.figures import FigureGenerator


# Import CLI function
from claire.cli import get_interactive_config


# Fixed run_claire_analysis function
def run_claire_analysis(args):
    """
    Run complete CLAIRE analysis with given parameters
    FIXED VERSION - matches original exactly
    """
    print("="*70)
    print("CLAIRE - Conserved Lipid Analysis with Interaction and Redistribution Evaluation")
    print("="*70)
    
    # Display configuration
    print("\nConfiguration:")
    print(f"  Topology:      {args.topology}")
    print(f"  Trajectory:    {args.trajectory}")
    print(f"  Output dir:    {args.output}")
    print(f"  Frames:        {args.start} to {args.stop} (step {args.step})")
    print(f"  Leaflet frame: {args.leaflet_frame}")
    print(f"  Mediator:      {args.mediator}")
    print(f"  Targets:       {', '.join(args.targets) if args.targets else 'Auto-detect'}")
    print(f"  Leaflet:       {args.leaflet}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save configuration
    import json
    config = vars(args)
    with open(os.path.join(args.output, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    # ============= PHASE 1: System Setup =============
    print("\n" + "="*50)
    print("PHASE 1: System Setup")
    print("="*50)
    
    # Load system
    print("\nLoading system...")
    membrane = MembraneSystem(args.topology, args.trajectory, verbose=args.verbose)
    
    # Check trajectory length
    n_frames = len(membrane.universe.trajectory)
    if args.stop == -1 or args.stop > n_frames:
        args.stop = n_frames
    print(f"  Trajectory has {n_frames} frames")
    print(f"  Will process frames {args.start} to {args.stop} (step {args.step})")
    
    # Identify components
    print("\nIdentifying membrane components...")
    membrane.identify_lipids()
    membrane.identify_proteins()
    
    # Identify leaflets
    print(f"\nIdentifying leaflets at frame {args.leaflet_frame}...")
    upper_leaflet, lower_leaflet = membrane.identify_leaflets(frame=args.leaflet_frame)
    
    if upper_leaflet is None or lower_leaflet is None:
        print("ERROR: Could not identify leaflets!")
        print("  Trying alternative leaflet identification...")
        # Fallback: use simple z-based selection
        membrane.universe.trajectory[args.leaflet_frame]
        z_center = membrane.universe.atoms.center_of_mass()[2]
        upper_leaflet = membrane.universe.select_atoms(f"prop z > {z_center}")
        lower_leaflet = membrane.universe.select_atoms(f"prop z < {z_center}")
    
    # Select analysis leaflet
    if args.leaflet == 'upper':
        analysis_leaflet = upper_leaflet
    elif args.leaflet == 'lower':
        analysis_leaflet = lower_leaflet
    else:  # both
        analysis_leaflet = upper_leaflet
    
    # Select lipids from leaflet
    print(f"\nSelecting lipids from {args.leaflet} leaflet...")
    
    # Get lipids in the selected leaflet
    lipid_selections = {}
    available_lipids = []
    
    for lipid_name in membrane.lipids.keys():
        selection = analysis_leaflet.select_atoms(f"resname {lipid_name}")
        if len(selection) > 0:
            lipid_selections[lipid_name] = selection
            available_lipids.append(lipid_name)
            n_mol = len(selection.residues)
            print(f"  {lipid_name}: {n_mol} molecules")
    
    # Determine mediator and targets
    if args.mediator not in lipid_selections:
        print(f"\nWARNING: Mediator {args.mediator} not found in {args.leaflet} leaflet")
        # Try to find GM3/DPG3
        for name in ['DPG3', 'GM3', 'DXG3', 'PNG3']:
            if name in lipid_selections:
                args.mediator = name
                print(f"  Using {name} as mediator instead")
                break
    
    # Auto-detect targets if not specified
    if not args.targets:
        args.targets = [name for name in available_lipids if name != args.mediator]
        print(f"  Auto-detected targets: {', '.join(args.targets)}")
    else:
        # Filter targets to only those present
        args.targets = [t for t in args.targets if t in available_lipids]
        print(f"  Using targets: {', '.join(args.targets)}")
    
    # Check if mediator exists
    if args.mediator not in lipid_selections:
        print(f"\nERROR: Mediator lipid {args.mediator} not found!")
        return False
    
    # Handle proteins
    if len(membrane.proteins) == 0:
        print("\nNo proteins found - using membrane center as reference")
        membrane.proteins = {
            'Membrane_Center': analysis_leaflet
        }
    
    # ============= PHASE 2: Frame Processing =============
    print("\n" + "="*50)
    print("PHASE 2: Frame Processing")
    print("="*50)
    
    n_frames_to_process = (args.stop - args.start) // args.step
    print(f"\nProcessing {n_frames_to_process} frames...")
    
    # Initialize processors
    frame_processor = FrameProcessor()
    box_dimensions = membrane.universe.dimensions[:3]
    
    # Process frames - CRITICAL: Process ALL frames for enough data!
    all_frame_data = []
    frame_count = 0
    
    for frame_idx in range(args.start, args.stop, args.step):
        if frame_count % 100 == 0 or args.verbose:
            print(f"  Processing frame {frame_idx} ({frame_count+1}/{n_frames_to_process})...")
        
        try:
            # CRITICAL: Pass target_lipids parameter!
            frame_results = frame_processor.process_frame_complete(
                frame_idx,
                membrane.universe,
                membrane.proteins,
                lipid_selections,
                box_dimensions,
                mediator_lipid=args.mediator,
                target_lipids=args.targets  # <-- MUST PASS THIS!
            )
            
            if frame_results:
                all_frame_data.extend(frame_results)
            
        except Exception as e:
            if args.verbose:
                print(f"    Error in frame {frame_idx}: {e}")
            continue
        
        frame_count += 1
    
    # Check if we got data
    if len(all_frame_data) == 0:
        print("\nERROR: No data collected from frames!")
        print("  Check if lipids are properly selected")
        return False
    
    # Convert to DataFrame
    df = pd.DataFrame(all_frame_data)
    print(f"\nCollected {len(df)} data points from {frame_count} frames")
    
    # Save raw data
    df.to_csv(os.path.join(args.output, 'raw_frame_data.csv'), index=False)
    print(f"  Saved raw data ({len(df)} rows)")
    
    # ============= PHASE 3: Analysis =============
    print("\n" + "="*50)
    print("PHASE 3: Analysis")
    print("="*50)
    
    # Aggregate by frame
    print("\nAggregating frame data...")
    agg_dict = {}
    for col in df.columns:
        if col not in ['frame', 'protein', 'time']:
            if pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = 'mean'
    
    if not agg_dict:
        print("ERROR: No numeric columns to aggregate!")
        return False
    
    frame_df = df.groupby('frame').agg(agg_dict).reset_index()
    print(f"  Aggregated to {len(frame_df)} frames")
    
    # Temporal analysis
    print("\nApplying temporal smoothing...")
    temporal = TemporalAnalyzer()
    numeric_cols = frame_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if args.smooth_window > 0 and len(frame_df) > args.smooth_window:
        smoothed_df = temporal.apply_smoothing(frame_df, numeric_cols, window=args.smooth_window)
    else:
        smoothed_df = frame_df
        print("  Skipping smoothing (insufficient frames or window=0)")
    
    # Diagnostics
    print("\nRunning diagnostics...")
    diagnostics = DiagnosticAnalyzer()
    diag_results, recommendations = diagnostics.diagnose_trajectory_data(df, smoothed_df)
    
    if recommendations and args.verbose:
        print("\nRecommendations:")
        for rec in recommendations[:5]:  # Show first 5
            print(f"  - {rec}")
    
    # ML Analysis - CRITICAL: Pass correct parameters!
    print("\nRunning ML analysis...")
    ml_analyzer = MLAnalyzer()
    ml_results, frame_df_ml = ml_analyzer.advanced_ml_analysis(
        df,
        target_lipids=args.targets,  # <-- PASS TARGETS!
        mediator_lipid=args.mediator  # <-- PASS MEDIATOR!
    )
    
    if not ml_results:
        print("WARNING: ML analysis returned no results")
        ml_results = {}
    
    # Conservation analysis
    print("\nRunning conservation analysis...")
    conservation = ConservationAnalyzer(enforce_conservation=True)
    
    # Get lipid columns for targets only
    lipid_columns = []
    for target in args.targets:
        col = f'{target}_contact_count'
        if col in frame_df_ml.columns:
            lipid_columns.append(col)
    
    conservation_results = {}
    if lipid_columns:
        # Find mediator column
        mediator_col = None
        for col_name in [f'{args.mediator.lower()}_contact_strength', 
                        f'{args.mediator.lower()}_contact_count',
                        'gm3_contact_strength', 
                        'gm3_contact_count']:
            if col_name in frame_df_ml.columns:
                mediator_col = col_name
                break
        
        if mediator_col:
            conservation_results = conservation.analyze_redistribution(
                frame_df_ml,
                mediator_col,
                lipid_columns,
                method='quartile'
            )
        else:
            print("  Could not find mediator column for conservation analysis")
    
    # Statistical analysis
    print("\nCalculating correlations...")
    stats_analyzer = StatisticalAnalyzer()
    corr_matrix = stats_analyzer.calculate_correlation_matrix(smoothed_df)
    
    # In run_claire_analysis, replace the visualization section with:

    # ============= PHASE 4: Visualization =============
    print("\n" + "="*50)
    print("PHASE 4: Visualization")
    print("="*50)
    
    print("\nGenerating figures...")
    
    # Initialize figure generator
    fig_gen = FigureGenerator()
    
    # 1 & 2: Generate main figures (figure_observed_results.png and figure_temporal_analysis.png)
    try:
        ml_effects, correlations = fig_gen.create_nature_quality_figures(
            ml_results,
            frame_df_ml,
            args.output,
            target_lipids=args.targets,
            mediator_lipid=args.mediator
        )
        print("  ✓ Generated figure_observed_results.png")
        print("  ✓ Generated figure_temporal_analysis.png")
    except Exception as e:
        print(f"  ✗ Error generating main figures: {e}")
    
    # Calculate composition ratios for additional plots
    frame_df_ml = fig_gen.calculate_composition_ratios(frame_df_ml, args.targets)
    composition_results = fig_gen.analyze_composition_changes(
        frame_df_ml, args.targets, args.mediator
    )
    
    # Analyze each protein separately for peptide plots
    peptide_results = {}
    if len(membrane.proteins) > 1:
        print("\n  Analyzing individual proteins...")
        for protein_name in membrane.proteins.keys():
            peptide_df = df[df['protein'] == protein_name].copy()
            
            # Aggregate by frame
            agg_dict = {}
            for col in peptide_df.columns:
                if pd.api.types.is_numeric_dtype(peptide_df[col]) and col not in ['frame', 'time']:
                    agg_dict[col] = 'mean'
            
            if agg_dict:
                peptide_frame_df = peptide_df.groupby('frame').agg(agg_dict).reset_index()
                
                # Apply smoothing
                temporal = TemporalAnalyzer()
                numeric_cols = peptide_frame_df.select_dtypes(include=[np.number]).columns.tolist()
                peptide_smoothed = temporal.apply_smoothing(
                    peptide_frame_df, numeric_cols, window=args.smooth_window
                )
                
                # Calculate composition ratios
                peptide_smoothed = fig_gen.calculate_composition_ratios(
                    peptide_smoothed, args.targets
                )
                
                # Analyze composition changes
                peptide_composition = fig_gen.analyze_composition_changes(
                    peptide_smoothed, args.targets, args.mediator
                )
                
                peptide_results[protein_name] = peptide_composition
    
    # 3: Generate composition analysis with peptides plot
    try:
        fig_gen.plot_composition_analysis_with_peptides(
            frame_df_ml,
            composition_results,
            df,
            peptide_results,
            args.output,
            target_lipids=args.targets,
            mediator_lipid=args.mediator
        )
        print("  ✓ Generated composition_analysis_with_peptides.png")
    except Exception as e:
        print(f"  ✗ Error generating composition analysis: {e}")
    
    # 4: Generate peptide comparison plot (only if multiple proteins)
    if len(peptide_results) > 1:
        try:
            fig_gen.plot_peptide_comparison(
                peptide_results,
                args.output,
                target_lipids=args.targets
            )
            print("  ✓ Generated peptide_comparison.png")
        except Exception as e:
            print(f"  ✗ Error generating peptide comparison: {e}")
    else:
        print("  ℹ Skipping peptide_comparison.png (only one protein/reference)")
    
    # ============= PHASE 5: Summary =============
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"\nProcessed:")
    print(f"  Frames: {frame_count}")
    print(f"  Data points: {len(df)}")
    print(f"  Mediator: {args.mediator}")
    print(f"  Targets: {', '.join(args.targets)}")
    
    if ml_results:
        print("\nML Analysis Results:")
        for lipid, results in ml_results.items():
            effect_key = 'gm3_effect' if 'gm3_effect' in results else f'{args.mediator.lower()}_effect'
            print(f"  {lipid}:")
            print(f"    Effect: {results.get(effect_key, 0):.3f}")
            print(f"    Correlation: {results.get('correlation', 0):.3f}")
            print(f"    P-value: {results.get('p_value', 1):.4f}")
    
    if conservation_results:
        print("\nConservation Analysis:")
        for lipid, results in conservation_results.items():
            print(f"  {lipid.replace('_contact_count', '')}:")
            print(f"    Change: {results['absolute_change']:+.3f} ({results['percent_change']:+.1f}%)")
            print(f"    P-value: {results['p_value']:.4f}")
    
    print(f"\nResults saved to: {args.output}/")
    
    # List output files
    if os.path.exists(args.output):
        output_files = os.listdir(args.output)
        print("\nGenerated files:")
        
        file_types = {
            'Data': ['.csv', '.json'],
            'Figures': ['.png', '.pdf', '.svg']
        }
        
        for category, extensions in file_types.items():
            files = [f for f in output_files if any(f.endswith(ext) for ext in extensions)]
            if files:
                print(f"\n  {category}:")
                for f in sorted(files):
                    filepath = os.path.join(args.output, f)
                    if os.path.exists(filepath):
                        size = os.path.getsize(filepath) / 1024
                        print(f"    - {f} ({size:.1f} KB)")
    
    # Final check
    expected_plots = ['figure_observed_results.png', 'figure_temporal_analysis.png']
    plots_found = sum(1 for f in expected_plots if f in output_files)
    
    if plots_found == len(expected_plots):
        print(f"\n✅ Success! All plots generated")
        return True
    else:
        print(f"\n⚠️  Warning: Only {plots_found}/{len(expected_plots)} expected plots found")
        return True  # Still return True if analysis completed


def main():
    """Main entry point with both interactive and command-line modes"""
    
    # Check if any command-line arguments were provided
    if len(sys.argv) == 1:
        # No arguments - run interactive mode
        args_dict = get_interactive_config()
        
        # Convert dict to args object
        class Args:
            pass
        args = Args()
        for key, value in args_dict.items():
            setattr(args, key, value)
        
        # Set defaults for missing values
        args.leaflet_frame = getattr(args, 'leaflet_frame', args.start)
        args.leaflet = getattr(args, 'leaflet', 'upper')
        args.smooth_window = getattr(args, 'smooth_window', 20)
    else:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(
            description='CLAIRE - Lipid redistribution analysis',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Run without arguments for interactive mode:
  python run_analysis.py

Or specify parameters via command line:
  python run_analysis.py -s system.psf -t trajectory.xtc --start 5000 --stop 10000
            """
        )
        
        parser.add_argument('-s', '--topology', 
                           help='Topology file (PSF, PDB, GRO, etc.)')
        parser.add_argument('-t', '--trajectory',
                           help='Trajectory file (XTC, DCD, TRR, etc.)')
        parser.add_argument('-o', '--output', default='claire_output',
                           help='Output directory')
        parser.add_argument('--start', type=int, default=0)
        parser.add_argument('--stop', type=int, default=-1)
        parser.add_argument('--step', type=int, default=1)
        parser.add_argument('--leaflet-frame', type=int, default=0)
        parser.add_argument('--mediator', default='DPG3')
        parser.add_argument('--targets', nargs='+', default=None)
        parser.add_argument('--leaflet', choices=['upper', 'lower', 'both'], default='upper')
        parser.add_argument('--smooth-window', type=int, default=20)
        parser.add_argument('-v', '--verbose', action='store_true')
        
        args = parser.parse_args()
        
        # Check if minimum required arguments are provided
        if not args.topology or not args.trajectory:
            print("Error: Topology and trajectory files are required in command-line mode")
            print("Run without arguments for interactive mode")
            sys.exit(1)
    
    # Run analysis
    try:
        success = run_claire_analysis(args)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()