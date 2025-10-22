#!/usr/bin/env python3
"""
CLAIRE - Composition-based Lipid Analysis with Temporal and Spatial Resolution
Main analysis script
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.trajectory_loader import load_universe, identify_lipid_leaflets, select_proteins, select_lipids
from core.frame_processor import calculate_frame_composition
from utils.parallel import test_multiprocessing, process_frames_parallel, process_frames_serial
from analysis.composition import CompositionAnalyzer
from analysis.temporal import TemporalAnalyzer
from analysis.spatial import SpatialAnalyzer
from analysis.ml_predict import CompositionPredictor
from analysis.causal_inference import CausalInference
from analysis.microscopy_prediction import MicroscopyAnalyzer
from visualization.plots import (plot_composition_changes, plot_temporal_composition,
                                 plot_temporal_composition_per_protein,
                                 plot_radial_profiles, plot_ml_predictions,
                                 plot_comprehensive_summary,
                                 plot_target_dependent_radial_profiles,
                                 plot_target_position_correlation,
                                 plot_lag_correlation)
from visualization.plots_radial_new import plot_radial_profiles_both
from visualization.plots_target_position import plot_target_position_effect_improved
from visualization.plots_causal import plot_causal_inference_results, plot_causal_network
from visualization.plots_microscopy import (plot_microscopy_predictions, plot_microscopy_beads,
                                           plot_microscopy_comparison, plot_multicolor_overlay)
from analysis.run_timeseries_ml import run_timeseries_ml_analysis
from analysis.bulk_composition import (calculate_bulk_vs_protein_composition,
                                       calculate_bulk_vs_protein_gm3_dependent)
from visualization.plots_bulk import (plot_bulk_vs_protein_comparison,
                                     plot_bulk_protein_gm3_comparison)
from config import *


def main():
    """Main CLAIRE analysis"""

    parser = argparse.ArgumentParser(description='CLAIRE - Lipid Composition Analysis')
    parser.add_argument('--topology', default=None, help='Topology file (PSF, PDB, GRO). If not specified, uses config.TOPOLOGY_FILE')
    parser.add_argument('--trajectory', default=None, help='Trajectory file (XTC, DCD, TRR). If not specified, uses config.TRAJECTORY_FILE')
    parser.add_argument('--output', default=None, help='Output directory (default: config.OUTPUT_DIR)')
    parser.add_argument('--lipids', nargs='+', default=None,
                       help='Lipid types to analyze (default: config.DEFAULT_LIPID_TYPES)')
    parser.add_argument('--target-lipid', default=None, help='Target/mediator lipid (default: config.TARGET_LIPID)')
    parser.add_argument('--start', type=int, default=None, help='Start frame (default: config.FRAME_START)')
    parser.add_argument('--stop', type=int, default=None, help='Stop frame (default: config.FRAME_STOP)')
    parser.add_argument('--step', type=int, default=None, help='Frame step (default: config.FRAME_STEP)')
    parser.add_argument('--cutoff', type=float, default=None, help='Composition cutoff distance (Å, default: config.COMPOSITION_CUTOFF=15.0)')
    parser.add_argument('--parallel', action='store_true', default=None, help='Use parallel processing (default: config.USE_PARALLEL)')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--n-workers', type=int, default=None, help='Number of workers')
    parser.add_argument('--skip-temporal', action='store_true', help='Skip temporal analysis')
    parser.add_argument('--skip-spatial', action='store_true', help='Skip spatial analysis')
    parser.add_argument('--skip-ml', action='store_true', help='Skip ML analysis')
    parser.add_argument('--skip-causal', action='store_true', help='Skip causal inference analysis')
    parser.add_argument('--skip-microscopy', action='store_true', help='Skip microscopy prediction')
    parser.add_argument('--skip-timeseries-ml', action='store_true', help='Skip time-series ML prediction')
    parser.add_argument('--skip-bulk', action='store_true', help='Skip bulk vs protein composition analysis')
    parser.add_argument('--leaflet', choices=['upper', 'lower', 'both'], default='upper',
                       help='Which leaflet to analyze')

    args = parser.parse_args()

    # Get topology and trajectory from config if not specified
    topology = args.topology if args.topology else TOPOLOGY_FILE
    trajectory = args.trajectory if args.trajectory else TRAJECTORY_FILE

    # Get lipid types from config if not specified
    lipid_types = args.lipids if args.lipids else DEFAULT_LIPID_TYPES
    target_lipid = args.target_lipid if args.target_lipid else TARGET_LIPID

    # Get frame range from config if not specified
    frame_start = args.start if args.start is not None else FRAME_START
    frame_stop = args.stop if args.stop is not None else FRAME_STOP
    frame_step = args.step if args.step is not None else FRAME_STEP

    # Get cutoff from config if not specified
    cutoff = args.cutoff if args.cutoff is not None else COMPOSITION_CUTOFF

    # Get output directory from config if not specified
    output_dir = args.output if args.output else OUTPUT_DIR

    # Determine parallel processing
    if args.no_parallel:
        use_parallel = False
    elif args.parallel:
        use_parallel = True
    else:
        use_parallel = USE_PARALLEL

    # Validate input files
    if topology is None:
        print("ERROR: No topology file specified. Use --topology or set TOPOLOGY_FILE in config.py")
        return
    if trajectory is None:
        print("ERROR: No trajectory file specified. Use --trajectory or set TRAJECTORY_FILE in config.py")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Clear all cache files before starting
    print("\n" + "="*70)
    print("CLEARING CACHE")
    print("="*70)

    import shutil

    cleared_count = 0

    # Clear temp_files directory
    if os.path.exists('temp_files'):
        try:
            shutil.rmtree('temp_files')
            print(f"  Removed directory: temp_files")
            cleared_count += 1
        except Exception as e:
            print(f"  Warning: Could not remove temp_files: {e}")

    # Clear claire_output directory
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
            print(f"  Removed directory: {output_dir}")
            cleared_count += 1
        except Exception as e:
            print(f"  Warning: Could not remove {output_dir}: {e}")

    if cleared_count == 0:
        print("  No cache files found (clean start)")
    else:
        print(f"  Total: {cleared_count} cache file(s)/directory cleared")
    print("="*70)

    # Recreate output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('temp_files', exist_ok=True)

    print("\n" + "="*80)
    print("CLAIRE - Composition-based Lipid Analysis")
    print("="*80)
    print(f"Topology: {topology}")
    print(f"Trajectory: {trajectory}")
    print(f"Output: {output_dir}")
    print(f"Lipids: {lipid_types}")
    print(f"Target lipid: {target_lipid}")
    print(f"Frame range: {frame_start} to {frame_stop if frame_stop else 'end'} (step {frame_step})")
    print(f"Composition cutoff: {cutoff} Å (GM3 binding uses 6Å)")
    print(f"Parallel processing: {use_parallel}")
    print("="*80)

    # Load trajectory
    print("\n### STEP 1: Loading Trajectory ###")
    u = load_universe(topology, trajectory)
    if u is None:
        print("ERROR: Failed to load trajectory")
        return

    # Determine frames to analyze
    n_frames = len(u.trajectory)
    stop = frame_stop if frame_stop is not None else n_frames
    frames = list(range(frame_start, stop, frame_step))
    print(f"Analyzing {len(frames)} frames: {frames[0]} to {frames[-1]} (step {frame_step})")

    # Identify leaflets (include target_lipid in leaflet detection)
    print("\n### STEP 2: Identifying Leaflets ###")
    lipid_types_with_target = lipid_types.copy()
    if target_lipid and target_lipid not in lipid_types_with_target:
        lipid_types_with_target.append(target_lipid)

    upper_leaflet, lower_leaflet = identify_lipid_leaflets(u, lipid_types_with_target,
                                                            leaflet_cutoff=LEAFLET_CUTOFF)

    if args.leaflet == 'upper':
        leaflet_to_analyze = upper_leaflet
        print("Analyzing upper leaflet only")
    elif args.leaflet == 'lower':
        leaflet_to_analyze = lower_leaflet
        print("Analyzing lower leaflet only")
    else:
        print("WARNING: Both leaflets not yet implemented, using upper leaflet")
        leaflet_to_analyze = upper_leaflet

    # Select proteins and lipids
    print("\n### STEP 3: Selecting Proteins and Lipids ###")
    proteins = select_proteins(u)

    lipid_selections = select_lipids(u, leaflet_to_analyze, lipid_types_with_target)

    protein_segids = ['PROA', 'PROB', 'PROC', 'PROD'][:len(proteins)]
    leaflet_resids = [res.resid for res in leaflet_to_analyze.residues]

    # Process frames
    print("\n### STEP 4: Processing Frames ###")

    cache_file = os.path.join(output_dir, 'frame_data.pkl')

    if os.path.exists(cache_file):
        print(f"Loading cached frame data from {cache_file}")
        with open(cache_file, 'rb') as f:
            frame_data_list = pickle.load(f)
    else:
        if use_parallel:
            # Test multiprocessing
            success, ctx_method = test_multiprocessing()
            if not success:
                print("WARNING: Multiprocessing test failed, falling back to serial")
                use_parallel = False

        if use_parallel:
            frame_data_list = process_frames_parallel(
                frames, topology, trajectory, protein_segids,
                leaflet_resids, lipid_types_with_target, cutoff, target_lipid,
                TM_RESIDUES, args.n_workers, ctx_method
            )
        else:
            frame_data_list = process_frames_serial(
                frames, u, proteins, lipid_selections, leaflet_to_analyze,
                cutoff, target_lipid, TM_RESIDUES
            )

        # Cache results
        with open(cache_file, 'wb') as f:
            pickle.dump(frame_data_list, f)
        print(f"✓ Cached frame data to {cache_file}")

    if len(frame_data_list) == 0:
        print("ERROR: No frames processed successfully")
        return

    # Composition analysis
    print("\n### STEP 5: Composition Analysis ###")

    analyzer = CompositionAnalyzer(lipid_types, target_lipid)
    df = analyzer.frames_to_dataframe(frame_data_list)
    df = analyzer.calculate_conservation_ratios(df)

    # Save dataframe
    df_file = os.path.join(output_dir, 'composition_data.csv')
    df.to_csv(df_file, index=False)
    print(f"✓ Saved composition data to {df_file}")

    # Analyze composition changes
    if target_lipid:
        comp_results = analyzer.analyze_composition_changes(df, 'target_lipid_bound', method='binary')
    else:
        print("No target lipid specified, skipping composition change analysis")
        comp_results = None

    # Plot composition changes
    if comp_results:
        fig = plot_composition_changes(comp_results, analyzer.comp_lipids,
                                      os.path.join(output_dir, 'composition_changes.png'))
        plt.close(fig)

    # Bulk vs Protein composition analysis
    if not args.skip_bulk:
        print("\n### STEP 5B: Bulk vs Protein Vicinity Composition ###")

        # Calculate bulk composition and compare with protein vicinity
        bulk_results = calculate_bulk_vs_protein_composition(
            df, u, lipid_selections, leaflet_to_analyze,
            analyzer.comp_lipids, proteins, n_frames=200
        )

        # Plot comparison
        fig = plot_bulk_vs_protein_comparison(
            bulk_results,
            os.path.join(output_dir, 'bulk_vs_protein.png')
        )
        plt.close(fig)

        # GM3-dependent bulk vs protein comparison
        if target_lipid:
            gm3_bulk_results = calculate_bulk_vs_protein_gm3_dependent(
                df, u, lipid_selections, leaflet_to_analyze,
                analyzer.comp_lipids, proteins, target_lipid, n_frames=200
            )

            fig = plot_bulk_protein_gm3_comparison(
                bulk_results, gm3_bulk_results, analyzer.comp_lipids,
                os.path.join(output_dir, 'bulk_vs_protein_gm3.png')
            )
            plt.close(fig)

    # Per-protein analysis
    print("\n### STEP 6: Per-Protein Analysis ###")
    if target_lipid:
        protein_results = analyzer.analyze_per_protein(df, 'target_lipid_bound')

    # Temporal analysis
    if not args.skip_temporal:
        print("\n### STEP 7: Per-Protein Temporal Analysis ###")
        temporal = TemporalAnalyzer(window_size=min(100, len(frames)//5), step_size=10)

        # STEP 7A: Per-protein sliding window composition
        protein_windows = temporal.sliding_window_composition_per_protein(df, analyzer.comp_lipids)

        # Save per-protein temporal data
        for protein_name, window_df in protein_windows.items():
            window_file = os.path.join(output_dir, f'temporal_windows_{protein_name}.csv')
            window_df.to_csv(window_file, index=False)

        # STEP 7B: Per-protein GM3 binding event detection
        protein_events = None
        if target_lipid:
            protein_events = temporal.detect_binding_events_per_protein(df, 'target_lipid_bound')

            # Save per-protein events to individual CSV files
            for protein_name, events in protein_events.items():
                binding_df_data = []
                for event in events['binding']:
                    binding_df_data.append({
                        'frame': event['frame'],
                        'protein': protein_name,
                        'event': 'binding'
                    })
                for event in events['unbinding']:
                    binding_df_data.append({
                        'frame': event['frame'],
                        'protein': protein_name,
                        'event': 'unbinding'
                    })

                if binding_df_data:
                    binding_df = pd.DataFrame(binding_df_data).sort_values('frame')
                    event_file = os.path.join(output_dir, f'gm3_events_{protein_name}.csv')
                    binding_df.to_csv(event_file, index=False)

            print(f"\n✓ Saved per-protein GM3 binding events")

        # STEP 7C: Per-protein lag correlation analysis
        if target_lipid:
            protein_lag_results = temporal.calculate_lag_correlation_per_protein(
                df, 'target_lipid_bound', analyzer.comp_lipids, max_lag=50
            )

            # Save summary of lag results
            lag_summary_data = []
            for protein_name, lipid_results in protein_lag_results.items():
                for lipid_type, result in lipid_results.items():
                    lags = result['lags']
                    corrs = result['correlations']
                    p_vals = result['p_values']

                    finite_mask = np.isfinite(corrs)
                    if np.any(finite_mask):
                        max_idx = np.argmax(np.abs(corrs[finite_mask]))
                        finite_lags = lags[finite_mask]
                        finite_corrs = corrs[finite_mask]
                        finite_pvals = p_vals[finite_mask]

                        lag_summary_data.append({
                            'protein': protein_name,
                            'lipid': lipid_type,
                            'max_lag': finite_lags[max_idx],
                            'max_correlation': finite_corrs[max_idx],
                            'p_value': finite_pvals[max_idx]
                        })

            if lag_summary_data:
                lag_summary_df = pd.DataFrame(lag_summary_data)
                lag_summary_file = os.path.join(output_dir, 'lag_correlation_summary.csv')
                lag_summary_df.to_csv(lag_summary_file, index=False)
                print(f"\n✓ Saved lag correlation summary to {lag_summary_file}")

        # STEP 7D: Plot per-protein temporal composition
        if protein_windows and protein_events:
            fig = plot_temporal_composition_per_protein(
                protein_windows, protein_events, analyzer.comp_lipids,
                output_path=os.path.join(output_dir, 'temporal_composition_per_protein.png')
            )
            plt.close(fig)
            print(f"\n✓ Saved per-protein temporal composition plot")

        # STEP 7E: Also create aggregated plot for overview
        if target_lipid:
            # Aggregate all events for overview plot
            aggregated_events = temporal.detect_binding_events(df, 'target_lipid_bound')

            # Aggregate all windows
            all_windows = pd.concat(protein_windows.values(), ignore_index=True)
            all_windows = all_windows.groupby('center_frame')[
                [f'{lt}_ratio' for lt in analyzer.comp_lipids]
            ].mean().reset_index()

            fig = plot_temporal_composition(
                all_windows, analyzer.comp_lipids,
                binding_events=aggregated_events,
                output_path=os.path.join(output_dir, 'temporal_composition_aggregated.png')
            )
            plt.close(fig)
            print(f"✓ Saved aggregated temporal composition plot")

    # Spatial analysis
    if not args.skip_spatial:
        print("\n### STEP 8: Spatial Analysis ###")
        spatial = SpatialAnalyzer(radii=SPATIAL_RADII, shell_type=SPATIAL_SHELL_TYPE)

        # Calculate radial profiles for subset of frames
        if SPATIAL_N_FRAMES is None:
            n_spatial_frames = len(frames)
        else:
            n_spatial_frames = min(SPATIAL_N_FRAMES, len(frames))
        spatial_frames = np.linspace(frames[0], frames[-1], n_spatial_frames, dtype=int)

        print(f"Calculating radial profiles for {n_spatial_frames} frames...")
        radial_data_list = []

        for i, frame_idx in enumerate(spatial_frames):
            if i % 100 == 0:
                print(f"  Frame {i+1}/{n_spatial_frames}")

            radial_data = spatial.calculate_radial_composition(
                u, frame_idx, proteins, lipid_selections,
                leaflet_to_analyze, analyzer.comp_lipids,
                target_lipid=target_lipid
            )
            radial_data_list.append(radial_data)

        # Overall average profiles
        radial_profiles = spatial.calculate_average_radial_profiles(
            radial_data_list, analyzer.comp_lipids
        )

        # Use new plotting function that supports both shell types and config order
        if SPATIAL_SHELL_TYPE == 'both':
            fig = plot_radial_profiles_both(radial_profiles, analyzer.comp_lipids,
                                           os.path.join(output_dir, 'radial_profiles_overall.png'))
        else:
            fig = plot_radial_profiles(radial_profiles, analyzer.comp_lipids,
                                      os.path.join(output_dir, 'radial_profiles_overall.png'))
        plt.close(fig)

        # Target lipid-dependent analysis
        if target_lipid:
            print("\n### STEP 8B: Target Lipid-Dependent Spatial Analysis ###")

            # Bound vs unbound profiles
            target_results = spatial.calculate_target_dependent_profiles(
                radial_data_list, analyzer.comp_lipids, target_lipid
            )

            if target_results:
                fig = plot_target_dependent_radial_profiles(
                    target_results, analyzer.comp_lipids,
                    os.path.join(output_dir, 'radial_profiles_target_dependent.png')
                )
                plt.close(fig)

            # Target position effect
            position_df = spatial.analyze_target_position_effect(
                radial_data_list, analyzer.comp_lipids, target_lipid
            )

            if len(position_df) > 0:
                # Save position data
                position_file = os.path.join(output_dir, 'target_position_data.csv')
                position_df.to_csv(position_file, index=False)
                print(f"✓ Saved target position data to {position_file}")

                # Plot improved target position effect
                fig = plot_target_position_effect_improved(
                    position_df, analyzer.comp_lipids, SPATIAL_RADII,
                    os.path.join(output_dir, 'target_position_effect.png')
                )
                if fig:
                    plt.close(fig)

    # ML prediction
    if not args.skip_ml and target_lipid:
        print("\n### STEP 9: Machine Learning Prediction ###")

        predictor = CompositionPredictor(test_size=0.3, random_seed=42)

        # Merge target position data for ML features
        ml_df = df.copy()
        if 'position_df' in locals() and len(position_df) > 0:
            # Merge target_bound_count from position_df into composition_df
            position_merge = position_df[['frame', 'protein', 'target_bound_count', 'target_distance']].copy()
            ml_df = ml_df.merge(position_merge, on=['frame', 'protein'], how='left')
            print(f"✓ Merged target position data for ML analysis")

        # Use target lipid count as predictor
        mediator_cols = []
        if 'target_bound_count' in ml_df.columns:
            mediator_cols.append('target_bound_count')
        if 'target_distance' in ml_df.columns:
            mediator_cols.append('target_distance')

        if f'{target_lipid}_count' in ml_df.columns:
            mediator_cols.append(f'{target_lipid}_count')

        if mediator_cols:
            print(f"Using features for ML: {mediator_cols}")
            ml_results = predictor.predict_composition_changes(
                ml_df, mediator_cols, analyzer.comp_lipids
            )

            fig = plot_ml_predictions(ml_results, analyzer.comp_lipids,
                                     os.path.join(output_dir, 'ml_predictions.png'))
            if fig:
                plt.close(fig)
        else:
            print("WARNING: No mediator columns found for ML prediction")

    # Causal inference
    if not args.skip_causal and target_lipid:
        print("\n### STEP 10: Causal Inference Analysis ###")

        causal_analyzer = CausalInference(max_lag=10, significance=0.05)

        # Perform causal analysis
        causal_results = causal_analyzer.analyze_causality(
            df, target_lipid=target_lipid, composition_lipids=analyzer.comp_lipids
        )

        if causal_results:
            # Save summary table
            summary_table = causal_analyzer.get_summary_table(causal_results)
            summary_path = os.path.join(output_dir, 'causal_inference_summary.csv')
            summary_table.to_csv(summary_path, index=False)
            print(f"✓ Saved causal inference summary to {summary_path}")

            # Plot causal inference results
            fig = plot_causal_inference_results(
                causal_results,
                os.path.join(output_dir, 'causal_inference.png')
            )
            if fig:
                plt.close(fig)

            # Plot causal network
            fig = plot_causal_network(
                causal_results,
                significance=0.05,
                output_path=os.path.join(output_dir, 'causal_network.png')
            )
            if fig:
                plt.close(fig)

    # Microscopy prediction
    if not args.skip_microscopy and target_lipid:
        print("\n### STEP 11: Microscopy Prediction ###")

        microscopy_analyzer = MicroscopyAnalyzer()

        # Predict and simulate microscopy at multiple scales
        microscopy_results = microscopy_analyzer.analyze_and_simulate(
            df, target_radii=[50.0, 100.0, 200.0]
        )

        if microscopy_results:
            # Plot continuous distribution predictions
            fig = plot_microscopy_predictions(
                microscopy_results,
                lipid_types=analyzer.comp_lipids,
                output_path=os.path.join(output_dir, 'microscopy_predictions.png')
            )
            if fig:
                plt.close(fig)

            # Plot bead-based (single molecule) simulations
            fig = plot_microscopy_beads(
                microscopy_results,
                lipid_types=analyzer.comp_lipids,
                output_path=os.path.join(output_dir, 'microscopy_beads.png')
            )
            if fig:
                plt.close(fig)

            # Plot comparison for main lipid (DPSM)
            if 'DPSM' in analyzer.comp_lipids:
                fig = plot_microscopy_comparison(
                    microscopy_results,
                    lipid_type='DPSM',
                    output_path=os.path.join(output_dir, 'microscopy_comparison_DPSM.png')
                )
                if fig:
                    plt.close(fig)

            # Plot multi-color overlay at 100 Å scale
            fig = plot_multicolor_overlay(
                microscopy_results,
                radius=100.0,
                mode='beads',
                output_path=os.path.join(output_dir, 'microscopy_multicolor.png')
            )
            if fig:
                plt.close(fig)

    # Time-series ML prediction
    if not args.skip_timeseries_ml and target_lipid:
        print("\n### STEP 12: Time-Series ML for Nanodomain Dynamics ###")

        try:
            timeseries_results = run_timeseries_ml_analysis(
                df,
                lipid_types=analyzer.comp_lipids,
                target_lipid=target_lipid,
                output_dir=output_dir,
                lookback=10,
                prediction_horizon=1,
                model_type='lstm',
                test_size=0.2,
                epochs=100,
                batch_size=32,
                lr=0.001
            )

            print("  Time-series ML analysis complete!")

        except Exception as e:
            print(f"  WARNING: Time-series ML failed: {str(e)}")
            import traceback
            traceback.print_exc()

    # Comprehensive summary
    print("\n### STEP 13: Creating Summary Figure ###")

    if comp_results and not args.skip_temporal and not args.skip_spatial:
        fig = plot_comprehensive_summary(
            comp_results, window_df, radial_profiles, analyzer.comp_lipids,
            os.path.join(output_dir, 'comprehensive_summary.png')
        )
        plt.close(fig)

    # Summary report
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print("\nGenerated files:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")
    print("="*80)


if __name__ == '__main__':
    main()
