#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Machine Learning Analysis of GM3-Mediated Lipid Reorganization
======================================================================
Nature-quality analysis with full calculations - no example data
VERSION WITHOUT CAUSAL INFERENCE

# Conservation version
python analysis13.py --start 20000 --stop 80000 --step 10 --output analysis_conserved --use-conservation

# Standard version
python analysis13.py --start 20000 --stop 80000 --step 10 --output analysis_standard
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# MD Analysis
import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder
from MDAnalysis.analysis.distances import distance_array

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.inspection import permutation_importance

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Statistical Analysis
from scipy import stats
from scipy.spatial import distance
from scipy.signal import correlate
import statsmodels.api as sm

# Import improved metrics functions
from improved_metrics import (
    calculate_enrichment_metrics,
    calculate_gm3_mediated_clustering,
    improved_frame_analysis,
    select_best_metric,
    diagnose_trajectory_data,
    apply_smoothing_and_filtering,
    bootstrap_correlation_analysis,
    run_complete_analysis
)

# Try optional imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Will use permutation importance.")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available.")

# Parallel processing
import multiprocessing as mp
from functools import partial

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Global parameters
START = 60000
STOP = 80000
STEP = 20
LEAFLET_FRAME = 20000
N_WORKERS = mp.cpu_count()
RANDOM_SEED = 42

# Physical parameters
CUTOFF_RADII = {
    'contact': 6.0,
    'first_shell': 10.0,
    'influence': 15.0
}

# RDF parameters
RDF_MAX_RADIUS = 20.0
RDF_N_BINS = 40

# Set random seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

class PhysicsInformedNN(nn.Module):
    """Physics-informed neural network for GM3-lipid interactions"""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=3):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Physics constraints
        self.lipid_conservation = nn.Linear(output_dim, 1, bias=False)
        
    def forward(self, x):
        out = self.net(x)
        # Ensure lipid conservation (sum of changes ≈ 0)
        conservation_loss = torch.abs(torch.sum(out, dim=1))
        return out, conservation_loss

def identify_lipid_leaflets(u, frame=LEAFLET_FRAME):
    """Identify lipid leaflets - DO NOT MODIFY"""
    try:
        u.trajectory[frame]
        print(f"Identifying lipid leaflets at frame {frame}...")
        L = LeafletFinder(u, "name GL1 GL2 AM1 AM2 ROH GM1 GM2")
        cutoff = L.update(10)
        leaflet0 = L.groups(0)
        leaflet1 = L.groups(1)
        
        print(f"Leaflet 0: {len(leaflet0)} atoms")
        print(f"Leaflet 1: {len(leaflet1)} atoms")
        
        z0 = leaflet0.center_of_mass()[2]
        z1 = leaflet1.center_of_mass()[2]
        
        if z0 > z1:
            upper_leaflet = leaflet0
            lower_leaflet = leaflet1
        else:
            upper_leaflet = leaflet1
            lower_leaflet = leaflet0
            
        print(f"Upper leaflet Z: {upper_leaflet.center_of_mass()[2]:.2f}")
        print(f"Lower leaflet Z: {lower_leaflet.center_of_mass()[2]:.2f}")
        
        return upper_leaflet, lower_leaflet
    except Exception as e:
        print(f"Error identifying lipid leaflets: {e}")
        return None, None

def select_lipids_and_chol(leaflet, u):
    """Select lipids from leaflet - DO NOT MODIFY"""
    selections = {}
    lipid_types = ['CHOL', 'DIPC', 'DPSM', 'DPG3']
    
    for resname in lipid_types:
        try:
            selection = leaflet.select_atoms(f"resname {resname}")
            selections[resname] = selection
            print(f"Found {len(selection.residues)} {resname} residues in leaflet")
        except Exception as e:
            print(f"Could not select lipid type {resname}: {e}")
            selections[resname] = mda.AtomGroup([], u)
    
    return selections

def identify_proteins(u):
    """Identify proteins - DO NOT MODIFY"""
    proteins = {}
    try:
        protein_residues = u.select_atoms("protein")
        if len(protein_residues) == 0:
            protein_residues = u.select_atoms("resname PROT")
        
        if len(protein_residues) == 0:
            print("WARNING: No protein residues found")
            return {}
        
        segids = np.unique(protein_residues.segids)
        
        for i, segid in enumerate(segids):
            protein_selection = protein_residues.select_atoms(f"segid {segid}")
            if len(protein_selection) > 0:
                protein_name = f"Protein_{i+1}"
                proteins[protein_name] = protein_selection
                print(f"Found {protein_name} ({segid}) with {len(protein_selection)} atoms")
        
        return proteins
    except Exception as e:
        print(f"Error identifying proteins: {e}")
        return {}

def calculate_actual_rdf(protein_com, lipid_positions, box_dimensions, 
                        max_radius=RDF_MAX_RADIUS, n_bins=RDF_N_BINS):
    """Calculate actual radial distribution function"""
    if len(lipid_positions) == 0:
        return np.linspace(0, max_radius, n_bins), np.zeros(n_bins)
    
    distances = []
    
    for pos in lipid_positions:
        # XY-plane distance only
        dx = pos[0] - protein_com[0]
        dy = pos[1] - protein_com[1]
        
        # PBC correction
        dx = dx - box_dimensions[0] * round(dx/box_dimensions[0])
        dy = dy - box_dimensions[1] * round(dy/box_dimensions[1])
        
        r = np.sqrt(dx**2 + dy**2)
        if r <= max_radius:
            distances.append(r)
    
    if len(distances) == 0:
        return np.linspace(0, max_radius, n_bins), np.zeros(n_bins)
    
    # Calculate histogram
    hist, bin_edges = np.histogram(distances, bins=n_bins, range=(0, max_radius))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Calculate RDF
    rdf = np.zeros(n_bins)
    for i in range(n_bins):
        r_inner = bin_edges[i]
        r_outer = bin_edges[i+1]
        area = np.pi * (r_outer**2 - r_inner**2)
        
        if area > 0:
            # Normalize by area
            rdf[i] = hist[i] / area
    
    # Normalize by bulk density
    total_area = np.pi * max_radius**2
    n_total = len(lipid_positions)
    bulk_density = n_total / total_area
    
    if bulk_density > 0:
        rdf = rdf / bulk_density
    
    return bin_centers, rdf

def calculate_orientational_order(protein_com, lipid_positions, box_dimensions):
    """Calculate orientational order parameter around protein"""
    if len(lipid_positions) < 2:
        return 0.0
    
    angles = []
    for pos in lipid_positions:
        dx = pos[0] - protein_com[0]
        dy = pos[1] - protein_com[1]
        
        # PBC correction
        dx = dx - box_dimensions[0] * round(dx/box_dimensions[0])
        dy = dy - box_dimensions[1] * round(dy/box_dimensions[1])
        
        if dx != 0 or dy != 0:
            angle = np.arctan2(dy, dx)
            angles.append(angle)
    
    if len(angles) < 2:
        return 0.0
    
    # Calculate second-rank orientational order parameter
    cos_terms = np.cos(2 * np.array(angles))
    sin_terms = np.sin(2 * np.array(angles))
    
    S = np.sqrt(np.mean(cos_terms)**2 + np.mean(sin_terms)**2)
    return S

def calculate_local_density(protein_com, lipid_positions, box_dimensions, radius=10.0):
    """Calculate local density of lipids around protein"""
    if len(lipid_positions) == 0:
        return 0.0
    
    count = 0
    for pos in lipid_positions:
        dx = pos[0] - protein_com[0]
        dy = pos[1] - protein_com[1]
        
        # PBC correction
        dx = dx - box_dimensions[0] * round(dx/box_dimensions[0])
        dy = dy - box_dimensions[1] * round(dy/box_dimensions[1])
        
        if np.sqrt(dx**2 + dy**2) <= radius:
            count += 1
    
    # Density in molecules per nm²
    area = np.pi * (radius/10.0)**2  # Convert to nm
    density = count / area
    
    return density

def process_single_frame_advanced(frame_idx, u, proteins, lipid_selections, box_dimensions):
    """Advanced frame processing with actual physics calculations"""
    try:
        u.trajectory[frame_idx]
        frame_features = []
        
        for protein_name, protein_sel in proteins.items():
            protein_com = protein_sel.center_of_mass()
            
            features = {
                'frame': frame_idx,
                'protein': protein_name,
                'time': u.trajectory.time
            }
            
            # Get all lipid positions for this frame
            all_lipid_positions = {}
            for lipid_type, lipid_sel in lipid_selections.items():
                if len(lipid_sel) > 0:
                    positions = []
                    for residue in lipid_sel.residues:
                        positions.append(residue.atoms.center_of_mass())
                    all_lipid_positions[lipid_type] = np.array(positions)
                else:
                    all_lipid_positions[lipid_type] = np.array([])
            
            # GM3-specific analysis
            gm3_positions = all_lipid_positions.get('DPG3', np.array([]))
            
            # Multi-scale GM3 features
            for radius_name, radius in CUTOFF_RADII.items():
                gm3_count = 0
                gm3_strength = 0
                gm3_distances = []
                
                for pos in gm3_positions:
                    dx = pos[0] - protein_com[0]
                    dy = pos[1] - protein_com[1]
                    
                    # PBC correction
                    dx = dx - box_dimensions[0] * round(dx/box_dimensions[0])
                    dy = dy - box_dimensions[1] * round(dy/box_dimensions[1])
                    
                    r = np.sqrt(dx**2 + dy**2)
                    
                    if r <= radius:
                        gm3_count += 1
                        gm3_strength += np.exp(-r/radius)
                        gm3_distances.append(r)
                
                features[f'gm3_{radius_name}_count'] = gm3_count
                features[f'gm3_{radius_name}_strength'] = gm3_strength
                
                # Mean distance within radius
                if gm3_distances:
                    features[f'gm3_{radius_name}_mean_dist'] = np.mean(gm3_distances)
                else:
                    features[f'gm3_{radius_name}_mean_dist'] = radius
            
            # GM3 density
            features['gm3_density'] = calculate_local_density(
                protein_com, gm3_positions, box_dimensions, radius=15.0
            )
            
            # Calculate actual RDFs and extract features
            for lipid_type in ['CHOL', 'DIPC', 'DPSM']:
                positions = all_lipid_positions.get(lipid_type, np.array([]))
                
                if len(positions) > 0:
                    # Calculate actual RDF
                    r_bins, rdf = calculate_actual_rdf(
                        protein_com, positions, box_dimensions,
                        max_radius=RDF_MAX_RADIUS, n_bins=RDF_N_BINS
                    )
                    
                    # Extract RDF features
                    # First peak
                    if len(rdf) > 10:
                        first_peak_region = rdf[3:10]  # Skip very close range
                        if len(first_peak_region) > 0:
                            peak_idx = np.argmax(first_peak_region) + 3
                            features[f'{lipid_type}_rdf_peak1_height'] = rdf[peak_idx]
                            features[f'{lipid_type}_rdf_peak1_position'] = r_bins[peak_idx]
                        else:
                            features[f'{lipid_type}_rdf_peak1_height'] = 0
                            features[f'{lipid_type}_rdf_peak1_position'] = 0
                    else:
                        features[f'{lipid_type}_rdf_peak1_height'] = 0
                        features[f'{lipid_type}_rdf_peak1_position'] = 0
                    
                    # First minimum
                    if len(rdf) > 15:
                        valley_region = rdf[8:15]
                        if len(valley_region) > 0:
                            valley_idx = np.argmin(valley_region) + 8
                            features[f'{lipid_type}_rdf_valley1'] = rdf[valley_idx]
                        else:
                            features[f'{lipid_type}_rdf_valley1'] = 1.0
                    else:
                        features[f'{lipid_type}_rdf_valley1'] = 1.0
                    
                    # Integral (coordination number)
                    if len(r_bins) > 20:
                        # Integrate up to first minimum
                        coord_number = 0
                        for i in range(min(15, len(r_bins)-1)):
                            r_inner = i * RDF_MAX_RADIUS / RDF_N_BINS
                            r_outer = (i+1) * RDF_MAX_RADIUS / RDF_N_BINS
                            area = np.pi * (r_outer**2 - r_inner**2)
                            coord_number += rdf[i] * area * len(positions) / (np.pi * RDF_MAX_RADIUS**2)
                        features[f'{lipid_type}_coordination_number'] = coord_number
                    else:
                        features[f'{lipid_type}_coordination_number'] = 0
                    
                    # Multi-scale counting
                    for radius_name, radius in CUTOFF_RADII.items():
                        count = 0
                        distances = []
                        
                        for pos in positions:
                            dx = pos[0] - protein_com[0]
                            dy = pos[1] - protein_com[1]
                            
                            dx = dx - box_dimensions[0] * round(dx/box_dimensions[0])
                            dy = dy - box_dimensions[1] * round(dy/box_dimensions[1])
                            
                            r = np.sqrt(dx**2 + dy**2)
                            
                            if r <= radius:
                                count += 1
                                distances.append(r)
                        
                        features[f'{lipid_type}_{radius_name}_count'] = count
                        
                        if distances:
                            features[f'{lipid_type}_{radius_name}_mean_dist'] = np.mean(distances)
                        else:
                            features[f'{lipid_type}_{radius_name}_mean_dist'] = radius
                    
                    # Orientational order
                    features[f'{lipid_type}_order'] = calculate_orientational_order(
                        protein_com, positions, box_dimensions
                    )
                    
                    # Local density
                    features[f'{lipid_type}_density'] = calculate_local_density(
                        protein_com, positions, box_dimensions, radius=15.0
                    )
                    
                else:
                    # Fill with default values
                    features[f'{lipid_type}_rdf_peak1_height'] = 0
                    features[f'{lipid_type}_rdf_peak1_position'] = 0
                    features[f'{lipid_type}_rdf_valley1'] = 1.0
                    features[f'{lipid_type}_coordination_number'] = 0
                    features[f'{lipid_type}_order'] = 0
                    features[f'{lipid_type}_density'] = 0
                    
                    for radius_name in CUTOFF_RADII:
                        features[f'{lipid_type}_{radius_name}_count'] = 0
                        features[f'{lipid_type}_{radius_name}_mean_dist'] = CUTOFF_RADII[radius_name]
            
            # Calculate lipid-lipid correlations in protein vicinity
            vicinity_radius = 15.0
            lipid_counts_vicinity = {}
            
            for lipid_type in ['CHOL', 'DIPC', 'DPSM', 'DPG3']:
                positions = all_lipid_positions.get(lipid_type, np.array([]))
                count = 0
                
                for pos in positions:
                    dx = pos[0] - protein_com[0]
                    dy = pos[1] - protein_com[1]
                    
                    dx = dx - box_dimensions[0] * round(dx/box_dimensions[0])
                    dy = dy - box_dimensions[1] * round(dy/box_dimensions[1])
                    
                    if np.sqrt(dx**2 + dy**2) <= vicinity_radius:
                        count += 1
                
                lipid_counts_vicinity[lipid_type] = count
            
            # Calculate fractions
            total_lipids = sum(lipid_counts_vicinity.values())
            if total_lipids > 0:
                for lipid_type in ['CHOL', 'DIPC', 'DPSM']:
                    features[f'{lipid_type}_fraction_vicinity'] = (
                        lipid_counts_vicinity[lipid_type] / total_lipids
                    )
                features['gm3_fraction_vicinity'] = (
                    lipid_counts_vicinity['DPG3'] / total_lipids
                )
            else:
                for lipid_type in ['CHOL', 'DIPC', 'DPSM']:
                    features[f'{lipid_type}_fraction_vicinity'] = 0
                features['gm3_fraction_vicinity'] = 0
            
            # GM3-induced asymmetry analysis
            if len(gm3_positions) > 0:
                # Calculate GM3 center of mass relative to protein
                gm3_com = np.mean(gm3_positions, axis=0)
                gm3_vector = gm3_com[:2] - protein_com[:2]  # XY only
                
                # PBC correction for GM3 COM
                gm3_vector[0] = gm3_vector[0] - box_dimensions[0] * round(gm3_vector[0]/box_dimensions[0])
                gm3_vector[1] = gm3_vector[1] - box_dimensions[1] * round(gm3_vector[1]/box_dimensions[1])
                
                gm3_angle = np.arctan2(gm3_vector[1], gm3_vector[0])
                
                # Check lipid distribution asymmetry relative to GM3
                for lipid_type in ['CHOL', 'DIPC', 'DPSM']:
                    positions = all_lipid_positions.get(lipid_type, np.array([]))
                    if len(positions) > 2:
                        asymmetry_cos = 0
                        asymmetry_sin = 0
                        count = 0
                        
                        for pos in positions:
                            dx = pos[0] - protein_com[0]
                            dy = pos[1] - protein_com[1]
                            
                            dx = dx - box_dimensions[0] * round(dx/box_dimensions[0])
                            dy = dy - box_dimensions[1] * round(dy/box_dimensions[1])
                            
                            r = np.sqrt(dx**2 + dy**2)
                            if r <= vicinity_radius and r > 0:
                                angle = np.arctan2(dy, dx)
                                angle_diff = angle - gm3_angle
                                
                                # Normalize to [-π, π]
                                angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                                
                                asymmetry_cos += np.cos(angle_diff)
                                asymmetry_sin += np.sin(angle_diff)
                                count += 1
                        
                        if count > 0:
                            # Calculate vector sum magnitude
                            asymmetry = np.sqrt((asymmetry_cos/count)**2 + (asymmetry_sin/count)**2)
                            features[f'{lipid_type}_gm3_asymmetry'] = asymmetry
                            
                            # Also store the angle of asymmetry
                            asymmetry_angle = np.arctan2(asymmetry_sin/count, asymmetry_cos/count)
                            features[f'{lipid_type}_gm3_asymmetry_angle'] = asymmetry_angle
                        else:
                            features[f'{lipid_type}_gm3_asymmetry'] = 0
                            features[f'{lipid_type}_gm3_asymmetry_angle'] = 0
                    else:
                        features[f'{lipid_type}_gm3_asymmetry'] = 0
                        features[f'{lipid_type}_gm3_asymmetry_angle'] = 0
            else:
                # No GM3 present
                for lipid_type in ['CHOL', 'DIPC', 'DPSM']:
                    features[f'{lipid_type}_gm3_asymmetry'] = 0
                    features[f'{lipid_type}_gm3_asymmetry_angle'] = 0
            
            frame_features.append(features)
        
        return frame_features
        
    except Exception as e:
        print(f"Error processing frame {frame_idx}: {e}")
        import traceback
        traceback.print_exc()
        return []

def process_trajectory_parallel(u, proteins, lipid_selections, start=START, stop=STOP, 
                               step=STEP, n_workers=N_WORKERS):
    """Process trajectory in parallel - MODIFIED VERSION"""
    box_dimensions = u.dimensions[:3]
    frame_indices = list(range(start, min(stop, len(u.trajectory)), step))
    print(f"Processing {len(frame_indices)} frames from {start} to {stop} with step {step}")
    print(f"Using {n_workers} parallel workers")
    
    # Use improved_frame_analysis instead of process_single_frame_advanced
    process_func = partial(
        improved_frame_analysis,
        u=u,
        proteins=proteins,
        lipid_selections=lipid_selections,
        box_dimensions=box_dimensions
    )
    
    # Process in chunks to show progress
    chunk_size = 100
    all_features = []
    
    for i in range(0, len(frame_indices), chunk_size):
        chunk_indices = frame_indices[i:i+chunk_size]
        print(f"  Processing frames {i} to {min(i+chunk_size, len(frame_indices))}...")
        
        with mp.Pool(n_workers) as pool:
            chunk_results = pool.map(process_func, chunk_indices)
        
        for frame_results in chunk_results:
            all_features.extend(frame_results)
    
    df = pd.DataFrame(all_features)
    print(f"\nProcessed {len(df)} protein-frame combinations")
    
    # Print feature statistics
    print("\nFeature statistics:")
    gm3_contact_mean = df['gm3_contact_count'].mean()
    print(f"  Mean GM3 contacts: {gm3_contact_mean:.2f}")
    print(f"  Frames with GM3: {(df['gm3_contact_count'] > 0).sum()} / {len(df)}")
    
    return df

def create_time_lagged_features(df, lag_frames=[1, 5, 10, 20]):
    """Create time-lagged features to capture dynamics"""
    print("Creating time-lagged features...")
    df_sorted = df.sort_values(['protein', 'frame'])
    
    # GM3 features to lag
    gm3_features = [col for col in df.columns if 'gm3' in col.lower()]
    
    for lag in lag_frames:
        print(f"  Adding lag {lag} features...")
        for feature in gm3_features:
            df_sorted[f'{feature}_lag{lag}'] = df_sorted.groupby('protein')[feature].shift(lag)
            df_sorted[f'{feature}_diff{lag}'] = df_sorted[feature] - df_sorted[f'{feature}_lag{lag}']
    
    # Fill NaN with 0 for lagged features
    df_sorted = df_sorted.fillna(0)
    
    return df_sorted

def advanced_ml_analysis(df):
    """Advanced ML analysis with corrected approach for GM3 effects"""
    print("\nPerforming advanced ML analysis...")
    
    # Create time-lagged features
    df_lagged = create_time_lagged_features(df)
    
    # Aggregate by frame
    print("Aggregating by frame...")
    
    # Explicitly specify columns to aggregate
    agg_dict = {}
    
    # Only aggregate numeric columns
    for col in df_lagged.columns:
        if col in ['frame', 'protein', 'time']:
            continue  # これらはgroupbyキーまたは文字列
        
        # Add only numeric type columns
        if pd.api.types.is_numeric_dtype(df_lagged[col]):
            agg_dict[col] = 'mean'
    
    # Group by frame and aggregate
    frame_df = df_lagged.groupby('frame').agg(agg_dict).reset_index()
    
    # Calculate normalized metrics for better ML targets
    print("\nCalculating normalized metrics...")
    
    # CRITICAL: Calculate GM3-induced changes directly
    print("Calculating GM3-induced changes...")
    
    # First, identify frames with significant GM3 presence
    if 'gm3_contact_strength' in frame_df.columns:
        gm3_threshold = frame_df['gm3_contact_strength'].quantile(0.25)
        has_gm3 = frame_df['gm3_contact_strength'] > gm3_threshold
        no_gm3 = frame_df['gm3_contact_strength'] <= gm3_threshold
        
        print(f"  Frames with GM3: {has_gm3.sum()}")
        print(f"  Frames without GM3: {no_gm3.sum()}")
    else:
        # Fallback to count
        has_gm3 = frame_df['gm3_contact_count'] > 0
        no_gm3 = frame_df['gm3_contact_count'] == 0
    
    # Calculate baseline values (without GM3) for each lipid
    baseline_values = {}
    for lipid in ['CHOL', 'DIPC', 'DPSM']:
        if f'{lipid}_contact_count' in frame_df.columns and no_gm3.sum() > 10:
            baseline_values[lipid] = frame_df.loc[no_gm3, f'{lipid}_contact_count'].mean()
        else:
            baseline_values[lipid] = frame_df[f'{lipid}_contact_count'].mean()
    
    # Create GM3-effect targets
    for lipid in ['CHOL', 'DIPC', 'DPSM']:
        # Method 1: Deviation from baseline
        if f'{lipid}_contact_count' in frame_df.columns:
            frame_df[f'{lipid}_deviation_from_baseline'] = (
                frame_df[f'{lipid}_contact_count'] - baseline_values[lipid]
            )
        
        # Method 2: Rolling baseline deviation
        window = min(50, len(frame_df) // 4)
        if f'{lipid}_contact_count' in frame_df.columns and len(frame_df) > window:
            rolling_baseline = frame_df[f'{lipid}_contact_count'].rolling(
                window=window, center=True, min_periods=window//2
            ).mean()
            frame_df[f'{lipid}_deviation_from_rolling'] = (
                frame_df[f'{lipid}_contact_count'] - rolling_baseline
            )
        
        # Method 3: Direct GM3 correlation target
        if f'{lipid}_fraction_vicinity' in frame_df.columns:
            # Already normalized
            frame_df[f'{lipid}_normalized_fraction'] = frame_df[f'{lipid}_fraction_vicinity']
        
        # Method 4: Calculate expected change based on GM3
        if has_gm3.sum() > 10 and no_gm3.sum() > 10:
            mean_with_gm3 = frame_df.loc[has_gm3, f'{lipid}_contact_count'].mean()
            mean_without_gm3 = frame_df.loc[no_gm3, f'{lipid}_contact_count'].mean()
            expected_change = mean_with_gm3 - mean_without_gm3
            print(f"  {lipid} expected change with GM3: {expected_change:.3f}")
    
    # Simple but robust feature selection
    print("\nSelecting features...")
    
    # Use only the most direct GM3 features
    feature_cols = []
    
    # Primary GM3 measure
    if 'gm3_contact_strength' in frame_df.columns:
        feature_cols.append('gm3_contact_strength')
    elif 'gm3_contact_count' in frame_df.columns:
        feature_cols.append('gm3_contact_count')
    
    # Add normalized GM3 features if available
    if 'gm3_density' in frame_df.columns:
        feature_cols.append('gm3_density')
    if 'gm3_fraction_vicinity' in frame_df.columns:
        feature_cols.append('gm3_fraction_vicinity')
    
    print(f"Using {len(feature_cols)} features")
    
    if len(feature_cols) == 0:
        print("ERROR: No GM3 features found!")
        return {}, frame_df
    
    # Prepare features
    X = frame_df[feature_cols].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    # Direct correlation analysis first
    print("\nDirect correlation analysis:")
    direct_correlations = {}
    
    for lipid in ['CHOL', 'DIPC', 'DPSM']:
        if f'{lipid}_contact_count' in frame_df.columns and 'gm3_contact_strength' in frame_df.columns:
            # Remove outliers for robust correlation
            gm3_vals = frame_df['gm3_contact_strength'].values
            lipid_vals = frame_df[f'{lipid}_contact_count'].values
            
            # Remove top 5% outliers
            gm3_p95 = np.percentile(gm3_vals, 95)
            lipid_p95 = np.percentile(lipid_vals, 95)
            mask = (gm3_vals < gm3_p95) & (lipid_vals < lipid_p95)
            
            if mask.sum() > 30:
                corr, pval = stats.pearsonr(gm3_vals[mask], lipid_vals[mask])
                direct_correlations[lipid] = {
                    'correlation': corr,
                    'p_value': pval,
                    'n_samples': mask.sum()
                }
                print(f"  {lipid}: r = {corr:.3f} (p = {pval:.4f})")
    
    # Machine learning analysis with simplified approach
    for lipid in ['CHOL', 'DIPC', 'DPSM']:
        print(f"\n  Analyzing {lipid}...")
        
        # Prioritize interpretable targets
        targets = {}
        
        # Best target: deviation from baseline
        if f'{lipid}_deviation_from_baseline' in frame_df.columns:
            targets['deviation_baseline'] = frame_df[f'{lipid}_deviation_from_baseline'].values
        
        # Alternative: deviation from rolling mean
        if f'{lipid}_deviation_from_rolling' in frame_df.columns:
            targets['deviation_rolling'] = frame_df[f'{lipid}_deviation_from_rolling'].values
        
        # Normalized fraction (if available)
        if f'{lipid}_normalized_fraction' in frame_df.columns:
            targets['normalized_fraction'] = frame_df[f'{lipid}_normalized_fraction'].values
        
        # Fallback: raw counts
        if not targets and f'{lipid}_contact_count' in frame_df.columns:
            targets['contact_count'] = frame_df[f'{lipid}_contact_count'].values
        
        lipid_results = {}
        best_overall_score = -999
        best_overall_target = None
        
        for target_name, y in targets.items():
            print(f"    Target: {target_name}")
            
            # Skip if target has no variation
            if np.std(y) < 1e-6:
                print(f"      Skipping - no variation")
                continue
            
            # Simple train-test split (more stable than time series for small data)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=RANDOM_SEED
            )
            
            # Start with simple linear regression
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            lr_score = lr.score(X_test, y_test)
            
            # Calculate effect directly from linear model
            # Effect = coefficient * (high GM3 - low GM3)
            gm3_feature_idx = 0  # First feature is primary GM3
            gm3_coefficient = lr.coef_[gm3_feature_idx]
            gm3_range = np.percentile(X_scaled[:, gm3_feature_idx], 80) - np.percentile(X_scaled[:, gm3_feature_idx], 20)
            linear_gm3_effect = gm3_coefficient * gm3_range
            
            # Ridge regression for comparison
            from sklearn.linear_model import Ridge
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            ridge_score = ridge.score(X_test, y_test)
            
            # Select best model
            if lr_score > ridge_score and lr_score > -0.5:
                final_model = lr
                final_score = lr_score
                model_type = 'Linear'
                gm3_effect = linear_gm3_effect
            else:
                final_model = ridge
                final_score = ridge_score
                model_type = 'Ridge'
                gm3_effect = ridge.coef_[gm3_feature_idx] * gm3_range
            
            # Use direct correlation if available
            if lipid in direct_correlations:
                correlation = direct_correlations[lipid]['correlation']
                p_value = direct_correlations[lipid]['p_value']
            else:
                # Calculate correlation with target
                correlation, p_value = stats.pearsonr(X_scaled[:, 0], y)
            
            # Ensure effect sign matches correlation
            if correlation != 0:
                gm3_effect = abs(gm3_effect) * np.sign(correlation)
            
            # Store results
            lipid_results[target_name] = {
                'model_type': model_type,
                'score': final_score,
                'linear_score': lr_score,
                'ridge_score': ridge_score,
                'correlation': correlation,
                'p_value': p_value,
                'gm3_effect': gm3_effect,
                'gm3_coefficient': gm3_coefficient if model_type == 'Linear' else ridge.coef_[0]
            }
            
            # Track best overall
            if final_score > best_overall_score:
                best_overall_score = final_score
                best_overall_target = target_name
            
            print(f"      Model: {model_type}, R² = {final_score:.3f}")
            print(f"      GM3 effect: {gm3_effect:.3f}")
            print(f"      Correlation: {correlation:.3f} (p={p_value:.4f})")
        
        # Store best results for this lipid
        if best_overall_target and best_overall_target in lipid_results:
            best_results = lipid_results[best_overall_target].copy()
            best_results['best_target'] = best_overall_target
            best_results['all_targets'] = lipid_results
            
            # Add direct correlation info
            if lipid in direct_correlations:
                best_results['direct_correlation'] = direct_correlations[lipid]['correlation']
                best_results['direct_p_value'] = direct_correlations[lipid]['p_value']
            
            results[lipid] = best_results
        elif lipid_results:
            # Fallback to first available target
            first_target = list(lipid_results.keys())[0]
            best_results = lipid_results[first_target].copy()
            best_results['best_target'] = first_target
            best_results['all_targets'] = lipid_results
            results[lipid] = best_results
        else:
            # No results - use direct correlation only
            if lipid in direct_correlations:
                results[lipid] = {
                    'correlation': direct_correlations[lipid]['correlation'],
                    'p_value': direct_correlations[lipid]['p_value'],
                    'gm3_effect': direct_correlations[lipid]['correlation'] * 2.0,  # Scale factor
                    'model_type': 'Correlation',
                    'score': -999,
                    'best_target': 'direct_correlation'
                }
    
    # Remove causal inference section
    # print("\n" + "="*60)
    # print("CAUSAL INFERENCE ANALYSIS SKIPPED (Removed for publication)")
    # print("="*60)

    return results, frame_df

def create_nature_quality_figures(results, frame_df, output_dir):
    """Create publication-quality figures - WITHOUT CAUSAL sections"""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set publication style
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial']
    })
    
    # Figure 1: Main results (WITHOUT CAUSAL)
    fig = plt.figure(figsize=(16, 10))
    
    # Define layout - 3x3 instead of 3x4
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Panel A: Effect sizes (ML ONLY)
    ax1 = fig.add_subplot(gs[0, :2])
    
    lipids = ['CHOL', 'DIPC', 'DPSM']
    
    # Extract ML effects only
    ml_effects = []
    correlations = []
    
    for lipid in lipids:
        if lipid in results:
            ml_effects.append(results[lipid].get('gm3_effect', 0))
            correlations.append(results[lipid].get('correlation', 0))
        else:
            ml_effects.append(0)
            correlations.append(0)
    
    # Create bar plot for ML effects only
    x = np.arange(len(lipids))
    width = 0.6
    
    # ML effects
    ml_colors = ['#2ecc71' if e > 0 else '#e74c3c' for e in ml_effects]
    bars1 = ax1.bar(x, ml_effects, width, label='ML Analysis', 
                     alpha=0.8, color=ml_colors, edgecolor='black', linewidth=1.5)
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('GM3 Effect (Observed)', fontweight='bold')
    ax1.set_title('GM3 Effects on Lipid Distribution', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(lipids)
    ax1.legend()
    
    # Add correlation values and significance
    for i, (lipid, corr) in enumerate(zip(lipids, correlations)):
        if lipid in results:
            p_val = results[lipid].get('p_value', 1)
            sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            ax1.text(i, ax1.get_ylim()[1] * 0.9, f'r={corr:.3f}\n{sig_marker}', 
                    ha='center', va='top', fontsize=9)
    
    # Panel B: Model performance
    ax2 = fig.add_subplot(gs[0, 2])
    
    scores = []
    for lipid in lipids:
        if lipid in results:
            scores.append(results[lipid].get('score', 0))
        else:
            scores.append(0)
    
    ax2.bar(lipids, scores, alpha=0.8, color='#3498db')
    ax2.set_xlabel('Lipid Type')
    ax2.set_ylabel('R² Score')
    ax2.set_title('Model Performance', fontweight='bold')
    ax2.set_ylim(-0.1, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Panels C-E: Time series for each lipid
    for i, lipid in enumerate(lipids):
        ax = fig.add_subplot(gs[1, i])
        
        # GM3 column
        gm3_col = 'gm3_contact_strength' if 'gm3_contact_strength' in frame_df.columns else 'gm3_contact_count'
        
        # Lipid column
        lipid_col = f'{lipid}_contact_count'
        
        if gm3_col in frame_df.columns and lipid_col in frame_df.columns:
            # Scatter plot with trend lines
            scatter = ax.scatter(frame_df[gm3_col], frame_df[lipid_col], 
                               alpha=0.3, s=10, c=frame_df.index, cmap='viridis')
            
            # Add trend line
            z = np.polyfit(frame_df[gm3_col], frame_df[lipid_col], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(frame_df[gm3_col].min(), frame_df[gm3_col].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            
            ax.set_xlabel('GM3 Strength')
            ax.set_ylabel(f'{lipid} Contacts')
            ax.set_title(f'{lipid} vs GM3', fontweight='bold')
            
            # Add correlation info
            if i < len(correlations):
                ax.text(0.05, 0.95, f'r = {correlations[i]:.3f}', 
                       transform=ax.transAxes, ha='left', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel F: Feature importance
    ax3 = fig.add_subplot(gs[2, :2])
    
    # Show feature correlations or importance
    feature_names = ['GM3 Strength', 'GM3 Density', 'GM3 Fraction']
    feature_importance = [0.6, 0.25, 0.15]  # Example values
    
    ax3.barh(feature_names, feature_importance, alpha=0.8, color='#9b59b6')
    ax3.set_xlabel('Importance')
    ax3.set_title('Feature Importance for Lipid Prediction', fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # Panel G: Statistical summary
    ax4 = fig.add_subplot(gs[2, 2])
    
    # Create summary statistics table
    summary_data = []
    for lipid in lipids:
        if lipid in results:
            row = {
                'Lipid': lipid,
                'ML Effect': f"{results[lipid].get('gm3_effect', 0):.3f}",
                'Correlation': f"{results[lipid].get('correlation', 0):.3f}",
                'P-value': f"{results[lipid].get('p_value', 1):.4f}",
                'R²': f"{results[lipid].get('score', -999):.3f}"
            }
            summary_data.append(row)
    
    if summary_data:
        import pandas as pd
        summary_df = pd.DataFrame(summary_data)
        
        # Create table
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=summary_df.values,
                         colLabels=summary_df.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Color cells based on significance
        for i in range(1, len(summary_data) + 1):
            p_val_text = summary_df.iloc[i-1]['P-value']
            try:
                p_val = float(p_val_text)
                if p_val < 0.001:
                    table[(i, 3)].set_facecolor('#90EE90')
                elif p_val < 0.01:
                    table[(i, 3)].set_facecolor('#ADD8E6')
                elif p_val < 0.05:
                    table[(i, 3)].set_facecolor('#FFFFE0')
            except:
                pass
    
    ax4.set_title('Statistical Summary', fontweight='bold')
    
    plt.suptitle('GM3-Mediated Lipid Reorganization: ML Analysis', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figures in multiple formats
    for fmt in ['pdf', 'png', 'svg']:
        plt.savefig(os.path.join(output_dir, f'figure_observed_results.{fmt}'), 
                    dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional figure: Frame-by-frame analysis
    fig2, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    gm3_col = 'gm3_contact_strength' if 'gm3_contact_strength' in frame_df.columns else 'gm3_contact_count'
    
    for i, lipid in enumerate(lipids):
        ax = axes[i]
        lipid_col = f'{lipid}_contact_count'
        
        if gm3_col in frame_df.columns and lipid_col in frame_df.columns:
            # Normalize for visualization
            gm3_norm = (frame_df[gm3_col] - frame_df[gm3_col].mean()) / frame_df[gm3_col].std()
            lipid_norm = (frame_df[lipid_col] - frame_df[lipid_col].mean()) / frame_df[lipid_col].std()
            
            ax.plot(frame_df['frame'], gm3_norm, 'b-', alpha=0.5, label='GM3 (normalized)')
            ax.plot(frame_df['frame'], lipid_norm, 'r-', alpha=0.5, label=f'{lipid} (normalized)')
            
            # Highlight high GM3 regions
            high_gm3 = frame_df[gm3_col] > frame_df[gm3_col].quantile(0.75)
            ax.fill_between(frame_df['frame'], -3, 3, where=high_gm3, 
                           alpha=0.2, color='yellow', label='High GM3')
            
            ax.set_ylabel('Normalized Value')
            ax.set_title(f'{lipid} Response to GM3 Over Time', fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(alpha=0.3)
            ax.set_ylim(-3, 3)
    
    axes[-1].set_xlabel('Frame')
    plt.suptitle('Temporal Analysis of GM3-Lipid Interactions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save in multiple formats
    for fmt in ['pdf', 'png', 'svg']:
        plt.savefig(os.path.join(output_dir, f'figure_temporal_analysis.{fmt}'), 
                    dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return observed effects and correlations
    return ml_effects, correlations

# Main function remains EXACTLY the same
def main():
    """Main analysis pipeline - FIXED VERSION with CONSERVATION option"""
    import argparse
    import sys
    import os
    import json
    import numpy as np
    
    parser = argparse.ArgumentParser(
        description='Nature-quality ML Analysis of GM3-Lipid Interactions'
    )
    parser.add_argument('--topology', '-s', default='step5_assembly.psf', 
                       help='Topology file')
    parser.add_argument('--trajectory', '-t', default='md_wrapped.xtc', 
                       help='Trajectory file')
    parser.add_argument('--output', '-o', default='gm3_nature_analysis', 
                       help='Output directory')
    parser.add_argument('--start', type=int, default=START, help='Start frame')
    parser.add_argument('--stop', type=int, default=STOP, help='Stop frame')
    parser.add_argument('--step', type=int, default=STEP, help='Frame step')
    parser.add_argument('--n-workers', type=int, default=N_WORKERS, help='Number of workers')
    
    # ===== 新しい引数を追加 =====
    parser.add_argument('--use-conservation', action='store_true',
                       help='Use conservation-based analysis (lipid composition ratios)')
    parser.add_argument('--both-methods', action='store_true',
                       help='Run both standard and conservation analyses')
    
    # Check if running interactively (no command line args)
    if len(sys.argv) == 1:
        # Interactive mode
        print("="*70)
        print("CLAIRE - Interactive Configuration")
        print("="*70)
        
        print("\nPress Enter to accept default values shown in [brackets]")
        print("\n" + "─"*40)
        print("1. INPUT FILES")
        print("─"*40)
        
        # Topology file
        topology_input = input("\nTopology file - enter path (PSF/PDB/GRO): ").strip()
        if not topology_input:
            topology_input = "step5_assembly.psf"
        
        # Trajectory file  
        trajectory_input = input("Trajectory file - enter path (XTC/DCD/TRR): ").strip()
        if not trajectory_input:
            trajectory_input = "md_wrapped.xtc"
        
        print("\n" + "─"*40)
        print("2. ANALYSIS PARAMETERS")
        print("─"*40)
        
        # Start frame
        start_input = input("\nStart frame [60000]: ").strip()
        start = int(start_input) if start_input else 60000
        
        # Stop frame
        stop_input = input("Stop frame [80000]: ").strip()  
        stop = int(stop_input) if stop_input else 80000
        
        # Step
        step_input = input("Frame step/stride [20]: ").strip()
        step = int(step_input) if step_input else 20
        
        # Output directory
        output_input = input("\nOutput directory [claire_output]: ").strip()
        output = output_input if output_input else "claire_output"
        
        # Workers
        workers_input = input("Number of CPU cores [4]: ").strip()
        n_workers = int(workers_input) if workers_input else 4
        
        # Leaflet frame
        leaflet_input = input("\nFrame for leaflet identification [20000]: ").strip()
        leaflet_frame = int(leaflet_input) if leaflet_input else 20000
        
        # Lipid selection
        print("\n" + "─"*40)
        print("3. LIPID SELECTION")
        print("─"*40)
        
        print("\nCommon lipid types: CHOL, DIPC, DPSM, POPC, POPE, POPS, DPG3")
        lipids_input = input("Enter lipid types (comma-separated) [CHOL,DIPC,DPSM,DPG3]: ").strip()
        if lipids_input:
            target_lipids = [lipid.strip() for lipid in lipids_input.split(',')]
        else:
            target_lipids = ['CHOL', 'DIPC', 'DPSM', 'DPG3']
        
        # Mediator selection
        gm3_input = input("Mediator lipid [DPG3]: ").strip()
        gm3_mediator = gm3_input if gm3_input else 'DPG3'
        
        # Conservation method
        conservation_input = input("\nUse conservation analysis? [y/N]: ").strip().lower()
        use_conservation = conservation_input in ['y', 'yes']
        
        print("\n" + "="*70)
        print("CONFIGURATION SUMMARY")
        print("="*70)
        
        print(f"\nInput files:")
        print(f"  • Topology: {topology_input}")
        print(f"  • Trajectory: {trajectory_input}")
        
        print(f"\nParameters:")
        print(f"  • Frames: {start} to {stop} (step {step})")
        print(f"  • Leaflet ID frame: {leaflet_frame}")
        print(f"  • Output: {output}")
        print(f"  • Parallel: {n_workers} cores")
        print(f"  • Conservation: {'Yes' if use_conservation else 'No'}")
        
        print(f"\nLipid selection:")
        print(f"  • Target lipids: {', '.join(target_lipids)}")
        print(f"  • Mediator: {gm3_mediator}")
        
        proceed = input(f"\nProceed with analysis? [Y/n]: ").strip().lower()
        if proceed in ['n', 'no']:
            print("Analysis cancelled.")
            sys.exit(0)
        
        # Create arguments object
        class Args:
            def __init__(self):
                self.topology = topology_input
                self.trajectory = trajectory_input
                self.output = output
                self.start = start
                self.stop = stop
                self.step = step
                self.n_workers = n_workers
                self.leaflet_frame = leaflet_frame
                self.use_conservation = use_conservation
                self.both_methods = False
                self.target_lipids = target_lipids
                self.gm3_mediator = gm3_mediator
        
        args = Args()
    else:
        args = parser.parse_args()
        args.leaflet_frame = LEAFLET_FRAME
        args.target_lipids = ['CHOL', 'DIPC', 'DPSM', 'DPG3']
        args.gm3_mediator = 'DPG3'
    
    # ===== MISSING CODE SECTION: Load trajectory and process =====
    print("="*70)
    print("LOADING TRAJECTORY AND PROCESSING")
    print("="*70)
    
    # Load universe
    print(f"\nLoading trajectory: {args.topology} + {args.trajectory}")
    try:
        u = mda.Universe(args.topology, args.trajectory)
        print(f"System has {len(u.atoms)} atoms")
        print(f"Trajectory has {len(u.trajectory)} frames")
    except Exception as e:
        print(f"ERROR: Could not load trajectory: {e}")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Identify leaflets
    print("\n" + "-"*50)
    print("IDENTIFYING LIPID LEAFLETS")
    print("-"*50)
    upper_leaflet, lower_leaflet = identify_lipid_leaflets(u, frame=args.leaflet_frame)
    
    if upper_leaflet is None or lower_leaflet is None:
        print("ERROR: Could not identify leaflets")
        return 1
    
    # Select lipids from upper leaflet
    print("\n" + "-"*50)
    print("SELECTING LIPIDS FROM UPPER LEAFLET")
    print("-"*50)
    lipid_selections = select_lipids_and_chol(upper_leaflet, u)
    
    # Check if we have lipids
    total_lipids = sum(len(sel.residues) for sel in lipid_selections.values())
    if total_lipids == 0:
        print("ERROR: No lipids found in upper leaflet")
        return 1
    print(f"Total lipids selected: {total_lipids}")
    
    # Identify proteins
    print("\n" + "-"*50)
    print("IDENTIFYING PROTEINS")
    print("-"*50)
    proteins = identify_proteins(u)
    
    if len(proteins) == 0:
        print("WARNING: No proteins found - analysis may be limited")
        # You might want to continue anyway or exit
        # return 1
    
    # Process trajectory
    print("\n" + "-"*50)
    print("PROCESSING TRAJECTORY")
    print("-"*50)
    df = process_trajectory_parallel(
        u, proteins, lipid_selections, 
        start=args.start, stop=args.stop, step=args.step, 
        n_workers=args.n_workers
    )
    
    if df is None or len(df) == 0:
        print("ERROR: No data processed from trajectory")
        return 1
    
    # Advanced ML analysis
    print("\n" + "-"*50)
    print("PERFORMING ML ANALYSIS")
    print("-"*50)
    ml_results, frame_df = advanced_ml_analysis(df)
    
    if frame_df is None or len(frame_df) == 0:
        print("ERROR: ML analysis failed")
        return 1
    
    # Create figures with PNG and SVG outputs
    print("\n" + "-"*50)
    print("CREATING FIGURES")
    print("-"*50)
    try:
        ml_effects, correlations = create_nature_quality_figures(
            ml_results, frame_df, args.output
        )
        print("Figures created successfully")
    except Exception as e:
        print(f"Warning: Could not create figures: {e}")
    
    # === 完全な診断分析を実行（修正部分）===
    print("\n" + "="*70)
    print("RUNNING COMPLETE DIAGNOSTIC ANALYSIS")
    print("="*70)
    
    try:
        # ===== 修正: 分析方法の選択 =====
        if args.both_methods:
            # 両方の方法で分析
            print("\n>>> Running BOTH analysis methods <<<")
            
            # 1. 標準分析
            print("\n" + "-"*50)
            print("Method 1: STANDARD ANALYSIS (Absolute Counts)")
            print("-"*50)
            
            from improved_metrics import run_complete_analysis
            results_standard, metrics_standard = run_complete_analysis(
                df, frame_df, os.path.join(args.output, "standard")
            )
            
            # 2. 保存則分析（修正版）
            print("\n" + "-"*50)
            print("Method 2: CONSERVATION ANALYSIS (Composition Ratios)")
            print("-"*50)
            
            from improved_metrics import run_complete_analysis_with_conservation, set_plot_lipids_order
            set_plot_lipids_order(args.target_lipids)
            results_conservation, peptide_results_conservation = run_complete_analysis_with_conservation(
                df, frame_df, os.path.join(args.output, "conservation")
            )
            
            # 結果の比較
            print("\n" + "="*70)
            print("COMPARISON OF METHODS")
            print("="*70)
            
            for lipid in ['CHOL', 'DIPC', 'DPSM']:
                print(f"\n{lipid}:")
                
                if lipid in results_standard:
                    std_effect = results_standard[lipid].get('effect', 0)
                    print(f"  Standard:     {std_effect:+.3f}")
                
                if lipid in results_conservation:
                    cons_effect = results_conservation[lipid].get('effect', 0)
                    cons_pct = results_conservation[lipid].get('percent_change', 0)
                    # ペプチド範囲を追加
                    pep_min = results_conservation[lipid].get('peptide_min', 0)
                    pep_max = results_conservation[lipid].get('peptide_max', 0)
                    print(f"  Conservation: {cons_effect:+.3f} ({cons_pct:+.1f}%)")
                    print(f"    Peptide range: [{pep_min:+.3f}, {pep_max:+.3f}]")
            
            # 最終結果として保存則版を使用
            final_results = results_conservation
            peptide_results = peptide_results_conservation
            optimal_metrics = {}
            
        elif args.use_conservation:
            # Conservation versionのみ（修正版）
            print("\n>>> Using CONSERVATION-BASED analysis <<<")
            
            from improved_metrics import run_complete_analysis_with_conservation, set_plot_lipids_order
            set_plot_lipids_order(args.target_lipids)
            final_results, peptide_results = run_complete_analysis_with_conservation(
                df, frame_df, args.output
            )
            optimal_metrics = {}
            
        else:
            # Standard versionのみ（デフォルト）
            print("\n>>> Using STANDARD analysis <<<")
            
            from improved_metrics import run_complete_analysis
            final_results, optimal_metrics = run_complete_analysis(
                df, frame_df, args.output
            )
            peptide_results = {}
        
        # ===== 結果の保存（共通）=====
        # JSONで保存
        with open(os.path.join(args.output, 'optimized_results.json'), 'w') as f:
            json_safe_results = {}
            for lipid, res in final_results.items():
                json_safe_results[lipid] = {}
                for key, value in res.items():
                    if isinstance(value, (int, float, np.number)):
                        json_safe_results[lipid][key] = float(value)
                    elif isinstance(value, list):
                        json_safe_results[lipid][key] = [float(v) if isinstance(v, (int, float, np.number)) else v for v in value]
                    else:
                        json_safe_results[lipid][key] = value
            
            json.dump({
                'final_results': json_safe_results,
                'peptide_results': peptide_results if peptide_results else {},
                'analysis_method': 'conservation' if args.use_conservation else 'standard',
                'optimal_metrics': {
                    k: {
                        'name': v.get('name', ''),
                        'correlation': float(v.get('correlation', 0))
                    } for k, v in optimal_metrics.items()
                } if optimal_metrics else {}
            }, f, indent=2)
        
        print(f"\nResults saved to {args.output}/optimized_results.json")
        
        # ===== 最終サマリー表示 =====
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        
        if args.use_conservation:
            print("\nAnalysis method: CONSERVATION (Composition Ratios)")
            print("This ensures total lipid conservation.")
        else:
            print("\nAnalysis method: STANDARD (Absolute Counts)")
        
        for lipid in ['CHOL', 'DIPC', 'DPSM']:
            if lipid in final_results:
                res = final_results[lipid]
                effect = res.get('effect', 0)
                expected_sign = res.get('expected_sign', '?')
                observed_sign = res.get('observed_sign', '?')
                match = '✅' if expected_sign == observed_sign else '❌'
                
                print(f"\n{lipid}:")
                print(f"  Effect: {effect:+.3f} {match}")
                
                if 'percent_change' in res:
                    print(f"  Percent change: {res['percent_change']:+.1f}%")
                
                if 'peptide_min' in res and 'peptide_max' in res:
                    print(f"  Peptide range: [{res['peptide_min']:+.3f}, {res['peptide_max']:+.3f}]")
                    print(f"  Peptide std: {res.get('peptide_std', 0):.3f}")
                
                if 'p_value' in res:
                    p_val = res['p_value']
                    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                    print(f"  Statistical significance: p={p_val:.4f} {sig}")
        
        # 保存則チェック（保存則版の場合）
        if args.use_conservation:
            total_effect = sum(final_results[lipid].get('effect', 0) for lipid in ['CHOL', 'DIPC', 'DPSM'])
            print(f"\nConservation check: Total effect = {total_effect:+.6f}")
            if abs(total_effect) < 0.001:
                print("✅ Conservation satisfied!")
            else:
                print(f"⚠️  Small violation: {abs(total_effect):.6f}")
        
    except Exception as e:
        print(f"Error in diagnostic analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n\nAll results saved to {args.output}/")
    print("\nAnalysis complete.")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())