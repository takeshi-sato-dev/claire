#!/usr/bin/env python3
"""
Frame-by-frame analysis processor - EXACT REPLICA of original_analysis_no_causal.py
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# Global parameters - EXACT COPY FROM ORIGINAL_ANALYSIS
CUTOFF_RADII = {
    'contact': 6.0,
    'first_shell': 10.0,
    'influence': 15.0
}

# RDF parameters
RDF_MAX_RADIUS = 20.0
RDF_N_BINS = 40


def calculate_actual_rdf(protein_com, lipid_positions, box_dimensions, 
                        max_radius=RDF_MAX_RADIUS, n_bins=RDF_N_BINS):
    """Calculate actual radial distribution function - EXACT COPY FROM ORIGINAL_ANALYSIS"""
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
    """Calculate orientational order parameter around protein - EXACT COPY FROM ORIGINAL_ANALYSIS"""
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
    """Calculate local density of lipids around protein - EXACT COPY FROM ORIGINAL_ANALYSIS"""
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
    """Advanced frame processing with actual physics calculations - EXACT COPY FROM ORIGINAL_ANALYSIS"""
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


class FrameProcessor:
    """
    Wrapper class for compatibility with existing CLAIRE structure
    """
    
    @staticmethod
    def process_frame_complete(frame_idx: int, 
                              universe,
                              proteins: Dict,
                              lipid_selections: Dict,
                              box_dimensions: np.ndarray,
                              mediator_lipid: str = 'DPG3',
                              target_lipids: Optional[List[str]] = None) -> List[Dict]:
        """
        Use the exact original_analysis processing function
        """
        return process_single_frame_advanced(frame_idx, universe, proteins, lipid_selections, box_dimensions)