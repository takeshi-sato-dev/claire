#!/usr/bin/env python3
"""
Frame-by-frame analysis processor
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from ..physics.calculations import PhysicsCalculator


class FrameProcessor:
    """
    Process individual frames with full physics analysis
    """
    
    # From original code
    CUTOFF_RADII = {
        'contact': 6.0,
        'first_shell': 10.0,
        'influence': 15.0
    }
    
    @staticmethod
    def process_frame_complete(frame_idx: int, 
                              universe,
                              proteins: Dict,
                              lipid_selections: Dict,
                              box_dimensions: np.ndarray,
                              mediator_lipid: str = 'DPG3',
                              target_lipids: Optional[List[str]] = None) -> List[Dict]:
        """
        Complete frame processing from original code
        
        Parameters
        ----------
        frame_idx : int
            Frame index
        universe : MDAnalysis.Universe
            Universe object
        proteins : dict
            Dictionary of proteins
        lipid_selections : dict
            Dictionary of lipid selections
        box_dimensions : numpy.ndarray
            Box dimensions
        mediator_lipid : str
            Mediator lipid name
        target_lipids : list, optional
            List of target lipid names. If None, use all non-mediator lipids
        """
        try:
            universe.trajectory[frame_idx]
            frame_features = []
            physics = PhysicsCalculator()
            
            # Determine target lipids dynamically
            if target_lipids is None:
                target_lipids = [lip for lip in lipid_selections.keys() 
                               if lip != mediator_lipid]
            
            for protein_name, protein_sel in proteins.items():
                protein_com = protein_sel.center_of_mass()
                
                features = {
                    'frame': frame_idx,
                    'protein': protein_name,
                    'time': universe.trajectory.time
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
                
                # Mediator (e.g., GM3) analysis
                mediator_positions = all_lipid_positions.get(mediator_lipid, np.array([]))
                
                # Multi-scale mediator features
                for radius_name, radius in FrameProcessor.CUTOFF_RADII.items():
                    mediator_count = 0
                    mediator_strength = 0
                    mediator_distances = []
                    
                    for pos in mediator_positions:
                        dx = pos[0] - protein_com[0]
                        dy = pos[1] - protein_com[1]
                        
                        # PBC correction
                        dx = dx - box_dimensions[0] * round(dx/box_dimensions[0])
                        dy = dy - box_dimensions[1] * round(dy/box_dimensions[1])
                        
                        r = np.sqrt(dx**2 + dy**2)
                        
                        if r <= radius:
                            mediator_count += 1
                            mediator_strength += np.exp(-r/radius)
                            mediator_distances.append(r)
                    
                    # Use generic naming for mediator
                    features[f'{mediator_lipid.lower()}_{radius_name}_count'] = mediator_count
                    features[f'{mediator_lipid.lower()}_{radius_name}_strength'] = mediator_strength
                    
                    if mediator_distances:
                        features[f'{mediator_lipid.lower()}_{radius_name}_mean_dist'] = np.mean(mediator_distances)
                    else:
                        features[f'{mediator_lipid.lower()}_{radius_name}_mean_dist'] = radius
                
                # Mediator density
                features[f'{mediator_lipid.lower()}_density'] = physics.calculate_local_density(
                    protein_com, mediator_positions, box_dimensions, radius=15.0
                )
                
                # For backward compatibility with gm3 naming
                if mediator_lipid.upper() in ['DPG3', 'GM3']:
                    features['gm3_contact_count'] = features[f'{mediator_lipid.lower()}_contact_count']
                    features['gm3_contact_strength'] = features[f'{mediator_lipid.lower()}_contact_strength']
                    features['gm3_density'] = features[f'{mediator_lipid.lower()}_density']
                
                # For each target lipid (dynamic list)
                for lipid_type in target_lipids:
                    positions = all_lipid_positions.get(lipid_type, np.array([]))
                    
                    if len(positions) > 0:
                        # Multi-scale counting
                        for radius_name, radius in FrameProcessor.CUTOFF_RADII.items():
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
                        features[f'{lipid_type}_order'] = physics.calculate_orientational_order(
                            protein_com, positions, box_dimensions
                        )
                        
                        # Local density
                        features[f'{lipid_type}_density'] = physics.calculate_local_density(
                            protein_com, positions, box_dimensions, radius=15.0
                        )
                        
                        # Mediator-induced asymmetry
                        if len(mediator_positions) > 0:
                            asymmetry_results = physics.calculate_gm3_induced_asymmetry(
                                mediator_positions, positions, protein_com, box_dimensions
                            )
                            features[f'{lipid_type}_{mediator_lipid.lower()}_asymmetry'] = asymmetry_results['asymmetry']
                            features[f'{lipid_type}_{mediator_lipid.lower()}_asymmetry_angle'] = asymmetry_results['asymmetry_angle']
                            # Backward compatibility
                            features[f'{lipid_type}_gm3_asymmetry'] = asymmetry_results['asymmetry']
                            features[f'{lipid_type}_gm3_asymmetry_angle'] = asymmetry_results['asymmetry_angle']
                        else:
                            features[f'{lipid_type}_{mediator_lipid.lower()}_asymmetry'] = 0
                            features[f'{lipid_type}_{mediator_lipid.lower()}_asymmetry_angle'] = 0
                            features[f'{lipid_type}_gm3_asymmetry'] = 0
                            features[f'{lipid_type}_gm3_asymmetry_angle'] = 0
                    else:
                        # Fill with default values
                        for radius_name in FrameProcessor.CUTOFF_RADII:
                            features[f'{lipid_type}_{radius_name}_count'] = 0
                            features[f'{lipid_type}_{radius_name}_mean_dist'] = FrameProcessor.CUTOFF_RADII[radius_name]
                        features[f'{lipid_type}_order'] = 0
                        features[f'{lipid_type}_density'] = 0
                        features[f'{lipid_type}_{mediator_lipid.lower()}_asymmetry'] = 0
                        features[f'{lipid_type}_{mediator_lipid.lower()}_asymmetry_angle'] = 0
                        features[f'{lipid_type}_gm3_asymmetry'] = 0
                        features[f'{lipid_type}_gm3_asymmetry_angle'] = 0
                
                # Calculate lipid fractions in protein vicinity
                vicinity_radius = 15.0
                lipid_counts_vicinity = {}
                
                # Include all lipids (mediator + targets)
                all_lipid_types = [mediator_lipid] + target_lipids
                for lipid_type in all_lipid_types:
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
                    for lipid_type in target_lipids:
                        features[f'{lipid_type}_fraction_vicinity'] = (
                            lipid_counts_vicinity[lipid_type] / total_lipids
                        )
                    features[f'{mediator_lipid.lower()}_fraction_vicinity'] = (
                        lipid_counts_vicinity[mediator_lipid] / total_lipids
                    )
                    # Backward compatibility
                    features['gm3_fraction_vicinity'] = features[f'{mediator_lipid.lower()}_fraction_vicinity']
                else:
                    for lipid_type in target_lipids:
                        features[f'{lipid_type}_fraction_vicinity'] = 0
                    features[f'{mediator_lipid.lower()}_fraction_vicinity'] = 0
                    features['gm3_fraction_vicinity'] = 0
                
                frame_features.append(features)
            
            return frame_features
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            return []