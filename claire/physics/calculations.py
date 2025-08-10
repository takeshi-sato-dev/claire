#!/usr/bin/env python3
"""
Physical calculations for membrane analysis
"""

import numpy as np
from typing import Tuple, Dict, List


class PhysicsCalculator:
    """
    Physics-based calculations for lipid analysis
    """
    
    @staticmethod
    def calculate_orientational_order(protein_com: np.ndarray, 
                                     lipid_positions: np.ndarray,
                                     box_dimensions: np.ndarray) -> float:
        """
        Calculate orientational order parameter around protein
        FROM ORIGINAL CODE - CRITICAL
        """
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
    
    @staticmethod
    def calculate_local_density(protein_com: np.ndarray,
                               lipid_positions: np.ndarray,
                               box_dimensions: np.ndarray,
                               radius: float = 10.0) -> float:
        """
        Calculate local density of lipids around protein
        FROM ORIGINAL CODE
        """
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
    
    @staticmethod
    def calculate_gm3_induced_asymmetry(gm3_positions: np.ndarray,
                                       lipid_positions: np.ndarray,
                                       protein_com: np.ndarray,
                                       box_dimensions: np.ndarray,
                                       vicinity_radius: float = 15.0) -> Dict:
        """
        GM3-induced asymmetry analysis from original code
        CRITICAL FUNCTION
        """
        if len(gm3_positions) == 0 or len(lipid_positions) == 0:
            return {
                'asymmetry': 0,
                'asymmetry_angle': 0,
                'n_lipids_affected': 0
            }
        
        # Calculate GM3 center of mass relative to protein
        gm3_com = np.mean(gm3_positions, axis=0)
        gm3_vector = gm3_com[:2] - protein_com[:2]  # XY only
        
        # PBC correction for GM3 COM
        gm3_vector[0] = gm3_vector[0] - box_dimensions[0] * round(gm3_vector[0]/box_dimensions[0])
        gm3_vector[1] = gm3_vector[1] - box_dimensions[1] * round(gm3_vector[1]/box_dimensions[1])
        
        gm3_angle = np.arctan2(gm3_vector[1], gm3_vector[0])
        
        # Check lipid distribution asymmetry relative to GM3
        asymmetry_cos = 0
        asymmetry_sin = 0
        count = 0
        
        for pos in lipid_positions:
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
            asymmetry_angle = np.arctan2(asymmetry_sin/count, asymmetry_cos/count)
        else:
            asymmetry = 0
            asymmetry_angle = 0
        
        return {
            'asymmetry': asymmetry,
            'asymmetry_angle': asymmetry_angle,
            'n_lipids_affected': count
        }