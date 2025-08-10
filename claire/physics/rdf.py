# claire/physics/rdf.py (NEW FILE - CRITICAL!)
#!/usr/bin/env python3
"""
Radial Distribution Function calculations
"""

import numpy as np
from typing import Tuple

class RDFCalculator:
    """
    Calculate RDF exactly as in original code
    """
    
    @staticmethod
    def calculate_actual_rdf(protein_com: np.ndarray, 
                            lipid_positions: np.ndarray,
                            box_dimensions: np.ndarray,
                            max_radius: float = 20.0,
                            n_bins: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate actual radial distribution function
        FROM ORIGINAL aiml13_no_causal.py - EXACT COPY
        """
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