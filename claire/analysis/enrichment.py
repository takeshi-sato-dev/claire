#!/usr/bin/env python3
"""
Enrichment analysis for lipid-protein interactions
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.spatial import distance


class EnrichmentAnalyzer:
    """
    Calculate enrichment metrics for lipids around proteins
    """
    
    @staticmethod
    def calculate_enrichment(protein_com: np.ndarray,
                           lipid_positions: np.ndarray,
                           all_lipid_positions: np.ndarray,
                           box_dimensions: np.ndarray,
                           radius: float = 15.0) -> Dict:
        """
        Calculate enrichment of specific lipid type around protein
        
        Parameters
        ----------
        protein_com : numpy.ndarray
            Protein center of mass
        lipid_positions : numpy.ndarray
            Positions of specific lipid type
        all_lipid_positions : numpy.ndarray
            Positions of all lipids
        box_dimensions : numpy.ndarray
            Box dimensions for PBC
        radius : float
            Analysis radius (Angstroms)
        
        Returns
        -------
        dict
            Enrichment metrics
        """
        if len(lipid_positions) == 0 or len(all_lipid_positions) == 0:
            return {
                'enrichment': 0,
                'local_density': 0,
                'global_density': 0,
                'local_fraction': 0,
                'local_count': 0
            }
        
        # Count local lipids (within radius)
        local_count = 0
        for pos in lipid_positions:
            dist = EnrichmentAnalyzer._calculate_2d_distance(
                pos, protein_com, box_dimensions
            )
            if dist <= radius:
                local_count += 1
        
        # Count total local lipids (all types)
        total_local = 0
        for pos in all_lipid_positions:
            dist = EnrichmentAnalyzer._calculate_2d_distance(
                pos, protein_com, box_dimensions
            )
            if dist <= radius:
                total_local += 1
        
        # Calculate densities
        area_nm2 = np.pi * (radius/10.0)**2  # Convert to nm²
        local_density = local_count / area_nm2
        
        # Global density
        membrane_area_nm2 = (box_dimensions[0] * box_dimensions[1]) / 100
        global_density = len(lipid_positions) / membrane_area_nm2
        
        # Enrichment factor
        if global_density > 0:
            enrichment = local_density / global_density
        else:
            enrichment = 0
        
        # Local fraction
        if total_local > 0:
            local_fraction = local_count / total_local
        else:
            local_fraction = 0
        
        return {
            'enrichment': enrichment,
            'local_density': local_density,
            'global_density': global_density,
            'local_fraction': local_fraction,
            'local_count': local_count
        }
    
    @staticmethod
    def _calculate_2d_distance(pos1: np.ndarray,
                              pos2: np.ndarray,
                              box_dimensions: np.ndarray) -> float:
        """
        Calculate 2D distance in XY plane with PBC
        
        Parameters
        ----------
        pos1 : numpy.ndarray
            First position
        pos2 : numpy.ndarray
            Second position
        box_dimensions : numpy.ndarray
            Box dimensions
        
        Returns
        -------
        float
            2D distance
        """
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        
        # PBC correction
        dx = dx - box_dimensions[0] * round(dx/box_dimensions[0])
        dy = dy - box_dimensions[1] * round(dy/box_dimensions[1])
        
        return np.sqrt(dx**2 + dy**2)
    
    @staticmethod
    def calculate_radial_distribution(protein_com: np.ndarray,
                                     lipid_positions: np.ndarray,
                                     box_dimensions: np.ndarray,
                                     max_radius: float = 30.0,
                                     n_bins: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate 2D radial distribution function
        
        Parameters
        ----------
        protein_com : numpy.ndarray
            Protein center of mass
        lipid_positions : numpy.ndarray
            Lipid positions
        box_dimensions : numpy.ndarray
            Box dimensions
        max_radius : float
            Maximum radius
        n_bins : int
            Number of bins
        
        Returns
        -------
        tuple
            (bin_centers, rdf_values)
        """
        if len(lipid_positions) == 0:
            return np.linspace(0, max_radius, n_bins), np.zeros(n_bins)
        
        # Calculate distances
        distances = []
        for pos in lipid_positions:
            dist = EnrichmentAnalyzer._calculate_2d_distance(
                pos, protein_com, box_dimensions
            )
            if dist <= max_radius:
                distances.append(dist)
        
        if len(distances) == 0:
            return np.linspace(0, max_radius, n_bins), np.zeros(n_bins)
        
        # Create histogram
        hist, bin_edges = np.histogram(distances, bins=n_bins, range=(0, max_radius))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate RDF (2D)
        rdf = np.zeros(n_bins)
        for i in range(n_bins):
            r_inner = bin_edges[i]
            r_outer = bin_edges[i+1]
            area = np.pi * (r_outer**2 - r_inner**2)
            
            if area > 0:
                rdf[i] = hist[i] / area
        
        # Normalize by bulk density
        total_area = np.pi * max_radius**2
        bulk_density = len(lipid_positions) / total_area
        
        if bulk_density > 0:
            rdf = rdf / bulk_density
        
        return bin_centers, rdf
    
    @staticmethod
    def calculate_clustering_index(lipid_positions: np.ndarray,
                                  reference_positions: np.ndarray,
                                  box_dimensions: np.ndarray,
                                  clustering_radius: float = 10.0) -> Dict:
        """
        Calculate clustering of lipids around reference molecules
        
        Parameters
        ----------
        lipid_positions : numpy.ndarray
            Positions of lipids to analyze
        reference_positions : numpy.ndarray
            Reference molecule positions (e.g., GM3)
        box_dimensions : numpy.ndarray
            Box dimensions
        clustering_radius : float
            Radius for clustering analysis
        
        Returns
        -------
        dict
            Clustering metrics
        """
        if len(lipid_positions) == 0 or len(reference_positions) == 0:
            return {
                'clustering_index': 0,
                'n_clustered': 0,
                'mean_cluster_size': 0
            }
        
        # Find lipids near reference molecules
        clustered_lipids = set()
        cluster_sizes = []
        
        for ref_pos in reference_positions:
            cluster = []
            for i, lipid_pos in enumerate(lipid_positions):
                dist = EnrichmentAnalyzer._calculate_2d_distance(
                    lipid_pos, ref_pos, box_dimensions
                )
                if dist <= clustering_radius:
                    cluster.append(i)
                    clustered_lipids.add(i)
            
            if cluster:
                cluster_sizes.append(len(cluster))
        
        # Calculate metrics
        n_clustered = len(clustered_lipids)
        clustering_index = n_clustered / len(lipid_positions)
        mean_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
        
        return {
            'clustering_index': clustering_index,
            'n_clustered': n_clustered,
            'mean_cluster_size': mean_cluster_size,
            'n_clusters': len(cluster_sizes)
        }