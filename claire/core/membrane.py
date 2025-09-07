#!/usr/bin/env python3
"""
Core membrane system analysis module
"""

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Global parameters from original_analysis
LEAFLET_FRAME = 20000


def identify_lipid_leaflets(u, frame=LEAFLET_FRAME):
    """Identify lipid leaflets - EXACT COPY FROM ORIGINAL_ANALYSIS"""
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
    """Select lipids from leaflet - EXACT COPY FROM ORIGINAL_ANALYSIS"""
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
    """Identify proteins - EXACT COPY FROM ORIGINAL_ANALYSIS"""
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


class MembraneSystem:
    """
    Main class for membrane system analysis with automatic lipid/protein detection
    """
    
    def __init__(self, topology: str, trajectory: str, verbose: bool = True):
        """
        Initialize membrane system
        
        Parameters
        ----------
        topology : str
            Path to topology file (PSF, PDB, GRO, etc.)
        trajectory : str
            Path to trajectory file (XTC, DCD, TRR, etc.)
        verbose : bool
            Print progress information
        """
        self.topology = topology
        self.trajectory = trajectory
        self.verbose = verbose
        
        # Load universe
        self.universe = mda.Universe(topology, trajectory)
        if self.verbose:
            print(f"Loaded system: {len(self.universe.atoms)} atoms, "
                  f"{len(self.universe.trajectory)} frames")
        
        # Storage for identified components
        self.lipids = {}
        self.proteins = {}
        self.leaflets = {'upper': None, 'lower': None}
        self.mediator_lipid = None
        self.target_lipids = []
        

    def identify_lipids(self) -> Dict:
        """
        Automatically identify lipid types
        MATCHES ORIGINAL CODE - returns AtomGroups, not dicts!
        
        Returns
        -------
        dict
            Dictionary of lipid selections by resname
        """
        lipids = {}
        
        # Common lipid residue names
        common_lipids = [
            'POPC', 'POPE', 'POPS', 'DOPC', 'DOPE', 'DOPS', 'DLPC', 'DLPE', 'DLPS',
            'DPPC', 'DPPE', 'DPPS', 'DSPC', 'DSPE', 'DSPS', 'DMPC', 'DMPE', 'DMPS',
            'CHOL', 'CHL1', 'ERG',
            'DPSM', 'DXSM', 'PGSM', 'PNSM', 'POSM', 'BNSM', 'XNSM',
            'DPG1', 'DPG3', 'DXG1', 'DXG3', 'PNG1', 'PNG3', 'XNG1', 'XNG3',
            'DPCE', 'DXCE', 'PNCE', 'POCE', 'XNCE',
            'DIPC',
            'DPGS', 'DPSM', 'DPG3', 'GM3'
        ]
        
        for lipid_name in common_lipids:
            selection = self.universe.select_atoms(f"resname {lipid_name}")
            if len(selection) > 0:
                lipids[lipid_name] = selection  # <-- Just the AtomGroup, NOT a dict!
                
                if self.verbose:
                    # Identify lipid type
                    if lipid_name in ['CHOL', 'CHL1', 'ERG']:
                        lipid_type = 'cholesterol'
                    elif 'PC' in lipid_name:
                        lipid_type = 'PC'
                    elif 'PE' in lipid_name:
                        lipid_type = 'PE'
                    elif 'PS' in lipid_name:
                        lipid_type = 'PS'
                    elif 'SM' in lipid_name:
                        lipid_type = 'SM'
                    elif 'G3' in lipid_name or 'GM3' in lipid_name:
                        lipid_type = 'GM3'
                    else:
                        lipid_type = 'other'
                    
                    print(f"  {lipid_name}: {len(selection.residues)} molecules ({lipid_type})")
        
        if self.verbose:
            print(f"\nIdentified {len(lipids)} lipid types:")
            for name, sel in lipids.items():
                print(f"  {name}: {len(sel.residues)} molecules")
        
        self.lipids = lipids
        return lipids
    


    def identify_proteins(self) -> Dict:
        """
        Identify protein segments - FIXED to detect proteins properly
        """
        proteins = {}
        
        # Try multiple selection methods
        # 1. Standard protein selection
        protein_selection = self.universe.select_atoms("protein")
        
        # 2. If no standard proteins, try by segment
        if len(protein_selection) == 0:
            # Get all segments
            all_segments = np.unique(self.universe.segments.segids)
            
            for segid in all_segments:
                seg = self.universe.select_atoms(f"segid {segid}")
                # Check if this segment is protein-like (not lipid)
                resnames = np.unique(seg.resnames)
                
                # If it contains non-lipid residues, might be protein
                lipid_resnames = ['CHOL', 'DOPC', 'DOPS', 'DPPC', 'DPPE', 'DPSM', 'DPG3', 
                                 'POPC', 'POPE', 'POPS', 'DLPC', 'DLPE', 'DLPS']
                
                is_lipid = all(res in lipid_resnames for res in resnames)
                
                if not is_lipid and len(seg) > 100:  # Likely a protein
                    proteins[f"Protein_{segid}"] = seg
        else:
            # Group proteins by segment
            segids = np.unique(protein_selection.segids)
            for i, segid in enumerate(segids):
                segment = protein_selection.select_atoms(f"segid {segid}")
                proteins[f"Protein_{i+1}"] = segment
        
        if self.verbose:
            print(f"\nIdentified {len(proteins)} proteins:")
            for name, prot in proteins.items():
                print(f"  {name}: {len(prot.residues)} residues, {len(prot)} atoms")
        
        self.proteins = proteins
        return proteins
    
    def identify_leaflets(self, frame: int = LEAFLET_FRAME) -> Tuple:
        """Use original_analysis's exact leaflet identification"""
        return identify_lipid_leaflets(self.universe, frame)
    
    def select_lipids_in_leaflet(self, leaflet: str = 'upper', 
                                 lipid_types: Optional[List[str]] = None) -> Dict:
        """
        Select specific lipid types from a leaflet
        
        Parameters
        ----------
        leaflet : str
            'upper' or 'lower'
        lipid_types : list, optional
            List of lipid resnames to select. If None, select all.
        
        Returns
        -------
        dict
            Dictionary of selected lipids by type
        """
        if self.leaflets[leaflet] is None:
            raise ValueError(f"Leaflet '{leaflet}' not identified. "
                           "Run identify_leaflets() first.")
        
        leaflet_atoms = self.leaflets[leaflet]
        
        if lipid_types is None:
            lipid_types = list(self.lipids.keys())
        
        selected_lipids = {}
        
        for resname in lipid_types:
            if resname in self.lipids:
                selection = leaflet_atoms.select_atoms(f"resname {resname}")
                if len(selection) > 0:
                    selected_lipids[resname] = selection
                    if self.verbose:
                        print(f"  Selected {len(selection.residues)} "
                              f"{resname} molecules in {leaflet} leaflet")
        
        return selected_lipids
    
    def set_analysis_targets(self, mediator_lipid: str, 
                            target_lipids: List[str],
                            leaflet: str = 'upper'):
        """
        Set the mediator lipid and target lipids for analysis
        
        Parameters
        ----------
        mediator_lipid : str
            Resname of the mediator lipid (e.g., 'DPG3' for GM3)
        target_lipids : list
            List of target lipid resnames
        leaflet : str
            Which leaflet to analyze
        """
        self.mediator_lipid = mediator_lipid
        self.target_lipids = target_lipids
        self.analysis_leaflet = leaflet
        
        if self.verbose:
            print(f"\nAnalysis configuration:")
            print(f"  Mediator lipid: {mediator_lipid}")
            print(f"  Target lipids: {', '.join(target_lipids)}")
            print(f"  Leaflet: {leaflet}")