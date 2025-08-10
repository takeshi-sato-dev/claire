#!/usr/bin/env python3
"""
Topology reading and processing utilities
"""

import numpy as np
from typing import Dict, List, Optional
import MDAnalysis as mda


class TopologyReader:
    """
    Reader for various topology formats with lipid/protein identification
    """
    
    @staticmethod
    def get_lipid_headgroup_atoms(forcefield: str = 'auto') -> List[str]:
        """
        Get headgroup atom names for different force fields
        
        Parameters
        ----------
        forcefield : str
            Force field type ('martini', 'charmm', 'amber', 'auto')
        
        Returns
        -------
        list
            List of headgroup atom names
        """
        headgroup_atoms = {
            'martini': ['GL1', 'GL2', 'AM1', 'AM2', 'ROH', 'GM1', 'GM2', 
                       'NC3', 'CNO', 'PO4'],
            'charmm': ['P', 'P8', 'P1', 'N', 'C2', 'C12', 'O11'],
            'amber': ['P', 'P8', 'P1', 'N4', 'C1', 'O11'],
            'auto': []  # Will be populated with all known
        }
        
        if forcefield == 'auto':
            # Combine all known headgroup atoms
            all_atoms = set()
            for ff_atoms in headgroup_atoms.values():
                if isinstance(ff_atoms, list):
                    all_atoms.update(ff_atoms)
            return list(all_atoms)
        
        return headgroup_atoms.get(forcefield, [])
    
    @staticmethod
    def guess_forcefield(universe: mda.Universe) -> str:
        """
        Guess the force field from atom/residue names
        
        Parameters
        ----------
        universe : MDAnalysis.Universe
            System to analyze
        
        Returns
        -------
        str
            Guessed force field name
        """
        atom_names = set(universe.atoms.names)
        
        # Check for characteristic atoms
        martini_atoms = {'BB', 'SC1', 'SC2', 'SC3', 'GL1', 'GL2'}
        charmm_atoms = {'CAY', 'CAT', 'CTL2', 'CTL3', 'CTL5'}
        
        if martini_atoms.intersection(atom_names):
            return 'martini'
        elif charmm_atoms.intersection(atom_names):
            return 'charmm'
        else:
            return 'auto'
    
    @staticmethod
    def identify_residue_types(universe: mda.Universe) -> Dict:
        """
        Classify all residues in the system
        
        Parameters
        ----------
        universe : MDAnalysis.Universe
            System to analyze
        
        Returns
        -------
        dict
            Dictionary with residue classifications
        """
        residue_types = {
            'lipids': [],
            'proteins': [],
            'nucleic_acids': [],
            'water': [],
            'ions': [],
            'other': []
        }
        
        # Common patterns
        lipid_patterns = ['PC', 'PE', 'PS', 'PA', 'PG', 'PI', 'CHOL', 
                         'SM', 'GM', 'DPG', 'DPS', 'DIP']
        protein_patterns = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 
                           'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 
                           'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 
                           'TYR', 'VAL']
        water_patterns = ['HOH', 'TIP3', 'TIP4', 'SPC', 'WAT', 'W']
        ion_patterns = ['NA', 'CL', 'K', 'CA', 'MG', 'ZN', 'ION', 
                       'SOD', 'CLA', 'POT', 'CAL']
        
        for resname in np.unique(universe.atoms.resnames):
            resname_upper = resname.upper()
            
            if any(pattern in resname_upper for pattern in lipid_patterns):
                residue_types['lipids'].append(resname)
            elif resname in protein_patterns:
                residue_types['proteins'].append(resname)
            elif resname in water_patterns:
                residue_types['water'].append(resname)
            elif resname in ion_patterns:
                residue_types['ions'].append(resname)
            else:
                residue_types['other'].append(resname)
        
        return residue_types