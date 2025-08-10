# claire/test_simple.py
#!/usr/bin/env python3
"""Simple test to verify everything works"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from claire.core.membrane import MembraneSystem

print("Simple test...")

# Load system
membrane = MembraneSystem('test_data/test_system.psf', 'test_data/test_trajectory.xtc')

# Identify components
membrane.identify_lipids()
print("\nLipids found:")
for name, selection in membrane.lipids.items():
    print(f"  {name}: {len(selection.residues)} molecules, type={type(selection)}")

# Get leaflets
upper, lower = membrane.identify_leaflets(frame=0)

print("\nUpper leaflet lipids:")
for lipid_name in membrane.lipids.keys():
    upper_selection = upper.select_atoms(f"resname {lipid_name}")
    if len(upper_selection) > 0:
        print(f"  {lipid_name}: {len(upper_selection.residues)} molecules")

print("\nLower leaflet lipids:")
for lipid_name in membrane.lipids.keys():
    lower_selection = lower.select_atoms(f"resname {lipid_name}")
    if len(lower_selection) > 0:
        print(f"  {lipid_name}: {len(lower_selection.residues)} molecules")