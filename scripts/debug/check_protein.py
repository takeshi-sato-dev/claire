# claire/check_proteins.py - FIXED
#!/usr/bin/env python3
"""Check what proteins are in the system"""

import MDAnalysis as mda
import numpy as np

u = mda.Universe('test_data/test_system.psf', 'test_data/test_trajectory.xtc')

print("System overview:")
print(f"  Total atoms: {len(u.atoms)}")
print(f"  Total residues: {len(u.residues)}")
print(f"  Total segments: {len(u.segments)}")

print("\nSegments in system:")
for seg in u.segments:
    print(f"  {seg.segid}: {len(seg.atoms)} atoms, {len(seg.residues)} residues")
    # Get residue names through residues
    if len(seg.residues) > 0:
        resnames = np.unique([res.resname for res in seg.residues])[:10]
        print(f"    First resnames: {resnames}")

print("\nTrying protein selection:")
proteins = u.select_atoms("protein")
print(f"  Standard 'protein' selection: {len(proteins)} atoms")

# Check each segment for protein residues
print("\nChecking each segment for protein residues:")
protein_resnames = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                   'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                   'THR', 'TRP', 'TYR', 'VAL', 'HSE', 'HSD', 'HSP']

for seg in u.segments:
    seg_atoms = u.select_atoms(f"segid {seg.segid}")
    resnames = np.unique([res.resname for res in seg_atoms.residues])
    
    # Check if any protein residues
    protein_residues = [r for r in resnames if r in protein_resnames]
    
    if protein_residues:
        print(f"  {seg.segid}: PROTEIN FOUND")
        print(f"    Protein residues: {protein_residues}")
        print(f"    Total residues: {len(seg.residues)}")
    else:
        # Show what it contains
        print(f"  {seg.segid}: Not protein")
        print(f"    Contains: {resnames[:5]}...")

# Try alternative protein selection
print("\nAlternative selections:")
# Try segname
for segid in ['P1', 'P2', 'PROA', 'PROB', 'PROT', 'PROT1', 'PROT2']:
    test = u.select_atoms(f"segid {segid}", updating=False)
    if len(test) > 0:
        print(f"  Found segment {segid}: {len(test)} atoms")

# Check for peptides
peptides = u.select_atoms("resname ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL")
if len(peptides) > 0:
    print(f"\nFound peptide atoms: {len(peptides)}")
    peptide_segs = np.unique(peptides.segids)
    print(f"  In segments: {peptide_segs}")