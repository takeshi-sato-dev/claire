# claire/debug_plot.py
#!/usr/bin/env python3
"""Debug why plots aren't being generated"""

import os
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("DEBUG: Plot Generation")
print("="*70)

# Check what files were created
output_dir = "claire_output"
print(f"\n1. Files in {output_dir}:")
if os.path.exists(output_dir):
    files = os.listdir(output_dir)
    for f in files:
        size = os.path.getsize(os.path.join(output_dir, f)) / 1024
        print(f"  - {f} ({size:.1f} KB)")
else:
    print(f"  Directory {output_dir} doesn't exist!")

# Check if CSV was saved
csv_path = os.path.join(output_dir, "raw_frame_data.csv")
if os.path.exists(csv_path):
    print(f"\n2. Checking raw_frame_data.csv:")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)[:10]}...")
    
    # Check for key columns
    print(f"\n3. Key columns check:")
    for col in ['gm3_contact_count', 'gm3_contact_strength', 
                'CHOL_contact_count', 'DIPC_contact_count', 'DPSM_contact_count']:
        if col in df.columns:
            mean_val = df[col].mean()
            non_zero = (df[col] > 0).sum()
            print(f"  {col}: mean={mean_val:.3f}, non-zero={non_zero}/{len(df)}")
        else:
            print(f"  {col}: NOT FOUND")
    
    # Check unique proteins
    if 'protein' in df.columns:
        print(f"\n4. Proteins: {df['protein'].unique()}")
    
    # Check frame range
    if 'frame' in df.columns:
        print(f"\n5. Frame range: {df['frame'].min()} to {df['frame'].max()}")
else:
    print(f"\n  raw_frame_data.csv not found!")

# Test the figure generator directly
print("\n6. Testing FigureGenerator import:")
try:
    from claire.visualization.figures import FigureGenerator
    print("  ✓ FigureGenerator imported")
    
    # Check if the method exists
    if hasattr(FigureGenerator, 'create_nature_quality_figures'):
        print("  ✓ create_nature_quality_figures method exists")
    else:
        print("  ✗ create_nature_quality_figures method NOT FOUND")
        
except Exception as e:
    print(f"  ✗ Error importing FigureGenerator: {e}")

# Check for expected plot files
print("\n7. Expected plot files:")
expected_plots = [
    'figure_observed_results.png',
    'figure_observed_results.pdf', 
    'figure_observed_results.svg',
    'figure_temporal_analysis.png',
    'figure_temporal_analysis.pdf',
    'figure_temporal_analysis.svg',
    'correlations.png'
]

for plot_file in expected_plots:
    path = os.path.join(output_dir, plot_file)
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024
        print(f"  ✓ {plot_file} ({size:.1f} KB)")
    else:
        print(f"  ✗ {plot_file} NOT FOUND")

# Check if it's a lipid name mismatch issue
print("\n8. Checking for hardcoded lipid names issue:")
print("  The code expects: CHOL, DIPC, DPSM")
print("  Your system has: Check raw_frame_data.csv columns above")
print("  If different, that's the problem!")