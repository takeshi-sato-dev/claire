# claire/debug_critical.py
#!/usr/bin/env python3
"""Critical debug - find where plotting fails"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("CRITICAL DEBUG: Why no plots?")
print("="*70)

# Step-by-step minimal test
try:
    # 1. Load system
    print("\n1. Loading system...")
    from claire.core.membrane import MembraneSystem
    membrane = MembraneSystem('test_data/test_system.psf', 'test_data/test_trajectory.xtc', verbose=False)
    print("  ✓ System loaded")
    
    # 2. Get leaflets
    print("\n2. Getting leaflets...")
    membrane.identify_lipids()
    upper, lower = membrane.identify_leaflets(frame=0)
    print(f"  ✓ Upper leaflet: {len(upper.atoms)} atoms")
    
    # 3. Select lipids from UPPER only
    print("\n3. Selecting lipids from UPPER leaflet...")
    upper_lipids = {}
    for lipid in membrane.lipids.keys():
        sel = upper.select_atoms(f"resname {lipid}")
        if len(sel) > 0:
            upper_lipids[lipid] = sel
            print(f"  ✓ {lipid}: {len(sel.residues)} molecules")
    
    if len(upper_lipids) == 0:
        print("  ✗ ERROR: No lipids found in upper leaflet!")
        sys.exit(1)
    
    # 4. Create minimal protein dict
    print("\n4. Creating protein reference...")
    proteins = {'Center': upper}  # Simple AtomGroup
    print("  ✓ Using upper leaflet center as reference")
    
    # 5. Process ONE frame
    print("\n5. Processing single frame...")
    from claire.analysis.frame_processor import FrameProcessor
    processor = FrameProcessor()
    
    # Pick mediator and targets
    mediator = 'DPG3' if 'DPG3' in upper_lipids else list(upper_lipids.keys())[0]
    targets = [l for l in upper_lipids.keys() if l != mediator][:3]  # Max 3 targets
    
    print(f"  Mediator: {mediator}")
    print(f"  Targets: {targets}")
    
    result = processor.process_frame_complete(
        0,
        membrane.universe,
        proteins,
        upper_lipids,
        membrane.universe.dimensions[:3],
        mediator_lipid=mediator,
        target_lipids=targets
    )
    
    if not result:
        print("  ✗ ERROR: Frame processor returned empty!")
        sys.exit(1)
    
    print(f"  ✓ Got {len(result)} data points")
    
    # 6. Create DataFrame
    print("\n6. Creating DataFrame...")
    df = pd.DataFrame(result)
    print(f"  ✓ DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)[:10]}...")
    
    # 7. Check critical columns exist
    print("\n7. Checking critical columns...")
    has_mediator = False
    has_targets = False
    
    for col in df.columns:
        if mediator.lower() in col or 'gm3' in col:
            has_mediator = True
        for target in targets:
            if target in col:
                has_targets = True
    
    print(f"  Mediator columns: {'✓' if has_mediator else '✗'}")
    print(f"  Target columns: {'✓' if has_targets else '✗'}")
    
    if not has_mediator or not has_targets:
        print("  ✗ ERROR: Missing required columns!")
        print(f"  All columns: {list(df.columns)}")
        sys.exit(1)
    
    # 8. Process 10 frames for ML
    print("\n8. Processing multiple frames for ML...")
    all_data = []
    for i in range(min(10, len(membrane.universe.trajectory))):
        frame_data = processor.process_frame_complete(
            i,
            membrane.universe,
            proteins,
            upper_lipids,
            membrane.universe.dimensions[:3],
            mediator_lipid=mediator,
            target_lipids=targets
        )
        all_data.extend(frame_data)
    
    df_full = pd.DataFrame(all_data)
    print(f"  ✓ Processed {len(df_full)} total data points")
    
    # 9. Run ML analysis
    print("\n9. Running ML analysis...")
    from claire.analysis.ml_analysis import MLAnalyzer
    ml = MLAnalyzer()
    
    ml_results, frame_df = ml.advanced_ml_analysis(
        df_full,
        target_lipids=targets,
        mediator_lipid=mediator
    )
    
    if not ml_results:
        print("  ✗ ERROR: ML returned no results!")
        sys.exit(1)
    
    print(f"  ✓ ML results for: {list(ml_results.keys())}")
    
    # 10. DIRECTLY test matplotlib
    print("\n10. Testing matplotlib directly...")
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    # Create simple test plot
    plt.figure()
    plt.plot([1,2,3], [1,2,3])
    plt.savefig('test_matplotlib.png')
    plt.close()
    
    if os.path.exists('test_matplotlib.png'):
        print("  ✓ Matplotlib works")
        os.remove('test_matplotlib.png')
    else:
        print("  ✗ ERROR: Matplotlib cannot create files!")
        sys.exit(1)
    
    # 11. Call figure generator DIRECTLY
    print("\n11. Calling FigureGenerator directly...")
    from claire.visualization.figures import FigureGenerator
    fig_gen = FigureGenerator()
    
    os.makedirs('debug_output', exist_ok=True)
    
    # Call with explicit parameters
    effects, corrs = fig_gen.create_nature_quality_figures(
        ml_results,
        frame_df,
        'debug_output',
        target_lipids=targets,
        mediator_lipid=mediator
    )
    
    print(f"  Effects: {effects}")
    print(f"  Correlations: {corrs}")
    
    # 12. Check what was created
    print("\n12. Checking output files...")
    if os.path.exists('debug_output'):
        files = os.listdir('debug_output')
        if len(files) == 0:
            print("  ✗ ERROR: No files created!")
        else:
            print("  ✓ Created files:")
            for f in files:
                size = os.path.getsize(f'debug_output/{f}') / 1024
                print(f"    - {f} ({size:.1f} KB)")
    else:
        print("  ✗ ERROR: Output directory not created!")
    
    print("\n" + "="*70)
    print("SUCCESS PATH:")
    print("  Data → ML → Figures → Files")
    print("Find where it breaks above!")
    
except Exception as e:
    print(f"\n✗ EXCEPTION: {e}")
    import traceback
    traceback.print_exc()