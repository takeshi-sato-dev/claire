#!/usr/bin/env python3
"""
Interactive CLI for CLAIRE analysis - Fixed version
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any


def get_interactive_config() -> Dict:
    """Get configuration through interactive prompts - FIXED"""
    
    print("=" * 70)
    print("CLAIRE - Interactive Configuration")
    print("=" * 70)
    print("\nPress Enter to accept default values shown in [brackets]\n")
    
    config = {}
    
    # Check for test data first
    if os.path.exists("test_data/test_system.psf"):
        print("Test data detected!")
        use_test = input("Use test data? [Y/n]: ").strip().lower()
        if use_test != 'n':
            return {
                'topology': 'test_data/test_system.psf',
                'trajectory': 'test_data/test_trajectory.xtc',
                'mediator': 'DPG3',
                'targets': ['DOPC', 'DPSM', 'CHOL'],
                'start': 0,
                'stop': 100,  # Process 100 frames for test data
                'step': 1,     # Process every frame for test
                'leaflet_frame': 0,  # Use first frame for leaflet identification
                'leaflet': 'upper',  # Analyze upper leaflet
                'output': 'test_output',
                'parallel': 4,
                'verbose': True,
                'smooth_window': 10,
                'bootstrap': 100
            }
    
    print("\n" + "─" * 40)
    print("1. INPUT FILES")
    print("─" * 40)
    
    # Helper function to list files
    def list_files(extensions):
        """List available files with given extensions"""
        files = []
        # Check test_data directory
        if os.path.exists("test_data"):
            for ext in extensions:
                files.extend(Path("test_data").glob(f"*.{ext}"))
        # Check current directory
        for ext in extensions:
            files.extend(Path(".").glob(f"*.{ext}"))
        return [str(f) for f in files]
    
    # Topology file
    print("\nTopology file options:")
    psf_files = list_files(['psf', 'pdb', 'gro'])
    if psf_files:
        print("  Available files:")
        for i, f in enumerate(psf_files, 1):
            print(f"    {i}. {f}")
        print("\n  You can enter a number from above OR type a full path")
        print("  Examples: '1' or 'test_data/system.psf' or '/home/user/sim.gro'")
    else:
        print("  No topology files found in current/test_data directories")
        print("  Please enter the full path to your topology file")
        print("  Example: '/home/user/simulations/membrane.psf'")
    
    while True:
        topology_input = input("Topology file - enter number or path (PSF/PDB/GRO): ").strip()
        
        if not topology_input:
            print("  ⚠️  This field is required!")
            print("  Please enter a number from the list or a file path")
            continue
            
        # Check if number was entered
        try:
            idx = int(topology_input) - 1
            if 0 <= idx < len(psf_files):
                config['topology'] = psf_files[idx]
                print(f"  ✓ Selected: {psf_files[idx]}")
                break
        except ValueError:
            pass
        
        # Check if file exists
        if os.path.exists(topology_input):
            config['topology'] = topology_input
            print(f"  ✓ Found: {topology_input}")
            break
        else:
            print(f"  ❌ File not found: {topology_input}")
            print("  Please try again with:")
            print("    • A number from the list above (e.g., '1')")
            print("    • A relative path (e.g., 'test_data/system.psf')")
            print("    • An absolute path (e.g., '/home/user/membrane.psf')")
    
    # Trajectory file
    print("\nTrajectory file options:")
    traj_files = list_files(['xtc', 'dcd', 'trr'])
    if traj_files:
        print("  Available files:")
        for i, f in enumerate(traj_files, 1):
            print(f"    {i}. {f}")
        print("\n  You can enter a number from above OR type a full path")
        print("  Examples: '1' or 'test_data/traj.xtc' or '/home/user/production.dcd'")
    else:
        print("  No trajectory files found in current/test_data directories")
        print("  Please enter the full path to your trajectory file")
        print("  Example: '/home/user/simulations/trajectory.xtc'")
    
    while True:
        traj_input = input("Trajectory file - enter number or path (XTC/DCD/TRR): ").strip()
        
        if not traj_input:
            print("  ⚠️  This field is required!")
            print("  Please enter a number from the list or a file path")
            continue
            
        # Check if number was entered
        try:
            idx = int(traj_input) - 1
            if 0 <= idx < len(traj_files):
                config['trajectory'] = traj_files[idx]
                print(f"  ✓ Selected: {traj_files[idx]}")
                break
        except ValueError:
            pass
        
        # Check if file exists
        if os.path.exists(traj_input):
            config['trajectory'] = traj_input
            print(f"  ✓ Found: {traj_input}")
            break
        else:
            print(f"  ❌ File not found: {traj_input}")
            print("  Please try again with:")
            print("    • A number from the list above (e.g., '1')")
            print("    • A relative path (e.g., 'test_data/trajectory.xtc')")
            print("    • An absolute path (e.g., '/home/user/production.dcd')")
    
    print("\n" + "─" * 40)
    print("2. LIPID SELECTION")
    print("─" * 40)
    
    print("\nCommon lipid types for reference:")
    print("  • Phospholipids: DIPC,DOPC, POPC, DPPC, DOPE, POPE, DOPS, POPS")
    print("  • Sphingolipids: DPSM, SSM, CER")
    print("  • Sterols: CHOL")
    print("  • Gangliosides: GM3 (often labeled as DPG3)")
    
    # Mediator lipid
    mediator = input("\nMediator lipid (e.g., GM3/DPG3) [DPG3]: ").strip()
    config['mediator'] = mediator if mediator else 'DPG3'
    
    # Target lipids
    print("\nTarget lipids to analyze (space-separated)")
    targets_input = input("Target lipids [DIPC DPSM CHOL]: ").strip()
    if targets_input:
        config['targets'] = targets_input.upper().split()
    else:
        config['targets'] = ['DIPC', 'DPSM', 'CHOL']
    
    print("\n" + "─" * 40)
    print("3. ANALYSIS PARAMETERS")
    print("─" * 40)
    
    # Frame selection
    start = input("\nStart frame [0]: ").strip()
    config['start'] = int(start) if start else 0
    
    stop = input("Stop frame (-1 for last) [100]: ").strip()  # Default to 100 frames
    config['stop'] = int(stop) if stop else 100
    
    step = input("Frame step/stride [1]: ").strip()  # Default to every frame
    config['step'] = int(step) if step else 1
    
    # Leaflet identification frame
    leaflet_frame = input("\nFrame for leaflet identification [0]: ").strip()
    config['leaflet_frame'] = int(leaflet_frame) if leaflet_frame else 0
    
    # Leaflet selection
    leaflet = input("Which leaflet to analyze (upper/lower/both) [upper]: ").strip().lower()
    config['leaflet'] = leaflet if leaflet in ['upper', 'lower', 'both'] else 'upper'
    
    # Output directory
    output = input("\nOutput directory [claire_output]: ").strip()
    config['output'] = output if output else 'claire_output'
    
    # Parallel processing
    parallel = input("Number of CPU cores for parallel processing [4]: ").strip()
    config['parallel'] = int(parallel) if parallel else 4
    
    # Additional parameters
    verbose = input("\nVerbose output? [Y/n]: ").strip().lower()
    config['verbose'] = verbose != 'n'
    
    # Set defaults for other parameters
    config['smooth_window'] = 10
    config['bootstrap'] = 100
    
    # Summary
    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"\nInput files:")
    print(f"  • Topology: {config['topology']}")
    print(f"  • Trajectory: {config['trajectory']}")
    print(f"\nLipids:")
    print(f"  • Mediator: {config['mediator']}")
    print(f"  • Targets: {', '.join(config['targets'])}")
    print(f"\nParameters:")
    print(f"  • Frames: {config['start']} to {config['stop']} (step {config['step']})")
    print(f"  • Leaflet ID frame: {config['leaflet_frame']}")
    print(f"  • Analyzing: {config['leaflet']} leaflet")
    print(f"  • Output: {config['output']}/")
    print(f"  • Parallel: {config['parallel']} cores")
    
    # Confirm
    proceed = input("\nProceed with analysis? [Y/n]: ").strip().lower()
    if proceed == 'n':
        print("Analysis cancelled.")
        sys.exit(0)
    
    return config


if __name__ == "__main__":
    # Test the CLI
    config = get_interactive_config()
    print(f"\nConfiguration ready: {config}")