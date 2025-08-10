# Test Data

This directory contains small test datasets for unit testing.

## Files

- `test_system.psf`: Topology file (5.09 MB)
- `test_trajectory.xtc`: Trajectory with 50 frames (7.34 MB)

## Generation

These files were generated from the full trajectory using `create_test_data.py`.
- Original PSF: step5_assembly.psf
- Original XTC: step7_production.xtc
- Original trajectory: 81158 frames
- Test trajectory: frames 60000 to 62450 (step 50)

## Usage

```python
import MDAnalysis as mda
u = mda.Universe('test_data/test_system.psf', 'test_data/test_trajectory.xtc')
print(f'Loaded {len(u.atoms)} atoms, {len(u.trajectory)} frames')
```
