# CLAIRE: Conserved Lipid Analysis with Interaction and Redistribution Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![CI](https://github.com/username/claire/actions/workflows/ci.yml/badge.svg)](https://github.com/username/claire/actions/workflows/ci.yml)

CLAIRE is a Python package for analyzing mediator-induced lipid redistribution around membrane proteins in molecular dynamics simulations, with a focus on conservation-aware analysis methods that respect fundamental physical laws.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Input Requirements](#input-requirements)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Features

- 🔬 **Conservation-aware analysis**: Ensures mass conservation in protein-lipid redistribution
- 🎯 **Protein-centric**: Analyzes lipid composition changes around individual proteins
- 📊 **Multi-scale metrics**: Analyzes from protein surface to bulk membrane
- 🤖 **Machine learning integration**: Random Forest models for predictive analysis
- 📈 **Statistical validation**: Bootstrap confidence intervals and permutation tests
- 🎨 **Publication-ready figures**: Automated generation of high-quality plots
- ⚡ **Parallel processing**: Efficient analysis of large trajectories
- 🔧 **Flexible**: Works with any protein, lipid types, and mediators

## Installation

### Prerequisites

- Python ≥ 3.10
- NumPy, SciPy, Pandas
- MDAnalysis ≥ 2.0
- scikit-learn ≥ 1.0
- Matplotlib, Seaborn

### Install from source

```bash
# Clone the repository
git clone https://github.com/takeshi-sato-dev/claire.git
cd claire

# Install in development mode
pip install -e .

# Or install directly
pip install .
```

### Verify installation

```bash
# Test import
python -c "import claire; print(f'CLAIRE version: {claire.__version__}')"

# Run tests
pytest tests/
```

## Quick Start

### Using test data (recommended for first-time users)

```bash
# Run with test data interactively
python claire/run_analysis.py

# When prompted, press Enter to use test data:
# Use test data? [Y/n]: <press Enter>
```

### Basic usage with your data

```bash
# Interactive mode (recommended)
python claire/run_analysis.py

# Command-line mode for protein-membrane system
python claire/run_analysis.py \
  --topology protein_membrane.psf \
  --trajectory your_trajectory.xtc \
  --mediator GM3 \
  --targets CHOL DOPC DPSM \
  --output results/
  
# CLAIRE will automatically:
# 1. Detect membrane proteins (GPCRs, channels, etc.)
# 2. Identify lipid types
# 3. Analyze redistribution around each protein
# 4. Generate protein-specific and comparison plots
```

## Detailed Usage

### 1. Interactive Mode (Recommended)

Interactive mode guides you through all parameters:

```bash
python claire/run_analysis.py
```

You will be prompted for:

#### Input Files
```
Topology file options:
  Available files:
    1. test_data/test_system.psf
    2. my_membrane.psf
    
  You can enter a number from above OR type a full path
  Examples: '1' or 'test_data/system.psf' or '/home/user/sim.gro'

Topology file - enter number or path (PSF/PDB/GRO): 1
```

#### Frame Selection
```
Start frame [0]: 0
Stop frame (-1 for last) [100]: 1000
Frame step/stride [1]: 10
Frame for leaflet identification [0]: 0
Which leaflet to analyze (upper/lower/both) [upper]: upper
```

#### Lipid Selection
```
Mediator lipid (e.g., GM3/DPG3) [DPG3]: GM3
Target lipids [DOPC DPSM CHOL]: CHOL DOPC DPSM POPE
```

### 2. Command-Line Mode

For automation and scripting:

```bash
python claire/run_analysis.py \
  --topology membrane.psf \
  --trajectory production.xtc \
  --start 0 \
  --stop 10000 \
  --step 10 \
  --leaflet-frame 0 \
  --leaflet upper \
  --mediator GM3 \
  --targets CHOL DOPC DPSM \
  --smooth-window 20 \
  --output results/ \
  --verbose
```

### 3. Python API

For integration into your scripts:

```python
from claire import MembraneSystem
from claire.analysis import MLAnalyzer, ConservationAnalyzer
from claire.visualization import FigureGenerator

# Load system
membrane = MembraneSystem('topology.psf', 'trajectory.xtc')
membrane.identify_lipids()
membrane.identify_proteins()

# Identify leaflets
upper, lower = membrane.identify_leaflets(frame=0)

# Select lipids from upper leaflet
lipid_selections = {}
for lipid_name in ['GM3', 'CHOL', 'DOPC', 'DPSM']:
    selection = upper.select_atoms(f"resname {lipid_name}")
    if len(selection) > 0:
        lipid_selections[lipid_name] = selection

# Analyze redistribution
analyzer = MLAnalyzer()
results = analyzer.analyze_redistribution(
    membrane,
    mediator='GM3',
    targets=['CHOL', 'DOPC', 'DPSM'],
    radii=[5, 10, 15, 20]  # Multi-scale
)

# Check conservation
conservation = ConservationAnalyzer()
conserved = conservation.check_conservation(results)
print(f"Conservation satisfied: {conserved}")

# Generate figures
fig_gen = FigureGenerator()
fig_gen.create_all_figures(results, 'output/')
```

## Input Requirements

### Topology Files
- **Supported formats**: PSF, PDB, GRO
- **Requirements**: Must contain lipid residues with standard names
- **Example lipid names**: DOPC, POPC, CHOL, DPSM, GM3, DPG3

### Trajectory Files
- **Supported formats**: XTC, DCD, TRR
- **Recommended**: At least 100 frames for statistical significance
- **Frame rate**: Can use stride to reduce computation

### System Requirements
- **Lipids**: Must be identifiable by residue name
- **Proteins**: Optional, automatically detected
- **Box**: Periodic boundary conditions required

## Output Files

CLAIRE generates comprehensive analysis results:

### Data Files
- `raw_frame_data.csv`: All computed metrics per frame
- `ml_results.json`: Machine learning model results
- `config.json`: Analysis parameters used

### Figures (PNG/PDF/SVG)
1. **figure_observed_results.png**: ML analysis summary
   - Effect sizes with error bars
   - Model performance (R² scores)
   - Time series scatter plots
   - Feature importance
   - Statistical summary table

2. **figure_temporal_analysis.png**: Time series analysis
   - Normalized mediator and lipid signals
   - Highlighted high-mediator regions
   - Temporal correlations

3. **composition_analysis_with_peptides.png**: Detailed composition
   - Raw vs smoothed data
   - Correlation analysis
   - Composition changes
   - Average composition pie chart

4. **peptide_comparison.png**: Protein-specific responses
   - Per-protein lipid changes
   - Statistical significance markers
   - Only generated if multiple proteins present

## Troubleshooting

### Common Issues

#### "ValueError: With n_samples=1..."
**Problem**: Too few frames processed
**Solution**: Increase frames or reduce step size
```bash
--stop 1000 --step 1  # Process more frames
```

#### "No proteins found"
**Problem**: Proteins not detected
**Solution**: CLAIRE will use membrane center as reference (this is fine)

#### "Mediator lipid not found"
**Problem**: Mediator name doesn't match residue names
**Solution**: Check lipid names in your topology:
```python
import MDAnalysis as mda
u = mda.Universe('your.psf')
print(np.unique(u.atoms.resnames))  # Shows all residue names
```

#### Memory issues
**Problem**: Large trajectory
**Solution**: Use larger step size:
```bash
--step 100  # Process every 100th frame
```

### Performance Tips

1. **Start with subset**: Test with first 1000 frames
2. **Use appropriate stride**: Step=10 usually sufficient
3. **Select specific leaflet**: Analyzing both doubles computation
4. **Limit target lipids**: Focus on key lipids of interest

## Example Workflows

### Analyzing GM3-Cholesterol Interactions Around GPCRs
```bash
python claire/run_analysis.py \
  --topology gpcr_membrane.psf \
  --trajectory md_production.xtc \
  --mediator GM3 \
  --targets CHOL SSM \
  --start 5000 \
  --stop 50000 \
  --step 10 \
  --output gpcr_gm3_chol_analysis/
```

### Comparing Lipid Environments Across Different Proteins
```bash
# Analyze multiple proteins in the same membrane
# CLAIRE will automatically detect and analyze each protein separately
python claire/run_analysis.py \
  --topology multi_protein_membrane.psf \
  --trajectory trajectory.xtc \
  --mediator GM3 \
  --targets CHOL DOPC DOPE DOPS \
  --output protein_comparison/
```

### Protein-Specific Mediator Effects
```python
import glob
from claire import run_analysis

# Compare how different mediators affect the same protein
protein_system = "gpcr_membrane.psf"
trajectory = "production.xtc"

for mediator in ['GM3', 'GM1', 'PIP2', 'CHOL']:
    run_analysis(
        topology=protein_system,
        trajectory=trajectory,
        mediator=mediator,
        targets=['DOPC', 'DOPE', 'DOPS', 'DPSM'],
        output=f'gpcr_{mediator}_effects/'
    )
```

## Advanced Features

### Custom Radii for Multi-scale Analysis
```python
analyzer.analyze_redistribution(
    membrane,
    radii=[3, 5, 7, 10, 15, 20, 25]  # Angstroms
)
```

### Specific Protein Analysis
```python
# Analyze each protein separately
for protein_name, protein_atoms in membrane.proteins.items():
    results = analyzer.analyze_protein_specific(
        membrane,
        protein=protein_name,
        mediator='GM3'
    )
```

### Statistical Validation
```python
# Increase bootstrap samples for publication
results = analyzer.analyze_with_bootstrap(
    membrane,
    n_bootstrap=10000,
    confidence_level=0.99
)
```

## Citation

If you use CLAIRE in your research, please cite:

```bibtex
@article{claire2025,
  title={CLAIRE: Conserved Lipid Analysis with Interaction and Redistribution Evaluation},
  author={Sato, Takeshi},
  journal={Journal of Open Source Software},
  year={2025},
  doi={10.21105/joss.XXXXX}
}
```

## License

MIT License - see [LICENSE](LICENSE) file

## Support

- **Issues**: [GitHub Issues](https://github.com/takeshi-sato-dev/claire/issues)
- **Documentation**: [Read the Docs](https://claire.readthedocs.io)
- **Email**: claire-support@example.com

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md)

## Acknowledgments

CLAIRE builds upon:
- [MDAnalysis](https://www.mdanalysis.org/) - Trajectory handling
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [NumPy](https://numpy.org/) & [SciPy](https://scipy.org/) - Numerical computing