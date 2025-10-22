# CLAIRE

**C**omposition-based **L**ipid **A**nalysis with **I**ntegrated **R**esolution and **E**nrichment

A comprehensive tool for analyzing lipid composition changes in molecular dynamics simulations with mass conservation, temporal tracking, spatial profiling, and machine learning prediction.

## Features

### Core Capabilities
- **Mass conservation-enforced composition analysis**: Ensures physically meaningful composition ratios
- **Leaflet-specific analysis**: Automatic detection and separate analysis of membrane leaflets
- **Parallel processing**: Efficient multi-core processing for large trajectories

### Advanced Analysis
1. **Temporal Analysis**
   - Sliding window composition tracking
   - Transition point detection
   - Autocorrelation analysis
   - Time-lagged correlation with mediators

2. **Spatial Analysis**
   - Radial composition profiles at multiple distances
   - Shell-by-shell composition comparison
   - Distance-dependent reorganization quantification

3. **Machine Learning**
   - Composition change prediction from mediator binding
   - Cross-system validation
   - Feature importance analysis
   - Multiple model comparison (Ridge, Lasso, Random Forest, Gradient Boosting)

### Visualization
- Publication-quality figures
- Comprehensive summary plots
- Temporal evolution plots
- Radial profile visualization
- ML prediction comparisons

## Installation

```bash
git clone https://github.com/yourusername/claire.git
cd claire
pip install -r requirements.txt
```

### Requirements
- Python >= 3.8
- MDAnalysis >= 2.0
- NumPy
- Pandas
- SciPy
- scikit-learn
- Matplotlib
- Seaborn

## Quick Start

### Configure Input Files

Edit `config.py` to set your default topology, trajectory, and frame range:

```python
# config.py
TOPOLOGY_FILE = "/path/to/your/system.psf"
TRAJECTORY_FILE = "/path/to/your/trajectory.xtc"

# Frame selection
FRAME_START = 20000  # Start frame
FRAME_STOP = 50000   # Stop frame (None = all)
FRAME_STEP = 10      # Frame step
```

### Basic Analysis

With config.py set up:
```bash
python run_claire.py \
    --lipids CHOL DIPC DPSM \
    --target-lipid DPG3 \
    --output results \
    --parallel
```

Or specify files directly:
```bash
python run_claire.py \
    --topology system.psf \
    --trajectory trajectory.xtc \
    --lipids CHOL DIPC DPSM \
    --target-lipid DPG3 \
    --output results \
    --parallel
```

### Python API

```python
from claire.core.trajectory_loader import load_universe, identify_lipid_leaflets
from claire.analysis.composition import CompositionAnalyzer
from claire.analysis.temporal import TemporalAnalyzer
from claire.analysis.spatial import SpatialAnalyzer

# Load trajectory
u = load_universe('system.psf', 'trajectory.xtc')
upper_leaflet, lower_leaflet = identify_lipid_leaflets(u)

# Analyze composition
analyzer = CompositionAnalyzer(['CHOL', 'DIPC', 'DPSM'], target_lipid='DPG3')
df = analyzer.frames_to_dataframe(frame_data_list)
results = analyzer.analyze_composition_changes(df)

# Temporal analysis
temporal = TemporalAnalyzer(window_size=100)
window_df = temporal.sliding_window_composition(df, ['CHOL', 'DIPC', 'DPSM'])

# Spatial analysis
spatial = SpatialAnalyzer(radii=[5.0, 10.0, 15.0, 20.0])
radial_profiles = spatial.calculate_radial_composition(u, frame_idx, ...)
```

## Command Line Options

```
--topology PATH         Topology file (required)
--trajectory PATH       Trajectory file (required)
--output DIR           Output directory (default: claire_output)
--lipids TYPE [TYPE...]  Lipid types to analyze (default: CHOL DIPC DPSM)
--target-lipid TYPE    Target/mediator lipid name
--start INT            Start frame (default: 0)
--stop INT             Stop frame (default: all)
--step INT             Frame step (default: 1)
--cutoff FLOAT         Contact cutoff in Angstroms (default: 15.0)
--parallel             Use parallel processing
--n-workers INT        Number of parallel workers (default: auto)
--skip-temporal        Skip temporal analysis
--skip-spatial         Skip spatial analysis
--skip-ml              Skip machine learning analysis
--leaflet CHOICE       Which leaflet: upper/lower/both (default: upper)
```

## Output Files

CLAIRE generates several output files:

**Data Files:**
- `composition_data.csv`: Frame-by-frame composition data
- `temporal_windows.csv`: Sliding window composition
- `frame_data.pkl`: Cached frame data for reanalysis

**Figures (saved as both PNG and SVG):**
- `composition_changes.png/.svg`: Bar chart of composition changes
- `temporal_composition.png/.svg`: Time evolution of composition
- `radial_profiles.png/.svg`: Distance-dependent composition
- `ml_predictions.png/.svg`: ML model predictions
- `comprehensive_summary.png/.svg`: Multi-panel summary figure

All figures are saved in both PNG (for presentations) and SVG (for publications) formats at 300 DPI.

## Key Concepts

### Mass Conservation

CLAIRE enforces mass conservation when calculating composition ratios:

```
For lipids L1, L2, L3:
ratio(L1) + ratio(L2) + ratio(L3) = 1.0
```

This ensures physically meaningful results when comparing high vs. low mediator conditions.

### Temporal Tracking

Sliding window analysis captures composition evolution over time:

```python
temporal = TemporalAnalyzer(window_size=100, step_size=10)
window_df = temporal.sliding_window_composition(df, lipid_types)
```

### Spatial Profiling

Radial composition profiles quantify distance-dependent reorganization:

```python
spatial = SpatialAnalyzer(radii=[5.0, 10.0, 15.0, 20.0])
radial_data = spatial.calculate_radial_composition(u, frame, ...)
```

### Machine Learning Prediction

Train models to predict composition changes:

```python
predictor = CompositionPredictor()
ml_results = predictor.predict_composition_changes(df, mediator_cols, lipid_types)
```

## Examples

See `examples/` directory for:
- `example_epha2.py`: EphA2 receptor analysis
- `example_notch.py`: Notch receptor analysis
- `example_cross_system.py`: Cross-system validation

## Citation

If you use CLAIRE in your research, please cite:

```
Sato, T. & Tamagaki-Asahina, H. (2025). CLAIRE: Composition-based Lipid Analysis
with Integrated Resolution and Enrichment for Membrane Reorganization Studies.
Journal of Computational Chemistry, XX(X), XXX-XXX.
```

## License

MIT License

## Contact

- Takeshi Sato: takeshi@mb.kyoto-phu.ac.jp
- GitHub: https://github.com/takeshi-sato-dev/claire
- Issues: https://github.com/takeshi-sato-dev/claire/issues

## Acknowledgments

CLAIRE builds upon insights from:
- LIPAC (Lipid-Protein interaction Analysis with Causal inference)
- MDAnalysis framework
- The membrane biophysics community
