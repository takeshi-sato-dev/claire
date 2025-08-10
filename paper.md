---
title: 'CLAIRE: Conserved Lipid Analysis with Interaction and Redistribution Evaluation'
tags:
  - Python
  - molecular dynamics
  - lipid bilayers
  - membrane proteins
  - GM3 ganglioside
  - machine learning
  - membrane biophysics
  - conservation laws
authors:
  - name: Takeshi Sato
    orcid: 0009-0006-9156-8655
    affiliation: 1
affiliations:
 - name: Kyoto Pharmaceutical University
   index: 1
date: 10 August 2025
bibliography: paper.bib
repository: https://github.com/takeshi-sato-dev/claire
archive_doi: 10.5281/zenodo.16787668  
---

# Summary

CLAIRE (Conserved Lipid Analysis with Interaction and Redistribution Evaluation) is a Python package for analyzing lipid redistribution around membrane proteins in molecular dynamics (MD) simulations. The package specifically addresses a fundamental problem in membrane analysis: quantifying how mediator molecules like GM3 ganglioside modulate the lipid composition in protein neighborhoods while respecting the physical constraint of mass conservation. When proteins are embedded in membranes, they create local perturbations that are further modified by mediator lipids, leading to complex redistribution patterns. Unlike traditional counting methods that can violate conservation principles, CLAIRE implements theoretically grounded algorithms that ensure the total lipid density remains constant during redistribution analysis, providing physically meaningful insights into protein-lipid and lipid-lipid interactions.

# Statement of need

Biological membranes are complex, heterogeneous structures where lipids organize into functional domains around membrane proteins, regulating crucial cellular processes [@Lingwood2010; @Sezgin2017]. The local lipid environment around proteins determines their structure, dynamics, and function. GM3 ganglioside, a key glycosphingolipid, plays a crucial role in modulating protein-lipid interactions by mediating the formation of lipid rafts and influencing the distribution of cholesterol, sphingomyelin, and other membrane components in protein neighborhoods [@Regina2018; @Sonnino2007]. Understanding how mediator lipids like GM3 reshape the lipid composition around specific proteins from MD simulations is essential for drug design, understanding disease mechanisms, and developing lipid-based therapeutics.

However, current analysis methods suffer from several fundamental limitations:

1. **Violation of conservation laws**: Traditional enrichment/depletion analysis counts lipids in specific regions without ensuring that total lipid number is conserved. This leads to physically impossible results where lipids appear to be created or destroyed.

2. **Lack of multi-scale analysis**: Lipid redistribution occurs across multiple length scales - from direct molecular contact (< 5 Å) to first solvation shells (5-10 Å) to longer-range influence zones (10-20 Å). Single-scale analysis misses important physics.

3. **Insufficient statistical rigor**: MD trajectories contain significant thermal fluctuations. Without proper statistical analysis, it's impossible to distinguish genuine redistribution from random noise.

4. **Protein heterogeneity**: Different membrane proteins induce distinct redistribution patterns. Averaging across all proteins masks important biological specificity.

CLAIRE addresses these fundamental challenges by implementing conservation-aware algorithms, multi-scale spatial analysis, rigorous statistical validation, and protein-specific characterization. The package provides a specialized framework for quantifying mediator-induced lipid redistribution that respects physical conservation laws while providing statistically robust results.

# Theory and Methods

## Conservation-Aware Analysis

The fundamental principle underlying CLAIRE is that lipid redistribution must conserve total mass. For a membrane patch containing N lipid types, the local composition at position **r** is:

$$\phi_i(\mathbf{r}) = \frac{n_i(\mathbf{r})}{\sum_{j=1}^{N} n_j(\mathbf{r})}$$

where $n_i(\mathbf{r})$ is the local number density of lipid type i. This formulation automatically satisfies the conservation constraint:

$$\sum_{i=1}^{N} \phi_i(\mathbf{r}) = 1 \quad \forall \mathbf{r}$$

When a mediator molecule M induces redistribution, the change in composition is:

$$\Delta\phi_i = \phi_i^{M+} - \phi_i^{M-}$$

where M+ and M- denote high and low mediator concentration regions. Conservation requires:

$$\sum_{i=1}^{N} \Delta\phi_i = 0$$

This means enrichment of one lipid type must be exactly balanced by depletion of others.

## Multi-Scale Spatial Analysis

CLAIRE implements a radial decomposition around protein centers to quantify lipid redistribution at different distances:

$E_i(r) = \frac{\rho_i^{local}(r)}{\rho_i^{bulk}}$

where the local density is calculated within shells around the protein:

$\rho_i^{local}(r) = \frac{1}{A(r)} \sum_{j \in \text{lipid } i} \Theta(||\mathbf{r}_j - \mathbf{r}_{protein}|| - r)$

with $\Theta$ being the Heaviside function, $\mathbf{r}_{protein}$ the protein center of mass, and A(r) the shell area. This allows analysis at multiple scales from the protein surface:
- Contact zone (r < 5 Å): Direct protein-lipid interactions
- First shell (5 Å < r < 10 Å): Solvation and boundary lipids
- Influence zone (10 Å < r < 20 Å): Mediator-modulated redistribution
- Bulk region (r > 20 Å): Unperturbed membrane composition

## Machine Learning for Effect Quantification

CLAIRE uses Random Forest regression to predict lipid responses from mediator features:

$$y_i = f(X_{mediator}) + \epsilon$$

where $y_i$ is the local density of lipid i and $X_{mediator}$ includes:
- Mediator contact strength: $S = \sum_j w_j \cdot \exp(-d_j/\lambda)$
- Mediator density: $\rho_{med} = N_{med}/V_{local}$
- Mediator clustering: $C = \langle d_{ij} \rangle^{-1}$

The model learns non-linear relationships between mediator presence and lipid redistribution, providing interpretable feature importance scores.

## Statistical Validation

CLAIRE implements bootstrap confidence intervals for all reported effects:

1. **Resampling**: Generate B bootstrap samples by resampling frames with replacement
2. **Effect calculation**: Compute $\Delta\phi_i^{(b)}$ for each bootstrap sample b
3. **Confidence interval**: Report the α/2 and 1-α/2 percentiles as CI bounds

Additionally, CLAIRE performs permutation tests to assess significance:
- Null hypothesis: No association between mediator and lipid distribution
- Test statistic: Correlation between mediator strength and lipid density
- P-value: Fraction of permutations with stronger correlation than observed

## Temporal Smoothing and Correlation

To handle temporal correlations in MD trajectories, CLAIRE implements:

1. **Savitzky-Golay filtering**: Preserves features while reducing noise
2. **Autocorrelation analysis**: Determines effective sample size
3. **Block averaging**: Ensures statistical independence

The effective number of independent samples is:

$$N_{eff} = \frac{N_{frames}}{1 + 2\tau_{int}}$$

where $\tau_{int}$ is the integrated autocorrelation time.

# Implementation

CLAIRE is built on MDAnalysis [@Gowers2016] for trajectory handling, with a modular architecture designed for extensibility:

**Core modules**:
- `membrane.py`: System setup, leaflet identification using graph-based clustering
- `physics.py`: Distance calculations with periodic boundary conditions
- `temporal.py`: Time series analysis, smoothing, autocorrelation
- `ml_analysis.py`: Random Forest models, feature engineering
- `conservation.py`: Conservation-enforcing redistribution metrics
- `figures.py`: Publication-quality visualizations

**Key algorithms**:
- **Leaflet identification**: Graph-based clustering using lipid connectivity
- **PBC handling**: Minimum image convention for all distance calculations
- **Parallel processing**: OpenMP-based frame processing with shared memory
- **Memory efficiency**: Streaming analysis for large trajectories

**Performance optimizations**:
- Vectorized NumPy operations for distance calculations
- KDTree spatial indexing for neighbor searches
- Cached membrane properties to avoid recomputation
- Incremental statistics for online analysis

# Validation and Testing

CLAIRE has been validated through:

1. **Conservation tests**: Verified that $\sum_i \Delta\phi_i = 0$ for all analyses
2. **Synthetic systems**: Known redistribution patterns correctly recovered
3. **Comparison with published data**: Reproduced GM3-cholesterol enrichment [@Lingwood2010]
4. **Statistical power analysis**: Determined minimum frames needed for significance

# Example Application

```python
from claire import MembraneSystem
from claire.analysis import MLAnalyzer, ConservationAnalyzer

# Load membrane system with embedded proteins
membrane = MembraneSystem('membrane_protein.psf', 'trajectory.xtc')
membrane.identify_proteins()  # Find GPCR, channels, etc.
membrane.identify_lipids()    # Find DOPC, CHOL, GM3, etc.

# Analyze GM3-mediated redistribution around each protein
analyzer = MLAnalyzer()
for protein_name, protein_atoms in membrane.proteins.items():
    results = analyzer.analyze_redistribution(
        membrane, 
        protein=protein_name,
        mediator='GM3',
        targets=['CHOL', 'DOPC', 'DPSM'],
        radii=[5, 10, 15, 20]  # Distance from protein surface
    )
    
    # Verify conservation around this protein
    conservation = ConservationAnalyzer()
    assert conservation.check_conservation(results), f"Conservation violated for {protein_name}!"

# Compare lipid environments between proteins
from claire.visualization import FigureGenerator
fig_gen = FigureGenerator()
fig_gen.create_protein_comparison_figures(all_results, bootstrap_samples=1000)
```

# Conclusions

CLAIRE provides a theoretically grounded, statistically robust framework for analyzing mediator-induced lipid redistribution around membrane proteins in MD simulations. By enforcing conservation laws and implementing multi-scale analysis with machine learning, CLAIRE enables quantitative characterization of how mediator lipids like GM3 modulate the local lipid environment of individual proteins. This protein-centric approach reveals how different proteins create distinct lipid microenvironments and how these are further shaped by mediator molecules. The package is designed to be extensible, allowing researchers to analyze any membrane protein system with any mediator molecule, providing insights into protein-lipid interactions critical for membrane biology and drug discovery.

# Acknowledgements

We acknowledge contributions and support from Kyoto Pharmaceutical University Fund for the Promotion of Collaborative Research.

# References