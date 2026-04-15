# CLAIRE reveals the temporal causal chain of lipid domain formation around a transmembrane protein

Takeshi Sato*

Kyoto Pharmaceutical University, 5 Misasaginakauchi-cho, Yamashina-ku, Kyoto 607-8414, Japan

*Email: takeshi@mb.kyoto-phu.ac.jp

---

## Abstract

The mechanism by which lipid binding to a transmembrane protein reorganizes the surrounding membrane remains unresolved. Total effect analyses cannot distinguish direct displacement from indirect pathways, and static decompositions cannot determine causal direction. Here, CLAIRE (Compositional Lipid Analysis with Inference of Remodeling Effects) combines hierarchical Bayesian mediation decomposition with Granger causality, conditional Granger causality, and transfer entropy to resolve the temporal causal structure of membrane compositional changes at two spatial scales: the first shell (6 Angstrom contact) and the surrounding cylinder (15 Angstrom radius). Applied to GM3 ganglioside binding to the EGFR transmembrane domain in coarse-grained molecular dynamics simulations (two independent 10 microsecond trajectories), CLAIRE identifies a four-step causal chain: GM3 binding excludes unsaturated lipids (DIPC) from the first shell, sphingomyelin (DPSM) fills the vacated space, local membrane ordering increases (measured by acyl chain order parameter S_CD), and cholesterol subsequently stabilizes the ordered environment. This chain is reproduced across replicates and absent in Notch. The framework provides the first temporal causal decomposition of lipid domain formation around a membrane protein.

---

## Main Text

### Introduction

Lipid domains in biological membranes, enriched in cholesterol and saturated sphingolipids, modulate the activity of transmembrane receptors.^1,2 GM3 ganglioside regulates EGFR kinase activation specifically in membranes poised for phase separation,^3 but whether GM3 binding actively constructs the local ordered environment or merely occurs within preexisting domains has remained an open question.^4,5 Molecular dynamics simulations of multicomponent membranes provide the spatiotemporal resolution to address this question, yet existing analytical frameworks report only total compositional differences between bound and unbound states,^6 which conflate direct displacement with indirect pathways involving third lipid species.

Two distinct gaps prevent mechanistic interpretation. The first is structural: the total compositional change cannot be decomposed into contributions from individual lipid species. The second is temporal: even when a static decomposition identifies an indirect pathway, the causal direction (does the binding event recruit the mediator, or does the mediator create the environment where binding occurs?) remains undetermined. Standard Granger causality^7 can detect temporal precedence but cannot distinguish genuine causation from spurious associations produced by domain dynamics, where a liquid-ordered domain arriving near the protein simultaneously changes both variables.

Here, CLAIRE addresses both gaps through a framework that operates at two spatial scales (6 Angstrom first shell contacts and 15 Angstrom local composition) and two analytical levels (static mediation decomposition and temporal causal analysis). The temporal analysis combines Granger causality with conditional Granger causality^8 (which controls for domain coarrival by conditioning on a third variable) and transfer entropy^9 (which captures nonlinear dependencies). Validated on synthetic data with known causal structure, CLAIRE is applied to GM3 binding in EGFR and Notch transmembrane domains, revealing a four-step causal chain from first shell lipid displacement to local domain formation.

### Results

**Static mediation decomposition separates direct and indirect pathways.** CLAIRE quantifies lipid composition within a 15 Angstrom cylinder around each protein copy at every trajectory frame and decomposes the total compositional change upon GM3 binding into direct and indirect components through mediation analysis on raw lipid counts (Fig. 1a, Methods). In EGFR, 42% of unsaturated lipid (DIPC) displacement upon GM3 binding is associated with an indirect pathway through cholesterol (CHOL count change +0.47, 4/4 copies positive in Replicate 2). In Notch, no indirect pathway exists (CHOL count change +0.09, 2+/2-), and the apparent cholesterol enrichment visible in mole fractions is an artifact of the unit sum constraint (Fig. 1b, Supplementary Fig. 1).

**Synthetic validation confirms detection of causal direction and chain structure.** Four basic scenarios (active, selective, independent, domain dynamics) are correctly classified by Granger causality with surrogate testing (Fig. 2a). Conditional Granger causality correctly identifies domain dynamics scenarios where standard Granger causality detects a spurious association. A synthetic causal chain (distance -> first shell DIPC -> S_CD -> CHOL) is correctly decomposed: each direct link shows forward Granger causality (F = 119 to 2462), F statistics decay with causal distance (direct > 1 hop > 2 hops), and null controls show no significant relationship (Fig. 2b,c). Transfer entropy, while mathematically equivalent to Granger causality for linear Gaussian systems,^10 requires substantially more data for equivalent detection power (Supplementary Fig. 2), consistent with theory. Granger causality serves as the primary detection method.

**GM3 contact distance does not predict compositional changes.** The minimum bead to bead distance between GM3 and the TM domain fluctuates continuously between 3 and 6 Angstrom during the bound state (occupancy 90 to 100%). Applied to all variable pairs across both replicates, temporal analysis detects no causal relationship between GM3 distance and any 15 Angstrom compositional variable (CHOL, DPSM, DIPC counts, local S_CD) in either direction (Fig. 3a, Supplementary Table 1). The static indirect pathway (42%) does not reflect a temporal causal relationship between GM3 contact fluctuations and cholesterol changes.

**First shell DIPC displacement precedes DPSM arrival.** CLAIRE measures first shell contacts (6 Angstrom bead to bead distance) for each lipid type at every frame. Temporal analysis of first shell variables reveals that DIPC departure from the first shell precedes DPSM arrival (Granger causality in the DIPC to DPSM direction, F = 104 to 284 in Replicate 1; F = 33 selective in Replicate 2; Fig. 3b). GM3 binding creates a vacancy by excluding unsaturated DIPC, and sphingomyelin fills the space.

**DPSM at the first shell orders the surrounding membrane.** First shell DPSM count predicts subsequent increases in local S_CD (forward/reverse F ratio 2 to 9, reproduced across both replicates; Fig. 3c). In Replicate 1, three of four copies show forward dominant Granger causality (F = 212 to 292 forward vs. F = 59 to 136 reverse). In Replicate 2, three of four copies are forward dominant (F = 99 to 145 forward, ratio 6 to 9), with two classified as active recruitment (forward significant, reverse not significant). At the 15 Angstrom scale, DPSM count changes predict S_CD changes in the DPSM to ordering direction (4/4 copies selective binding in Replicate 2, F = 99 to 206; Fig. 3d). DPSM, not cholesterol, is the primary cause of ordering.

**Cholesterol stabilizes the ordered environment.** S_CD to CHOL temporal analysis consistently shows the reverse direction: CHOL count changes precede S_CD changes (selective binding classification in both replicates; Fig. 3e). Cholesterol does not initiate ordering but arrives after DPSM has begun to order the membrane, stabilizing the nascent liquid-ordered environment. This is consistent with the observation that GM3 regulates EGFR only in cholesterol-containing membranes:^3 without cholesterol, DPSM alone cannot maintain a stable ordered domain.

**The causal chain is absent in Notch.** Notch exhibits no indirect pathway in the static decomposition, and temporal analysis detects no systematic causal relationships between first shell variables and local composition (Supplementary Table 2). GM3 binding to Notch displaces DIPC directly without triggering the DPSM replacement and ordering cascade observed in EGFR.

### Discussion

CLAIRE resolves a four-step causal chain linking a specific lipid binding event to local domain formation around a transmembrane protein (Fig. 4):

Step 1: GM3 binds the EGFR TM domain and excludes DIPC from the first shell (established by LIPAC^6).

Step 2: DPSM fills the space vacated by DIPC at the first shell (temporal: fs_DIPC -> fs_DPSM).

Step 3: DPSM at the first shell orders the surrounding membrane (temporal: fs_DPSM -> S_CD, forward/reverse ratio 2 to 9).

Step 4: Cholesterol arrives and stabilizes the ordered environment (temporal: CHOL -> S_CD, selective direction).

This chain provides a mechanistic basis for the experimental observation that GM3 inhibits EGFR activation only in membranes containing cholesterol.^3 Without cholesterol, Step 4 fails: DPSM initiates ordering but cannot maintain it. The domain dissipates. With cholesterol, the ordered environment is stabilized, the EGFR transmembrane domain is trapped in a liquid-ordered context, and kinase activation is suppressed.

The distinction between DPSM as initiator and cholesterol as stabilizer has not been previously accessible. Static analyses, including LIPAC v2 cooperative mode analysis,^6 detect that cholesterol and DPSM arrive together (cooperative response) but cannot resolve the temporal order. CLAIRE decomposes this cooperative response into its sequential components.

The absence of this chain in Notch demonstrates that the causal structure is specific to the receptor. Notch does not engage in DPSM replacement upon GM3 binding, and no ordering cascade follows. This specificity may reflect structural differences in the transmembrane domains of the two receptors.

The framework has limitations. The coarse-grained representation (MARTINI 2.2^11,12) captures lateral lipid organization but not atomistic detail. The 2 ns frame interval limits temporal resolution to processes slower than approximately 4 ns. First shell lipids are also counted in the 15 Angstrom cylinder, introducing partial measurement overlap between the two spatial scales. The number of independent protein copies (3 to 4 per trajectory) is modest, and several hierarchical Bayesian highest density intervals span zero. Granger causality detects linear temporal precedence; nonlinear causal mechanisms, if present, may require larger datasets for detection by transfer entropy.

The CLAIRE framework is applicable to any multicomponent membrane simulation where the causal structure of compositional changes upon ligand binding requires interpretation. The two-stage architecture (trajectory extraction followed by statistical analysis on a single CSV file) allows rapid exploration of variable combinations without reprocessing the trajectory. The analysis code is freely available at https://github.com/takeshi-sato-dev/CLAIRE.

---

## Methods

### Simulation systems

Two coarse-grained MARTINI 2.2^11,12 systems were simulated in GROMACS 2023.3.^13 (i) EGFR transmembrane and juxtamembrane domain (residues 610 to 673, 4 copies) in a 40 x 40 nm membrane (upper leaflet: CHOL/DIPC/DPSM 33/33/33 mol% with 1 mol% GM3; lower leaflet: CHOL/DIPC/DOPS 33/33/33 mol%). Two independent 10 microsecond trajectories with different velocity seeds. (ii) Notch transmembrane and juxtamembrane domain (4 copies), identical composition, 8 microsecond production. Analysis: 2 to 8 microsecond interval, 3000 frames per copy at 2 ns intervals.

### CLAIRE Stage 1: Variable extraction

At each frame, for each protein copy, the following variables are computed:

(a) Lipid counts within a 15 Angstrom lateral (xy) cylinder centered on the TM domain center of mass (residues 621 to 644). GM3 is excluded.

(b) First shell contacts: the number of lipid molecules of each type with any bead within 6.0 Angstrom of any TM domain bead.

(c) Local S_CD: the mean acyl chain order parameter of DIPC and DPSM within the 15 Angstrom cylinder. For each lipid residue, S_CD = (3 <cos^2 theta> - 1) / 2, where theta is the angle between consecutive tail bead vectors and the membrane normal (z axis). Cholesterol and GM3 are excluded.

(d) TM helix tilt: the angle between the principal axis of TM backbone beads and the z axis, computed by singular value decomposition.

(e) GM3 contact distance: the minimum bead to bead distance between GM3 and the TM domain, with periodic boundary correction and 15 Angstrom lateral prescreening.

(f) GM3 binding state: bound if any GM3 bead is within 6.0 Angstrom of any protein bead (LIPAC criterion^6).

### CLAIRE Stage 2: Static mediation decomposition

For each protein copy, the mean composition difference between bound and unbound states yields a per copy total effect with autocorrelation corrected standard error. A hierarchical Bayesian model estimates the population effect: Delta_j ~ Normal(mu, sqrt(sigma^2_between + se^2_j)), with weakly informative priors mu ~ Normal(0, 0.5) and sigma_between ~ HalfNormal(0.3). Posterior inference uses the No-U-Turn Sampler^14 (4 chains, 2000 warmup, 2000 sampling). The mediation decomposition operates on raw lipid counts: n_Y = a + beta_direct * T + beta_coupling * n_M + epsilon. The indirect effect is Total minus Direct. Mediator selection evaluates all lipid types by count change sign consistency and coupling magnitude.

### CLAIRE Stage 3: Temporal causal analysis

Bound state frames are extracted per copy. For each pair of variables (x, y), stationarity is ensured by Augmented Dickey Fuller testing (differencing if needed). Granger causality tests whether the past of x improves prediction of y beyond the autoregressive model of y alone, across lags 1 to 10 (Bonferroni corrected). Statistical significance is assessed by phase randomized surrogate testing (50 surrogates): the Fourier phases of x are randomized (preserving autocorrelation structure) and the test statistic is recomputed, yielding a nonparametric p value. Conditional Granger causality adds a third variable z (e.g., DIPC count when testing CHOL) to control for domain coarrival. Transfer entropy is computed as the conditional mutual information between x_past and y_future given y_past, validated against the same surrogate distribution. Classification: if forward Granger causality is significant and reverse is not, the pair is classified as active; reverse only as selective; both as feedback; neither as independent. If standard Granger causality is significant but conditional Granger causality is not, the classification is domain dynamics (apparent causality driven by a common factor).

---

## References

(1) Lingwood, D.; Simons, K. Lipid Rafts As a Membrane-Organizing Principle. *Science* **2010**, *327*, 46-50.

(2) Sezgin, E.; Levental, I.; Mayor, S.; Eggeling, C. The Mystery of Membrane Organization: Composition, Regulation and Roles of Lipid Rafts. *Nat. Rev. Mol. Cell Biol.* **2017**, *18*, 361-374.

(3) Coskun, U.; Grzybek, M.; Drechsel, D.; Simons, K. Regulation of Human EGF Receptor by Lipids. *Proc. Natl. Acad. Sci. U. S. A.* **2011**, *108*, 9044-9048.

(4) Veatch, S. L.; Keller, S. L. Separation of Liquid Phases in Giant Vesicles of Ternary Mixtures of Phospholipids and Cholesterol. *Biophys. J.* **2003**, *85*, 3074-3083.

(5) Ingolfsson, H. I.; Melo, M. N.; van Eerden, F. J.; Arnarez, C.; Lopez, C. A.; Wassenaar, T. A.; Periole, X.; de Vries, A. H.; Tieleman, D. P.; Marrink, S. J. Lipid Organization of the Plasma Membrane. *J. Am. Chem. Soc.* **2014**, *136*, 14554-14559.

(6) Sato, T. A Computational Framework for Causal Inference in Molecular Dynamics Analysis of Lipid-Protein Interactions. *J. Chem. Inf. Model.* **2026**, in press.

(7) Granger, C. W. J. Investigating Causal Relations by Econometric Models and Cross-Spectral Methods. *Econometrica* **1969**, *37*, 424-438.

(8) Geweke, J. Measurement of Linear Dependence and Feedback Between Multiple Time Series. *J. Am. Stat. Assoc.* **1982**, *77*, 304-313.

(9) Schreiber, T. Measuring Information Transfer. *Phys. Rev. Lett.* **2000**, *85*, 461-464.

(10) Barnett, L.; Barrett, A. B.; Seth, A. K. Granger Causality and Transfer Entropy Are Equivalent for Gaussian Variables. *Phys. Rev. Lett.* **2009**, *103*, 238701.

(11) Marrink, S. J.; Risselada, H. J.; Yefimov, S.; Tieleman, D. P.; de Vries, A. H. The MARTINI Force Field: Coarse Grained Model for Biomolecular Simulations. *J. Phys. Chem. B* **2007**, *111*, 7812-7824.

(12) Monticelli, L.; Kandasamy, S. K.; Periole, X.; Larson, R. G.; Tieleman, D. P.; Marrink, S. J. The MARTINI Coarse-Grained Force Field: Extension to Proteins. *J. Chem. Theory Comput.* **2008**, *4*, 819-834.

(13) Abraham, M. J.; Murtola, T.; Schulz, R.; Pall, S.; Smith, J. C.; Hess, B.; Lindahl, E. GROMACS: High Performance Molecular Simulations through Multi-Level Parallelism from Laptops to Supercomputers. *SoftwareX* **2015**, *1-2*, 19-25.

(14) Hoffman, M. D.; Gelman, A. The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. *J. Mach. Learn. Res.* **2014**, *15*, 1593-1623.

(15) Baron, R. M.; Kenny, D. A. The Moderator-Mediator Variable Distinction in Social Psychological Research. *J. Pers. Soc. Psychol.* **1986**, *51*, 1173-1182.

(16) Aitchison, J. The Statistical Analysis of Compositional Data. *J. R. Stat. Soc. Ser. B* **1982**, *44*, 139-160.

---

## Figure Legends

**Figure 1.** CLAIRE framework and static mediation decomposition. (a) Schematic of the two-stage pipeline: Stage 1 extracts variables at two spatial scales (6 Angstrom first shell, 15 Angstrom cylinder) from each trajectory frame; Stage 2 performs hierarchical Bayesian mediation decomposition and temporal causal analysis. (b) Mediation decomposition on counts: EGFR exhibits a 42% indirect pathway through cholesterol (left), while Notch shows direct displacement only (right). The apparent cholesterol enrichment in Notch fractions is an artifact of the unit sum constraint.

**Figure 2.** Synthetic validation. (a) Four basic scenarios correctly classified by Granger causality: active (F = 157 forward), selective (F = 179 reverse), independent, and domain dynamics. (b) Full causal chain (distance -> first shell DIPC -> S_CD -> CHOL): each direct link detected with F = 440 to 2462. F statistics decay with causal distance (direct > transitive). Null controls are independent. (c) Pathway diagram with F statistics for each link.

**Figure 3.** Temporal causal analysis of EGFR. (a) GM3 contact distance shows no temporal causal relationship with any compositional variable (all independent, both replicates). (b) First shell DIPC departure precedes DPSM arrival (Granger causality in reverse direction, F = 104 to 284). (c) First shell DPSM predicts ordering increase (forward/reverse F ratio 2 to 9, reproduced). (d) DPSM count predicts S_CD at 15 Angstrom scale (4/4 selective in Replicate 2). (e) Cholesterol arrival follows ordering: CHOL -> S_CD direction dominant (selective binding).

**Figure 4.** The causal chain of lipid domain formation around EGFR. GM3 binding excludes DIPC from the first shell (LIPAC). DPSM fills the vacated space (temporal causal). DPSM orders the surrounding membrane (temporal causal, forward/reverse ratio 2 to 9). Cholesterol arrives and stabilizes the ordered environment (temporal causal, selective direction). This chain is absent in Notch.
