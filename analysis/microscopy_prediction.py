#!/usr/bin/env python3
"""
Predict large-scale lipid organization and simulate fluorescence microscopy

Extrapolates from 15Å MD data to microscopy-observable scales (50-200 nm)
and generates simulated fluorescence images
"""

import numpy as np
import pandas as pd
from scipy import ndimage, spatial
from scipy.special import erf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import warnings


class MicroscopyPredictor:
    """Predict large-scale lipid organization from local MD data"""

    def __init__(self, base_radius=15.0, prediction_radii=[50.0, 100.0, 200.0]):
        """Initialize predictor

        Parameters
        ----------
        base_radius : float, default 15.0
            Base radius from MD simulation (Å)
        prediction_radii : list of float
            Target radii for prediction (Å)
        """
        self.base_radius = base_radius
        self.prediction_radii = prediction_radii
        self.gp_models = {}

    def extrapolate_composition(self, base_composition, target_radius, model='diffusion'):
        """Extrapolate composition from base radius to target radius

        Parameters
        ----------
        base_composition : dict
            Lipid composition at base radius (e.g., {'CHOL': 0.3, 'DPSM': 0.2, 'DIPC': 0.5})
        target_radius : float
            Target radius for extrapolation (Å)
        model : str, default 'diffusion'
            Extrapolation model ('diffusion', 'exponential', 'linear')

        Returns
        -------
        dict
            Predicted composition at target radius
        """
        if target_radius <= self.base_radius:
            return base_composition.copy()

        ratio = target_radius / self.base_radius

        predicted = {}

        if model == 'diffusion':
            # Diffusion-based decay: composition converges to bulk average
            # Assume bulk composition is more uniform
            for lipid, fraction in base_composition.items():
                # Decay factor based on diffusion equation
                # C(r) = C0 + (C_local - C0) * exp(-r/lambda)
                # where lambda is characteristic length scale
                lambda_scale = 50.0  # Å

                decay = np.exp(-(target_radius - self.base_radius) / lambda_scale)
                bulk_average = 1.0 / len(base_composition)  # Assume uniform bulk

                predicted[lipid] = bulk_average + (fraction - bulk_average) * decay

        elif model == 'exponential':
            # Simple exponential decay
            decay = np.exp(-0.01 * (target_radius - self.base_radius))
            for lipid, fraction in base_composition.items():
                predicted[lipid] = fraction * decay + (1.0 / len(base_composition)) * (1 - decay)

        elif model == 'linear':
            # Linear interpolation to uniform distribution
            alpha = min((target_radius - self.base_radius) / 200.0, 1.0)
            uniform = 1.0 / len(base_composition)
            for lipid, fraction in base_composition.items():
                predicted[lipid] = fraction * (1 - alpha) + uniform * alpha

        # Normalize to ensure sum = 1
        total = sum(predicted.values())
        if total > 0:
            predicted = {k: v / total for k, v in predicted.items()}

        return predicted

    def predict_spatial_distribution(self, df, protein_positions, box_size=400,
                                    grid_spacing=2.0, target_radius=100.0):
        """Predict spatial distribution of lipids at target scale

        Parameters
        ----------
        df : pandas.DataFrame
            Composition data with protein positions
        protein_positions : dict
            {protein_name: (x, y, z)} positions in Å
        box_size : float
            Size of simulation box (Å)
        grid_spacing : float
            Spacing of prediction grid (Å)
        target_radius : float
            Prediction radius (Å)

        Returns
        -------
        dict
            Predicted lipid distributions on 2D grid
        """
        # Create 2D grid
        n_grid = int(box_size / grid_spacing)
        x = np.linspace(0, box_size, n_grid)
        y = np.linspace(0, box_size, n_grid)
        X, Y = np.meshgrid(x, y)

        # Initialize composition grids
        lipid_types = [col.replace('_fraction', '') for col in df.columns if '_fraction' in col]
        composition_grids = {lipid: np.zeros((n_grid, n_grid)) for lipid in lipid_types}

        # For each grid point, predict composition based on nearby proteins
        for i in range(n_grid):
            for j in range(n_grid):
                grid_point = np.array([X[i, j], Y[i, j]])

                # Find contribution from each protein
                total_weight = 0
                weighted_composition = {lipid: 0.0 for lipid in lipid_types}

                for protein_name, (px, py, pz) in protein_positions.items():
                    protein_pos = np.array([px, py])

                    # Distance from grid point to protein (with PBC)
                    diff = grid_point - protein_pos
                    # PBC correction
                    diff = diff - box_size * np.round(diff / box_size)
                    dist = np.linalg.norm(diff)

                    if dist < target_radius * 2:  # Consider proteins within 2x target radius
                        # Weight by inverse distance
                        weight = np.exp(-dist / target_radius)

                        # Get base composition for this protein
                        protein_data = df[df['protein'] == protein_name]
                        if len(protein_data) > 0:
                            base_comp = {}
                            for lipid in lipid_types:
                                col = f'{lipid}_fraction'
                                if col in protein_data.columns:
                                    base_comp[lipid] = protein_data[col].mean()

                            # Extrapolate to target radius
                            predicted_comp = self.extrapolate_composition(
                                base_comp, dist, model='diffusion'
                            )

                            # Add weighted contribution
                            for lipid, fraction in predicted_comp.items():
                                weighted_composition[lipid] += weight * fraction

                            total_weight += weight

                # Normalize
                if total_weight > 0:
                    for lipid in lipid_types:
                        composition_grids[lipid][i, j] = weighted_composition[lipid] / total_weight
                else:
                    # Far from all proteins, use bulk average
                    for lipid in lipid_types:
                        composition_grids[lipid][i, j] = 1.0 / len(lipid_types)

        return {
            'grids': composition_grids,
            'x': x,
            'y': y,
            'box_size': box_size,
            'target_radius': target_radius
        }


class FluorescenceMicroscope:
    """Simulate fluorescence microscopy images"""

    def __init__(self, resolution=20.0, wavelength=550, NA=1.4, pixel_size=10.0):
        """Initialize microscope simulator

        Parameters
        ----------
        resolution : float, default 20.0
            Microscope resolution (nm) - FWHM of PSF
        wavelength : float, default 550
            Emission wavelength (nm)
        NA : float, default 1.4
            Numerical aperture
        pixel_size : float, default 10.0
            Detector pixel size (nm)
        """
        self.resolution = resolution  # nm
        self.wavelength = wavelength  # nm
        self.NA = NA
        self.pixel_size = pixel_size  # nm

        # Calculate PSF parameters
        # Airy disk radius: r = 0.61 * lambda / NA
        self.psf_radius = 0.61 * wavelength / NA  # nm

    def generate_psf(self, size=51):
        """Generate Point Spread Function

        Parameters
        ----------
        size : int
            Size of PSF kernel (pixels, should be odd)

        Returns
        -------
        numpy.ndarray
            Normalized PSF kernel
        """
        center = size // 2
        y, x = np.ogrid[-center:center+1, -center:center+1]

        # Convert to nm
        x_nm = x * self.pixel_size
        y_nm = y * self.pixel_size

        r = np.sqrt(x_nm**2 + y_nm**2)

        # Gaussian approximation of Airy disk
        sigma = self.resolution / 2.355  # Convert FWHM to sigma
        psf = np.exp(-(r**2) / (2 * sigma**2))

        # Normalize
        psf = psf / np.sum(psf)

        return psf

    def simulate_image(self, true_distribution, noise_level=0.1, photon_count=1000):
        """Simulate fluorescence microscopy image

        Parameters
        ----------
        true_distribution : numpy.ndarray
            True molecular distribution (concentration map)
        noise_level : float, default 0.1
            Relative noise level (0-1)
        photon_count : float, default 1000
            Average photon count per pixel

        Returns
        -------
        numpy.ndarray
            Simulated microscopy image
        """
        # Apply PSF (diffraction limit)
        psf = self.generate_psf()
        blurred = ndimage.convolve(true_distribution, psf, mode='constant')

        # Scale to photon counts
        image = blurred * photon_count / np.mean(blurred)

        # Add Poisson noise (shot noise)
        image = np.random.poisson(image).astype(float)

        # Add Gaussian noise (detector noise)
        gaussian_noise = np.random.normal(0, noise_level * photon_count, image.shape)
        image = image + gaussian_noise

        # Clip negative values
        image = np.maximum(image, 0)

        return image

    def simulate_bead_distribution(self, lipid_positions, box_size, grid_size=512):
        """Simulate individual lipid molecules as fluorescent beads

        Parameters
        ----------
        lipid_positions : numpy.ndarray
            N x 2 array of (x, y) positions in nm
        box_size : float
            Size of imaging area (nm)
        grid_size : int
            Size of output image (pixels)

        Returns
        -------
        numpy.ndarray
            Simulated image with individual beads
        """
        # Create empty image
        image = np.zeros((grid_size, grid_size))

        # Convert positions to pixels
        scale = grid_size / box_size
        pixel_positions = (lipid_positions * scale).astype(int)

        # Place beads
        psf = self.generate_psf(size=11)  # Smaller PSF for individual beads
        psf_half = psf.shape[0] // 2

        for px, py in pixel_positions:
            if 0 <= px < grid_size and 0 <= py < grid_size:
                # Add PSF around this position
                x_min = max(0, px - psf_half)
                x_max = min(grid_size, px + psf_half + 1)
                y_min = max(0, py - psf_half)
                y_max = min(grid_size, py + psf_half + 1)

                psf_x_min = psf_half - (px - x_min)
                psf_x_max = psf_half + (x_max - px)
                psf_y_min = psf_half - (py - y_min)
                psf_y_max = psf_half + (y_max - py)

                image[y_min:y_max, x_min:x_max] += psf[psf_y_min:psf_y_max, psf_x_min:psf_x_max]

        # Add noise
        image = self.simulate_image(image, noise_level=0.05, photon_count=500)

        return image

    def generate_bead_positions_from_composition(self, composition_grid, n_lipids_per_unit=10):
        """Generate individual lipid positions from composition grid

        Parameters
        ----------
        composition_grid : numpy.ndarray
            2D composition map (0-1)
        n_lipids_per_unit : int
            Number of lipids per unit composition

        Returns
        -------
        numpy.ndarray
            N x 2 array of lipid positions
        """
        positions = []

        grid_shape = composition_grid.shape
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                n_lipids = int(composition_grid[i, j] * n_lipids_per_unit)

                # Generate random positions within this grid cell
                for _ in range(n_lipids):
                    # Add random offset within the cell
                    x = j + np.random.uniform(0, 1)
                    y = i + np.random.uniform(0, 1)
                    positions.append([x, y])

        return np.array(positions)


class MicroscopyAnalyzer:
    """Comprehensive microscopy prediction and simulation"""

    def __init__(self):
        self.predictor = MicroscopyPredictor()
        self.microscope = FluorescenceMicroscope()

    def analyze_and_simulate(self, df, target_radii=[50.0, 100.0, 200.0]):
        """Complete analysis: predict large-scale and simulate microscopy

        Parameters
        ----------
        df : pandas.DataFrame
            Composition data
        target_radii : list of float
            Target prediction radii (Å)

        Returns
        -------
        dict
            Complete results including predictions and simulated images
        """
        print("\n" + "="*70)
        print("MICROSCOPY PREDICTION AND SIMULATION")
        print("="*70)

        results = {}

        # Get protein positions (use COM from first frame as representative)
        proteins = df['protein'].unique()

        # For simplicity, assume proteins are evenly spaced
        # In real analysis, use actual positions from trajectory
        box_size = 400.0  # Å (40 nm)
        n_proteins = len(proteins)

        protein_positions = {}
        for i, protein_name in enumerate(sorted(proteins)):
            angle = 2 * np.pi * i / n_proteins
            radius = box_size / 4
            x = box_size / 2 + radius * np.cos(angle)
            y = box_size / 2 + radius * np.sin(angle)
            z = 0
            protein_positions[protein_name] = (x, y, z)

        print(f"Analyzing {n_proteins} proteins in {box_size:.1f} Å box")

        for target_radius in target_radii:
            print(f"\nPredicting at {target_radius:.1f} Å ({target_radius/10:.1f} nm)...")

            # Predict spatial distribution
            spatial_dist = self.predictor.predict_spatial_distribution(
                df, protein_positions, box_size=box_size,
                grid_spacing=2.0, target_radius=target_radius
            )

            results[target_radius] = {
                'spatial_distribution': spatial_dist,
                'microscopy_images': {}
            }

            # Simulate microscopy for each lipid type
            for lipid_type, composition_grid in spatial_dist['grids'].items():
                # Convert grid spacing from Å to nm
                box_size_nm = box_size / 10.0

                # Simulate continuous distribution
                image_continuous = self.microscope.simulate_image(
                    composition_grid, noise_level=0.1, photon_count=1000
                )

                # Simulate bead-based distribution
                bead_positions = self.microscope.generate_bead_positions_from_composition(
                    composition_grid, n_lipids_per_unit=5
                )
                # Scale positions to nm
                scale_factor = box_size_nm / composition_grid.shape[0]
                bead_positions_nm = bead_positions * scale_factor

                image_beads = self.microscope.simulate_bead_distribution(
                    bead_positions_nm, box_size_nm, grid_size=512
                )

                results[target_radius]['microscopy_images'][lipid_type] = {
                    'continuous': image_continuous,
                    'beads': image_beads,
                    'bead_positions': bead_positions_nm
                }

                print(f"  ✓ Simulated {lipid_type}: {len(bead_positions_nm)} molecules")

        print("\n" + "="*70)
        print(f"Microscopy simulation complete for {len(target_radii)} scales")
        print("="*70)

        return results
