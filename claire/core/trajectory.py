#!/usr/bin/env python3
"""
Trajectory processing utilities
"""

import numpy as np
import multiprocessing as mp
from functools import partial
from typing import Dict, List, Optional, Tuple, Callable
import MDAnalysis as mda
from tqdm import tqdm


class TrajectoryProcessor:
    """
    Parallel trajectory processing with progress tracking
    """
    
    def __init__(self, universe: mda.Universe, n_workers: Optional[int] = None):
        """
        Initialize trajectory processor
        
        Parameters
        ----------
        universe : MDAnalysis.Universe
            System to process
        n_workers : int, optional
            Number of parallel workers (default: CPU count)
        """
        self.universe = universe
        self.n_workers = n_workers or mp.cpu_count()
        
    def process_frames(self, 
                      analysis_func: Callable,
                      start: int = 0,
                      stop: Optional[int] = None,
                      step: int = 1,
                      chunk_size: int = 100,
                      verbose: bool = True,
                      **kwargs) -> List:
        """
        Process trajectory frames in parallel
        
        Parameters
        ----------
        analysis_func : callable
            Function to apply to each frame
        start : int
            Start frame
        stop : int, optional
            Stop frame (None = end of trajectory)
        step : int
            Frame step
        chunk_size : int
            Frames per chunk for parallel processing
        verbose : bool
            Show progress bar
        **kwargs
            Additional arguments for analysis_func
        
        Returns
        -------
        list
            Results from all frames
        """
        if stop is None:
            stop = len(self.universe.trajectory)
        
        frame_indices = list(range(start, min(stop, len(self.universe.trajectory)), step))
        
        if verbose:
            print(f"Processing {len(frame_indices)} frames "
                  f"({start} to {stop}, step {step})")
            print(f"Using {self.n_workers} workers")
        
        # Create partial function with kwargs
        process_func = partial(analysis_func, universe=self.universe, **kwargs)
        
        all_results = []
        
        # Process in chunks with progress bar
        if verbose:
            pbar = tqdm(total=len(frame_indices), desc="Processing frames")
        
        for i in range(0, len(frame_indices), chunk_size):
            chunk_indices = frame_indices[i:i+chunk_size]
            
            if self.n_workers > 1:
                with mp.Pool(self.n_workers) as pool:
                    chunk_results = pool.map(process_func, chunk_indices)
            else:
                # Single-threaded for debugging
                chunk_results = [process_func(idx) for idx in chunk_indices]
            
            all_results.extend(chunk_results)
            
            if verbose:
                pbar.update(len(chunk_indices))
        
        if verbose:
            pbar.close()
        
        return all_results
    
    @staticmethod
    def apply_pbc_correction(positions: np.ndarray, 
                           reference: np.ndarray,
                           box_dimensions: np.ndarray) -> np.ndarray:
        """
        Apply periodic boundary condition corrections
        
        Parameters
        ----------
        positions : numpy.ndarray
            Positions to correct
        reference : numpy.ndarray
            Reference position
        box_dimensions : numpy.ndarray
            Box dimensions [x, y, z]
        
        Returns
        -------
        numpy.ndarray
            Corrected positions
        """
        corrected = positions.copy()
        
        for dim in range(3):
            delta = corrected[:, dim] - reference[dim]
            corrected[:, dim] -= box_dimensions[dim] * np.round(delta / box_dimensions[dim])
        
        return corrected
    
    @staticmethod
    def calculate_com_distance_2d(pos1: np.ndarray, 
                                 pos2: np.ndarray,
                                 box_dimensions: np.ndarray) -> float:
        """
        Calculate 2D distance (XY plane) with PBC
        
        Parameters
        ----------
        pos1 : numpy.ndarray
            First position
        pos2 : numpy.ndarray
            Second position
        box_dimensions : numpy.ndarray
            Box dimensions
        
        Returns
        -------
        float
            2D distance
        """
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        # PBC correction
        dx = dx - box_dimensions[0] * round(dx/box_dimensions[0])
        dy = dy - box_dimensions[1] * round(dy/box_dimensions[1])
        
        return np.sqrt(dx**2 + dy**2)