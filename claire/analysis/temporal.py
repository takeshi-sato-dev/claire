#!/usr/bin/env python3
"""
Temporal analysis of lipid dynamics
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from typing import Dict, Optional, Tuple, List


class TemporalAnalyzer:
    """
    Analyze temporal patterns in lipid redistribution
    """
    
    @staticmethod
    def apply_smoothing(data: pd.DataFrame,
                       columns: list,
                       window: int = 20,
                       method: str = 'rolling') -> pd.DataFrame:
        """
        Apply smoothing to time series data
        
        Parameters
        ----------
        data : pandas.DataFrame
            Time series data
        columns : list
            Columns to smooth
        window : int
            Window size
        method : str
            'rolling', 'savgol', or 'gaussian'
        
        Returns
        -------
        pandas.DataFrame
            Smoothed data
        """
        smoothed = data.copy()
        
        for col in columns:
            if col not in data.columns:
                continue
                
            if method == 'rolling':
                smoothed[col] = data[col].rolling(
                    window=window, center=True, min_periods=1
                ).mean()
                
            elif method == 'savgol':
                # Savitzky-Golay filter
                if len(data) > window:
                    smoothed[col] = signal.savgol_filter(
                        data[col].values, window, 3
                    )
                    
            elif method == 'gaussian':
                # Gaussian filter
                smoothed[col] = pd.Series(
                    signal.gaussian_filter1d(data[col].values, sigma=window/4)
                )
        
        return smoothed
    
    @staticmethod
    def create_time_lagged_features(df: pd.DataFrame, 
                                   lag_frames: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Create time-lagged features to capture dynamics
        FROM ORIGINAL CODE - CRITICAL FOR ML ANALYSIS
        
        Parameters
        ----------
        df : pandas.DataFrame
            Data with protein and frame columns
        lag_frames : list
            List of lag values to create
        
        Returns
        -------
        pandas.DataFrame
            Data with lagged features
        """
        print("Creating time-lagged features...")
        df_sorted = df.sort_values(['protein', 'frame'])
        
        # GM3 features to lag
        gm3_features = [col for col in df.columns if 'gm3' in col.lower()]
        
        for lag in lag_frames:
            print(f"  Adding lag {lag} features...")
            for feature in gm3_features:
                df_sorted[f'{feature}_lag{lag}'] = df_sorted.groupby('protein')[feature].shift(lag)
                df_sorted[f'{feature}_diff{lag}'] = df_sorted[f'{feature}_lag{lag}'] - df_sorted[feature]
        
        # Fill NaN with 0 for lagged features
        df_sorted = df_sorted.fillna(0)
        
        return df_sorted
    
    @staticmethod
    def detect_outliers(data: pd.DataFrame,
                       columns: list,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect and mark outliers in time series
        
        Parameters
        ----------
        data : pandas.DataFrame
            Time series data
        columns : list
            Columns to check
        method : str
            'iqr', 'zscore', or 'mad'
        threshold : float
            Threshold for outlier detection
        
        Returns
        -------
        pandas.DataFrame
            Data with outlier flags
        """
        outliers = pd.DataFrame(index=data.index)
        
        for col in columns:
            if col not in data.columns:
                continue
            
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                outliers[f'{col}_outlier'] = (
                    (data[col] < lower) | (data[col] > upper)
                )
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                outliers[f'{col}_outlier'] = z_scores > threshold
                
            elif method == 'mad':
                # Median Absolute Deviation
                median = data[col].median()
                mad = np.median(np.abs(data[col] - median))
                if mad > 0:
                    modified_z = 0.6745 * (data[col] - median) / mad
                    outliers[f'{col}_outlier'] = np.abs(modified_z) > threshold
        
        return pd.concat([data, outliers], axis=1)
    
    @staticmethod
    def apply_smoothing_and_filtering(frame_df: pd.DataFrame, 
                                    window: int = 10,
                                    outlier_percentile: float = 95) -> pd.DataFrame:
        """
        Noise reduction and smoothing from original improved_metrics.py
        
        Parameters
        ----------
        frame_df : pandas.DataFrame
            Frame data
        window : int
            Smoothing window
        outlier_percentile : float
            Percentile for outlier removal
        
        Returns
        -------
        pandas.DataFrame
            Smoothed and filtered data
        """
        print("\nApplying noise reduction...")
        
        smoothed_df = frame_df.copy()
        
        # Numeric columns only
        numeric_cols = frame_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['frame', 'time']:
                # Remove outliers
                percentile_val = np.percentile(frame_df[col], outlier_percentile)
                mask = frame_df[col] <= percentile_val
                
                # Moving average
                smoothed_df[col] = frame_df[col].rolling(
                    window=window, center=True, min_periods=1
                ).mean()
                
                # Clip outliers
                smoothed_df.loc[~mask, col] = percentile_val
        
        return smoothed_df
    
    @staticmethod
    def calculate_autocorrelation(data: pd.Series,
                                 max_lag: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate autocorrelation function
        
        Parameters
        ----------
        data : pandas.Series
            Time series
        max_lag : int
            Maximum lag
        
        Returns
        -------
        tuple
            (lags, autocorrelations)
        """
        # Remove NaN
        clean_data = data.dropna().values
        
        if len(clean_data) < max_lag * 2:
            max_lag = len(clean_data) // 2
        
        autocorr = []
        lags = range(max_lag + 1)
        
        for lag in lags:
            if lag == 0:
                autocorr.append(1.0)
            else:
                c = np.corrcoef(clean_data[:-lag], clean_data[lag:])[0, 1]
                autocorr.append(c if not np.isnan(c) else 0)
        
        return np.array(lags), np.array(autocorr)
    
    @staticmethod
    def analyze_temporal_stability(data: pd.DataFrame,
                                 columns: list,
                                 window_size: int = 100) -> Dict:
        """
        Analyze temporal stability of variables
        
        Parameters
        ----------
        data : pandas.DataFrame
            Time series data
        columns : list
            Columns to analyze
        window_size : int
            Window for stability analysis
        
        Returns
        -------
        dict
            Stability metrics
        """
        stability = {}
        
        for col in columns:
            if col not in data.columns:
                continue
            
            # Calculate rolling statistics
            rolling_mean = data[col].rolling(window=window_size, center=True).mean()
            rolling_std = data[col].rolling(window=window_size, center=True).std()
            
            # Coefficient of variation over time
            cv_series = rolling_std / (rolling_mean + 1e-10)
            
            stability[col] = {
                'mean_cv': cv_series.mean(),
                'cv_trend': np.polyfit(range(len(cv_series.dropna())), 
                                      cv_series.dropna(), 1)[0],
                'stable_fraction': (cv_series < 0.1).sum() / len(cv_series),
                'drift': (rolling_mean.iloc[-1] - rolling_mean.iloc[0]) / rolling_mean.iloc[0]
                        if rolling_mean.iloc[0] != 0 else 0
            }
        
        return stability
    
    @staticmethod
    def segment_trajectory(data: pd.DataFrame,
                         change_column: str,
                         n_segments: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Segment trajectory into phases
        
        Parameters
        ----------
        data : pandas.DataFrame
            Trajectory data
        change_column : str
            Column to use for segmentation
        n_segments : int
            Number of segments
        
        Returns
        -------
        tuple
            (data with segment labels, segment statistics)
        """
        data = data.copy()
        
        # Simple segmentation by equal time intervals
        segment_size = len(data) // n_segments
        data['segment'] = data.index // segment_size
        
        # Calculate segment statistics
        segment_stats = []
        for seg in range(n_segments):
            seg_data = data[data['segment'] == seg]
            if len(seg_data) > 0:
                segment_stats.append({
                    'segment': seg,
                    'mean': seg_data[change_column].mean(),
                    'std': seg_data[change_column].std(),
                    'start_frame': seg_data.index[0],
                    'end_frame': seg_data.index[-1]
                })
        
        return data, pd.DataFrame(segment_stats)