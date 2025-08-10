#!/usr/bin/env python3
"""
Conservation-based lipid redistribution analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple


class ConservationAnalyzer:
    """
    Analyze lipid redistribution with conservation constraints
    """
    
    def __init__(self, enforce_conservation: bool = True):
        """
        Initialize conservation analyzer
        
        Parameters
        ----------
        enforce_conservation : bool
            Whether to enforce strict conservation (sum = 1)
        """
        self.enforce_conservation = enforce_conservation
        
    def calculate_composition_ratios(self, 
                                    frame_data: pd.DataFrame,
                                    lipid_columns: List[str]) -> pd.DataFrame:
        """
        Calculate lipid composition ratios ensuring conservation
        
        Parameters
        ----------
        frame_data : pandas.DataFrame
            Frame data with lipid counts
        lipid_columns : list
            Column names for lipid counts
        
        Returns
        -------
        pandas.DataFrame
            Data with added composition ratio columns
        """
        # Calculate total lipids
        total_lipids = sum(frame_data[col] for col in lipid_columns)
        
        # Avoid division by zero
        total_lipids = total_lipids.replace(0, 1)
        
        # Calculate ratios
        for col in lipid_columns:
            ratio_col = col.replace('_count', '_ratio')
            frame_data[ratio_col] = frame_data[col] / total_lipids
        
        # Verify conservation
        if self.enforce_conservation:
            ratio_columns = [col.replace('_count', '_ratio') for col in lipid_columns]
            total_ratio = sum(frame_data[col] for col in ratio_columns)
            
            mean_total = total_ratio.mean()
            if abs(mean_total - 1.0) > 0.001:
                print(f"Warning: Conservation violated. Mean total ratio = {mean_total:.6f}")
        
        return frame_data
    
    def analyze_redistribution(self,
                             data: pd.DataFrame,
                             mediator_column: str,
                             lipid_columns: List[str],
                             method: str = 'quartile') -> Dict:
        """
        Analyze lipid redistribution caused by mediator
        
        Parameters
        ----------
        data : pandas.DataFrame
            Data with mediator and lipid information
        mediator_column : str
            Column name for mediator (e.g., GM3)
        lipid_columns : list
            List of lipid column names
        method : str
            'quartile', 'median', or 'continuous'
        
        Returns
        -------
        dict
            Redistribution analysis results
        """
        results = {}
        
        # First calculate composition ratios
        data = self.calculate_composition_ratios(data, lipid_columns)
        
        if method == 'quartile':
            # Split by quartiles
            q75 = data[mediator_column].quantile(0.75)
            q25 = data[mediator_column].quantile(0.25)
            
            high_mediator = data[data[mediator_column] > q75]
            low_mediator = data[data[mediator_column] < q25]
            
        elif method == 'median':
            # Split by median
            median = data[mediator_column].median()
            high_mediator = data[data[mediator_column] > median]
            low_mediator = data[data[mediator_column] <= median]
            
        else:  # continuous
            # Use correlation-based analysis
            return self._continuous_analysis(data, mediator_column, lipid_columns)
        
        # Compare high vs low mediator conditions
        for lipid_col in lipid_columns:
            lipid_name = lipid_col.replace('_count', '')
            ratio_col = lipid_col.replace('_count', '_ratio')
            
            if len(high_mediator) > 0 and len(low_mediator) > 0:
                # Calculate changes
                low_mean = low_mediator[ratio_col].mean()
                high_mean = high_mediator[ratio_col].mean()
                absolute_change = high_mean - low_mean
                
                # Percent change
                if low_mean > 0:
                    percent_change = 100 * absolute_change / low_mean
                else:
                    percent_change = 0
                
                # Statistical test
                if len(high_mediator) >= 2 and len(low_mediator) >= 2:
                    t_stat, p_value = stats.ttest_ind(
                        high_mediator[ratio_col].dropna(),
                        low_mediator[ratio_col].dropna(),
                        equal_var=False
                    )
                else:
                    p_value = 1.0
                
                # Determine significance
                if p_value < 0.001:
                    significance = '***'
                elif p_value < 0.01:
                    significance = '**'
                elif p_value < 0.05:
                    significance = '*'
                else:
                    significance = 'ns'
                
                results[lipid_name] = {
                    'low_mediator_ratio': low_mean,
                    'high_mediator_ratio': high_mean,
                    'absolute_change': absolute_change,
                    'percent_change': percent_change,
                    'p_value': p_value,
                    'significance': significance,
                    'n_high': len(high_mediator),
                    'n_low': len(low_mediator)
                }
        
        # Verify conservation of changes
        if self.enforce_conservation:
            total_change = sum(res['absolute_change'] for res in results.values())
            if abs(total_change) > 0.001:
                print(f"Warning: Total change = {total_change:.6f} (should be ~0)")
        
        return results
    
    def _continuous_analysis(self, 
                            data: pd.DataFrame,
                            mediator_column: str,
                            lipid_columns: List[str]) -> Dict:
        """
        Continuous correlation-based analysis
        
        Parameters
        ----------
        data : pandas.DataFrame
            Data frame with all columns
        mediator_column : str
            Mediator column name
        lipid_columns : list
            Lipid column names
        
        Returns
        -------
        dict
            Correlation-based results
        """
        results = {}
        
        for lipid_col in lipid_columns:
            lipid_name = lipid_col.replace('_count', '')
            ratio_col = lipid_col.replace('_count', '_ratio')
            
            # Calculate correlation
            mask = data[mediator_column].notna() & data[ratio_col].notna()
            if mask.sum() > 10:
                corr, p_value = stats.pearsonr(
                    data.loc[mask, mediator_column],
                    data.loc[mask, ratio_col]
                )
                
                # Calculate slope from linear regression
                from sklearn.linear_model import LinearRegression
                X = data.loc[mask, mediator_column].values.reshape(-1, 1)
                y = data.loc[mask, ratio_col].values
                
                lr = LinearRegression()
                lr.fit(X, y)
                slope = lr.coef_[0]
                
                # Estimate effect size
                mediator_range = (data[mediator_column].quantile(0.75) - 
                                data[mediator_column].quantile(0.25))
                effect_size = slope * mediator_range
                
                results[lipid_name] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'slope': slope,
                    'effect_size': effect_size,
                    'n_samples': mask.sum()
                }
        
        return results
    
    def analyze_temporal_redistribution(self,
                                       data: pd.DataFrame,
                                       time_column: str,
                                       window_size: int = 20) -> pd.DataFrame:
        """
        Analyze temporal patterns in redistribution
        
        Parameters
        ----------
        data : pandas.DataFrame
            Time series data
        time_column : str
            Time/frame column name
        window_size : int
            Window for rolling statistics
        
        Returns
        -------
        pandas.DataFrame
            Data with temporal analysis
        """
        # Sort by time
        data = data.sort_values(time_column)
        
        # Calculate rolling means
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != time_column:
                data[f'{col}_rolling_mean'] = data[col].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
                
                data[f'{col}_rolling_std'] = data[col].rolling(
                    window=window_size, center=True, min_periods=1
                ).std()
        
        return data