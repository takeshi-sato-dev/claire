#!/usr/bin/env python3
"""
Statistical analysis utilities
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class StatisticalAnalyzer:
    """
    Statistical analysis methods for lipid redistribution
    """
    
    @staticmethod
    def calculate_correlation_matrix(data: pd.DataFrame,
                                    columns: Optional[list] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix between variables
        
        Parameters
        ----------
        data : pandas.DataFrame
            Data frame
        columns : list, optional
            Columns to include
        
        Returns
        -------
        pandas.DataFrame
            Correlation matrix
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        return data[columns].corr()
    
    @staticmethod
    def bootstrap_correlation(x: np.ndarray,
                            y: np.ndarray,
                            n_bootstrap: int = 1000,
                            confidence_level: float = 0.95) -> Dict:
        """
        Calculate correlation with bootstrap confidence intervals
        
        Parameters
        ----------
        x : numpy.ndarray
            First variable
        y : numpy.ndarray
            Second variable
        n_bootstrap : int
            Number of bootstrap samples
        confidence_level : float
            Confidence level for CI
        
        Returns
        -------
        dict
            Correlation statistics
        """
        # Remove NaN values
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 30:
            return {
                'correlation': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'p_value': np.nan
            }
        
        # Original correlation
        original_corr, original_p = stats.pearsonr(x_clean, y_clean)
        
        # Bootstrap
        correlations = []
        n = len(x_clean)
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            boot_corr, _ = stats.pearsonr(x_clean[indices], y_clean[indices])
            correlations.append(boot_corr)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        ci_lower = np.percentile(correlations, 100 * alpha/2)
        ci_upper = np.percentile(correlations, 100 * (1 - alpha/2))
        
        return {
            'correlation': original_corr,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': original_p,
            'std': np.std(correlations)
        }
    
    @staticmethod
    def analyze_effect_size(data: pd.DataFrame,
                          mediator_column: str,
                          target_column: str,
                          method: str = 'multiple') -> Dict:
        """
        Calculate effect size using multiple methods
        
        Parameters
        ----------
        data : pandas.DataFrame
            Data frame
        mediator_column : str
            Mediator variable
        target_column : str
            Target variable
        method : str
            'quartile', 'regression', or 'multiple'
        
        Returns
        -------
        dict
            Effect size estimates
        """
        # Remove NaN
        mask = data[mediator_column].notna() & data[target_column].notna()
        clean_data = data.loc[mask, [mediator_column, target_column]]
        
        if len(clean_data) < 10:
            return {'error': 'Insufficient data'}
        
        effects = {}
        
        # Method 1: Quartile comparison
        if method in ['quartile', 'multiple']:
            q75 = clean_data[mediator_column].quantile(0.75)
            q25 = clean_data[mediator_column].quantile(0.25)
            
            high_group = clean_data[clean_data[mediator_column] > q75]
            low_group = clean_data[clean_data[mediator_column] < q25]
            
            if len(high_group) > 5 and len(low_group) > 5:
                effect_quartile = (high_group[target_column].mean() - 
                                 low_group[target_column].mean())
                effects['quartile'] = effect_quartile
                
                # Cohen's d
                pooled_std = np.sqrt(
                    (high_group[target_column].var() + 
                     low_group[target_column].var()) / 2
                )
                if pooled_std > 0:
                    effects['cohens_d'] = effect_quartile / pooled_std
        
        # Method 2: Linear regression
        if method in ['regression', 'multiple']:
            X = clean_data[mediator_column].values.reshape(-1, 1)
            y = clean_data[target_column].values
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit model
            lr = LinearRegression()
            lr.fit(X_scaled, y)
            
            # Effect = coefficient * typical change in mediator
            mediator_range = q75 - q25 if 'q75' in locals() else clean_data[mediator_column].std()
            effect_regression = lr.coef_[0] * (mediator_range / clean_data[mediator_column].std())
            
            effects['regression'] = effect_regression
            effects['r_squared'] = lr.score(X_scaled, y)
        
        # Method 3: Correlation-based
        if method in ['correlation', 'multiple']:
            corr, p_val = stats.pearsonr(
                clean_data[mediator_column], 
                clean_data[target_column]
            )
            effect_correlation = corr * clean_data[target_column].std()
            
            effects['correlation'] = corr
            effects['correlation_effect'] = effect_correlation
            effects['p_value'] = p_val
        
        # Determine consistency
        if len(effects) > 1:
            effect_values = [v for k, v in effects.items() 
                           if k not in ['p_value', 'r_squared', 'cohens_d']]
            effects['consistent'] = all(np.sign(v) == np.sign(effect_values[0]) 
                                       for v in effect_values)
        
        return effects
    
    @staticmethod
    def perform_significance_tests(data1: np.ndarray,
                                  data2: np.ndarray) -> Dict:
        """
        Perform multiple statistical tests
        
        Parameters
        ----------
        data1 : numpy.ndarray
            First dataset
        data2 : numpy.ndarray
            Second dataset
        
        Returns
        -------
        dict
            Test results
        """
        results = {}
        
        # T-test
        t_stat, t_pval = stats.ttest_ind(data1, data2, equal_var=False)
        results['t_test'] = {'statistic': t_stat, 'p_value': t_pval}
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pval = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        results['mann_whitney'] = {'statistic': u_stat, 'p_value': u_pval}
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(data1, data2)
        results['ks_test'] = {'statistic': ks_stat, 'p_value': ks_pval}
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(data1) - np.mean(data2)
        pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
        if pooled_std > 0:
            cohens_d = mean_diff / pooled_std
            results['cohens_d'] = cohens_d
        
        return results