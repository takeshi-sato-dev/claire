#!/usr/bin/env python3
"""
ML analysis matching original code
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Optional


class MLAnalyzer:
    """
    Simplified ML analysis matching original
    """
    
    @staticmethod
    def advanced_ml_analysis(df: pd.DataFrame, 
                            target_lipids: Optional[List[str]] = None,
                            mediator_lipid: str = 'DPG3') -> Tuple[Dict, pd.DataFrame]:
        """
        Advanced ML analysis - MATCHING ORIGINAL
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input data
        target_lipids : list, optional
            List of target lipid names. If None, try to detect from columns
        mediator_lipid : str
            Mediator lipid name
        """
        print("\nPerforming advanced ML analysis...")
        
        # Detect target lipids from columns if not provided
        if target_lipids is None:
            # Look for _contact_count columns
            target_lipids = []
            for col in df.columns:
                if col.endswith('_contact_count') and not col.startswith(mediator_lipid.lower()):
                    lipid_name = col.replace('_contact_count', '')
                    if lipid_name.upper() not in ['GM3', 'DPG3', mediator_lipid.upper()]:
                        target_lipids.append(lipid_name.upper())
            
            # Remove duplicates
            target_lipids = list(set(target_lipids))
            
            print(f"  Detected target lipids: {target_lipids}")
        
        # Create time-lagged features
        from .temporal import TemporalAnalyzer
        df_lagged = TemporalAnalyzer.create_time_lagged_features(df)
        
        # Aggregate by frame
        print("Aggregating by frame...")
        
        agg_dict = {}
        for col in df_lagged.columns:
            if col in ['frame', 'protein', 'time']:
                continue
            if pd.api.types.is_numeric_dtype(df_lagged[col]):
                agg_dict[col] = 'mean'
        
        frame_df = df_lagged.groupby('frame').agg(agg_dict).reset_index()
        
        # Calculate normalized metrics
        print("\nCalculating normalized metrics...")
        
        # Determine mediator column
        mediator_col = None
        for col_name in ['gm3_contact_strength', f'{mediator_lipid.lower()}_contact_strength',
                        'gm3_contact_count', f'{mediator_lipid.lower()}_contact_count']:
            if col_name in frame_df.columns:
                mediator_col = col_name
                break
        
        if mediator_col is None:
            print(f"ERROR: No mediator column found for {mediator_lipid}")
            return {}, frame_df
        
        print(f"  Using mediator column: {mediator_col}")
        
        # Calculate GM3-induced changes
        print("Calculating mediator-induced changes...")
        
        if 'strength' in mediator_col:
            mediator_threshold = frame_df[mediator_col].quantile(0.25)
            has_mediator = frame_df[mediator_col] > mediator_threshold
            no_mediator = frame_df[mediator_col] <= mediator_threshold
        else:
            has_mediator = frame_df[mediator_col] > 0
            no_mediator = frame_df[mediator_col] == 0
        
        print(f"  Frames with {mediator_lipid}: {has_mediator.sum()}")
        print(f"  Frames without {mediator_lipid}: {no_mediator.sum()}")
        
        # Calculate baseline values for each target lipid
        baseline_values = {}
        for lipid in target_lipids:
            col = f'{lipid}_contact_count'
            if col in frame_df.columns and no_mediator.sum() > 10:
                baseline_values[lipid] = frame_df.loc[no_mediator, col].mean()
            elif col in frame_df.columns:
                baseline_values[lipid] = frame_df[col].mean()
            else:
                baseline_values[lipid] = 0
        
        # Create effect targets for each lipid
        for lipid in target_lipids:
            col = f'{lipid}_contact_count'
            if col in frame_df.columns:
                frame_df[f'{lipid}_deviation_from_baseline'] = (
                    frame_df[col] - baseline_values[lipid]
                )
            
            window = min(50, len(frame_df) // 4)
            if col in frame_df.columns and len(frame_df) > window:
                rolling_baseline = frame_df[col].rolling(
                    window=window, center=True, min_periods=window//2
                ).mean()
                frame_df[f'{lipid}_deviation_from_rolling'] = (
                    frame_df[col] - rolling_baseline
                )
            
            fraction_col = f'{lipid}_fraction_vicinity'
            if fraction_col in frame_df.columns:
                frame_df[f'{lipid}_normalized_fraction'] = frame_df[fraction_col]
        
        # Feature selection
        print("\nSelecting features...")
        
        feature_cols = []
        # Add mediator features
        if mediator_col in frame_df.columns:
            feature_cols.append(mediator_col)
        
        # Add additional mediator features if available
        for col in [f'{mediator_lipid.lower()}_density', 'gm3_density',
                   f'{mediator_lipid.lower()}_fraction_vicinity', 'gm3_fraction_vicinity']:
            if col in frame_df.columns and col not in feature_cols:
                feature_cols.append(col)
        
        print(f"Using {len(feature_cols)} features: {feature_cols}")
        
        if len(feature_cols) == 0:
            print(f"ERROR: No {mediator_lipid} features found!")
            return {}, frame_df
        
        X = frame_df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = {}
        
        # Direct correlation analysis
        print("\nDirect correlation analysis:")
        direct_correlations = {}
        
        for lipid in target_lipids:
            col = f'{lipid}_contact_count'
            if col in frame_df.columns and mediator_col in frame_df.columns:
                mediator_vals = frame_df[mediator_col].values
                lipid_vals = frame_df[col].values
                
                # Remove outliers
                mediator_p95 = np.percentile(mediator_vals, 95)
                lipid_p95 = np.percentile(lipid_vals, 95)
                mask = (mediator_vals < mediator_p95) & (lipid_vals < lipid_p95)
                
                if mask.sum() > 30:
                    corr, pval = stats.pearsonr(mediator_vals[mask], lipid_vals[mask])
                    direct_correlations[lipid] = {
                        'correlation': corr,
                        'p_value': pval,
                        'n_samples': mask.sum()
                    }
                    print(f"  {lipid}: r = {corr:.3f} (p = {pval:.4f})")
        
        # Machine learning analysis for each target lipid
        for lipid in target_lipids:
            print(f"\n  Analyzing {lipid}...")
            
            targets = {}
            
            # Collect available targets
            for target_type in ['deviation_from_baseline', 'deviation_from_rolling', 
                              'normalized_fraction', 'contact_count']:
                col = f'{lipid}_{target_type}' if target_type != 'contact_count' else f'{lipid}_contact_count'
                if col in frame_df.columns:
                    targets[target_type] = frame_df[col].values
            
            if not targets:
                print(f"    No valid targets found for {lipid}")
                continue
            
            lipid_results = {}
            best_overall_score = -999
            best_overall_target = None
            
            for target_name, y in targets.items():
                print(f"    Target: {target_name}")
                
                if np.std(y) < 1e-6:
                    print(f"      Skipping - no variation")
                    continue
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.3, random_state=42
                )
                
                # Linear regression
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                lr_score = lr.score(X_test, y_test)
                
                # Calculate effect
                mediator_feature_idx = 0
                mediator_coefficient = lr.coef_[mediator_feature_idx]
                mediator_range = np.percentile(X_scaled[:, mediator_feature_idx], 80) - \
                               np.percentile(X_scaled[:, mediator_feature_idx], 20)
                linear_effect = mediator_coefficient * mediator_range
                
                # Ridge regression
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_train, y_train)
                ridge_score = ridge.score(X_test, y_test)
                
                # Select best model
                if lr_score > ridge_score and lr_score > -0.5:
                    final_model = lr
                    final_score = lr_score
                    model_type = 'Linear'
                    effect = linear_effect
                else:
                    final_model = ridge
                    final_score = ridge_score
                    model_type = 'Ridge'
                    effect = ridge.coef_[mediator_feature_idx] * mediator_range
                
                # Get correlation
                if lipid in direct_correlations:
                    correlation = direct_correlations[lipid]['correlation']
                    p_value = direct_correlations[lipid]['p_value']
                else:
                    correlation, p_value = stats.pearsonr(X_scaled[:, 0], y)
                
                # Ensure effect sign matches correlation
                if correlation != 0:
                    effect = abs(effect) * np.sign(correlation)
                
                lipid_results[target_name] = {
                    'model_type': model_type,
                    'score': final_score,
                    'linear_score': lr_score,
                    'ridge_score': ridge_score,
                    'correlation': correlation,
                    'p_value': p_value,
                    'gm3_effect': effect,  # Keep name for compatibility
                    f'{mediator_lipid.lower()}_effect': effect,
                    f'{mediator_lipid.lower()}_coefficient': mediator_coefficient if model_type == 'Linear' else ridge.coef_[0]
                }
                
                if final_score > best_overall_score:
                    best_overall_score = final_score
                    best_overall_target = target_name
                
                print(f"      Model: {model_type}, R² = {final_score:.3f}")
                print(f"      {mediator_lipid} effect: {effect:.3f}")
                print(f"      Correlation: {correlation:.3f} (p={p_value:.4f})")
            
            # Store best results for this lipid
            if best_overall_target and best_overall_target in lipid_results:
                best_results = lipid_results[best_overall_target].copy()
                best_results['best_target'] = best_overall_target
                best_results['all_targets'] = lipid_results
                
                if lipid in direct_correlations:
                    best_results['direct_correlation'] = direct_correlations[lipid]['correlation']
                    best_results['direct_p_value'] = direct_correlations[lipid]['p_value']
                
                results[lipid] = best_results
            elif lipid_results:
                first_target = list(lipid_results.keys())[0]
                best_results = lipid_results[first_target].copy()
                best_results['best_target'] = first_target
                best_results['all_targets'] = lipid_results
                results[lipid] = best_results
            else:
                if lipid in direct_correlations:
                    results[lipid] = {
                        'correlation': direct_correlations[lipid]['correlation'],
                        'p_value': direct_correlations[lipid]['p_value'],
                        'gm3_effect': direct_correlations[lipid]['correlation'] * 2.0,
                        f'{mediator_lipid.lower()}_effect': direct_correlations[lipid]['correlation'] * 2.0,
                        'model_type': 'Correlation',
                        'score': -999,
                        'best_target': 'direct_correlation'
                    }
        
        return results, frame_df