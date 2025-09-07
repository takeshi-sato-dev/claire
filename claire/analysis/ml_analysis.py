#!/usr/bin/env python3
"""
ML analysis - EXACT REPLICA of original_analysis_no_causal.py advanced_ml_analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Optional


def create_time_lagged_features(df, lag_frames=[1, 5, 10, 20]):
    """Create time-lagged features to capture dynamics - EXACT COPY FROM ORIGINAL_ANALYSIS"""
    print("Creating time-lagged features...")
    df_sorted = df.sort_values(['protein', 'frame'])
    
    # GM3 features to lag
    gm3_features = [col for col in df.columns if 'gm3' in col.lower()]
    
    for lag in lag_frames:
        print(f"  Adding lag {lag} features...")
        for feature in gm3_features:
            df_sorted[f'{feature}_lag{lag}'] = df_sorted.groupby('protein')[feature].shift(lag)
            df_sorted[f'{feature}_diff{lag}'] = df_sorted[feature] - df_sorted[f'{feature}_lag{lag}']
    
    # Fill NaN with 0 for lagged features
    df_sorted = df_sorted.fillna(0)
    
    return df_sorted


def advanced_ml_analysis(df):
    """Advanced ML analysis with corrected approach for GM3 effects - EXACT COPY FROM ORIGINAL_ANALYSIS"""
    print("\nPerforming advanced ML analysis...")
    
    # Create time-lagged features
    df_lagged = create_time_lagged_features(df)
    
    # Aggregate by frame
    print("Aggregating by frame...")
    
    # Explicitly specify columns to aggregate
    agg_dict = {}
    
    # Only aggregate numeric columns
    for col in df_lagged.columns:
        if col in ['frame', 'protein', 'time']:
            continue  # These are groupby keys or strings
        
        # Add only numeric type columns
        if pd.api.types.is_numeric_dtype(df_lagged[col]):
            agg_dict[col] = 'mean'
    
    # frameでグループ化して集計
    frame_df = df_lagged.groupby('frame').agg(agg_dict).reset_index()
    
    # Calculate normalized metrics for better ML targets
    print("\nCalculating normalized metrics...")
    
    # CRITICAL: Calculate GM3-induced changes directly
    print("Calculating GM3-induced changes...")
    
    # First, identify frames with significant GM3 presence
    if 'gm3_contact_strength' in frame_df.columns:
        gm3_threshold = frame_df['gm3_contact_strength'].quantile(0.25)
        has_gm3 = frame_df['gm3_contact_strength'] > gm3_threshold
        no_gm3 = frame_df['gm3_contact_strength'] <= gm3_threshold
        
        print(f"  Frames with GM3: {has_gm3.sum()}")
        print(f"  Frames without GM3: {no_gm3.sum()}")
    else:
        # Fallback to count
        has_gm3 = frame_df['gm3_contact_count'] > 0
        no_gm3 = frame_df['gm3_contact_count'] == 0
    
    # Calculate baseline values (without GM3) for each lipid
    baseline_values = {}
    for lipid in ['CHOL', 'DIPC', 'DPSM']:
        if f'{lipid}_contact_count' in frame_df.columns and no_gm3.sum() > 10:
            baseline_values[lipid] = frame_df.loc[no_gm3, f'{lipid}_contact_count'].mean()
        else:
            baseline_values[lipid] = frame_df[f'{lipid}_contact_count'].mean()
    
    # Create GM3-effect targets
    for lipid in ['CHOL', 'DIPC', 'DPSM']:
        # Method 1: Deviation from baseline
        if f'{lipid}_contact_count' in frame_df.columns:
            frame_df[f'{lipid}_deviation_from_baseline'] = (
                frame_df[f'{lipid}_contact_count'] - baseline_values[lipid]
            )
        
        # Method 2: Rolling baseline deviation
        window = min(50, len(frame_df) // 4)
        if f'{lipid}_contact_count' in frame_df.columns and len(frame_df) > window:
            rolling_baseline = frame_df[f'{lipid}_contact_count'].rolling(
                window=window, center=True, min_periods=window//2
            ).mean()
            frame_df[f'{lipid}_deviation_from_rolling'] = (
                frame_df[f'{lipid}_contact_count'] - rolling_baseline
            )
        
        # Method 3: Direct GM3 correlation target
        if f'{lipid}_fraction_vicinity' in frame_df.columns:
            # Already normalized
            frame_df[f'{lipid}_normalized_fraction'] = frame_df[f'{lipid}_fraction_vicinity']
        
        # Method 4: Calculate expected change based on GM3
        if has_gm3.sum() > 10 and no_gm3.sum() > 10:
            mean_with_gm3 = frame_df.loc[has_gm3, f'{lipid}_contact_count'].mean()
            mean_without_gm3 = frame_df.loc[no_gm3, f'{lipid}_contact_count'].mean()
            expected_change = mean_with_gm3 - mean_without_gm3
            print(f"  {lipid} expected change with GM3: {expected_change:.3f}")
    
    # Simple but robust feature selection
    print("\nSelecting features...")
    
    # Use only the most direct GM3 features
    feature_cols = []
    
    # Primary GM3 measure
    if 'gm3_contact_strength' in frame_df.columns:
        feature_cols.append('gm3_contact_strength')
    elif 'gm3_contact_count' in frame_df.columns:
        feature_cols.append('gm3_contact_count')
    
    # Add normalized GM3 features if available
    if 'gm3_density' in frame_df.columns:
        feature_cols.append('gm3_density')
    if 'gm3_fraction_vicinity' in frame_df.columns:
        feature_cols.append('gm3_fraction_vicinity')
    
    print(f"Using {len(feature_cols)} features")
    
    if len(feature_cols) == 0:
        print("ERROR: No GM3 features found!")
        return {}, frame_df
    
    # Prepare features
    X = frame_df[feature_cols].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    # Direct correlation analysis first
    print("\nDirect correlation analysis:")
    direct_correlations = {}
    
    for lipid in ['CHOL', 'DIPC', 'DPSM']:
        if f'{lipid}_contact_count' in frame_df.columns and 'gm3_contact_strength' in frame_df.columns:
            # Remove outliers for robust correlation
            gm3_vals = frame_df['gm3_contact_strength'].values
            lipid_vals = frame_df[f'{lipid}_contact_count'].values
            
            # Remove top 5% outliers
            gm3_p95 = np.percentile(gm3_vals, 95)
            lipid_p95 = np.percentile(lipid_vals, 95)
            mask = (gm3_vals < gm3_p95) & (lipid_vals < lipid_p95)
            
            if mask.sum() > 30:
                corr, pval = stats.pearsonr(gm3_vals[mask], lipid_vals[mask])
                direct_correlations[lipid] = {
                    'correlation': corr,
                    'p_value': pval,
                    'n_samples': mask.sum()
                }
                print(f"  {lipid}: r = {corr:.3f} (p = {pval:.4f})")
    
    # Machine learning analysis with simplified approach
    for lipid in ['CHOL', 'DIPC', 'DPSM']:
        print(f"\n  Analyzing {lipid}...")
        
        # Prioritize interpretable targets
        targets = {}
        
        # Best target: deviation from baseline
        if f'{lipid}_deviation_from_baseline' in frame_df.columns:
            targets['deviation_baseline'] = frame_df[f'{lipid}_deviation_from_baseline'].values
        
        # Alternative: deviation from rolling mean
        if f'{lipid}_deviation_from_rolling' in frame_df.columns:
            targets['deviation_rolling'] = frame_df[f'{lipid}_deviation_from_rolling'].values
        
        # Normalized fraction (if available)
        if f'{lipid}_normalized_fraction' in frame_df.columns:
            targets['normalized_fraction'] = frame_df[f'{lipid}_normalized_fraction'].values
        
        # Fallback: raw counts
        if not targets and f'{lipid}_contact_count' in frame_df.columns:
            targets['contact_count'] = frame_df[f'{lipid}_contact_count'].values
        
        lipid_results = {}
        best_overall_score = -999
        best_overall_target = None
        
        for target_name, y in targets.items():
            print(f"    Target: {target_name}")
            
            # Skip if target has no variation
            if np.std(y) < 1e-6:
                print(f"      Skipping - no variation")
                continue
            
            # Simple train-test split (more stable than time series for small data)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42  # Fixed seed from ORIGINAL_ANALYSIS
            )
            
            # Start with simple linear regression
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            lr_score = lr.score(X_test, y_test)
            
            # Calculate effect directly from linear model
            # Effect = coefficient * (high GM3 - low GM3)
            gm3_feature_idx = 0  # First feature is primary GM3
            gm3_coefficient = lr.coef_[gm3_feature_idx]
            gm3_range = np.percentile(X_scaled[:, gm3_feature_idx], 80) - np.percentile(X_scaled[:, gm3_feature_idx], 20)
            linear_gm3_effect = gm3_coefficient * gm3_range
            
            # Ridge regression for comparison
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            ridge_score = ridge.score(X_test, y_test)
            
            # Select best model
            if lr_score > ridge_score and lr_score > -0.5:
                final_model = lr
                final_score = lr_score
                model_type = 'Linear'
                gm3_effect = linear_gm3_effect
            else:
                final_model = ridge
                final_score = ridge_score
                model_type = 'Ridge'
                gm3_effect = ridge.coef_[gm3_feature_idx] * gm3_range
            
            # Use direct correlation if available
            if lipid in direct_correlations:
                correlation = direct_correlations[lipid]['correlation']
                p_value = direct_correlations[lipid]['p_value']
            else:
                # Calculate correlation with target
                correlation, p_value = stats.pearsonr(X_scaled[:, 0], y)
            
            # Ensure effect sign matches correlation
            if correlation != 0:
                gm3_effect = abs(gm3_effect) * np.sign(correlation)
            
            # Store results
            lipid_results[target_name] = {
                'model_type': model_type,
                'score': final_score,
                'linear_score': lr_score,
                'ridge_score': ridge_score,
                'correlation': correlation,
                'p_value': p_value,
                'gm3_effect': gm3_effect,
                'gm3_coefficient': gm3_coefficient if model_type == 'Linear' else ridge.coef_[0]
            }
            
            # Track best overall
            if final_score > best_overall_score:
                best_overall_score = final_score
                best_overall_target = target_name
            
            print(f"      Model: {model_type}, R² = {final_score:.3f}")
            print(f"      GM3 effect: {gm3_effect:.3f}")
            print(f"      Correlation: {correlation:.3f} (p={p_value:.4f})")
        
        # Store best results for this lipid
        if best_overall_target and best_overall_target in lipid_results:
            best_results = lipid_results[best_overall_target].copy()
            best_results['best_target'] = best_overall_target
            best_results['all_targets'] = lipid_results
            
            # Add direct correlation info
            if lipid in direct_correlations:
                best_results['direct_correlation'] = direct_correlations[lipid]['correlation']
                best_results['direct_p_value'] = direct_correlations[lipid]['p_value']
            
            results[lipid] = best_results
        elif lipid_results:
            # Fallback to first available target
            first_target = list(lipid_results.keys())[0]
            best_results = lipid_results[first_target].copy()
            best_results['best_target'] = first_target
            best_results['all_targets'] = lipid_results
            results[lipid] = best_results
        else:
            # No results - use direct correlation only
            if lipid in direct_correlations:
                results[lipid] = {
                    'correlation': direct_correlations[lipid]['correlation'],
                    'p_value': direct_correlations[lipid]['p_value'],
                    'gm3_effect': direct_correlations[lipid]['correlation'] * 2.0,  # Scale factor
                    'model_type': 'Correlation',
                    'score': -999,
                    'best_target': 'direct_correlation'
                }
    
    return results, frame_df


class MLAnalyzer:
    """
    Wrapper class to match CLAIRE structure
    """
    
    @staticmethod
    def advanced_ml_analysis(df: pd.DataFrame, 
                            target_lipids: Optional[List[str]] = None,
                            mediator_lipid: str = 'DPG3') -> Tuple[Dict, pd.DataFrame]:
        """
        Wrapper for the exact original_analysis function
        """
        return advanced_ml_analysis(df)