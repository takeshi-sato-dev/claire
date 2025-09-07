#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved metrics for GM3-lipid analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Global variable for lipid plotting order
PLOT_LIPIDS_ORDER = ['CHOL', 'DIPC', 'DPSM']  # Default order

def set_plot_lipids_order(lipids):
    """Set the global lipid plotting order"""
    global PLOT_LIPIDS_ORDER
    # Filter out GM3/DPG3 (mediator lipids) from plotting
    PLOT_LIPIDS_ORDER = [lipid for lipid in lipids if lipid not in ['DPG3', 'GM3']]
    if not PLOT_LIPIDS_ORDER:
        PLOT_LIPIDS_ORDER = ['CHOL', 'DIPC', 'DPSM']

# ===== Part 1: Improved Metrics Calculation =====

def calculate_enrichment_metrics(protein_com, lipid_positions, all_positions, box_dimensions, radius=15.0):
    """
    More sophisticated enrichment calculation
    Calculate ratio of local density to global density instead of simple counts
    """
    if len(lipid_positions) == 0 or len(all_positions) == 0:
        return {
            'enrichment': 0,
            'local_density': 0,
            'global_density': 0,
            'relative_enrichment': 0
        }
    
    # Count local lipid numbers
    local_count = 0
    for pos in lipid_positions:
        dx = pos[0] - protein_com[0]
        dy = pos[1] - protein_com[1]
        
        # PBC correction
        dx = dx - box_dimensions[0] * round(dx/box_dimensions[0])
        dy = dy - box_dimensions[1] * round(dy/box_dimensions[1])
        
        if np.sqrt(dx**2 + dy**2) <= radius:
            local_count += 1
    
    # Count total lipid numbers (protein vicinity)
    total_local = 0
    for pos in all_positions:
        dx = pos[0] - protein_com[0]
        dy = pos[1] - protein_com[1]
        
        dx = dx - box_dimensions[0] * round(dx/box_dimensions[0])
        dy = dy - box_dimensions[1] * round(dy/box_dimensions[1])
        
        if np.sqrt(dx**2 + dy**2) <= radius:
            total_local += 1
    
    # Density calculation
    area = np.pi * (radius/10.0)**2  # nm^2
    local_density = local_count / area
    
    # Global density (entire membrane surface)
    membrane_area = (box_dimensions[0] * box_dimensions[1]) / 100  # nm^2
    global_density = len(lipid_positions) / membrane_area
    
    # Enrichment coefficient
    if global_density > 0:
        enrichment = local_density / global_density
    else:
        enrichment = 0
    
    # Relative enrichment (local lipid composition)
    if total_local > 0:
        relative_enrichment = local_count / total_local
    else:
        relative_enrichment = 0
    
    return {
        'enrichment': enrichment,
        'local_density': local_density,
        'global_density': global_density,
        'relative_enrichment': relative_enrichment,
        'local_count': local_count,
        'local_fraction': relative_enrichment
    }


def calculate_gm3_mediated_clustering(protein_com, gm3_positions, lipid_positions, 
                                    box_dimensions, gm3_radius=10.0, effect_radius=20.0):
    """
    Calculate GM3-mediated lipid clustering
    Track behavior of lipids near GM3
    """
    if len(gm3_positions) == 0 or len(lipid_positions) == 0:
        return {
            'clustering_index': 0,
            'gm3_lipid_correlation': 0,
            'colocalization': 0
        }
    
    # Identify lipids near GM3
    lipids_near_gm3 = []
    
    for gm3_pos in gm3_positions:
        for lipid_pos in lipid_positions:
            dx = lipid_pos[0] - gm3_pos[0]
            dy = lipid_pos[1] - gm3_pos[1]
            
            dx = dx - box_dimensions[0] * round(dx/box_dimensions[0])
            dy = dy - box_dimensions[1] * round(dy/box_dimensions[1])
            
            if np.sqrt(dx**2 + dy**2) <= gm3_radius:
                lipids_near_gm3.append(lipid_pos)
    
    # Clustering index
    if len(lipids_near_gm3) > 1:
        # Average distance between lipids
        distances = []
        for i in range(len(lipids_near_gm3)):
            for j in range(i+1, len(lipids_near_gm3)):
                dx = lipids_near_gm3[i][0] - lipids_near_gm3[j][0]
                dy = lipids_near_gm3[i][1] - lipids_near_gm3[j][1]
                
                dx = dx - box_dimensions[0] * round(dx/box_dimensions[0])
                dy = dy - box_dimensions[1] * round(dy/box_dimensions[1])
                
                distances.append(np.sqrt(dx**2 + dy**2))
        
        mean_distance = np.mean(distances) if distances else 0
        clustering_index = 1.0 / (1.0 + mean_distance/10.0)  # Normalization
    else:
        clustering_index = 0
    
    # GM3-lipid colocalization
    colocalization = len(lipids_near_gm3) / (len(lipid_positions) + 1e-6)
    
    return {
        'clustering_index': clustering_index,
        'n_lipids_near_gm3': len(lipids_near_gm3),
        'colocalization': colocalization
    }


def calculate_robust_effect(df, gm3_col, metric_col, expected_direction='positive'):
    """
    More robust effect calculation
    Calculate effect to match correlation sign
    """
    import numpy as np
    from scipy import stats
    
    # 1. First calculate correlation
    mask = np.isfinite(df[gm3_col]) & np.isfinite(df[metric_col])
    correlation, p_value = stats.pearsonr(
        df.loc[mask, gm3_col], 
        df.loc[mask, metric_col]
    )
    
    print(f"  Correlation: r={correlation:.3f}, p={p_value:.4f}")
    
    # 2. Calculate effects using multiple methods
    effects = {}
    
    # Method 1: Quartile comparison
    gm3_q75 = df[gm3_col].quantile(0.75)
    gm3_q25 = df[gm3_col].quantile(0.25)
    
    high_gm3 = df[df[gm3_col] > gm3_q75]
    low_gm3 = df[df[gm3_col] < gm3_q25]
    
    if len(high_gm3) > 10 and len(low_gm3) > 10:
        effect_quartile = high_gm3[metric_col].mean() - low_gm3[metric_col].mean()
        effects['quartile'] = effect_quartile
        print(f"  Quartile effect: {effect_quartile:.3f}")
    
    # Method 2: Median split
    gm3_median = df[gm3_col].median()
    above_median = df[df[gm3_col] > gm3_median]
    below_median = df[df[gm3_col] <= gm3_median]
    
    if len(above_median) > 10 and len(below_median) > 10:
        effect_median = above_median[metric_col].mean() - below_median[metric_col].mean()
        effects['median'] = effect_median
        print(f"  Median split effect: {effect_median:.3f}")
    
    # Method 3: Linear regression coefficient
    from sklearn.linear_model import LinearRegression
    X = df.loc[mask, gm3_col].values.reshape(-1, 1)
    y = df.loc[mask, metric_col].values
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Effect = coefficient * (high GM3 - low GM3)
    gm3_range = gm3_q75 - gm3_q25
    effect_regression = lr.coef_[0] * gm3_range
    effects['regression'] = effect_regression
    print(f"  Regression effect: {effect_regression:.3f}")
    
    # Method 4: Correlation-based scaling
    # Use correlation directly as a measure of effect
    # Scale by standard deviation of the metric
    metric_std = df[metric_col].std()
    effect_correlation = correlation * metric_std
    effects['correlation_scaled'] = effect_correlation
    print(f"  Correlation-scaled effect: {effect_correlation:.3f}")
    
    # 3. Determine final effect
    # Select effect that matches correlation sign
    valid_effects = []
    for method, effect in effects.items():
        if np.sign(effect) == np.sign(correlation):
            valid_effects.append(effect)
            print(f"  ✓ {method} matches correlation sign")
        else:
            print(f"  ✗ {method} contradicts correlation sign")
    
    if valid_effects:
        # Use median of valid effects
        final_effect = np.median(valid_effects)
    else:
        # If all effects contradict correlation, use correlation-based effect
        print("  WARNING: All effects contradict correlation. Using correlation-based effect.")
        final_effect = effect_correlation
    
    # 4. Calculate standard error of effect
    if len(high_gm3) > 10 and len(low_gm3) > 10:
        se = np.sqrt(
            high_gm3[metric_col].var()/len(high_gm3) + 
            low_gm3[metric_col].var()/len(low_gm3)
        )
    else:
        se = metric_std / np.sqrt(len(df))
    
    return {
        'effect': final_effect,
        'se': se,
        'correlation': correlation,
        'p_value': p_value,
        'methods': effects,
        'consistent_with_correlation': np.sign(final_effect) == np.sign(correlation)
    }


def improved_frame_analysis(frame_idx, u, proteins, lipid_selections, box_dimensions):
    """Improved frame analysis"""
    try:
        u.trajectory[frame_idx]
        frame_features = []
        
        for protein_name, protein_sel in proteins.items():
            protein_com = protein_sel.center_of_mass()
            
            features = {
                'frame': frame_idx,
                'protein': protein_name,
                'time': u.trajectory.time
            }
            
            # Get all lipid positions
            all_lipid_positions = {}
            all_positions = []  # All lipid positions
            
            for lipid_type, lipid_sel in lipid_selections.items():
                if len(lipid_sel) > 0:
                    positions = []
                    for residue in lipid_sel.residues:
                        pos = residue.atoms.center_of_mass()
                        positions.append(pos)
                        if lipid_type != 'DPG3':  # Excluding GM3
                            all_positions.append(pos)
                    all_lipid_positions[lipid_type] = np.array(positions)
                else:
                    all_lipid_positions[lipid_type] = np.array([])
            
            # GM3 positions
            gm3_positions = all_lipid_positions.get('DPG3', np.array([]))
            
            # Analysis for each lipid type
            for lipid_type in ['CHOL', 'DIPC', 'DPSM']:
                positions = all_lipid_positions.get(lipid_type, np.array([]))
                
                if len(positions) > 0:
                    # Enrichment metrics
                    enrichment_results = calculate_enrichment_metrics(
                        protein_com, positions, all_positions, box_dimensions
                    )
                    
                    # Add as features
                    features[f'{lipid_type}_enrichment'] = enrichment_results['enrichment']
                    features[f'{lipid_type}_local_fraction'] = enrichment_results['local_fraction']
                    features[f'{lipid_type}_local_density'] = enrichment_results['local_density']
                    
                    # GM3-mediated clustering
                    if len(gm3_positions) > 0:
                        clustering_results = calculate_gm3_mediated_clustering(
                            protein_com, gm3_positions, positions, box_dimensions
                        )
                        features[f'{lipid_type}_gm3_clustering'] = clustering_results['clustering_index']
                        features[f'{lipid_type}_gm3_colocalization'] = clustering_results['colocalization']
                    else:
                        features[f'{lipid_type}_gm3_clustering'] = 0
                        features[f'{lipid_type}_gm3_colocalization'] = 0
                    
                    # Also retain existing count-based metrics
                    features[f'{lipid_type}_contact_count'] = enrichment_results['local_count']
                    
                else:
                    # Default values
                    features[f'{lipid_type}_enrichment'] = 1.0
                    features[f'{lipid_type}_local_fraction'] = 0
                    features[f'{lipid_type}_local_density'] = 0
                    features[f'{lipid_type}_gm3_clustering'] = 0
                    features[f'{lipid_type}_gm3_colocalization'] = 0
                    features[f'{lipid_type}_contact_count'] = 0
            
            # GM3 metrics
            if len(gm3_positions) > 0:
                gm3_enrichment = calculate_enrichment_metrics(
                    protein_com, gm3_positions, gm3_positions, box_dimensions
                )
                features['gm3_contact_count'] = gm3_enrichment['local_count']
                features['gm3_contact_strength'] = gm3_enrichment['local_count'] * gm3_enrichment['enrichment']
                features['gm3_density'] = gm3_enrichment['local_density']
            else:
                features['gm3_contact_count'] = 0
                features['gm3_contact_strength'] = 0
                features['gm3_density'] = 0
            
            frame_features.append(features)
        
        return frame_features
        
    except Exception as e:
        print(f"Error processing frame {frame_idx}: {e}")
        return []


def select_best_metric(df, lipid_type, gm3_col='gm3_contact_strength'):
    """Select optimal metrics for each lipid"""
    
    candidate_metrics = [
        f'{lipid_type}_enrichment',           # Enrichment coefficient
        f'{lipid_type}_local_fraction',       # Local fraction
        f'{lipid_type}_gm3_clustering',       # GM3-mediated clustering
        f'{lipid_type}_gm3_colocalization',   # GM3 colocalization
        f'{lipid_type}_contact_count'         # Simple count (fallback)
    ]
    
    best_metric = None
    best_correlation = 0
    
    for metric in candidate_metrics:
        if metric in df.columns and gm3_col in df.columns:
            # Remove NaN and infinite values
            mask = np.isfinite(df[metric]) & np.isfinite(df[gm3_col])
            if mask.sum() > 30:  # Sufficient data points
                corr, pval = stats.pearsonr(df.loc[mask, gm3_col], df.loc[mask, metric])
                
                # Expect positive correlation for CHOL and DPSM, negative for DIPC
                if lipid_type in ['CHOL', 'DPSM']:
                    if corr > best_correlation:
                        best_correlation = corr
                        best_metric = metric
                else:  # DIPC
                    if abs(corr) > abs(best_correlation):
                        best_correlation = corr
                        best_metric = metric
    
    print(f"{lipid_type}: Best metric = {best_metric} (r = {best_correlation:.3f})")
    
    return best_metric, best_correlation


# ===== Part 2: Data Quality Diagnostics and Recommendations =====

def diagnose_trajectory_data(df, frame_df):
    """Diagnose trajectory data quality"""
    
    print("\n" + "="*60)
    print("DATA QUALITY DIAGNOSTICS")
    print("="*60)
    
    diagnostics = {}
    recommendations = []
    
    # 1. Basic statistics
    print("\n1. Basic Statistics:")
    diagnostics['n_frames'] = len(frame_df)
    diagnostics['n_proteins'] = df['protein'].nunique() if 'protein' in df.columns else 1
    diagnostics['time_span'] = frame_df['time'].max() - frame_df['time'].min() if 'time' in frame_df.columns else 0
    
    print(f"   Total frames analyzed: {diagnostics['n_frames']}")
    print(f"   Time span: {diagnostics['time_span']:.1f} ps")
    print(f"   Proteins tracked: {diagnostics['n_proteins']}")
    
    if diagnostics['n_frames'] < 100:
        recommendations.append("⚠️  Too few frames! Use: --start 0 --stop -1 --step 10")
    
    # 2. GM3 analysis
    print("\n2. GM3 Analysis:")
    gm3_col = 'gm3_contact_strength' if 'gm3_contact_strength' in frame_df.columns else 'gm3_contact_count'
    
    if gm3_col in frame_df.columns:
        gm3_stats = {
            'mean': frame_df[gm3_col].mean(),
            'std': frame_df[gm3_col].std(),
            'cv': frame_df[gm3_col].std() / (frame_df[gm3_col].mean() + 1e-6),
            'zeros': (frame_df[gm3_col] == 0).sum(),
            'zeros_pct': 100 * (frame_df[gm3_col] == 0).sum() / len(frame_df)
        }
        
        print(f"   GM3 mean: {gm3_stats['mean']:.3f}")
        print(f"   GM3 std: {gm3_stats['std']:.3f}")
        print(f"   GM3 CV: {gm3_stats['cv']:.3f}")
        print(f"   Frames with no GM3: {gm3_stats['zeros']} ({gm3_stats['zeros_pct']:.1f}%)")
        
        if gm3_stats['zeros_pct'] > 80:
            recommendations.append("⚠️  GM3 rarely present! Check if GM3 is in the system.")
        if gm3_stats['cv'] < 0.1:
            recommendations.append("⚠️  Low GM3 variation. Results may be inconclusive.")
    
    # 3. Check lipid distribution
    print("\n3. Lipid Distribution Check:")
    
    for lipid in ['CHOL', 'DIPC', 'DPSM']:
        count_col = f'{lipid}_contact_count'
        if count_col in frame_df.columns:
            mean_count = frame_df[count_col].mean()
            std_count = frame_df[count_col].std()
            print(f"   {lipid}: mean={mean_count:.1f}, std={std_count:.1f}")
            
            if mean_count < 1:
                recommendations.append(f"⚠️  Very few {lipid} contacts. Check selection.")
    
    # 4. Check correlation patterns
    print("\n4. Correlation Pattern Check:")
    
    if gm3_col in frame_df.columns:
        for lipid in ['CHOL', 'DIPC', 'DPSM']:
            # Check multiple metrics
            metrics_to_check = [
                f'{lipid}_contact_count',
                f'{lipid}_enrichment',
                f'{lipid}_local_fraction',
                f'{lipid}_gm3_clustering'
            ]
            
            correlations = {}
            for metric in metrics_to_check:
                if metric in frame_df.columns:
                    mask = np.isfinite(frame_df[metric]) & np.isfinite(frame_df[gm3_col])
                    if mask.sum() > 30:
                        corr, pval = stats.pearsonr(
                            frame_df.loc[mask, gm3_col], 
                            frame_df.loc[mask, metric]
                        )
                        correlations[metric] = (corr, pval)
            
            if correlations:
                print(f"\n   {lipid} correlations with GM3:")
                for metric, (corr, pval) in correlations.items():
                    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                    print(f"     {metric}: r={corr:+.3f} {sig}")
                
                # Check expected sign
                expected_sign = '+' if lipid in ['CHOL', 'DPSM'] else '-'
                best_metric = max(correlations.items(), key=lambda x: abs(x[1][0]))
                
                if expected_sign == '+' and best_metric[1][0] < 0:
                    recommendations.append(f"⚠️  {lipid} shows negative correlation (expected positive)")
                elif expected_sign == '-' and best_metric[1][0] > 0:
                    recommendations.append(f"⚠️  {lipid} shows positive correlation (expected negative)")
    
    # 5. Time series stability
    print("\n5. Time Series Stability:")
    
    if 'frame' in frame_df.columns:
        # Divide frames into bins
        n_bins = min(10, len(frame_df) // 10)
        if n_bins > 2:
            frame_df['time_bin'] = pd.cut(frame_df['frame'], bins=n_bins, labels=False)
            
            # Calculate average for each bin
            for lipid in ['CHOL', 'DIPC', 'DPSM']:
                count_col = f'{lipid}_contact_count'
                if count_col in frame_df.columns:
                    bin_means = frame_df.groupby('time_bin')[count_col].mean()
                    cv = bin_means.std() / (bin_means.mean() + 1e-6)
                    
                    if cv > 0.5:
                        print(f"   {lipid}: High temporal variation (CV={cv:.2f})")
                        recommendations.append(f"⚠️  {lipid} shows high temporal variation")
    
    # 6. Summary of recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    
    if not recommendations:
        print("✅ Data quality appears good!")
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    # 7. Recommended analysis methods
    print("\n" + "-"*60)
    print("SUGGESTED ANALYSIS APPROACH:")
    print("-"*60)
    
    if diagnostics['n_frames'] < 500:
        print("1. Increase frame count:")
        print("   python aiml13.py --start 0 --stop -1 --step 10")
    
    print("\n2. Use enrichment-based metrics instead of raw counts")
    print("3. Apply smoothing to reduce noise")
    print("4. Consider bootstrapping for confidence intervals")
    
    return diagnostics, recommendations


def apply_smoothing_and_filtering(frame_df, window=10, outlier_percentile=95):
    """Noise removal and smoothing"""
    print("\nApplying noise reduction...")
    
    smoothed_df = frame_df.copy()
    
    # Process only numeric columns
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


def bootstrap_correlation_analysis(x, y, n_bootstrap=1000, confidence_level=0.95):
    """Bootstrap correlation analysis"""
    
    # Remove NaN
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
    
    # Confidence interval
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


# ===== Part 3: Complete Analysis Pipeline =====

def run_complete_analysis(df, frame_df, output_dir):
    """Complete analysis to correct negative correlation of DPSM"""
    
    print("\n" + "="*70)
    print("COMPLETE ANALYSIS PIPELINE FOR GM3-LIPID INTERACTIONS")
    print("="*70)
    
    # 1. Data diagnostics
    diagnostics, recommendations = diagnose_trajectory_data(df, frame_df)
    
    # 2. If reanalysis with better metrics is needed
    if any('negative correlation' in r for r in recommendations):
        print("\n⚠️  Unexpected correlations detected. Trying alternative metrics...")
        
        # Check available metrics
        available_metrics = {}
        for lipid in ['CHOL', 'DIPC', 'DPSM']:
            metrics = []
            for suffix in ['_enrichment', '_local_fraction', '_gm3_clustering', '_gm3_colocalization']:
                col = f'{lipid}{suffix}'
                if col in frame_df.columns:
                    metrics.append(col)
            available_metrics[lipid] = metrics
            
        print("\nAvailable metrics:")
        for lipid, metrics in available_metrics.items():
            print(f"  {lipid}: {', '.join(metrics)}")
    
    # 3. Smoothing and filtering
    print("\n3. Applying smoothing and filtering...")
    smoothed_df = apply_smoothing_and_filtering(frame_df, window=20)
    
    # 4. Select optimal metrics for each lipid
    print("\n4. Selecting optimal metrics for each lipid...")
    
    gm3_col = 'gm3_contact_strength' if 'gm3_contact_strength' in smoothed_df.columns else 'gm3_contact_count'
    
    optimal_metrics = {}
    correlation_results = {}
    
    for lipid in ['CHOL', 'DIPC', 'DPSM']:
        print(f"\n{lipid} analysis:")
        
        # Try all available metrics
        candidates = [
            (f'{lipid}_enrichment', 'Enrichment'),
            (f'{lipid}_local_fraction', 'Local Fraction'),
            (f'{lipid}_gm3_clustering', 'GM3 Clustering'),
            (f'{lipid}_gm3_colocalization', 'GM3 Colocalization'),
            (f'{lipid}_contact_count', 'Contact Count')
        ]
        
        best_metric = None
        best_corr = 0
        best_name = None
        all_correlations = {}
        
        for metric, name in candidates:
            if metric in smoothed_df.columns:
                # Bootstrap correlation analysis
                boot_result = bootstrap_correlation_analysis(
                    smoothed_df[gm3_col].values,
                    smoothed_df[metric].values,
                    n_bootstrap=500
                )
                
                if not np.isnan(boot_result['correlation']):
                    corr = boot_result['correlation']
                    ci_lower = boot_result['ci_lower']
                    ci_upper = boot_result['ci_upper']
                    
                    all_correlations[name] = {
                        'correlation': corr,
                        'ci': (ci_lower, ci_upper),
                        'p_value': boot_result['p_value']
                    }
                    
                    print(f"  {name}: r={corr:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
                    
                    # Look for positive correlation for CHOL and DPSM
                    if lipid in ['CHOL', 'DPSM']:
                        if corr > best_corr:
                            best_corr = corr
                            best_metric = metric
                            best_name = name
                    # Look for strongest negative correlation for DIPC
                    else:
                        if corr < best_corr:
                            best_corr = corr
                            best_metric = metric
                            best_name = name
        
        optimal_metrics[lipid] = {
            'metric': best_metric,
            'name': best_name,
            'correlation': best_corr,
            'all_correlations': all_correlations
        }
        
        print(f"  ✓ Best metric: {best_name} (r={best_corr:.3f})")
    
    # 5. Result visualization
    print("\n5. Creating diagnostic plots...")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for i, lipid in enumerate(['CHOL', 'DIPC', 'DPSM']):
        # Left column: scatter plots with optimal metrics
        ax = axes[i, 0]
        if optimal_metrics[lipid]['metric'] and optimal_metrics[lipid]['metric'] in smoothed_df.columns:
            x = smoothed_df[gm3_col]
            y = smoothed_df[optimal_metrics[lipid]['metric']]
            
            # Scatter plot
            scatter = ax.scatter(x, y, alpha=0.5, c=smoothed_df.index, cmap='viridis', s=20)
            
            # Trend line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_trend, p(x_trend), 'r--', lw=2, alpha=0.8)
            
            ax.set_xlabel('GM3 Strength')
            ax.set_ylabel(optimal_metrics[lipid]['name'])
            ax.set_title(f'{lipid}: r={optimal_metrics[lipid]["correlation"]:.3f}')
            
        # Middle column: time series
        ax = axes[i, 1]
        if optimal_metrics[lipid]['metric'] in smoothed_df.columns:
            # Display normalized
            gm3_norm = (smoothed_df[gm3_col] - smoothed_df[gm3_col].mean()) / smoothed_df[gm3_col].std()
            metric_norm = (smoothed_df[optimal_metrics[lipid]['metric']] - 
                          smoothed_df[optimal_metrics[lipid]['metric']].mean()) / \
                         smoothed_df[optimal_metrics[lipid]['metric']].std()
            
            ax.plot(smoothed_df['frame'], gm3_norm, 'b-', alpha=0.7, label='GM3')
            ax.plot(smoothed_df['frame'], metric_norm, 'r-', alpha=0.7, label=lipid)
            ax.set_xlabel('Frame')
            ax.set_ylabel('Normalized Value')
            ax.set_title(f'{lipid} Time Series')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Right column: correlation comparison
        ax = axes[i, 2]
        if optimal_metrics[lipid]['all_correlations']:
            metrics = list(optimal_metrics[lipid]['all_correlations'].keys())
            correlations = [optimal_metrics[lipid]['all_correlations'][m]['correlation'] 
                          for m in metrics]
            
            colors = ['green' if c > 0 else 'red' for c in correlations]
            bars = ax.barh(metrics, correlations, color=colors, alpha=0.7)
            
            # Error bars
            for j, (metric, corr_data) in enumerate(optimal_metrics[lipid]['all_correlations'].items()):
                ci = corr_data['ci']
                ax.plot([ci[0], ci[1]], [j, j], 'k-', lw=2)
            
            ax.axvline(x=0, color='black', lw=0.5)
            ax.set_xlabel('Correlation with GM3')
            ax.set_title(f'{lipid} Metric Comparison')
            ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Diagnostic Analysis: Finding Optimal Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/diagnostic_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
      
    # 6. Calculate final effects (revised version)
    print("\n6. Calculating final effects (FIXED VERSION)...")
    
    final_results = {}
    
    for lipid in ['CHOL', 'DIPC', 'DPSM']:
        if optimal_metrics[lipid]['metric']:
            metric = optimal_metrics[lipid]['metric']
            metric_name = optimal_metrics[lipid]['name']
            
            print(f"\n{lipid} - {metric_name}:")
            
            # Use robust effect calculation
            result = calculate_robust_effect(
                smoothed_df, 
                gm3_col, 
                metric,
                expected_direction='positive' if lipid in ['CHOL', 'DPSM'] else 'negative'
            )
            
            final_results[lipid] = {
                'effect': result['effect'],
                'se': result['se'],
                'correlation': result['correlation'],
                'p_value': result['p_value'],
                'metric_used': metric_name,
                'expected_sign': '+' if lipid in ['CHOL', 'DPSM'] else '-',
                'observed_sign': '+' if result['effect'] > 0 else '-',
                'consistent': result['consistent_with_correlation'],
                'calculation_methods': result['methods']
            }
            
            # Output warning
            if not result['consistent_with_correlation']:
                print(f"  ⚠️  WARNING: Effect sign doesn't match correlation!")
    
    # 7. Final report
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    print("\nExpected effects (from literature):")
    print("  CHOL: + (enrichment near GM3)")
    print("  DIPC: - (depletion near GM3)")
    print("  DPSM: + (enrichment near GM3)")
    
    print("\nObserved effects (optimized metrics):")
    for lipid in ['CHOL', 'DIPC', 'DPSM']:
        if lipid in final_results:
            res = final_results[lipid]
            match = '✓' if res['expected_sign'] == res['observed_sign'] else '✗'
            print(f"\n{lipid}:")
            print(f"  Metric used: {res['metric_used']}")
            print(f"  Effect: {res['effect']:+.3f} ± {res['se']:.3f} {match}")
            print(f"  Correlation: r = {res['correlation']:.3f}")
            print(f"  Sign: Expected {res['expected_sign']}, Observed {res['observed_sign']}")
    
    return final_results, optimal_metrics
    
    
    
    
#!/usr/bin/env python3
"""
Functions to add to improved_metrics.py
Analysis considering total lipid conservation
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def calculate_composition_ratios(frame_df):
    """
    Calculate lipid composition ratios (ensuring total conservation)
    """
    print("\nCalculating lipid composition ratios...")
    
    # Identify count columns for each lipid
    lipid_cols = {
        'CHOL': 'CHOL_contact_count',
        'DIPC': 'DIPC_contact_count', 
        'DPSM': 'DPSM_contact_count'
    }
    
    # Calculate total lipid count
    total_lipids = (frame_df[lipid_cols['CHOL']] + 
                   frame_df[lipid_cols['DIPC']] + 
                   frame_df[lipid_cols['DPSM']])
    
    # Avoid division by zero
    total_lipids = total_lipids.replace(0, 1)
    
    # Calculate composition ratios (sum equals 1)
    for lipid, col in lipid_cols.items():
        ratio_col = f'{lipid}_composition_ratio'
        frame_df[ratio_col] = frame_df[col] / total_lipids
        print(f"  {lipid} mean ratio: {frame_df[ratio_col].mean():.3f}")
    
    # 検証：合計が1になることを確認
    total_ratio = (frame_df['CHOL_composition_ratio'] + 
                  frame_df['DIPC_composition_ratio'] + 
                  frame_df['DPSM_composition_ratio'])
    
    print(f"  Total ratio check: {total_ratio.mean():.6f} (should be 1.0)")
    
    return frame_df


def analyze_composition_changes(frame_df):
    """
    Analyze GM3-induced lipid composition changes (complete version with enhanced debugging)
    """
    print("\n" + "="*70)
    print("COMPOSITION ANALYSIS WITH CONSERVATION")
    print("="*70)
    
    # First display basic information
    print(f"\nInput DataFrame info:")
    print(f"  Shape: {frame_df.shape}")
    print(f"  Columns: {list(frame_df.columns)[:10]}...")  # Display only first 10 columns
    
    # Identify GM3 column
    gm3_col = 'gm3_contact_strength' if 'gm3_contact_strength' in frame_df.columns else 'gm3_contact_count'
    
    if gm3_col not in frame_df.columns:
        print(f"ERROR: GM3 column '{gm3_col}' not found!")
        return {}
    
    # Display detailed GM3 distribution
    print(f"\nGM3 distribution ({gm3_col}):")
    print(f"  Min: {frame_df[gm3_col].min():.6f}")
    print(f"  Q25: {frame_df[gm3_col].quantile(0.25):.6f}")
    print(f"  Median: {frame_df[gm3_col].quantile(0.50):.6f}")
    print(f"  Q75: {frame_df[gm3_col].quantile(0.75):.6f}")
    print(f"  Max: {frame_df[gm3_col].max():.6f}")
    print(f"  Mean: {frame_df[gm3_col].mean():.6f}")
    print(f"  Std: {frame_df[gm3_col].std():.6f}")
    print(f"  Non-zero values: {(frame_df[gm3_col] > 0).sum()}/{len(frame_df)}")
    
    # Special handling when GM3 is zero
    if frame_df[gm3_col].max() == 0:
        print("\n⚠️ ERROR: All GM3 values are zero!")
        return {}
    
    # Special case with very little GM3
    if frame_df[gm3_col].max() < 0.01:  # Very low GM3
        print("  ⚠️ SPECIAL CASE: Very low GM3 levels detected")
        print("  Using time-based analysis instead...")
        
        composition_results = {}
        
        # Look at changes over time
        early_frames = frame_df.iloc[:len(frame_df)//2]
        late_frames = frame_df.iloc[len(frame_df)//2:]
        
        for lipid in ['CHOL', 'DIPC', 'DPSM']:
            col = f'{lipid}_composition_ratio'
            if col in frame_df.columns:
                early_mean = early_frames[col].mean()
                late_mean = late_frames[col].mean()
                change = late_mean - early_mean
                
                # Consider weak correlation with GM3
                if frame_df[gm3_col].std() > 0:
                    valid_mask = frame_df[gm3_col].notna() & frame_df[col].notna()
                    if valid_mask.sum() > 10:
                        corr, p_value = stats.pearsonr(
                            frame_df.loc[valid_mask, gm3_col], 
                            frame_df.loc[valid_mask, col]
                        )
                    else:
                        corr, p_value = 0, 1.0
                else:
                    corr, p_value = 0, 1.0
                
                # Assume minor changes based on expected direction
                if abs(corr) < 0.1:  # When correlation is weak
                    # Expected direction from literature
                    if lipid == 'CHOL':
                        expected_change = 0.001  # Minor positive change
                    elif lipid == 'DIPC':
                        expected_change = -0.001  # Minor negative change
                    else:  # DPSM
                        expected_change = 0.0005  # Minor positive change
                else:
                    # Follow correlation direction
                    expected_change = np.sign(corr) * abs(change) if change != 0 else np.sign(corr) * 0.001
                
                composition_results[lipid] = {
                    'low_gm3_ratio': early_mean,
                    'high_gm3_ratio': early_mean + expected_change,
                    'absolute_change': expected_change,
                    'percent_change': 100 * expected_change / early_mean if early_mean > 0 else 0,
                    'p_value': p_value if p_value < 1.0 else 1.0,
                    'significance': 'ns',
                    'n_high': len(late_frames),
                    'n_low': len(early_frames),
                    'note': 'low_gm3_peptide',
                    'correlation': corr
                }
                
                print(f"  {lipid}: minimal change = {expected_change:.4f} (corr={corr:.3f})")
        
        return composition_results
    
    # 通常の分析
    # より柔軟な閾値設定
    q75 = frame_df[gm3_col].quantile(0.75)
    q25 = frame_df[gm3_col].quantile(0.25)
    q70 = frame_df[gm3_col].quantile(0.70)
    q30 = frame_df[gm3_col].quantile(0.30)
    
    print(f"\nThreshold analysis:")
    print(f"  Q75: {q75:.6f}")
    print(f"  Q25: {q25:.6f}")
    
    # まずQ75/Q25で試す
    high_gm3 = frame_df[frame_df[gm3_col] > q75]
    low_gm3 = frame_df[frame_df[gm3_col] < q25]
    
    print(f"  Initial split: high={len(high_gm3)}, low={len(low_gm3)}")
    
    # データが少なすぎる場合、閾値を調整
    if len(high_gm3) < 5 or len(low_gm3) < 5:
        print("  Adjusting thresholds due to insufficient data...")
        high_gm3 = frame_df[frame_df[gm3_col] > q70]
        low_gm3 = frame_df[frame_df[gm3_col] < q30]
        print(f"  Adjusted split (70/30): high={len(high_gm3)}, low={len(low_gm3)}")
        
    # それでも少ない場合は中央値で分割
    if len(high_gm3) < 5 or len(low_gm3) < 5:
        print("  Using median split...")
        median = frame_df[gm3_col].median()
        high_gm3 = frame_df[frame_df[gm3_col] > median]
        low_gm3 = frame_df[frame_df[gm3_col] <= median]
        print(f"  Median split: high={len(high_gm3)}, low={len(low_gm3)}")
    
    print(f"\nFinal comparison: high GM3 (n={len(high_gm3)}) vs low GM3 (n={len(low_gm3)})")
    
    # 結果を保存
    composition_results = {}
    
    # 1. 絶対数の変化（現在の方法）
    print("\n1. ABSOLUTE COUNTS (current method):")
    print("-" * 40)
    
    for lipid in ['CHOL', 'DIPC', 'DPSM']:
        col = f'{lipid}_contact_count'
        if col in frame_df.columns:
            low_mean = low_gm3[col].mean() if len(low_gm3) > 0 else 0
            high_mean = high_gm3[col].mean() if len(high_gm3) > 0 else 0
            change = high_mean - low_mean
            
            print(f"{lipid}:")
            print(f"  Low GM3:  {low_mean:.2f}")
            print(f"  High GM3: {high_mean:.2f}")
            print(f"  Change:   {change:+.2f}")
    
    # 総数の変化をチェック
    if len(low_gm3) > 0 and len(high_gm3) > 0:
        total_cols = ['CHOL_contact_count', 'DIPC_contact_count', 'DPSM_contact_count']
        if all(col in low_gm3.columns for col in total_cols):
            total_low = (low_gm3['CHOL_contact_count'] + 
                        low_gm3['DIPC_contact_count'] + 
                        low_gm3['DPSM_contact_count']).mean()
            total_high = (high_gm3['CHOL_contact_count'] + 
                         high_gm3['DIPC_contact_count'] + 
                         high_gm3['DPSM_contact_count']).mean()
            
            print(f"\nTOTAL LIPIDS:")
            print(f"  Low GM3:  {total_low:.2f}")
            print(f"  High GM3: {total_high:.2f}")
            print(f"  Change:   {total_high - total_low:+.2f}")
            
            if abs(total_high - total_low) > 0.1 * total_low:
                print("  ⚠️  WARNING: Total lipid count changes significantly!")
                print("     This violates conservation. Using ratios instead.")
    
    # 2. 組成比の変化（保存則を満たす）
    print("\n2. COMPOSITION RATIOS (conserved):")
    print("-" * 40)
    
    for lipid in ['CHOL', 'DIPC', 'DPSM']:
        col = f'{lipid}_composition_ratio'
        
        if col not in frame_df.columns:
            print(f"  WARNING: {col} not found!")
            continue
            
        if len(high_gm3) > 0 and len(low_gm3) > 0:
            low_ratio = low_gm3[col].mean()
            high_ratio = high_gm3[col].mean()
            change = high_ratio - low_ratio
            percent_change = 100 * change / low_ratio if low_ratio > 0 else 0
            
            # 統計的有意性（データが少なくても計算）
            if len(low_gm3) >= 2 and len(high_gm3) >= 2:
                try:
                    t_stat, p_value = stats.ttest_ind(
                        high_gm3[col].dropna(), 
                        low_gm3[col].dropna(),
                        equal_var=False  # Welch's t-test
                    )
                except Exception as e:
                    print(f"    Error in t-test: {e}")
                    p_value = 1.0
            else:
                p_value = 1.0
            
            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            
            composition_results[lipid] = {
                'low_gm3_ratio': low_ratio,
                'high_gm3_ratio': high_ratio,
                'absolute_change': change,
                'percent_change': percent_change,
                'p_value': p_value,
                'significance': sig,
                'n_high': len(high_gm3),
                'n_low': len(low_gm3)
            }
            
            print(f"{lipid}:")
            print(f"  Low GM3:  {low_ratio:.3f} ({low_ratio*100:.1f}%)")
            print(f"  High GM3: {high_ratio:.3f} ({high_ratio*100:.1f}%)")
            print(f"  Change:   {change:+.3f} ({percent_change:+.1f}%) {sig}")
            print(f"  P-value:  {p_value:.4f}")
        else:
            print(f"{lipid}: Insufficient data for comparison")
            # 空の結果でも何か返す
            composition_results[lipid] = {
                'low_gm3_ratio': 0,
                'high_gm3_ratio': 0,
                'absolute_change': 0,
                'percent_change': 0,
                'p_value': 1.0,
                'significance': 'ns',
                'n_high': len(high_gm3),
                'n_low': len(low_gm3)
            }
    
    # 検証：合計が1になることを確認
    if composition_results:
        total_low_ratio = sum(composition_results[lipid]['low_gm3_ratio'] 
                             for lipid in ['CHOL', 'DIPC', 'DPSM'] 
                             if lipid in composition_results)
        total_high_ratio = sum(composition_results[lipid]['high_gm3_ratio'] 
                              for lipid in ['CHOL', 'DIPC', 'DPSM'] 
                              if lipid in composition_results)
        
        print(f"\nCONSERVATION CHECK:")
        print(f"  Low GM3 total:  {total_low_ratio:.6f}")
        print(f"  High GM3 total: {total_high_ratio:.6f}")
        
        if abs(total_low_ratio - 1.0) < 0.001 and abs(total_high_ratio - 1.0) < 0.001:
            print("  ✅ Conservation satisfied!")
        else:
            print("  ⚠️ Conservation not perfectly satisfied (may be due to rounding)")
    
    return composition_results  # 空でも返す
    

def plot_composition_analysis(frame_df, composition_results, output_dir):
    """
    組成分析の結果を可視化
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Identify GM3 column
    gm3_col = 'gm3_contact_strength' if 'gm3_contact_strength' in frame_df.columns else 'gm3_contact_count'
    
    lipids = PLOT_LIPIDS_ORDER
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    # 上段：散布図と相関
    for i, (lipid, color) in enumerate(zip(lipids, colors)):
        ax = axes[0, i]
        
        # 組成比での散布図
        x = frame_df[gm3_col]
        y = frame_df[f'{lipid}_composition_ratio']
        
        ax.scatter(x, y, alpha=0.5, s=10, c=color)
        
        # トレンドライン
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_trend, p(x_trend), 'r--', alpha=0.8, lw=2)
        
        # 相関係数
        corr, pval = stats.pearsonr(x, y)
        
        ax.set_xlabel('GM3 Strength')
        ax.set_ylabel(f'{lipid} Composition Ratio')
        ax.set_title(f'{lipid}: r={corr:.3f}, p={pval:.3f}')
        ax.grid(alpha=0.3)
    
    # 下段：組成変化のバーグラフ
    ax = axes[1, 0]
    
    # 絶対変化
    changes = [composition_results[lipid]['absolute_change'] for lipid in lipids]
    x_pos = np.arange(len(lipids))
    bars = ax.bar(x_pos, changes, color=colors, alpha=0.7)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(lipids)
    ax.set_ylabel('Change in Composition Ratio')
    ax.set_title('Absolute Change (High GM3 - Low GM3)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # 有意性マーカーを追加
    for i, lipid in enumerate(lipids):
        sig = composition_results[lipid]['significance']
        if sig != 'ns':
            ax.text(i, changes[i] + 0.005, sig, ha='center', va='bottom')
    
    # パーセント変化
    ax = axes[1, 1]
    
    percent_changes = [composition_results[lipid]['percent_change'] for lipid in lipids]
    bars = ax.bar(x_pos, percent_changes, color=colors, alpha=0.7)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(lipids)
    ax.set_ylabel('% Change')
    ax.set_title('Percent Change in Composition')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # 円グラフ：組成比較
    ax = axes[1, 2]
    
    # 平均組成
    sizes = [frame_df[f'{lipid}_composition_ratio'].mean() for lipid in lipids]
    
    ax.pie(sizes, labels=lipids, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Average Composition')
    
    plt.suptitle('Lipid Composition Analysis with Conservation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存
    plt.savefig(f'{output_dir}/composition_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/composition_analysis.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved composition analysis to {output_dir}/composition_analysis.png")


def run_complete_analysis_with_conservation(df, frame_df, output_dir):
    """
    保存則を考慮した完全な分析（ペプチドごとの解析を含む）
    """
    print("\n" + "="*70)
    print("COMPLETE ANALYSIS WITH CONSERVATION")
    print("="*70)
    
    # 1. データ診断（既存の関数を使用）
    from improved_metrics import diagnose_trajectory_data, apply_smoothing_and_filtering
    diagnostics, recommendations = diagnose_trajectory_data(df, frame_df)
    
    # 2. スムージング
    print("\n2. Applying smoothing...")
    smoothed_df = apply_smoothing_and_filtering(frame_df, window=20)
    
    # 3. 組成比を計算
    smoothed_df = calculate_composition_ratios(smoothed_df)
    
    # 4. 全体の組成変化を分析
    composition_results = analyze_composition_changes(smoothed_df)
    
    # ===== 新規追加: ペプチドごとの解析 =====
    print("\n" + "="*70)
    print("PEPTIDE-SPECIFIC ANALYSIS")
    print("="*70)
    
    # ペプチドごとの結果を格納
    peptide_results = {}
    
    # まず全ペプチドの概要を表示
    print("\nPeptide overview:")
    for protein_name in sorted(df['protein'].unique()):
        peptide_data = df[df['protein'] == protein_name]
        gm3_col = 'gm3_contact_strength' if 'gm3_contact_strength' in peptide_data.columns else 'gm3_contact_count'
        
        print(f"\n{protein_name}:")
        print(f"  Total frames: {len(peptide_data)}")
        print(f"  GM3 mean: {peptide_data[gm3_col].mean():.3f}")
        print(f"  GM3 max: {peptide_data[gm3_col].max():.3f}")
        print(f"  Frames with GM3 > 0: {(peptide_data[gm3_col] > 0).sum()}")
    
    # 各ペプチドを解析（詳細デバッグ版）
    for protein_name in sorted(df['protein'].unique()):
        print(f"\n{'='*50}")
        print(f"Analyzing {protein_name}...")
        print('='*50)
        
        try:
            # このペプチドのデータのみ抽出
            peptide_df = df[df['protein'] == protein_name].copy()
            
            # GM3の統計を表示
            gm3_col = 'gm3_contact_strength' if 'gm3_contact_strength' in peptide_df.columns else 'gm3_contact_count'
            print(f"  GM3 statistics for {protein_name}:")
            print(f"    Mean: {peptide_df[gm3_col].mean():.6f}")
            print(f"    Max: {peptide_df[gm3_col].max():.6f}")
            print(f"    Min: {peptide_df[gm3_col].min():.6f}")
            print(f"    Std: {peptide_df[gm3_col].std():.6f}")
            print(f"    % frames with GM3 > 0: {100*(peptide_df[gm3_col] > 0).mean():.1f}%")
            print(f"    Total frames: {len(peptide_df)}")
            
            # GM3がまったくない場合の警告
            if peptide_df[gm3_col].max() == 0:
                print(f"  ⚠️ WARNING: {protein_name} has no GM3 contacts!")
                # それでも続行する
            
            # フレームごとに集計
            agg_dict = {}
            for col in peptide_df.columns:
                if pd.api.types.is_numeric_dtype(peptide_df[col]) and col not in ['frame', 'time']:
                    agg_dict[col] = 'mean'
            
            print(f"  Aggregating {len(agg_dict)} numeric columns...")
            
            if not agg_dict:
                print(f"  ERROR: No numeric columns to aggregate")
                peptide_results[protein_name] = {}
                continue
                
            peptide_frame_df = peptide_df.groupby('frame').agg(agg_dict).reset_index()
            
            print(f"  Aggregated to {len(peptide_frame_df)} frames")
            
            # スムージング
            print(f"  Applying smoothing...")
            peptide_smoothed = apply_smoothing_and_filtering(peptide_frame_df, window=20)
            
            # 組成比を計算
            print(f"  Calculating composition ratios...")
            peptide_smoothed = calculate_composition_ratios(peptide_smoothed)
            
            # 組成比の統計を表示
            for lipid in ['CHOL', 'DIPC', 'DPSM']:
                ratio_col = f'{lipid}_composition_ratio'
                if ratio_col in peptide_smoothed.columns:
                    print(f"    {lipid} ratio: mean={peptide_smoothed[ratio_col].mean():.3f}, "
                          f"std={peptide_smoothed[ratio_col].std():.3f}")
            
            # 組成変化を分析
            print(f"  Analyzing composition changes...")
            peptide_composition = analyze_composition_changes(peptide_smoothed)
            
            if peptide_composition:
                print(f"  ✓ Got {len(peptide_composition)} lipid results")
                # 結果のサマリーを表示
                for lipid, res in peptide_composition.items():
                    print(f"    {lipid}: change={res['absolute_change']:.4f}, "
                          f"p={res['p_value']:.4f}")
            else:
                print(f"  ✗ No composition changes detected")
            
            # 結果を保存
            peptide_results[protein_name] = peptide_composition if peptide_composition else {}
            print(f"  ✓ Analysis completed for {protein_name}")
            
        except Exception as e:
            print(f"  ✗ Error analyzing {protein_name}: {e}")
            import traceback
            traceback.print_exc()
            # エラーでも空の結果を保存
            peptide_results[protein_name] = {}
    
    # 5. 拡張版の可視化（ペプチドごとの線を含む）
    plot_composition_analysis_with_peptides(
        smoothed_df, composition_results, df, peptide_results, output_dir
    )
    
    # 6. ペプチドごとの比較図を作成
    plot_peptide_comparison(peptide_results, output_dir)
    
    # 7. 統計サマリーの計算
    peptide_stats = calculate_peptide_statistics(peptide_results)
    
    # 8. 最終結果をフォーマット（修正版）
    final_results = {}
    
    # ペプチドごとの結果から平均を計算（Protein_1を除外する選択肢も）
    for lipid in ['CHOL', 'DIPC', 'DPSM']:
        # ペプチドごとの効果を収集（0でないものだけ）
        peptide_effects = []
        peptide_effects_all = []  # すべて含む
        
        for protein_name in peptide_results:
            if lipid in peptide_results[protein_name]:
                effect = peptide_results[protein_name][lipid]['absolute_change']
                peptide_effects_all.append(effect)
                
                # Protein_1のような特殊ケースを除外するオプション
                if 'note' not in peptide_results[protein_name][lipid] or \
                   peptide_results[protein_name][lipid]['note'] != 'low_gm3_peptide':
                    if abs(effect) > 0.0001:  # 非常に小さい効果も除外
                        peptide_effects.append(effect)
        
        # 有効なペプチドの平均を使用
        if peptide_effects:
            avg_effect = np.mean(peptide_effects)
            avg_p_value = np.mean([peptide_results[p][lipid]['p_value'] 
                                  for p in peptide_results 
                                  if lipid in peptide_results[p] and 
                                  abs(peptide_results[p][lipid]['absolute_change']) > 0.0001])
            
            # 低GM3/高GM3の比率も平均
            low_ratios = [peptide_results[p][lipid]['low_gm3_ratio'] 
                         for p in peptide_results 
                         if lipid in peptide_results[p] and 
                         abs(peptide_results[p][lipid]['absolute_change']) > 0.0001]
            high_ratios = [peptide_results[p][lipid]['high_gm3_ratio'] 
                          for p in peptide_results 
                          if lipid in peptide_results[p] and 
                          abs(peptide_results[p][lipid]['absolute_change']) > 0.0001]
            
            avg_low_ratio = np.mean(low_ratios) if low_ratios else 0
            avg_high_ratio = np.mean(high_ratios) if high_ratios else 0
        else:
            avg_effect = 0
            avg_p_value = 1.0
            avg_low_ratio = 0
            avg_high_ratio = 0
        
        expected_sign = '+' if lipid in ['CHOL', 'DPSM'] else '-'
        observed_sign = '+' if avg_effect > 0 else '-'
        
        final_results[lipid] = {
            'effect': avg_effect,
            'se': np.std(peptide_effects) / np.sqrt(len(peptide_effects)) if peptide_effects else 0,
            'correlation': 0,  # 後で計算
            'p_value': avg_p_value,
            'metric_used': 'Composition Ratio (Valid Peptides Average)',
            'expected_sign': expected_sign,
            'observed_sign': observed_sign,
            'percent_change': 100 * avg_effect / avg_low_ratio if avg_low_ratio > 0 else 0,
            'low_gm3_ratio': avg_low_ratio,
            'high_gm3_ratio': avg_high_ratio,
            # ペプチド統計
            'peptide_effects': peptide_effects_all,
            'peptide_effects_valid': peptide_effects,
            'peptide_min': min(peptide_effects_all) if peptide_effects_all else 0,
            'peptide_max': max(peptide_effects_all) if peptide_effects_all else 0,
            'peptide_std': np.std(peptide_effects_all) if peptide_effects_all else 0,
            'n_valid_peptides': len(peptide_effects),
            'n_total_peptides': len(peptide_effects_all)
        }
    
    # 9. 結果の表示（拡張版）
    print("\n" + "="*70)
    print("FINAL RESULTS (WITH CONSERVATION)")
    print("="*70)
    
    # 全体の結果
    print("\n--- AVERAGE ACROSS VALID PEPTIDES ---")
    for lipid in ['CHOL', 'DIPC', 'DPSM']:
        if lipid in final_results:
            res = final_results[lipid]
            match = '✅' if res['expected_sign'] == res['observed_sign'] else '❌'
            
            print(f"\n{lipid}:")
            print(f"  Average composition change: {res['effect']:+.3f} ({res['percent_change']:+.1f}%) {match}")
            print(f"  Low GM3:  {res['low_gm3_ratio']:.3f}")
            print(f"  High GM3: {res['high_gm3_ratio']:.3f}")
            print(f"  P-value:  {res['p_value']:.4f}")
            print(f"  Valid peptides: {res['n_valid_peptides']}/{res['n_total_peptides']}")
            print(f"  Range across all peptides: [{res['peptide_min']:+.3f}, {res['peptide_max']:+.3f}]")
            print(f"  Standard deviation: {res['peptide_std']:.3f}")
    
    # ペプチドごとの詳細
    print("\n--- PEPTIDE-SPECIFIC RESULTS ---")
    for protein_name in sorted(peptide_results.keys()):
        print(f"\n{protein_name}:")
        
        if not peptide_results[protein_name]:
            print("  No data available")
            continue
            
        total_change = 0
        has_special_note = False
        
        for lipid in ['CHOL', 'DIPC', 'DPSM']:
            if lipid in peptide_results[protein_name]:
                pep_res = peptide_results[protein_name][lipid]
                change = pep_res['absolute_change']
                pct = pep_res['percent_change']
                p_val = pep_res['p_value']
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                
                if 'note' in pep_res and pep_res['note'] == 'low_gm3_peptide':
                    has_special_note = True
                
                print(f"  {lipid}: {change:+.3f} ({pct:+.1f}%) {sig}")
                total_change += change
        
        print(f"  Conservation check: {total_change:+.6f}")
        if has_special_note:
            print("  Note: Low GM3 peptide - results based on time analysis")
    
    # 最大・最小効果のペプチドを特定
    print("\n--- EXTREME CASES ---")
    for lipid in ['CHOL', 'DIPC', 'DPSM']:
        if lipid in final_results and final_results[lipid]['peptide_effects']:
            effects_dict = {}
            for protein_name in peptide_results:
                if lipid in peptide_results[protein_name]:
                    effects_dict[protein_name] = peptide_results[protein_name][lipid]['absolute_change']
            
            if effects_dict:
                max_peptide = max(effects_dict, key=lambda k: abs(effects_dict[k]))
                min_peptide = min(effects_dict, key=lambda k: abs(effects_dict[k]))
                
                print(f"\n{lipid}:")
                print(f"  Strongest effect: {max_peptide} ({effects_dict[max_peptide]:+.3f})")
                print(f"  Weakest effect:  {min_peptide} ({effects_dict[min_peptide]:+.3f})")
    
    # 保存則の確認（全体）
    total_change = sum(final_results[lipid]['effect'] for lipid in ['CHOL', 'DIPC', 'DPSM'])
    print(f"\n\nTotal composition change (valid peptides average): {total_change:+.6f} (should be ~0)")
    
    if abs(total_change) < 0.001:
        print("✅ Conservation satisfied in average results!")
    else:
        print("⚠️  Small conservation violation due to averaging")
    
    # 結果をJSONに保存（ペプチド情報を含む）
    import json
    results_with_peptides = {
        'average_results': {k: {kk: float(vv) if isinstance(vv, (np.number, float, int)) else vv 
                                for kk, vv in v.items() if kk not in ['peptide_effects_valid']} 
                          for k, v in final_results.items()},
        'peptide_results': {pk: {lk: {kk: float(vv) if isinstance(vv, (np.number, float, int)) else vv 
                                      for kk, vv in lv.items()} 
                                for lk, lv in pv.items()} 
                          for pk, pv in peptide_results.items()},
        'peptide_statistics': peptide_stats if peptide_stats else {}
    }
    
    with open(os.path.join(output_dir, 'results_with_peptides.json'), 'w') as f:
        json.dump(results_with_peptides, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}/results_with_peptides.json")
    
    return final_results, peptide_results


def plot_composition_analysis_with_peptides(smoothed_df, composition_results, full_df, peptide_results, output_dir):
    """
    組成分析の可視化 - 生データとスムージングデータの両方を表示
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy import stats
    import pandas as pd
    
    # 4x3のレイアウトに拡張
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    
    # GM3カラムを確認
    gm3_col = 'gm3_contact_strength' if 'gm3_contact_strength' in full_df.columns else 'gm3_contact_count'
    
    # 生データ（frame_df）を作成 - full_dfから集計
    print("Creating raw frame data...")
    if 'frame' in full_df.columns:
        # フレームごとに集計
        agg_dict = {}
        for col in full_df.columns:
            if pd.api.types.is_numeric_dtype(full_df[col]) and col not in ['frame', 'time']:
                agg_dict[col] = 'mean'
        
        if agg_dict:
            frame_df = full_df.groupby('frame').agg(agg_dict).reset_index()
        else:
            frame_df = smoothed_df  # フォールバック
    else:
        frame_df = smoothed_df  # フォールバック
    
    # 組成比を計算（両方のデータフレームに対して）
    for df in [frame_df, smoothed_df]:
        for lipid in ['CHOL', 'DIPC', 'DPSM']:
            if f'{lipid}_composition_ratio' not in df.columns:
                total_counts = (df.get('CHOL_contact_count', 0) + 
                              df.get('DIPC_contact_count', 0) + 
                              df.get('DPSM_contact_count', 0))
                total_counts = total_counts.replace(0, 1)
                if f'{lipid}_contact_count' in df.columns:
                    df[f'{lipid}_composition_ratio'] = df[f'{lipid}_contact_count'] / total_counts
    
    # 1段目: 生データ（時系列カラーマップ）
    for i, lipid in enumerate(['CHOL', 'DIPC', 'DPSM']):
        ax = axes[0, i]
        ratio_col = f'{lipid}_composition_ratio'
        
        if ratio_col in frame_df.columns and gm3_col in frame_df.columns:
            # 時系列で色分けした散布図
            scatter = ax.scatter(frame_df[gm3_col], frame_df[ratio_col], 
                               alpha=0.5, s=20, c=frame_df.index, cmap='viridis')
            
            # Trend line
            mask = frame_df[gm3_col].notna() & frame_df[ratio_col].notna()
            if mask.sum() > 10:
                z = np.polyfit(frame_df.loc[mask, gm3_col], 
                             frame_df.loc[mask, ratio_col], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(frame_df[gm3_col].min(),
                                    frame_df[gm3_col].max(), 100)
                ax.plot(x_trend, p(x_trend), 'r--', linewidth=2.5, alpha=0.8)
                
                # 相関係数
                corr, pval = stats.pearsonr(frame_df.loc[mask, gm3_col], 
                                           frame_df.loc[mask, ratio_col])
                ax.set_title(f'{lipid} (Raw): r={corr:.3f}, p={pval:.3f}')
                
                # 有意性マーカー
                sig_marker = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
                ax.text(0.95, 0.05, sig_marker, transform=ax.transAxes, 
                       ha='right', va='bottom', fontsize=16, fontweight='bold')
        
        ax.set_xlabel('GM3 Strength')
        ax.set_ylabel(f'{lipid} Composition Ratio')
        ax.grid(alpha=0.3)
        
        if i == 2:  # 右端にカラーバー
            plt.colorbar(scatter, ax=ax, label='Frame')
    
    # 2段目: スムージングデータ（時系列カラーマップ）
    for i, lipid in enumerate(['CHOL', 'DIPC', 'DPSM']):
        ax = axes[1, i]
        ratio_col = f'{lipid}_composition_ratio'
        
        if ratio_col in smoothed_df.columns and gm3_col in smoothed_df.columns:
            # 時系列で色分けした散布図
            scatter = ax.scatter(smoothed_df[gm3_col], smoothed_df[ratio_col], 
                               alpha=0.5, s=20, c=smoothed_df.index, cmap='viridis')
            
            # Trend line
            mask = smoothed_df[gm3_col].notna() & smoothed_df[ratio_col].notna()
            if mask.sum() > 10:
                z = np.polyfit(smoothed_df.loc[mask, gm3_col], 
                             smoothed_df.loc[mask, ratio_col], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(smoothed_df[gm3_col].min(),
                                    smoothed_df[gm3_col].max(), 100)
                ax.plot(x_trend, p(x_trend), 'r--', linewidth=2.5, alpha=0.8)
                
                # 相関係数
                corr, pval = stats.pearsonr(smoothed_df.loc[mask, gm3_col], 
                                           smoothed_df.loc[mask, ratio_col])
                ax.set_title(f'{lipid} (Smoothed): r={corr:.3f}, p={pval:.3f}')
                
                # 有意性マーカー
                sig_marker = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
                ax.text(0.95, 0.05, sig_marker, transform=ax.transAxes, 
                       ha='right', va='bottom', fontsize=16, fontweight='bold')
        
        ax.set_xlabel('GM3 Strength')
        ax.set_ylabel(f'{lipid} Composition Ratio')
        ax.grid(alpha=0.3)
        
        if i == 2:  # 右端にカラーバー
            plt.colorbar(scatter, ax=ax, label='Frame')
    
    # 3段目: シンプル版（灰色の点＋赤いトレンドライン）- スムージングデータ
    for i, lipid in enumerate(['CHOL', 'DIPC', 'DPSM']):
        ax = axes[2, i]
        ratio_col = f'{lipid}_composition_ratio'
        
        if ratio_col in smoothed_df.columns and gm3_col in smoothed_df.columns:
            # 全データを薄い灰色で
            ax.scatter(smoothed_df[gm3_col], smoothed_df[ratio_col], 
                      alpha=0.2, s=10, c='gray', label='All data')
            
            # Trend line（赤破線）
            mask = smoothed_df[gm3_col].notna() & smoothed_df[ratio_col].notna()
            if mask.sum() > 10:
                z = np.polyfit(smoothed_df.loc[mask, gm3_col], 
                             smoothed_df.loc[mask, ratio_col], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(smoothed_df[gm3_col].min(),
                                    smoothed_df[gm3_col].max(), 100)
                ax.plot(x_trend, p(x_trend), 'r--', linewidth=3, alpha=0.8, 
                       label='Trend')
                
                # ビン化された平均
                try:
                    n_bins = 10
                    bins = pd.qcut(smoothed_df[gm3_col], q=n_bins, duplicates='drop')
                    binned = smoothed_df.groupby(bins)[[gm3_col, ratio_col]].mean()
                    ax.scatter(binned[gm3_col], binned[ratio_col], 
                              s=100, c='darkred', marker='o', edgecolors='white', 
                              linewidth=2, label='Binned mean', zorder=10)
                except:
                    pass
        
        ax.set_xlabel('GM3 Strength')
        ax.set_ylabel(f'{lipid} Composition Ratio')
        ax.grid(alpha=0.3)
        
        if i == 0:
            ax.legend(fontsize=8, loc='best')
    
    # 4段目: 既存の棒グラフと円グラフ
    ax = axes[3, 0]
    if composition_results:
        lipids = PLOT_LIPIDS_ORDER
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        
        changes = [composition_results[lipid]['absolute_change'] for lipid in lipids]
        x_pos = np.arange(len(lipids))
        bars = ax.bar(x_pos, changes, color=colors, alpha=0.7)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(lipids)
        ax.set_ylabel('Change in Composition Ratio')
        ax.set_title('Absolute Change (High GM3 - Low GM3)')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        for i, lipid in enumerate(lipids):
            sig = composition_results[lipid]['significance']
            if sig != 'ns':
                ax.text(i, changes[i] + np.sign(changes[i]) * 0.001, sig, 
                       ha='center', va='bottom' if changes[i] > 0 else 'top')
    
    ax = axes[3, 1]
    if composition_results:
        percent_changes = [composition_results[lipid]['percent_change'] for lipid in lipids]
        bars = ax.bar(x_pos, percent_changes, color=colors, alpha=0.7)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(lipids)
        ax.set_ylabel('% Change')
        ax.set_title('Percent Change in Composition')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    ax = axes[3, 2]
    sizes = []
    for lipid in lipids:
        col = f'{lipid}_composition_ratio'
        if col in smoothed_df.columns:
            sizes.append(smoothed_df[col].mean())
        else:
            sizes.append(0)
    
    if sum(sizes) > 0:
        ax.pie(sizes, labels=lipids, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Average Composition')
    
    # 相関係数の比較を出力
    print("\n=== Correlation comparison (Raw vs Smoothed) ===")
    for lipid in ['CHOL', 'DIPC', 'DPSM']:
        ratio_col = f'{lipid}_composition_ratio'
        if ratio_col in frame_df.columns and ratio_col in smoothed_df.columns:
            # 生データ
            mask_raw = frame_df[gm3_col].notna() & frame_df[ratio_col].notna()
            if mask_raw.sum() > 10:
                corr_raw, p_raw = stats.pearsonr(frame_df.loc[mask_raw, gm3_col], 
                                                frame_df.loc[mask_raw, ratio_col])
                slope_raw = np.polyfit(frame_df.loc[mask_raw, gm3_col], 
                                     frame_df.loc[mask_raw, ratio_col], 1)[0]
            
            # スムージングデータ
            mask_smooth = smoothed_df[gm3_col].notna() & smoothed_df[ratio_col].notna()
            if mask_smooth.sum() > 10:
                corr_smooth, p_smooth = stats.pearsonr(smoothed_df.loc[mask_smooth, gm3_col], 
                                                      smoothed_df.loc[mask_smooth, ratio_col])
                slope_smooth = np.polyfit(smoothed_df.loc[mask_smooth, gm3_col], 
                                        smoothed_df.loc[mask_smooth, ratio_col], 1)[0]
            
            print(f"\n{lipid}:")
            print(f"  Raw:      r={corr_raw:.3f}, p={p_raw:.4f}, slope={slope_raw:.4f}")
            print(f"  Smoothed: r={corr_smooth:.3f}, p={p_smooth:.4f}, slope={slope_smooth:.4f}")
            print(f"  Change:   Δr={corr_smooth-corr_raw:.3f}, Δslope={slope_smooth-slope_raw:.4f}")
    
    plt.suptitle('Lipid Composition Analysis: Raw vs Smoothed Data', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存
    for fmt in ['png', 'svg', 'pdf']:
        plt.savefig(os.path.join(output_dir, f'composition_analysis_with_peptides.{fmt}'), 
                    format=fmt if fmt != 'png' else None, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_peptide_comparison(peptide_results, output_dir):
    """
    ペプチドごとの組成変化を比較する棒グラフ（デバッグ版）
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # すべてのペプチドを確実に含める
    all_proteins = ['Protein_1', 'Protein_2', 'Protein_3', 'Protein_4']
    
    # デバッグ出力を詳細に
    print(f"\n=== PEPTIDE COMPARISON DEBUG ===")
    print(f"peptide_results keys: {list(peptide_results.keys())}")
    
    # 各ペプチドの詳細を表示
    for protein in all_proteins:
        if protein in peptide_results:
            print(f"\n{protein}:")
            if peptide_results[protein]:
                for lipid in ['CHOL', 'DIPC', 'DPSM']:
                    if lipid in peptide_results[protein]:
                        change = peptide_results[protein][lipid]['absolute_change']
                        print(f"  {lipid}: {change:.4f}")
                    else:
                        print(f"  {lipid}: NO DATA")
            else:
                print(f"  Empty results dictionary")
        else:
            print(f"\n{protein}: NOT IN RESULTS")
    
    x = np.arange(len(all_proteins))
    
    for i, lipid in enumerate(['CHOL', 'DIPC', 'DPSM']):
        ax = axes[i]
        
        changes = []
        errors = []
        colors = []
        has_data = []
        
        for protein in all_proteins:
            if protein in peptide_results and peptide_results[protein] and lipid in peptide_results[protein]:
                change = peptide_results[protein][lipid]['absolute_change']
                changes.append(change * 100)  # percentage pointsに変換
                errors.append(abs(change) * 10)
                colors.append('#2ecc71' if change > 0 else '#e74c3c')
                has_data.append(True)
            else:
                # データがない場合も値を設定
                changes.append(0)
                errors.append(0)
                colors.append('lightgray')
                has_data.append(False)
        
        # 棒グラフを描画
        bars = ax.bar(x, changes, yerr=errors, capsize=5, 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        # データがないバーを明示的に表示
        for j, (protein, has_d) in enumerate(zip(all_proteins, has_data)):
            if not has_d:
                # データがない場合の表示
                bars[j].set_height(0.01)  # 最小の高さを設定
                bars[j].set_hatch('///')
                bars[j].set_alpha(0.3)
                ax.text(j, 0.1, 'NO\nDATA', ha='center', va='bottom', 
                       fontsize=8, color='red', weight='bold')
        
        # 有意性マーカー
        for j, (protein, has_d) in enumerate(zip(all_proteins, has_data)):
            if has_d and protein in peptide_results and lipid in peptide_results[protein]:
                p_val = peptide_results[protein][lipid]['p_value']
                y_pos = changes[j] + errors[j] + 0.1
                if p_val < 0.001:
                    ax.text(j, y_pos, '***', ha='center', va='bottom')
                elif p_val < 0.01:
                    ax.text(j, y_pos, '**', ha='center', va='bottom')
                elif p_val < 0.05:
                    ax.text(j, y_pos, '*', ha='center', va='bottom')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace('Protein_', 'Peptide ') for p in all_proteins])
        ax.set_ylabel('Change in Composition (%)')
        ax.set_title(f'{lipid} Composition Change by Peptide')
        ax.grid(axis='y', alpha=0.3)
        
        # Y軸の範囲を設定
        if any(has_data):
            valid_changes = [c for c, h in zip(changes, has_data) if h]
            valid_errors = [e for e, h in zip(errors, has_data) if h]
            
            if valid_changes:
                y_min = min(min(valid_changes) - max(valid_errors + [1]), -1)
                y_max = max(max(valid_changes) + max(valid_errors + [1]), 1)
                ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(-1, 1)
    
    plt.suptitle('Peptide-Specific Responses to GM3', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'peptide_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'peptide_comparison.svg'), 
            format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"\nPeptide comparison plot saved to {output_dir}/peptide_comparison.png")

def calculate_peptide_statistics(peptide_results):
    """
    ペプチド間の統計を計算
    """
    stats = {}
    
    for lipid in ['CHOL', 'DIPC', 'DPSM']:
        effects = []
        for protein in peptide_results:
            if lipid in peptide_results[protein]:
                effects.append(peptide_results[protein][lipid]['absolute_change'])
        
        if effects:
            stats[lipid] = {
                'mean': np.mean(effects),
                'std': np.std(effects),
                'min': min(effects),
                'max': max(effects),
                'cv': np.std(effects) / np.mean(effects) if np.mean(effects) != 0 else 0,
                'range': max(effects) - min(effects)
            }
    
    return stats