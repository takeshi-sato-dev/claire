#!/usr/bin/env python3
"""
Data quality diagnostics from improved_metrics.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple


class DiagnosticAnalyzer:
    """
    Diagnose data quality and provide recommendations
    """
    
    @staticmethod
    def diagnose_trajectory_data(df: pd.DataFrame, frame_df: pd.DataFrame) -> Tuple[Dict, List]:
        """
        Complete diagnostic analysis from original code
        """
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
        
        # 3. Lipid distribution check
        print("\n3. Lipid Distribution Check:")
        
        for lipid in ['CHOL', 'DIPC', 'DPSM']:
            count_col = f'{lipid}_contact_count'
            if count_col in frame_df.columns:
                mean_count = frame_df[count_col].mean()
                std_count = frame_df[count_col].std()
                print(f"   {lipid}: mean={mean_count:.1f}, std={std_count:.1f}")
                
                if mean_count < 1:
                    recommendations.append(f"⚠️  Very few {lipid} contacts. Check selection.")
        
        # 4. Correlation pattern check
        print("\n4. Correlation Pattern Check:")
        
        if gm3_col in frame_df.columns:
            for lipid in ['CHOL', 'DIPC', 'DPSM']:
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
        
        # 6. Recommendations summary
        print("\n" + "="*60)
        print("RECOMMENDATIONS:")
        print("="*60)
        
        if not recommendations:
            print("✅ Data quality appears good!")
        else:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        
        return diagnostics, recommendations
    
    @staticmethod
    def select_best_metric(df: pd.DataFrame, lipid_type: str, gm3_col: str = 'gm3_contact_strength') -> Tuple[str, float]:
        """
        Select best metric for each lipid
        FROM ORIGINAL improved_metrics.py
        """
        candidate_metrics = [
            f'{lipid_type}_enrichment',
            f'{lipid_type}_local_fraction',
            f'{lipid_type}_gm3_clustering',
            f'{lipid_type}_gm3_colocalization',
            f'{lipid_type}_contact_count'
        ]
        
        best_metric = None
        best_correlation = 0
        
        for metric in candidate_metrics:
            if metric in df.columns and gm3_col in df.columns:
                mask = np.isfinite(df[metric]) & np.isfinite(df[gm3_col])
                if mask.sum() > 30:
                    corr, pval = stats.pearsonr(df.loc[mask, gm3_col], df.loc[mask, metric])
                    
                    # CHOL and DPSM expect positive correlation, DIPC expects negative
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