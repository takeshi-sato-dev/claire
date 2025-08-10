#!/usr/bin/env python3
"""
Generate figures - GENERALIZED for any lipid types
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Fix for Mac Preview
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional


class FigureGenerator:
    """
    Create publication figures for any lipid system
    """
    
    @staticmethod
    def create_nature_quality_figures(results: Dict, 
                                     frame_df: pd.DataFrame, 
                                     output_dir: str,
                                     target_lipids: Optional[List[str]] = None,
                                     mediator_lipid: str = 'DPG3') -> Tuple[List, List]:
        """
        Creates figure_observed_results.png and figure_temporal_analysis.png
        Works with ANY lipid types
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Set publication style
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial']
        })
        
        # Determine lipids from results or parameters
        if target_lipids is None:
            target_lipids = list(results.keys())
        
        lipids = target_lipids[:6]  # Max 6 for display
        
        # Figure 1: Main results
        n_cols = min(3, len(lipids))
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Panel A: Effect sizes
        ax1 = fig.add_subplot(gs[0, :2])
        
        ml_effects = []
        correlations = []
        
        for lipid in lipids:
            if lipid in results:
                # Look for effect with mediator name or default to gm3
                effect_key = f'{mediator_lipid.lower()}_effect'
                if effect_key not in results[lipid]:
                    effect_key = 'gm3_effect'
                ml_effects.append(results[lipid].get(effect_key, 0))
                correlations.append(results[lipid].get('correlation', 0))
            else:
                ml_effects.append(0)
                correlations.append(0)
        
        x = np.arange(len(lipids))
        width = 0.6
        
        ml_colors = ['#2ecc71' if e > 0 else '#e74c3c' for e in ml_effects]
        bars1 = ax1.bar(x, ml_effects, width, label='ML Analysis', 
                       alpha=0.8, color=ml_colors, edgecolor='black', linewidth=1.5)
        
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_ylabel(f'{mediator_lipid} Effect (Observed)', fontweight='bold')
        ax1.set_title(f'{mediator_lipid} Effects on Lipid Distribution', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(lipids, rotation=45 if len(lipids) > 3 else 0)
        ax1.legend()
        
        for i, (lipid, corr) in enumerate(zip(lipids, correlations)):
            if lipid in results:
                p_val = results[lipid].get('p_value', 1)
                sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                ax1.text(i, ax1.get_ylim()[1] * 0.9, f'r={corr:.3f}\n{sig_marker}', 
                        ha='center', va='top', fontsize=9)
        
        # Panel B: Model performance
        ax2 = fig.add_subplot(gs[0, 2])
        
        scores = []
        for lipid in lipids:
            if lipid in results:
                scores.append(results[lipid].get('score', 0))
            else:
                scores.append(0)
        
        ax2.bar(range(len(lipids)), scores, alpha=0.8, color='#3498db')
        ax2.set_xticks(range(len(lipids)))
        ax2.set_xticklabels(lipids, rotation=45)
        ax2.set_ylabel('R² Score')
        ax2.set_title('Model Performance', fontweight='bold')
        ax2.set_ylim(-0.1, 1)
        ax2.grid(axis='y', alpha=0.3)
        
        # Panels C-E: Time series (up to 3 lipids)
        for i in range(min(3, len(lipids))):
            ax = fig.add_subplot(gs[1, i])
            lipid = lipids[i]
            
            # Find mediator column
            mediator_col = None
            for col_name in [f'{mediator_lipid.lower()}_contact_strength', 
                           f'{mediator_lipid.lower()}_contact_count',
                           'gm3_contact_strength', 'gm3_contact_count']:
                if col_name in frame_df.columns:
                    mediator_col = col_name
                    break
            
            lipid_col = f'{lipid}_contact_count'
            
            if mediator_col and lipid_col in frame_df.columns:
                scatter = ax.scatter(frame_df[mediator_col], frame_df[lipid_col], 
                                   alpha=0.3, s=10, c=frame_df.index, cmap='viridis')
                
                mask = frame_df[mediator_col].notna() & frame_df[lipid_col].notna()
                if mask.sum() > 2:
                    z = np.polyfit(frame_df.loc[mask, mediator_col], 
                                 frame_df.loc[mask, lipid_col], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(frame_df[mediator_col].min(), 
                                        frame_df[mediator_col].max(), 100)
                    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
                
                ax.set_xlabel(f'{mediator_lipid} Strength')
                ax.set_ylabel(f'{lipid} Contacts')
                ax.set_title(f'{lipid} vs {mediator_lipid}', fontweight='bold')
                
                if i < len(correlations):
                    ax.text(0.05, 0.95, f'r = {correlations[i]:.3f}', 
                           transform=ax.transAxes, ha='left', va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Rest of panels...
        plt.suptitle(f'{mediator_lipid}-Mediated Lipid Reorganization: ML Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        for fmt in ['pdf', 'png', 'svg']:
            filepath = os.path.join(output_dir, f'figure_observed_results.{fmt}')
            plt.savefig(filepath, dpi=300, bbox_inches='tight', format=fmt)
        plt.close()
        
        # Figure 2: Temporal analysis
        n_lipids = min(len(lipids), 5)
        fig2, axes = plt.subplots(n_lipids, 1, figsize=(12, 4*n_lipids), sharex=True)
        
        if n_lipids == 1:
            axes = [axes]
        
        for i in range(n_lipids):
            ax = axes[i]
            lipid = lipids[i]
            lipid_col = f'{lipid}_contact_count'
            
            if mediator_col and lipid_col in frame_df.columns and 'frame' in frame_df.columns:
                mediator_norm = (frame_df[mediator_col] - frame_df[mediator_col].mean()) / (frame_df[mediator_col].std() + 1e-10)
                lipid_norm = (frame_df[lipid_col] - frame_df[lipid_col].mean()) / (frame_df[lipid_col].std() + 1e-10)
                
                ax.plot(frame_df['frame'], mediator_norm, 'b-', alpha=0.5, label=mediator_lipid)
                ax.plot(frame_df['frame'], lipid_norm, 'r-', alpha=0.5, label=lipid)
                
                high_mediator = frame_df[mediator_col] > frame_df[mediator_col].quantile(0.75)
                ax.fill_between(frame_df['frame'], -3, 3, where=high_mediator, 
                              alpha=0.2, color='yellow', label=f'High {mediator_lipid}')
                
                ax.set_ylabel('Normalized Value')
                ax.set_title(f'{lipid} Response to {mediator_lipid} Over Time', fontweight='bold')
                ax.legend(loc='upper right')
                ax.grid(alpha=0.3)
                ax.set_ylim(-3, 3)
        
        axes[-1].set_xlabel('Frame')
        plt.suptitle(f'Temporal Analysis of {mediator_lipid}-Lipid Interactions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        for fmt in ['pdf', 'png', 'svg']:
            filepath = os.path.join(output_dir, f'figure_temporal_analysis.{fmt}')
            plt.savefig(filepath, dpi=300, bbox_inches='tight', format=fmt)
        plt.close()
        
        return ml_effects, correlations
    
    @staticmethod
    def calculate_composition_ratios(frame_df, target_lipids: List[str]):
        """
        Calculate composition ratios for ANY set of lipids
        """
        print("\nCalculating lipid composition ratios...")
        
        # Build column mapping for available lipids
        lipid_cols = {}
        for lipid in target_lipids:
            col = f'{lipid}_contact_count'
            if col in frame_df.columns:
                lipid_cols[lipid] = col
        
        if not lipid_cols:
            print("  No lipid contact count columns found!")
            return frame_df
        
        # Calculate total
        total_lipids = sum(frame_df.get(col, 0) for col in lipid_cols.values())
        total_lipids = total_lipids.replace(0, 1)
        
        # Calculate ratios
        for lipid, col in lipid_cols.items():
            ratio_col = f'{lipid}_composition_ratio'
            frame_df[ratio_col] = frame_df[col] / total_lipids
            print(f"  {lipid} mean ratio: {frame_df[ratio_col].mean():.3f}")
        
        # Verify conservation
        total_ratio = sum(frame_df[f'{lipid}_composition_ratio'] 
                         for lipid in lipid_cols.keys() 
                         if f'{lipid}_composition_ratio' in frame_df.columns)
        print(f"  Total ratio check: {total_ratio.mean():.6f} (should be 1.0)")
        
        return frame_df
    
    @staticmethod
    def analyze_composition_changes(frame_df, target_lipids: List[str], mediator_lipid: str = 'DPG3'):
        """
        Analyze composition changes for ANY lipids
        """
        print("\n" + "="*70)
        print("COMPOSITION ANALYSIS WITH CONSERVATION")
        print("="*70)
        
        # Find mediator column
        mediator_col = None
        for col_name in [f'{mediator_lipid.lower()}_contact_strength',
                        f'{mediator_lipid.lower()}_contact_count',
                        'gm3_contact_strength', 'gm3_contact_count']:
            if col_name in frame_df.columns:
                mediator_col = col_name
                break
        
        if not mediator_col:
            print(f"ERROR: No mediator column found for {mediator_lipid}!")
            return {}
        
        print(f"Using mediator column: {mediator_col}")
        
        # Check mediator distribution
        if frame_df[mediator_col].max() == 0:
            print(f"ERROR: All {mediator_lipid} values are zero!")
            return {}
        
        # Split high/low mediator
        q75 = frame_df[mediator_col].quantile(0.75)
        q25 = frame_df[mediator_col].quantile(0.25)
        
        high_mediator = frame_df[frame_df[mediator_col] > q75]
        low_mediator = frame_df[frame_df[mediator_col] < q25]
        
        if len(high_mediator) < 5 or len(low_mediator) < 5:
            median = frame_df[mediator_col].median()
            high_mediator = frame_df[frame_df[mediator_col] > median]
            low_mediator = frame_df[frame_df[mediator_col] <= median]
        
        print(f"Comparison: high {mediator_lipid} (n={len(high_mediator)}) vs low (n={len(low_mediator)})")
        
        composition_results = {}
        
        for lipid in target_lipids:
            col = f'{lipid}_composition_ratio'
            
            if col in frame_df.columns and len(high_mediator) > 0 and len(low_mediator) > 0:
                low_ratio = low_mediator[col].mean()
                high_ratio = high_mediator[col].mean()
                change = high_ratio - low_ratio
                percent_change = 100 * change / low_ratio if low_ratio > 0 else 0
                
                if len(low_mediator) >= 2 and len(high_mediator) >= 2:
                    try:
                        t_stat, p_value = stats.ttest_ind(
                            high_mediator[col].dropna(), 
                            low_mediator[col].dropna(),
                            equal_var=False
                        )
                    except:
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
                    'n_high': len(high_mediator),
                    'n_low': len(low_mediator)
                }
                
                print(f"{lipid}: Change = {change:+.3f} ({percent_change:+.1f}%) {sig}")
        
        return composition_results
    
    @staticmethod
    def plot_composition_analysis_with_peptides(smoothed_df, composition_results, full_df, 
                                               peptide_results, output_dir,
                                               target_lipids: List[str], 
                                               mediator_lipid: str = 'DPG3'):
        """
        Creates composition_analysis_with_peptides.png for ANY lipids
        """
        n_lipids = min(len(target_lipids), 4)  # Max 4 lipids for display
        lipids = target_lipids[:n_lipids]
        
        # Dynamic layout based on number of lipids
        fig, axes = plt.subplots(4, n_lipids, figsize=(6*n_lipids, 20))
        
        # Ensure axes is 2D
        if n_lipids == 1:
            axes = axes.reshape(-1, 1)
        
        # Find mediator column
        mediator_col = None
        for col_name in [f'{mediator_lipid.lower()}_contact_strength',
                        f'{mediator_lipid.lower()}_contact_count',
                        'gm3_contact_strength', 'gm3_contact_count']:
            if col_name in full_df.columns:
                mediator_col = col_name
                break
        
        if not mediator_col:
            print(f"WARNING: No mediator column found for plots")
            return
        
        # Create raw frame data
        if 'frame' in full_df.columns:
            agg_dict = {}
            for col in full_df.columns:
                if pd.api.types.is_numeric_dtype(full_df[col]) and col not in ['frame', 'time']:
                    agg_dict[col] = 'mean'
            
            if agg_dict:
                frame_df = full_df.groupby('frame').agg(agg_dict).reset_index()
            else:
                frame_df = smoothed_df
        else:
            frame_df = smoothed_df
        
        # Calculate composition ratios
        for df in [frame_df, smoothed_df]:
            # Get all contact count columns
            count_cols = [c for c in df.columns if c.endswith('_contact_count')]
            if count_cols:
                total_counts = sum(df[col] for col in count_cols)
                total_counts = total_counts.replace(0, 1)
                
                for lipid in lipids:
                    count_col = f'{lipid}_contact_count'
                    if count_col in df.columns:
                        df[f'{lipid}_composition_ratio'] = df[count_col] / total_counts
        
        # Generate colors for lipids
        colors = plt.cm.Set1(np.linspace(0, 1, n_lipids))
        
        # Row 1: Raw data
        for i, lipid in enumerate(lipids):
            ax = axes[0, i]
            ratio_col = f'{lipid}_composition_ratio'
            
            if ratio_col in frame_df.columns and mediator_col in frame_df.columns:
                scatter = ax.scatter(frame_df[mediator_col], frame_df[ratio_col], 
                                   alpha=0.5, s=20, c=frame_df.index, cmap='viridis')
                
                mask = frame_df[mediator_col].notna() & frame_df[ratio_col].notna()
                if mask.sum() > 10:
                    z = np.polyfit(frame_df.loc[mask, mediator_col], 
                                 frame_df.loc[mask, ratio_col], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(frame_df[mediator_col].min(),
                                        frame_df[mediator_col].max(), 100)
                    ax.plot(x_trend, p(x_trend), 'r--', linewidth=2.5, alpha=0.8)
                    
                    corr, pval = stats.pearsonr(frame_df.loc[mask, mediator_col], 
                                               frame_df.loc[mask, ratio_col])
                    ax.set_title(f'{lipid} (Raw): r={corr:.3f}, p={pval:.3f}')
                    
                    sig_marker = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
                    ax.text(0.95, 0.05, sig_marker, transform=ax.transAxes, 
                           ha='right', va='bottom', fontsize=16, fontweight='bold')
            
            ax.set_xlabel(f'{mediator_lipid} Strength')
            ax.set_ylabel(f'{lipid} Composition Ratio')
            ax.grid(alpha=0.3)
        
        # Row 2: Smoothed data
        for i, lipid in enumerate(lipids):
            ax = axes[1, i]
            ratio_col = f'{lipid}_composition_ratio'
            
            if ratio_col in smoothed_df.columns and mediator_col in smoothed_df.columns:
                scatter = ax.scatter(smoothed_df[mediator_col], smoothed_df[ratio_col], 
                                   alpha=0.5, s=20, c=smoothed_df.index, cmap='viridis')
                
                mask = smoothed_df[mediator_col].notna() & smoothed_df[ratio_col].notna()
                if mask.sum() > 10:
                    z = np.polyfit(smoothed_df.loc[mask, mediator_col], 
                                 smoothed_df.loc[mask, ratio_col], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(smoothed_df[mediator_col].min(),
                                        smoothed_df[mediator_col].max(), 100)
                    ax.plot(x_trend, p(x_trend), 'r--', linewidth=2.5, alpha=0.8)
                    
                    corr, pval = stats.pearsonr(smoothed_df.loc[mask, mediator_col], 
                                               smoothed_df.loc[mask, ratio_col])
                    ax.set_title(f'{lipid} (Smoothed): r={corr:.3f}, p={pval:.3f}')
            
            ax.set_xlabel(f'{mediator_lipid} Strength')
            ax.set_ylabel(f'{lipid} Composition Ratio')
            ax.grid(alpha=0.3)
        
        # Row 3: Simplified view
        for i, lipid in enumerate(lipids):
            ax = axes[2, i]
            ratio_col = f'{lipid}_composition_ratio'
            
            if ratio_col in smoothed_df.columns and mediator_col in smoothed_df.columns:
                ax.scatter(smoothed_df[mediator_col], smoothed_df[ratio_col], 
                          alpha=0.2, s=10, c='gray', label='All data')
                
                mask = smoothed_df[mediator_col].notna() & smoothed_df[ratio_col].notna()
                if mask.sum() > 10:
                    z = np.polyfit(smoothed_df.loc[mask, mediator_col], 
                                 smoothed_df.loc[mask, ratio_col], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(smoothed_df[mediator_col].min(),
                                        smoothed_df[mediator_col].max(), 100)
                    ax.plot(x_trend, p(x_trend), 'r--', linewidth=3, alpha=0.8, label='Trend')
            
            ax.set_xlabel(f'{mediator_lipid} Strength')
            ax.set_ylabel(f'{lipid} Composition Ratio')
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)
        
        # Row 4: Summary statistics
        # Bar chart for absolute changes
        if n_lipids >= 2:
            ax = axes[3, 0]
            if composition_results:
                changes = [composition_results.get(lipid, {}).get('absolute_change', 0) for lipid in lipids]
                x_pos = np.arange(len(lipids))
                bar_colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in changes]
                bars = ax.bar(x_pos, changes, color=bar_colors, alpha=0.7)
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(lipids)
                ax.set_ylabel('Change in Composition Ratio')
                ax.set_title(f'Absolute Change (High vs Low {mediator_lipid})')
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax.grid(axis='y', alpha=0.3)
        
        # Percent changes
        if n_lipids >= 3:
            ax = axes[3, 1]
            if composition_results:
                percent_changes = [composition_results.get(lipid, {}).get('percent_change', 0) for lipid in lipids]
                bars = ax.bar(x_pos, percent_changes, color=bar_colors, alpha=0.7)
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(lipids)
                ax.set_ylabel('% Change')
                ax.set_title('Percent Change in Composition')
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax.grid(axis='y', alpha=0.3)
        
        # Pie chart for average composition
        if n_lipids >= 3:
            ax = axes[3, min(2, n_lipids-1)]
            sizes = []
            for lipid in lipids:
                col = f'{lipid}_composition_ratio'
                if col in smoothed_df.columns:
                    sizes.append(smoothed_df[col].mean())
                else:
                    sizes.append(0)
            
            if sum(sizes) > 0:
                ax.pie(sizes, labels=lipids, colors=colors[:len(lipids)], 
                      autopct='%1.1f%%', startangle=90)
                ax.set_title('Average Composition')
        
        plt.suptitle(f'Lipid Composition Analysis: {mediator_lipid} Effects', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        for fmt in ['png', 'svg', 'pdf']:
            plt.savefig(os.path.join(output_dir, f'composition_analysis_with_peptides.{fmt}'), 
                       format=fmt if fmt != 'png' else None, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_peptide_comparison(peptide_results, output_dir, target_lipids: List[str]):
        """
        Creates peptide_comparison.png for ANY lipids and ANY proteins
        """
        if not peptide_results:
            print("No peptide results to plot")
            return
        
        # Get all proteins
        all_proteins = sorted(peptide_results.keys())
        n_proteins = len(all_proteins)
        n_lipids = min(len(target_lipids), 6)  # Max 6 lipids
        lipids = target_lipids[:n_lipids]
        
        # Dynamic figure size
        fig_width = max(15, n_lipids * 5)
        fig, axes = plt.subplots(1, n_lipids, figsize=(fig_width, 5))
        
        if n_lipids == 1:
            axes = [axes]
        
        x = np.arange(n_proteins)
        
        for i, lipid in enumerate(lipids):
            ax = axes[i]
            
            changes = []
            errors = []
            colors = []
            has_data = []
            
            for protein in all_proteins:
                if protein in peptide_results and peptide_results[protein] and lipid in peptide_results[protein]:
                    change = peptide_results[protein][lipid]['absolute_change']
                    changes.append(change * 100)  # Convert to percentage
                    errors.append(abs(change) * 10)
                    colors.append('#2ecc71' if change > 0 else '#e74c3c')
                    has_data.append(True)
                else:
                    changes.append(0)
                    errors.append(0)
                    colors.append('lightgray')
                    has_data.append(False)
            
            bars = ax.bar(x, changes, yerr=errors, capsize=5, 
                          color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Mark missing data
            for j, (protein, has_d) in enumerate(zip(all_proteins, has_data)):
                if not has_d:
                    bars[j].set_height(0.01)
                    bars[j].set_hatch('///')
                    bars[j].set_alpha(0.3)
            
            # Add significance markers
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
            
            # Format protein labels
            labels = [p.replace('Protein_', 'P').replace('_', ' ') for p in all_proteins]
            ax.set_xticklabels(labels, rotation=45 if n_proteins > 4 else 0)
            ax.set_ylabel('Change in Composition (%)')
            ax.set_title(f'{lipid} Composition Change')
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Protein-Specific Lipid Responses', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        for fmt in ['png', 'svg']:
            plt.savefig(os.path.join(output_dir, f'peptide_comparison.{fmt}'), 
                       format=fmt if fmt != 'png' else None, dpi=300, bbox_inches='tight')
        plt.close()