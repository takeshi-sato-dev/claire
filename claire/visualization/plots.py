#!/usr/bin/env python3
"""
Visualization utilities for CLAIRE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os


class Visualizer:
    """
    Create publication-quality figures for lipid redistribution analysis
    """
    
    def __init__(self, style: str = 'publication'):
        """
        Initialize visualizer
        
        Parameters
        ----------
        style : str
            Plot style ('publication', 'presentation', 'notebook')
        """
        self.set_style(style)
        
    def set_style(self, style: str):
        """
        Set plotting style
        
        Parameters
        ----------
        style : str
            Style name
        """
        if style == 'publication':
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
        elif style == 'presentation':
            plt.rcParams.update({
                'font.size': 14,
                'axes.labelsize': 16,
                'axes.titlesize': 18,
                'xtick.labelsize': 14,
                'ytick.labelsize': 14,
                'legend.fontsize': 14,
                'figure.titlesize': 20
            })
        else:  # notebook
            plt.rcParams.update(plt.rcParamsDefault)
        
        sns.set_style("whitegrid")
    
    def plot_conservation_analysis(self,
                                  composition_results: Dict,
                                  data: pd.DataFrame,
                                  output_dir: str,
                                  formats: List[str] = ['png', 'svg', 'pdf']):
        """
        Create conservation analysis figure
        
        Parameters
        ----------
        composition_results : dict
            Results from conservation analysis
        data : pandas.DataFrame
            Raw data
        output_dir : str
            Output directory
        formats : list
            Output formats
        """
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        lipids = ['CHOL', 'DIPC', 'DPSM']
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        
        # Row 1: Composition changes
        ax = axes[0, 0]
        changes = [composition_results[lipid]['absolute_change'] for lipid in lipids]
        x_pos = np.arange(len(lipids))
        bars = ax.bar(x_pos, changes, color=colors, alpha=0.7)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(lipids)
        ax.set_ylabel('Change in Composition Ratio')
        ax.set_title('Absolute Change (High GM3 - Low GM3)')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Add significance markers
        for i, lipid in enumerate(lipids):
            sig = composition_results[lipid]['significance']
            if sig != 'ns':
                y_pos = changes[i] + np.sign(changes[i]) * 0.001
                ax.text(i, y_pos, sig, ha='center', 
                       va='bottom' if changes[i] > 0 else 'top')
        
        # Row 1: Percent changes
        ax = axes[0, 1]
        percent_changes = [composition_results[lipid]['percent_change'] for lipid in lipids]
        bars = ax.bar(x_pos, percent_changes, color=colors, alpha=0.7)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(lipids)
        ax.set_ylabel('% Change')
        ax.set_title('Percent Change in Composition')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Row 1: Pie chart of average composition
        ax = axes[0, 2]
        sizes = []
        for lipid in lipids:
            col = f'{lipid}_ratio'
            if col in data.columns:
                sizes.append(data[col].mean())
            else:
                sizes.append(0)
        
        if sum(sizes) > 0:
            ax.pie(sizes, labels=lipids, colors=colors, 
                  autopct='%1.1f%%', startangle=90)
            ax.set_title('Average Composition')
        
        # Row 2: Time series for each lipid
        for i, lipid in enumerate(lipids):
            ax = axes[1, i]
            
            if 'frame' in data.columns:
                ratio_col = f'{lipid}_ratio'
                if ratio_col in data.columns and 'gm3_strength' in data.columns:
                    # Normalize for visualization
                    gm3_norm = (data['gm3_strength'] - data['gm3_strength'].mean()) / \
                              data['gm3_strength'].std()
                    lipid_norm = (data[ratio_col] - data[ratio_col].mean()) / \
                                data[ratio_col].std()
                    
                    ax.plot(data['frame'], gm3_norm, 'b-', alpha=0.5, label='GM3')
                    ax.plot(data['frame'], lipid_norm, 'r-', alpha=0.5, label=lipid)
                    
                    ax.set_xlabel('Frame')
                    ax.set_ylabel('Normalized Value')
                    ax.set_title(f'{lipid} vs GM3 Over Time')
                    ax.legend()
                    ax.grid(alpha=0.3)
        
        plt.suptitle('Lipid Composition Analysis with Conservation', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save in multiple formats
        for fmt in formats:
            filename = os.path.join(output_dir, f'conservation_analysis.{fmt}')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_enrichment_profiles(self,
                                enrichment_data: Dict,
                                output_dir: str):
        """
        Plot enrichment profiles around proteins
        
        Parameters
        ----------
        enrichment_data : dict
            Enrichment analysis results
        output_dir : str
            Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        n_lipids = len(enrichment_data)
        fig, axes = plt.subplots(1, n_lipids, figsize=(5*n_lipids, 5))
        
        if n_lipids == 1:
            axes = [axes]
        
        for i, (lipid, data) in enumerate(enrichment_data.items()):
            ax = axes[i]
            
            if 'radial_distribution' in data:
                r, rdf = data['radial_distribution']
                ax.plot(r, rdf, 'b-', linewidth=2)
                ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel('Distance (Å)')
                ax.set_ylabel('g(r)')
                ax.set_title(f'{lipid} RDF')
                ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'enrichment_profiles.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_matrix(self,
                               corr_matrix: pd.DataFrame,
                               output_dir: str,
                               filename: str = 'correlation_matrix'):
        """
        Plot correlation matrix heatmap
        
        Parameters
        ----------
        corr_matrix : pandas.DataFrame
            Correlation matrix
        output_dir : str
            Output directory
        filename : str
            Output filename (without extension)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5)
        
        plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        for fmt in ['png', 'svg']:
            plt.savefig(os.path.join(output_dir, f'{filename}.{fmt}'),
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_temporal_analysis(self,
                              temporal_data: pd.DataFrame,
                              variables: List[str],
                              output_dir: str):
        """
        Plot temporal analysis results
        
        Parameters
        ----------
        temporal_data : pandas.DataFrame
            Temporal analysis data
        variables : list
            Variables to plot
        output_dir : str
            Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        n_vars = len(variables)
        fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4*n_vars), sharex=True)
        
        if n_vars == 1:
            axes = [axes]
        
        for i, var in enumerate(variables):
            ax = axes[i]
            
            if var in temporal_data.columns:
                # Plot raw data
                ax.plot(temporal_data.index, temporal_data[var], 
                       'gray', alpha=0.3, label='Raw')
                
                # Plot smoothed if available
                smooth_col = f'{var}_smooth'
                if smooth_col in temporal_data.columns:
                    ax.plot(temporal_data.index, temporal_data[smooth_col],
                           'b-', linewidth=2, label='Smoothed')
                
                ax.set_ylabel(var)
                ax.legend()
                ax.grid(alpha=0.3)
        
        axes[-1].set_xlabel('Frame')
        plt.suptitle('Temporal Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'temporal_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()