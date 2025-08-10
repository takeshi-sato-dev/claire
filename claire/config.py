#!/usr/bin/env python3
"""
Configuration management for CLAIRE
"""

import os
import yaml
from typing import Dict, Any, Optional


class Config:
    """
    Configuration manager for CLAIRE
    """
    
    # Default configuration
    DEFAULTS = {
        'analysis': {
            'cutoff_radii': {
                'contact': 6.0,
                'first_shell': 10.0,
                'influence': 15.0
            },
            'smoothing_window': 20,
            'bootstrap_iterations': 1000,
            'confidence_level': 0.95
        },
        'trajectory': {
            'chunk_size': 100,
            'parallel': True
        },
        'visualization': {
            'dpi': 300,
            'formats': ['png', 'svg'],
            'style': 'publication'
        },
        'lipids': {
            'headgroup_atoms': {
                'martini': ['GL1', 'GL2', 'AM1', 'AM2', 'ROH', 'GM1', 'GM2'],
                'charmm': ['P', 'P8', 'P1', 'N', 'C2'],
                'amber': ['P', 'P8', 'P1', 'N4', 'C1']
            }
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration
        
        Parameters
        ----------
        config_file : str, optional
            Path to configuration file
        """
        self.config = self.DEFAULTS.copy()
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """
        Load configuration from file
        
        Parameters
        ----------
        config_file : str
            Path to YAML configuration file
        """
        with open(config_file, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Update defaults with user configuration
        self._update_nested(self.config, user_config)
    
    def _update_nested(self, base: Dict, update: Dict):
        """
        Recursively update nested dictionary
        
        Parameters
        ----------
        base : dict
            Base dictionary
        update : dict
            Update dictionary
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                self._update_nested(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Parameters
        ----------
        key : str
            Configuration key (dot-separated for nested)
        default : any
            Default value if key not found
        
        Returns
        -------
        any
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value
        
        Parameters
        ----------
        key : str
            Configuration key (dot-separated for nested)
        value : any
            Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, filename: str):
        """
        Save configuration to file
        
        Parameters
        ----------
        filename : str
            Output filename
        """
        with open(filename, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)