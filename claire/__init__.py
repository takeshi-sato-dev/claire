#!/usr/bin/env python3
"""
CLAIRE - Conserved Lipid Analysis with Interaction and Redistribution Evaluation

A Python package for analyzing lipid redistribution in membranes with conservation laws.
"""

from .core.membrane import MembraneSystem
from .analysis.conservation import ConservationAnalyzer
from .analysis.enrichment import EnrichmentAnalyzer
from .visualization.plots import Visualizer

__all__ = [
    'MembraneSystem',
    'ConservationAnalyzer',
    'EnrichmentAnalyzer',
    'Visualizer'
]

__version__ = "0.1.0"