"""
CLAIRE - Composition-based Lipid Analysis with Integrated Resolution and Enrichment
"""

__version__ = "1.0.0"
__author__ = "Takeshi Sato"
__email__ = "takeshi@mb.kyoto-phu.ac.jp"

from .core.trajectory_loader import load_universe, identify_lipid_leaflets, select_proteins, select_lipids
from .analysis.composition import CompositionAnalyzer
from .analysis.temporal import TemporalAnalyzer
from .analysis.spatial import SpatialAnalyzer
from .analysis.ml_predict import CompositionPredictor

__all__ = [
    'load_universe',
    'identify_lipid_leaflets',
    'select_proteins',
    'select_lipids',
    'CompositionAnalyzer',
    'TemporalAnalyzer',
    'SpatialAnalyzer',
    'CompositionPredictor',
]
