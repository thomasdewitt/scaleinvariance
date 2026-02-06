"""
Simulation module for multifractal fields

Contains methods for generating synthetic multifractal time series and fields.
"""

from .FIF import FIF_1D, FIF_ND
from .fbm import fBm_1D, fBm_1D_circulant, fBm_2D_circulant, fBm_ND_circulant
from .generalized_scale_invariance import canonical_scale_metric