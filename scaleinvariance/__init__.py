"""
ScaleInvariance: Simulation and analysis of multifractal fields

This package provides tools for:
- Hurst exponent estimation using multiple methods
- Fractionally integrated flux simulation
- Fractional Brownian motion simulation
"""

__version__ = "0.6.0"
__author__ = "Thomas DeWitt"

from . import backend

# Initialize backend with default threading (90% CPU count)
backend.set_num_threads(backend._num_threads)

from . import analysis, simulation

# Easy access to main functions
from .analysis import structure_function_hurst, haar_fluctuation_hurst, spectral_hurst, haar_fluctuation_analysis, structure_function_analysis, spectral_analysis, two_point_intermittency_exponent, K
from .simulation import FIF_1D, FIF_ND, fBm_1D, fBm_1D_circulant, fBm_ND_circulant, canonical_scale_metric

# Backend configuration
from .backend import set_backend, get_backend, set_num_threads
