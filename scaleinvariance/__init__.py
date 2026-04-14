"""
ScaleInvariance: Simulation and analysis of multifractal fields

This package provides tools for:
- Hurst exponent estimation using multiple methods
- Fractionally integrated flux simulation
- Fractional Brownian motion simulation
"""

__version__ = "0.10.0"
__author__ = "Thomas DeWitt"

from . import backend

# Initialize backend with default threading (90% CPU count)
backend.set_num_threads(backend._num_threads)

from . import analysis, simulation

# Easy access to main functions
from .analysis import (
    structure_function, structure_function_hurst,
    haar_fluctuation, haar_fluctuation_hurst,
    power_spectrum_binned, spectral_hurst,
    K_analytic, K_empirical, two_point_C1,
    # Deprecated aliases
    structure_function_analysis, haar_fluctuation_analysis, spectral_analysis,
    two_point_intermittency_exponent, K, compute_K_q_function,
)
from .simulation import (
    FIF_1D,
    FIF_ND,
    fBm_1D,
    fBm_1D_circulant,
    fBm_ND_circulant,
    canonical_scale_metric,
    fractional_integral_spectral,
    broken_fractional_integral_spectral,
)

# Backend configuration
from .backend import (
    set_backend,
    get_backend,
    set_num_threads,
    set_numerical_precision,
    get_numerical_precision,
    set_device,
    get_device,
    to_numpy,
)
