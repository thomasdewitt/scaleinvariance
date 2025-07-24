"""
ScaleInvariance: Simulation and analysis of multifractal fields

This package provides tools for:
- Hurst exponent estimation using multiple methods
- Fractionally integrated flux simulation
- Fractional Brownian motion simulation
"""

__version__ = "0.1.0"
__author__ = "Thomas DeWitt"

import torch
import os

# Set threading for both PyTorch and BLAS
torch.set_num_threads(8)
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

device = torch.device('cpu')

from . import analysis, simulation

# Easy access to main functions
from .analysis import structure_function_hurst, haar_fluctuation_hurst, spectral_hurst, haar_fluctuation_analysis, structure_function_analysis, spectral_analysis
from .simulation import fractionally_integrated_flux, fractional_brownian_motion, acausal_fBm_1D, acausal_fBm_2D