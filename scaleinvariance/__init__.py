"""
ScaleInvariance: Simulation and analysis of multifractal fields

This package provides tools for:
- Hurst exponent estimation using multiple methods
- Fractionally integrated flux simulation
- Fractional Brownian motion simulation
"""

__version__ = "0.3.2"
__author__ = "Thomas DeWitt"

import torch
import os

# Set threading for both PyTorch and BLAS to 9/10 of available CPU cores
num_threads = max(1, int(os.cpu_count() * 0.9))
torch.set_num_threads(num_threads)
os.environ['OMP_NUM_THREADS'] = str(num_threads)
os.environ['MKL_NUM_THREADS'] = str(num_threads)
os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)

device = torch.device('cpu')

from . import analysis, simulation

# Easy access to main functions
from .analysis import structure_function_hurst, haar_fluctuation_hurst, spectral_hurst, haar_fluctuation_analysis, structure_function_analysis, spectral_analysis
from .simulation import FIF_1D, FIF_2D, acausal_fBm_1D, acausal_fBm_2D