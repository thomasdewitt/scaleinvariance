"""
Analysis module for multifractal fields

Contains methods for estimating scaling exponents and analyzing 
multifractal properties.
"""

from .structure_function import structure_function, structure_function_hurst, structure_function_analysis
from .haar_fluctuation import haar_fluctuation, haar_fluctuation_hurst, haar_fluctuation_analysis
from .spectral import power_spectrum_binned, spectral_hurst, spectral_analysis
from .intermittency import K_analytic, K_empirical, two_point_C1, K, compute_K_q_function, two_point_intermittency_exponent