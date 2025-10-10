"""
Analysis module for multifractal fields

Contains methods for estimating scaling exponents and analyzing 
multifractal properties.
"""

from .structure_function import structure_function_hurst, structure_function_analysis
from .haar_fluctuation import haar_fluctuation_hurst, haar_fluctuation_analysis
from .spectral import spectral_hurst, spectral_analysis
from .intermittency import two_point_intermittency_exponent, K