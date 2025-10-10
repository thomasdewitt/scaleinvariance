import numpy as np

from .structure_function import structure_function_analysis
from .haar_fluctuation import haar_fluctuation_analysis
from ..utils import estimate_hurst_from_scaling


def K(q, C1, alpha):
    """
    Multifractal K(q) function.

    K(q) = (C1/(alpha-1)) * (q^alpha - q)

    Parameters
    ----------
    q : float
        Order of the moment.
    C1 : float
        Codimension of the mean (intermittency parameter).
    alpha : float
        Lévy stability parameter.

    Returns
    -------
    float
        K(q) value.
    """
    return (C1 / (alpha - 1)) * (q**alpha - q)


def two_point_intermittency_exponent(data, order=2, assumed_alpha=2.0, scaling_method='structure_function',
                                      min_sep=None, max_sep=None, axis=0):
    """
    Calculate the intermittency parameter C1 from multifractal scaling.

    Uses the relationship: xi_q = qH - K(q)
    where K(q) = (C1/(alpha-1)) * (q^alpha - q)

    Computes scaling exponents for order 1 and arbitrary order q, then solves for C1:
    C1 = (alpha-1) * (q*xi_1 - xi_q) / (q^alpha - q)

    Parameters
    ----------
    data : array-like
        N-dimensional scalar data array.
    order : float, optional
        Second order for scaling exponent calculation (default: 2).
    assumed_alpha : float, optional
        Lévy stability parameter (default: 2.0).
    scaling_method : str, optional
        Method for calculating scaling exponents. Options:
        - 'structure_function' (default)
        - 'haar_fluctuation'
    min_sep : int, optional
        Minimum separation for scaling analysis.
    max_sep : int, optional
        Maximum separation for scaling analysis.
    axis : int, optional
        Axis along which to compute scaling (default: 0).

    Returns
    -------
    C1 : float
        Estimated intermittency parameter (codimension of the mean).
    uncertainty : float
        Uncertainty in C1 estimate from error propagation.
        NOTE: This uncertainty is from standard error propagation and should
        not be taken too seriously. The true uncertainty is likely larger due
        finite sample effects (convergence is very slow)
    """
    # Select analysis function
    if scaling_method == 'structure_function':
        analysis_func = structure_function_analysis
    elif scaling_method == 'haar_fluctuation':
        analysis_func = haar_fluctuation_analysis
    else:
        raise ValueError(f"Unknown scaling_method: {scaling_method}. Use 'structure_function' or 'haar_fluctuation'")

    # Calculate scaling exponents for order 1 and q
    lags_1, values_1 = analysis_func(data, order=1, max_sep=max_sep, axis=axis)
    xi_1, sigma_1 = estimate_hurst_from_scaling(lags_1, values_1, min_sep, max_sep, return_fit=False)

    lags_q, values_q = analysis_func(data, order=order, max_sep=max_sep, axis=axis)
    xi_q, sigma_q = estimate_hurst_from_scaling(lags_q, values_q, min_sep, max_sep, return_fit=False)

    # Calculate K(q) from  xi_q = q * xi_1 - K(q)
    Kq = order * xi_1 - xi_q

    # Error propagation for K(q)
    sigma_Kq = np.sqrt((order * sigma_1)**2 + sigma_q**2)

    # Calculate C1
    denominator = order**assumed_alpha - order
    C1 = (assumed_alpha - 1) * Kq / denominator

    # Error propagation for C1
    sigma_C1 = abs((assumed_alpha - 1) / denominator) * sigma_Kq

    return C1, sigma_C1
