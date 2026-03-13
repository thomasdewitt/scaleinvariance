import numpy as np
from scipy.optimize import curve_fit

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


def compute_K_q_function(data, q_values=None, scaling_method='structure_function',
                         hurst_fit_method='fixed', min_sep=None, max_sep=None, axis=0,
                         nan_behavior='raise'):
    """
    Estimate the multifractal K(q) scaling function empirically.

    For each moment order q, fits the scaling exponent zeta(q) from the
    q-th order structure function or Haar fluctuation scaling.

    Two fitting paths are available (controlled by hurst_fit_method):

    hurst_fit_method='fixed' (default):
        H is estimated from the order-1 scaling exponent zeta(1).
        K(q) = q * H - zeta(q) is then fit to recover C1 and alpha.

    hurst_fit_method='joint':
        H, C1, and alpha are all estimated jointly by fitting
        zeta(q) = q*H - K(q, C1, alpha) to the empirical zeta(q) values.

    Parameters
    ----------
    data : array-like
        N-dimensional scalar data array.
    q_values : array-like, optional
        Moment orders at which to evaluate K(q). Default: np.arange(0.1, 2.51, 0.1).
    scaling_method : str, optional
        'structure_function' (default) or 'haar_fluctuation'.
    hurst_fit_method : str, optional
        'fixed' (default): H from order-1 scaling, fit C1 and alpha only.
        'joint': H, C1, and alpha all free in a joint fit to zeta(q).
    min_sep : int, optional
        Minimum lag for scaling fits.
    max_sep : int, optional
        Maximum lag for scaling fits.
    axis : int, optional
        Axis along which to compute (default: 0).
    nan_behavior : str, optional
        Passed to haar_fluctuation_analysis when scaling_method='haar_fluctuation'.
        'raise' (default) or 'ignore'.

    Returns
    -------
    q_values : np.ndarray
        Moment orders used.
    K_empirical : np.ndarray
        Empirical K(q) = q * H_est - zeta(q), where H_est is the order-1 scaling exponent.
    H_fit : float
        Fitted Hurst exponent (equals H_est when hurst_fit_method='fixed').
    C1_fit : float
        Best-fit C1 parameter.
    alpha_fit : float
        Best-fit alpha parameter.
    """
    if q_values is None:
        q_values = np.arange(0.1, 2.51, 0.1)
    q_values = np.asarray(q_values)

    n_params = 3 if hurst_fit_method == 'joint' else 2
    if len(q_values) < n_params + 1:
        raise ValueError(
            f"hurst_fit_method='{hurst_fit_method}' requires at least {n_params + 1} q values "
            f"(fits {n_params} parameters); got {len(q_values)}."
        )

    if scaling_method == 'structure_function':
        analysis_kwargs = {}
        analysis_func = structure_function_analysis
    elif scaling_method == 'haar_fluctuation':
        analysis_kwargs = {'nan_behavior': nan_behavior}
        analysis_func = haar_fluctuation_analysis
    else:
        raise ValueError(f"Unknown scaling_method: '{scaling_method}'. Use 'structure_function' or 'haar_fluctuation'")

    lags_1, values_1 = analysis_func(data, order=1, max_sep=max_sep, axis=axis, **analysis_kwargs)
    H_est, _ = estimate_hurst_from_scaling(lags_1, values_1, min_sep, max_sep)

    zeta_empirical = np.zeros(len(q_values))
    for i, q in enumerate(q_values):
        lags_q, values_q = analysis_func(data, order=q, max_sep=max_sep, axis=axis, **analysis_kwargs)
        zeta_empirical[i], _ = estimate_hurst_from_scaling(lags_q, values_q, min_sep, max_sep)

    K_empirical = q_values * H_est - zeta_empirical

    if hurst_fit_method == 'joint':
        zeta_model = lambda q, H, C1, alpha: q * H - K(q, C1, alpha)
        popt, _ = curve_fit(zeta_model, q_values, zeta_empirical,
                            p0=[H_est, 0.1, 1.8],
                            bounds=([H_est - 0.5, 0.0, 0.5], [H_est + 0.5, 1.0, 2.0]),
                            max_nfev=10000)
        H_fit, C1_fit, alpha_fit = popt
    elif hurst_fit_method == 'fixed':
        popt, _ = curve_fit(K, q_values, K_empirical, p0=[0.1, 1.8],
                            bounds=([0.0, 0.5], [1.0, 2.0]), max_nfev=10000)
        C1_fit, alpha_fit = popt
        H_fit = H_est
    else:
        raise ValueError(f"Unknown hurst_fit_method: '{hurst_fit_method}'. Use 'fixed' or 'joint'")

    return q_values, K_empirical, H_fit, C1_fit, alpha_fit


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
