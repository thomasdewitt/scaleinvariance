import numpy as np
import warnings
from scipy.signal import convolve

from ..utils import process_lags, estimate_hurst_from_scaling


def haar_fluctuation_analysis(data, order=1, max_sep=None, axis=0, lags='powers of 1.2', nan_behavior='raise'):
    """
    Calculate the mean absolute Haar fluctuation for regularly spaced scalar data along a given axis.

    Uses scipy.signal.convolve for efficient computation with BLAS parallelization.
    
    Parameters
    -----------
    data : array-like
        N-dimensional scalar data array.
    order : int, optional
        Order of the fluctuation (default: 1).
    max_sep : int, optional
        Maximum separation/lag to compute. If None, defaults to (size along axis) - 1.
    axis : int, optional
        Axis along which to compute the Haar fluctuation (default: 0).
    lags : str, list, or np.ndarray, optional
        Option for selecting lag values. Can be:
          - A list or array of lags.
          - 'all': all integer lags from 2 to max_lag.
          - 'powers of 10': lags as powers of 10.
          - 'powers of 3': lags as powers of 3.
          - 'powers of 2': lags as powers of 2.
          - 'powers of 1.2': lags as powers of 1.2. (~ 8 per decade)
          - 'powers of 1.05': lags as powers of 1.05. (~ 25 per decade)
    nan_behavior : str, optional
        Behavior when encountering NaN values in the data. Options are:
            - 'raise': Raise an error if NaN values are found (default).
            - 'ignore': Ignore NaN values in the calculation.
    
    Returns
    --------
    tuple (np.ndarray, np.ndarray)
        lags : 1-D array of lag values.
        haar : 1-D array of mean absolute haar fluctuation values corresponding to each lag.
    """
    if nan_behavior not in ['ignore','raise']: 
        raise ValueError("nan_behavior must be 'raise' or 'ignore'")
    
    data = np.asarray(data)
    if (np.count_nonzero(np.isnan(data)) > 0) and (nan_behavior == 'raise'):
        raise ValueError("Input data contains NaN values; change nan_behavior to 'ignore' or check inputs.")
    
    # Validate axis
    if axis < 0: 
        axis += data.ndim
    if axis < 0 or axis >= data.ndim:
        raise ValueError("Axis is out of bounds for data dimensions.")
    if max_sep is not None and (not isinstance(max_sep, int) or max_sep <= 0):
        raise ValueError("'max_sep' must be a positive integer or None.")
    
    L = data.shape[axis]
    if max_sep is None:
        max_sep = L - 1
    else:
        max_sep = min(max_sep, L - 1)
    
    # Process lag options
    lags = process_lags(lags, max_sep, even_only=True)

    # Determine convolution method
    if np.any(np.isnan(data)):
        conv_method = 'direct'
    else:
        conv_method = 'auto'

    haar_flucs = []

    for i, lag in enumerate(lags):
        # If lag is odd or too large, assign NaN and skip processing.
        if lag % 2 != 0 or lag >= max_sep:
            haar_flucs.append(np.nan)
            continue
            
        # Construct Haar kernel
        kernel = np.ones(lag) / (lag/2)
        kernel[:lag//2] *= -1

        # Embed the 1D kernel into an n-D kernel with its values along the specified axis.
        kernel_shape = [1] * data.ndim
        kernel_shape[axis] = lag
        kernel = kernel.reshape(kernel_shape)

        # Convolution using scipy (parallelized with BLAS)
        abs_conv = np.abs(convolve(data, kernel, mode='valid', method=conv_method))

        if (abs_conv.size == 0) or np.all(np.isnan(abs_conv)):
            haar_flucs.append(np.nan)
            continue
        
        if order != 1:
            abs_conv = abs_conv**order

        mean_absolute_haar_fluctuation = np.nanmean(abs_conv)
        haar_flucs.append(mean_absolute_haar_fluctuation)

    return np.array(lags), np.array(haar_flucs)


def haar_fluctuation_hurst(data, min_sep=None, max_sep=None, axis=0, return_fit=False):
    """
    Calculate the Hurst exponent using Haar fluctuation analysis.
    
    For a self-similar process with Hurst exponent H, the Haar fluctuation scales as:
    F_H(r) ~ r^H
    
    This function estimates H by performing a linear regression on log(F_H(r)) vs log(r).
    
    Parameters
    -----------
    data : array-like
        N-dimensional scalar data array.
    min_sep : int, optional
        Minimum separation/lag to include in the fit.
    max_sep : int, optional
        Maximum separation/lag to include in the fit. If None, defaults to array_size/8
        or array_size if array_size < 128 (with warning).
    axis : int, optional
        Axis along which to compute the Haar fluctuation (default: 0).
    return_fit : bool, optional
        If True, return the lags, fluctuation values, and fitted line.
        
    Returns
    --------
    If return_fit is False:
        H : float
            Estimated Hurst exponent.
        uncertainty : float
            Standard error of the Hurst exponent estimate.
    If return_fit is True:
        H : float 
            Estimated Hurst exponent.
        uncertainty : float
            Standard error of the Hurst exponent estimate.
        lags : np.ndarray
            The lags used in the calculation.
        haar_values : np.ndarray
            The Haar fluctuation values.
        fit_line : np.ndarray
            The fitted line used to estimate H.
    """
    data = np.asarray(data)
    array_size = data.shape[axis]
    
    # Check minimum array size for reliable Hurst estimation
    if array_size < 16:
        raise ValueError(f"Array size along axis {axis} is {array_size}, but minimum 16 points required for Hurst estimation")
    
    # Set default separations to isolate scaling range >> grid size and << domain size
    if max_sep is None:
        if array_size <= 512:
            max_sep = array_size - 1
            warnings.warn(f"Array size ({array_size}) is small. Using max_sep={max_sep}. "
                         "Consider using larger arrays for more reliable Hurst estimation.",
                         UserWarning)
        else:
            max_sep = array_size // 8
    
    if min_sep is None:
        if array_size <= 512:
            min_sep = 1
        else:
            min_sep = 4
    
    # Calculate Haar fluctuations
    lags, haar_values = haar_fluctuation_analysis(data, order=1, max_sep=max_sep,
                                                 axis=axis, lags='powers of 1.2')
    
    # Use common Hurst estimation utility
    return estimate_hurst_from_scaling(lags, haar_values, min_sep, max_sep, return_fit)