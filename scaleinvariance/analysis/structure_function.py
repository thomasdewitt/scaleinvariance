import numpy as np
import warnings

from ..utils import process_lags, estimate_hurst_from_scaling


def structure_function_analysis(data, order=1, max_sep=None, axis=0, lags='powers of 1.2'):
    """
    Calculate the mean structure function for regularly spaced scalar data along a given axis.
    
    The structure function S_n(r) of order n is defined as:
        S_n(r) = <|u(x + r) - u(x)|^n>
    where <> denotes the average over all valid pairs separated by lag r.
    
    Parameters
    -----------
    data : array-like
        N-dimensional scalar data array.
    order : int, optional
        Order of the structure function (default: 1). Must be positive.
    max_sep : int, optional
        Maximum lag to compute. If None, defaults to (size along axis) - 1.
    axis : int, optional
        Axis along which to compute the structure function (default: 0).
    lags : str, list, or np.ndarray, optional
        Option for selecting lag values. Can be:
          - A list or array of lags.
          - 'all': all integer lags from 1 to max_sep.
          - 'powers of 10': lags as powers of 10.
          - 'powers of 3': lags as powers of 3.
          - 'powers of 2': lags as powers of 2.
          - 'powers of 1.2': lags as powers of 1.2.
          - 'powers of 1.05': lags as powers of 1.05.
    
    Returns
    --------
    tuple (np.ndarray, np.ndarray)
        lags : 1-D array of lag values.
        sf   : 1-D array of structure function values corresponding to each lag.
    """
    data = np.asarray(data)
    if order <= 0:
        raise ValueError("Order must be a positive float.")
    
    # Validate axis
    if axis < 0: 
        axis += data.ndim
    if axis < 0 or axis >= data.ndim:
        raise ValueError("Axis is out of bounds for data dimensions.")
    
    L = data.shape[axis]
    if max_sep is None:
        max_sep = L - 1
    else:
        max_sep = min(max_sep, L - 1)
    
    # Process lag options
    lags = process_lags(lags, max_sep, even_only=False)
    
    sf_values = np.empty(lags.size, dtype=np.float64)
    
    # Compute structure function values for each lag along the specified axis
    for i, lag in enumerate(lags):
        # Build slices for the given axis
        slice1 = [slice(None)] * data.ndim
        slice2 = [slice(None)] * data.ndim
        slice1[axis] = slice(lag, None)
        slice2[axis] = slice(None, -lag)
        
        diff = np.abs(data[tuple(slice1)] - data[tuple(slice2)])
        
        if (diff.size == 0) or np.all(np.isnan(diff)):
            sf_values[i] = np.nan
            continue
            
        sf_values[i] = np.nanmean(diff**order)
    
    return lags, sf_values


def structure_function_hurst(data, min_sep=None, max_sep=None, axis=0, return_fit=False):
    """
    Calculate the Hurst exponent using structure functions.
    
    For a self-similar process with Hurst exponent H, the first order structure function scales as:
    S_1(r) ~ r^H
    
    This function estimates H by performing a linear regression on log(S_1(r)) vs log(r).
    
    Parameters
    -----------
    data : array-like
        N-dimensional scalar data array.
    min_sep : int, optional
        Minimum separation/lag to include in the fit. If None, defaults to 1.
    max_sep : int, optional
        Maximum separation/lag to include in the fit. If None, defaults to array_size/8
        or array_size if array_size < 128 (with warning).
    axis : int, optional
        Axis along which to compute the structure function (default: 0).
    return_fit : bool, optional
        If True, return the lags, structure function values, and fitted line.
        
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
        sf_values : np.ndarray
            The structure function values.
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
    
    # Calculate structure function
    lags, sf_values = structure_function_analysis(data, order=1, max_sep=max_sep, 
                                                 axis=axis, lags='powers of 1.2')
    
    # Use common Hurst estimation utility
    return estimate_hurst_from_scaling(lags, sf_values, min_sep, max_sep, return_fit)