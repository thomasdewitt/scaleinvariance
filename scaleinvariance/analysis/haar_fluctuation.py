import numpy as np
import warnings

from .. import backend as B
from ..utils import process_lags, estimate_hurst_from_scaling


def haar_fluctuation(data, order=1, max_sep=None, axis=0, lags='powers of 1.2', nan_behavior='raise'):
    """
    Calculate the mean absolute Haar fluctuation for regularly spaced scalar data along a given axis.

    Uses FFT-based convolution via the backend for efficient computation,
    with a direct fallback when NaNs are present.

    Parameters
    -----------
    data : array-like
        N-dimensional scalar data array.
    order : float or 1-D array-like of float, optional
        Order(s) of the fluctuation (default: 1). Each order must be positive.
        When an array is given, the absolute Haar convolution |conv(data, kernel)|
        is computed once per lag and reused across all orders (avoids redundant
        convolutions).
    max_sep : int, optional
        Maximum separation/lag to compute. If None, defaults to (size along axis) - 1.
    axis : int, optional
        Axis along which to compute the Haar fluctuation (default: 0).
    lags : str, list, or np.ndarray, optional
        Option for selecting lag values. Can be:
          - A list or array of lags.
          - 'all': all integer lags from 1 to max_sep.
          - 'powers of X': lags as integers of powers of X, where X is any number (e.g., 'powers of 2', 'powers of 1.2').
    nan_behavior : str, optional
        Behavior when encountering NaN values in the data. Options are:
            - 'raise': Raise an error if NaN values are found (default).
            - 'ignore': Ignore NaN values in the calculation.

    Returns
    --------
    tuple (np.ndarray, np.ndarray)
        lags : 1-D array of lag values.
        haar : mean absolute haar fluctuation values.
               Shape is (n_lags,) if `order` is a scalar, or (n_orders, n_lags)
               if `order` is a 1-D array. The lag axis is always last.
    """
    if nan_behavior not in ['ignore','raise']:
        raise ValueError("nan_behavior must be 'raise' or 'ignore'")

    # Check for integer dtype before converting to backend array
    if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
        data = B.asarray(data, dtype=B._active_real_dtype_np())
    else:
        data = B.asarray(data)
    has_nans = B.any(B.isnan(data))

    if has_nans and nan_behavior == 'raise':
        raise ValueError("Input data contains NaN values; change nan_behavior to 'ignore' or check inputs.")

    order_is_scalar = np.ndim(order) == 0
    order_arr = np.atleast_1d(np.asarray(order, dtype=np.float64))
    if order_arr.ndim != 1:
        raise ValueError("Order must be a scalar or 1-D array.")
    if np.any(order_arr <= 0):
        raise ValueError("All orders must be positive.")

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

    haar_flucs = np.empty((order_arr.size, lags.size), dtype=np.float64)

    for i, lag in enumerate(lags):
        # If lag is odd or too large, assign NaN and skip processing.
        if lag % 2 != 0 or lag > max_sep:
            haar_flucs[:, i] = np.nan
            continue

        # Construct Haar kernel
        kernel = B.ones(lag) / (lag/2)
        kernel[:lag//2] = kernel[:lag//2] * -1

        # Convolution via backend
        abs_conv = B.abs(B.convolve1d(data, kernel, axis=axis, nan_safe=has_nans))

        if (B.numel(abs_conv) == 0) or B.all(B.isnan(abs_conv)):
            haar_flucs[:, i] = np.nan
            del abs_conv
            continue

        for j, o in enumerate(order_arr):
            powered = abs_conv if o == 1 else abs_conv ** o
            haar_flucs[j, i] = float(B.nanmean(powered))
        del abs_conv

    if order_is_scalar:
        haar_flucs = haar_flucs[0]
    return np.array(lags, dtype=np.int64), haar_flucs


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
        Maximum separation/lag to include in the fit. Defaults to array_size - 1
        (a warning is emitted for small arrays).
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
    data = B.asarray(data)
    array_size = data.shape[axis]

    # Check minimum array size for reliable Hurst estimation
    if array_size < 16:
        raise ValueError(f"Array size along axis {axis} is {array_size}, but minimum 16 points required for Hurst estimation")

    # Set default separations
    if max_sep is None:
        if array_size <= 512:
            max_sep = array_size - 1
            warnings.warn(f"Array size ({array_size}) is small. Using max_sep={max_sep}. "
                         "Consider using larger arrays for more reliable Hurst estimation.",
                         UserWarning)
        else:
            max_sep = array_size - 1

    if min_sep is None:
        if array_size <= 512:
            min_sep = 1
        else:
            min_sep = 4

    # Calculate Haar fluctuations
    lags, haar_values = haar_fluctuation(data, order=1, max_sep=max_sep,
                                                 axis=axis, lags='powers of 1.2')

    # Use common Hurst estimation utility
    return estimate_hurst_from_scaling(lags, haar_values, min_sep, max_sep, return_fit)


def haar_fluctuation_analysis(*args, **kwargs):
    """Deprecated: use haar_fluctuation() instead."""
    import warnings
    warnings.warn("haar_fluctuation_analysis is deprecated, use haar_fluctuation instead",
                  DeprecationWarning, stacklevel=2)
    return haar_fluctuation(*args, **kwargs)
