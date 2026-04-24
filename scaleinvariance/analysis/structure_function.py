import numpy as np
import warnings

from .. import backend as B
from ..utils import process_lags, estimate_hurst_from_scaling


def structure_function(data, order=1, max_sep=None, axis=0, lags='powers of 1.2'):
    """
    Calculate the mean structure function for regularly spaced scalar data along a given axis.

    The structure function S_n(r) of order n is defined as:
        S_n(r) = <|u(x + r) - u(x)|^n>
    where <> denotes the average over all valid pairs separated by lag r.

    Parameters
    -----------
    data : array-like
        N-dimensional scalar data array.
    order : float or 1-D array-like of float, optional
        Order(s) of the structure function (default: 1). Each order must be positive.
        When an array is given, the absolute increments |u(x+r) - u(x)| are computed
        once per lag and reused across all orders (avoids redundant work).
    max_sep : int, optional
        Maximum lag to compute. If None, defaults to (size along axis) - 1.
    axis : int, optional
        Axis along which to compute the structure function (default: 0).
    lags : str, list, or np.ndarray, optional
        Option for selecting lag values. Can be:
          - A list or array of lags.
          - 'all': all integer lags from 1 to max_sep.
          - 'powers of X': lags as integers of powers of X, where X is any number (e.g., 'powers of 2', 'powers of 1.2').

    Returns
    --------
    tuple (np.ndarray, np.ndarray)
        lags : 1-D array of lag values.
        sf   : structure function values.
               Shape is (n_lags,) if `order` is a scalar, or (n_orders, n_lags)
               if `order` is a 1-D array. The lag axis is always last.
    """
    data = B.asarray(data)
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

    L = data.shape[axis]
    if max_sep is None:
        max_sep = L - 1
    else:
        max_sep = min(max_sep, L - 1)

    # Process lag options
    lags = process_lags(lags, max_sep, even_only=False)

    sf_values = np.empty((order_arr.size, lags.size), dtype=np.float64)

    # Compute structure function values for each lag along the specified axis
    for i, lag in enumerate(lags):
        # Build slices for the given axis
        slice1 = [slice(None)] * data.ndim
        slice2 = [slice(None)] * data.ndim
        slice1[axis] = slice(lag, None)
        slice2[axis] = slice(None, -lag)

        diff = B.abs(data[tuple(slice1)] - data[tuple(slice2)])

        if (B.numel(diff) == 0) or B.all(B.isnan(diff)):
            sf_values[:, i] = np.nan
            continue

        for j, o in enumerate(order_arr):
            sf_values[j, i] = float(B.nanmean(diff ** o))

    if order_is_scalar:
        sf_values = sf_values[0]
    return lags, sf_values


def costructure_function(data1, data2, order1=1, order2=1, max_sep=None,
                         axis=0, lags='powers of 1.2'):
    """
    Calculate the mixed (two-field, two-exponent) structure function.

    Defined as:
        S_{p,q}(r) = <|u_1(x + r) - u_1(x)|^p * |u_2(x + r) - u_2(x)|^q>
    where <> denotes the average over positions where both increments are finite.

    The diagonal p = q case recovers a single-exponent co-structure function;
    the p -> 0 or q -> 0 slices recover the single-field structure functions
    of the other field (up to the caveat that 0 is not a permitted order here).

    Parameters
    -----------
    data1, data2 : array-like
        N-dimensional scalar data arrays. Must have identical shape.
    order1, order2 : float or 1-D array-like of float, optional
        Exponent(s) applied to |delta f_1| and |delta f_2| respectively (default: 1).
        Each order must be positive. Arrays are allowed on either or both; the
        absolute increments of each field are computed once per lag and reused
        across all (order1, order2) combinations.
    max_sep : int, optional
        Maximum lag to compute. If None, defaults to (size along axis) - 1.
    axis : int, optional
        Axis along which to compute increments (default: 0).
    lags : str, list, or np.ndarray, optional
        Lag selection (see `structure_function` for accepted forms).

    Returns
    --------
    tuple (np.ndarray, np.ndarray)
        lags : 1-D array of lag values.
        cosf : co-structure function values.
               Shape depends on order1/order2 being scalar vs array:
               - (n_lags,)                        both scalar
               - (n_order1, n_lags)               order1 array, order2 scalar
               - (n_order2, n_lags)               order1 scalar, order2 array
               - (n_order1, n_order2, n_lags)     both arrays
               The lag axis is always last.

    Notes
    -----
    The average is over positions where both increments are finite. This is
    asymmetric with `structure_function`, which averages over positions where
    the single field is finite. If one field has NaNs and the other does not,
    the effective sample count here is the intersection of valid positions.
    """
    data1 = B.asarray(data1)
    data2 = B.asarray(data2)
    if data1.shape != data2.shape:
        raise ValueError(
            f"data1 and data2 must have the same shape; got {data1.shape} vs {data2.shape}"
        )

    o1_is_scalar = np.ndim(order1) == 0
    o2_is_scalar = np.ndim(order2) == 0
    o1 = np.atleast_1d(np.asarray(order1, dtype=np.float64))
    o2 = np.atleast_1d(np.asarray(order2, dtype=np.float64))
    if o1.ndim != 1 or o2.ndim != 1:
        raise ValueError("order1 and order2 must each be scalar or 1-D.")
    if np.any(o1 <= 0) or np.any(o2 <= 0):
        raise ValueError("All orders must be positive.")

    if axis < 0:
        axis += data1.ndim
    if axis < 0 or axis >= data1.ndim:
        raise ValueError("Axis is out of bounds for data dimensions.")

    L = data1.shape[axis]
    if max_sep is None:
        max_sep = L - 1
    else:
        max_sep = min(max_sep, L - 1)

    lags = process_lags(lags, max_sep, even_only=False)

    cosf = np.empty((o1.size, o2.size, lags.size), dtype=np.float64)

    for i, lag in enumerate(lags):
        slice1 = [slice(None)] * data1.ndim
        slice2 = [slice(None)] * data1.ndim
        slice1[axis] = slice(lag, None)
        slice2[axis] = slice(None, -lag)
        sl1 = tuple(slice1)
        sl2 = tuple(slice2)

        d1 = B.abs(data1[sl1] - data1[sl2])
        d2 = B.abs(data2[sl1] - data2[sl2])

        if (B.numel(d1) == 0) or B.all(B.isnan(d1)) or B.all(B.isnan(d2)):
            cosf[:, :, i] = np.nan
            continue

        for j, p in enumerate(o1):
            d1p = d1 ** p
            for k, q in enumerate(o2):
                cosf[j, k, i] = float(B.nanmean(d1p * (d2 ** q)))

    # Squeeze scalar order axes. Lags axis is last and is never squeezed.
    if o1_is_scalar and o2_is_scalar:
        cosf = cosf[0, 0]
    elif o1_is_scalar:
        cosf = cosf[0]
    elif o2_is_scalar:
        cosf = cosf[:, 0]

    return lags, cosf


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
    data = B.asarray(data)
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
            max_sep = array_size - 1

    if min_sep is None:
        if array_size <= 512:
            min_sep = 1
        else:
            min_sep = 4

    # Calculate structure function
    lags, sf_values = structure_function(data, order=1, max_sep=max_sep,
                                                 axis=axis, lags='powers of 1.2')

    # Use common Hurst estimation utility
    return estimate_hurst_from_scaling(lags, sf_values, min_sep, max_sep, return_fit)


def structure_function_analysis(*args, **kwargs):
    """Deprecated: use structure_function() instead."""
    import warnings
    warnings.warn("structure_function_analysis is deprecated, use structure_function instead",
                  DeprecationWarning, stacklevel=2)
    return structure_function(*args, **kwargs)
