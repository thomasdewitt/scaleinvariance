import warnings

import numpy as np

from .. import backend as B
from ..utils import process_lags, estimate_hurst_from_scaling
from .structure_function import structure_function
from .wavelets import get_wavelet


def wavelet_fluctuation(data, wavelet='haar', order=1, max_sep=None, axis=0,
                        lags='powers of 1.2', nan_behavior='raise', periodic=False):
    """
    Mean absolute wavelet fluctuation of regularly spaced data along an axis.

    At each scale (lag) ``r`` the data is convolved with the chosen wavelet's
    kernel ``g_r`` and the q-th moment of the modulus is averaged over all
    positions (and over every other axis, treated as independent
    realizations):

        F_q(r) = < |g_r * data|^q >.

    For a self-affine field with Hurst exponent ``H`` this scales as
    ``F_q(r) ~ r^{xi(q)}`` with ``xi(q) = qH - K(q)``. All kernels share the
    L1 ("fluctuation") normalization (``sum|g_r|`` fixed across scales, mean
    removed), so the *slope* is the same physical exponent for every wavelet,
    while the amplitude (intercept) differs by wavelet. See `wavelets` for the
    kernel definitions and the lag->scale convention.

    Parameters
    ----------
    data : array-like
        N-dimensional scalar data array.
    wavelet : str or Wavelet, optional
        Which wavelet to use (default: 'haar'). One of the names in
        ``scaleinvariance.analysis.wavelets.WAVELETS``: 'haar',
        'structure_function', 'mexican_hat', 'morlet'. An unknown name raises.
        ('haar' reproduces the previous `haar_fluctuation` exactly;
        'structure_function' reproduces `structure_function` to numerical
        precision via the general path -- the dedicated functions remain the
        faster route.)
    order : float or 1-D array-like of float, optional
        Moment order(s) (default: 1). Each must be positive. With an array the
        modulus convolution ``|g_r * data|`` is computed once per lag and
        reused across all orders.
    max_sep : int, optional
        Maximum lag to compute. If None, defaults to (size along axis) - 1.
    axis : int, optional
        Axis along which to compute (default: 0).
    lags : str, list, or np.ndarray, optional
        'all', 'powers of X', or an explicit array of lags. Even-width
        wavelets (Haar) keep only even lags.
    nan_behavior : str, optional
        'raise' (default) errors on NaN input; 'ignore' propagates NaN through
        a NaN-safe convolution (a window containing any NaN yields NaN there),
        then the moment is a ``nanmean`` over the surviving positions. The rule
        is identical for every wavelet (and bit-identical to the old
        ``haar_fluctuation``), but two consequences follow from the kernel
        widths (see `wavelets`):

        - A single NaN poisons a span as wide as the kernel: ``~r`` for 'haar'
          / 'structure_function', but ``~5.8 r`` (Mexican hat) and ``~15 r``
          (Morlet). Gappy data therefore loses far more output to the localized
          wavelets at the same lag, and large lags can go entirely NaN before
          ``r`` approaches the domain length.
        - ``wavelet='structure_function'`` would be *stricter* under NaN than
          the dedicated `structure_function`: the validity mask counts the full
          width-``r+1`` window (including the zero taps between the two ``+/-1``
          endpoints), so an interior NaN would poison the output even though it
          is multiplied by zero. To avoid silently discarding those pairs, when
          NaNs are present this case **delegates to** `structure_function`
          (which needs only the two endpoints finite) and emits a
          ``UserWarning`` saying so. The delegate reuses the same resolved lag
          array, so the only difference from the wavelet path is the (more
          correct) NaN handling.
    periodic : bool, optional
        If True, treat the domain as periodic (toroidal) along ``axis`` and use
        circular convolution, giving all L wraparound windows per lag. Unlike
        the structure function there is no F(r) = F(L - r) symmetry, so the
        full lag range is kept; the constraint is simply that the kernel fit
        the domain.

    Returns
    -------
    lags : np.ndarray
        1-D int64 array of lag values.
    fluctuations : np.ndarray
        Shape (n_lags,) for scalar ``order``, else (n_orders, n_lags); the lag
        axis is last. Lags whose kernel exceeds the domain (the localized
        wavelets have kernels several times wider than their nominal scale)
        are returned as NaN.
    """
    if nan_behavior not in ['ignore', 'raise']:
        raise ValueError("nan_behavior must be 'raise' or 'ignore'")

    wv = get_wavelet(wavelet)

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

    # Process lag options (even-only wavelets keep only even lags)
    lags = process_lags(lags, max_sep, even_only=wv.even_only)

    # The structure-function wavelet convolves with [1, 0, ..., 0, -1] (width
    # r+1), but the NaN-safe validity mask requires the WHOLE window finite --
    # including the interior zero taps the first difference never touches. So a
    # NaN strictly between the two endpoints needlessly discards a pair that the
    # dedicated structure_function() keeps. When NaNs are present we hand off to
    # that function (only the two endpoints must be finite). The resolved lag
    # array is passed through with max_sep=None so the delegate reproduces the
    # exact same lags as the wavelet path -- including, for periodic=True, lags
    # above L // 2 (process_lags leaves an explicit array untouched, so the
    # periodic max_sep <= L // 2 guard is not triggered). On NaN-free data the
    # two paths already agree, so no delegation is needed there.
    if has_nans and wv.name == 'structure_function':
        warnings.warn(
            "wavelet='structure_function' on NaN-containing data: the wavelet "
            "path flags an increment as NaN whenever its width-(r+1) convolution "
            "window contains a NaN -- including the interior zero taps the first "
            "difference never uses -- discarding more pairs than necessary. "
            "Delegating to structure_function(), which averages over every pair "
            "whose two endpoints are finite. (On NaN-free data the two paths are "
            "identical.)",
            UserWarning, stacklevel=2)
        # Delegate only the *reachable* lags (lag <= max_sep). Lags whose kernel
        # exceeds the domain are NaN in the wavelet path; structure_function
        # would instead compute finite wraparound aliases for periodic lags >= L
        # (roll by lag % L), so we reinstate NaN columns for them rather than
        # pass them through. Return int64 lags to match the wavelet path exactly.
        reachable = lags <= max_sep
        _, sf_vals = structure_function(data, order=order, max_sep=None, axis=axis,
                                        lags=lags[reachable], periodic=periodic)
        out = np.full((order_arr.size, lags.size), np.nan, dtype=np.float64)
        out[:, reachable] = np.asarray(sf_vals).reshape(order_arr.size, -1)
        if order_is_scalar:
            out = out[0]
        return np.array(lags, dtype=np.int64), out

    flucs = np.empty((order_arr.size, lags.size), dtype=np.float64)

    for i, lag in enumerate(lags):
        # Skip lags this wavelet cannot represent at this scale.
        if (wv.even_only and lag % 2 != 0) or lag > max_sep:
            flucs[:, i] = np.nan
            continue

        # Pass lag as-is (np.int64). Builders rely on numpy's dtype promotion
        # in float32 mode (e.g. the Haar kernel becomes float64 via int64/2),
        # which keeps wavelet='haar' bit-identical to the original.
        kernel = wv.build_kernel(lag)

        # The kernel must fit the domain. Localized wavelets have kernels wider
        # than their nominal scale, so the largest lags are unreachable; lags
        # are sorted ascending, so once one exceeds L all larger ones do too.
        if kernel.shape[0] > L:
            flucs[:, i:] = np.nan
            break

        # Convolution via backend (complex kernel -> complex result, real ->
        # real; circular when periodic). abs() is the modulus for complex.
        abs_conv = B.abs(B.convolve1d(data, kernel, axis=axis, nan_safe=has_nans,
                                      circular=periodic))

        if (B.numel(abs_conv) == 0) or B.all(B.isnan(abs_conv)):
            flucs[:, i] = np.nan
            del abs_conv
            continue

        for j, o in enumerate(order_arr):
            powered = abs_conv if o == 1 else abs_conv ** o
            flucs[j, i] = float(B.nanmean(powered))
        del abs_conv

    if order_is_scalar:
        flucs = flucs[0]
    return np.array(lags, dtype=np.int64), flucs


def wavelet_fluctuation_hurst(data, wavelet='haar', min_sep=None, max_sep=None,
                              axis=0, return_fit=False, periodic=False):
    """
    Estimate the Hurst exponent from first-order wavelet-fluctuation scaling.

    For a self-similar process the order-1 fluctuation scales as
    ``F(r) ~ r^H``; H is the slope of log F(r) vs log r. The accessible H range
    depends on the wavelet's vanishing moments (Haar: roughly 0 < H < 1; the
    Mexican hat and Morlet reach higher H).

    Parameters
    ----------
    data : array-like
        N-dimensional scalar data array.
    wavelet : str or Wavelet, optional
        Wavelet to use (default: 'haar').
    min_sep, max_sep : int, optional
        Lag window for the linear fit. Defaults match `haar_fluctuation_hurst`.
    axis : int, optional
        Axis along which to compute (default: 0).
    return_fit : bool, optional
        If True, also return (lags, fluctuation values, fit_line).
    periodic : bool, optional
        Periodic (toroidal) circular convolution along ``axis``.

    Returns
    -------
    H, uncertainty[, lags, values, fit_line]
    """
    data = B.asarray(data)
    array_size = data.shape[axis]

    if array_size < 16:
        raise ValueError(f"Array size along axis {axis} is {array_size}, but minimum 16 points required for Hurst estimation")

    if max_sep is None:
        max_sep = array_size - 1
        if array_size <= 512:
            warnings.warn(f"Array size ({array_size}) is small. Using max_sep={max_sep}. "
                          "Consider using larger arrays for more reliable Hurst estimation.",
                          UserWarning)

    if min_sep is None:
        if array_size <= 512:
            min_sep = 1
        else:
            min_sep = 4

    lags, values = wavelet_fluctuation(data, wavelet=wavelet, order=1,
                                       max_sep=max_sep, axis=axis,
                                       lags='powers of 1.2', periodic=periodic)

    return estimate_hurst_from_scaling(lags, values, min_sep, max_sep, return_fit)


def haar_fluctuation(data, order=1, max_sep=None, axis=0, lags='powers of 1.2',
                     nan_behavior='raise', periodic=False):
    """Mean absolute Haar fluctuation.

    Thin wrapper for ``wavelet_fluctuation(..., wavelet='haar')`` (the Haar
    fluctuation is the difference of the means of the two halves of a width-r
    window). Kept for convenience and backward compatibility; see
    `wavelet_fluctuation` for the full parameter documentation.
    """
    return wavelet_fluctuation(data, wavelet='haar', order=order, max_sep=max_sep,
                               axis=axis, lags=lags, nan_behavior=nan_behavior,
                               periodic=periodic)


def haar_fluctuation_hurst(data, min_sep=None, max_sep=None, axis=0,
                           return_fit=False, periodic=False):
    """Hurst exponent from Haar-fluctuation scaling.

    Thin wrapper for ``wavelet_fluctuation_hurst(..., wavelet='haar')``.
    """
    return wavelet_fluctuation_hurst(data, wavelet='haar', min_sep=min_sep,
                                     max_sep=max_sep, axis=axis,
                                     return_fit=return_fit, periodic=periodic)


def haar_fluctuation_analysis(*args, **kwargs):
    """Deprecated: use haar_fluctuation() instead."""
    warnings.warn("haar_fluctuation_analysis is deprecated, use haar_fluctuation instead",
                  DeprecationWarning, stacklevel=2)
    return haar_fluctuation(*args, **kwargs)
