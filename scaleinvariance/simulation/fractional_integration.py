import math

import numpy as np

from .. import backend as B
from .kernels import _normalize_axis_flag


def _isotropic_rfft_frequencies(shape):
    """Return isotropic Fourier wavenumber magnitudes on the rfftn grid.

    Builds ``sqrt(sum f_i^2)`` via broadcasted 1D axes (rfftfreq for the
    last axis, fftfreq for the rest) to avoid the memory cost of a full
    meshgrid.
    """
    ndim = len(shape)
    freqs_squared = None
    for axis_idx in range(ndim):
        if axis_idx == ndim - 1:
            axis = B.rfftfreq(shape[-1], d=1.0)
        else:
            axis = B.fftfreq(shape[axis_idx], d=1.0)
        broadcast_shape = [1] * ndim
        broadcast_shape[axis_idx] = axis.shape[0]
        term = axis.reshape(broadcast_shape) ** 2
        freqs_squared = term if freqs_squared is None else freqs_squared + term
    return B.sqrt(freqs_squared)


def _pow_or_inf(base, exp):
    """Return ``base ** exp`` as a Python float, returning ``inf`` on overflow."""
    try:
        return float(base) ** float(exp)
    except OverflowError:
        return math.inf


def _check_kernel_max(kernel_max_bound, context_msg):
    """Raise OverflowError if ``kernel_max_bound`` risks overflowing the
    active real dtype (within a factor-of-10 headroom for signal amplitude
    after rfft).
    """
    real_dtype_name = B.get_numerical_precision()
    max_finite = float(np.finfo(real_dtype_name).max)
    if not math.isfinite(kernel_max_bound) or kernel_max_bound > max_finite / 10.0:
        raise OverflowError(
            f"Fourier-kernel peak magnitude ~{kernel_max_bound:.2e} "
            f"({context_msg}) exceeds the safe range for {real_dtype_name}. "
            "Reduce H, reduce outer_scale, or call "
            "scaleinvariance.set_numerical_precision('float64')."
        )


def _guard_single_kernel_overflow(outer_scale, H, ndim):
    """Bound the max magnitude of ``|f|^(-H)`` over the rfft grid.

    Candidate extrema: lowest resolvable frequency ``1/outer_scale`` (after
    the outer-scale clip) and highest isotropic frequency ``sqrt(ndim)/2``
    (rfft corner). Whichever gives the larger magnitude wins — this covers
    H of either sign and dimensions where ``sqrt(ndim)/2 > 1`` (d >= 5).
    """
    min_freq = 1.0 / float(outer_scale)
    max_freq = math.sqrt(ndim) / 2.0
    kernel_max_bound = max(
        _pow_or_inf(min_freq, -H),
        _pow_or_inf(max_freq, -H),
    )
    _check_kernel_max(
        kernel_max_bound,
        f"outer_scale={outer_scale}, H={H}, ndim={ndim}",
    )


def _guard_broken_kernel_overflow(
    outer_scale, H_small, H_large, transition_wavelength, ndim
):
    """Bound the max magnitude of the two-branch kernel over the rfft grid.

    Small branch carries a prefactor ``k_t^(H_small - H_large)`` that can
    itself be huge when ``transition_wavelength`` is far from 1, so we must
    evaluate both branches at all candidate extrema and take the max.
    Candidate extrema: ``1/outer_scale``, ``k_t``, and ``sqrt(ndim)/2``.
    """
    min_freq = 1.0 / float(outer_scale)
    max_freq = math.sqrt(ndim) / 2.0
    k_t = 1.0 / float(transition_wavelength)
    small_prefactor = _pow_or_inf(k_t, H_small - H_large)

    large_at = lambda f: _pow_or_inf(f, -H_large)
    small_at = lambda f: (small_prefactor * _pow_or_inf(f, -H_small)
                          if math.isfinite(small_prefactor) else math.inf)

    # Over-estimate by evaluating both branches at all candidates. The actual
    # kernel uses whichever branch applies per-bin, so this is an upper bound.
    candidates = [
        large_at(min_freq), large_at(k_t), large_at(max_freq),
        small_at(min_freq), small_at(k_t), small_at(max_freq),
    ]
    kernel_max_bound = max(candidates)
    _check_kernel_max(
        kernel_max_bound,
        f"outer_scale={outer_scale}, H_small={H_small}, H_large={H_large}, "
        f"transition_wavelength={transition_wavelength}, ndim={ndim}",
    )


def _build_odd_phase_factor(shape, odd_tuple):
    """Build the per-axis sign product ``Π_{j in odd_axes} sign(f_j)`` on the
    rfftn grid via broadcasted 1D axes.

    Last axis uses ``rfftfreq`` (non-negative, sign vanishes only at DC); all
    other axes use ``fftfreq`` (signed). Returns a real-valued array broadcast
    over the rfft shape. Caller multiplies by the magnitude and the ``i^k``
    phase to assemble the full complex kernel.
    """
    ndim = len(shape)
    sign_product = None
    for axis_idx, is_odd in enumerate(odd_tuple):
        if not is_odd:
            continue
        if axis_idx == ndim - 1:
            f_axis = B.rfftfreq(shape[axis_idx], d=1.0)
        else:
            f_axis = B.fftfreq(shape[axis_idx], d=1.0)
        sign_1d = B.sign(f_axis)
        broadcast_shape = [1] * ndim
        broadcast_shape[axis_idx] = f_axis.shape[0]
        sign_factor = sign_1d.reshape(broadcast_shape)
        sign_product = sign_factor if sign_product is None else sign_product * sign_factor
    return sign_product


def _zero_odd_axis_nyquist(kernel, shape, odd_tuple):
    """Zero out the Nyquist bin along each odd axis with even length.

    For an axis ``j`` in ``odd_axes`` with even size, the Nyquist bin aliases
    to itself under negation. The full-grid Hermitian-symmetry requirement at
    that bin is inconsistent with the antisymmetric kernel (it would demand
    ``sign(f_j) = -sign(f_j)``), so we zero it. DC is handled automatically by
    ``sign(0) = 0``.
    """
    ndim = len(shape)
    for axis_idx, is_odd in enumerate(odd_tuple):
        if not is_odd or shape[axis_idx] % 2 != 0:
            continue
        nyq_idx = shape[axis_idx] // 2
        slicer = [slice(None)] * ndim
        slicer[axis_idx] = nyq_idx
        kernel[tuple(slicer)] = 0


def fractional_integral_spectral(signal, H, outer_scale=None, odd_axes=None):
    """
    Apply an acausal spectral fractional integral of order ``H`` to a real signal.

    Computes ``irfftn(rfftn(signal) * K(f))`` where the default kernel is the
    radial Riesz response ``K(f) = |f|^(-H)`` and ``|f|`` is the isotropic
    Fourier wavenumber ``sqrt(sum f_i^2)``. Works for 1D and N-D real signals.
    Low frequencies are regularized by clipping ``|f|`` to ``1/outer_scale``;
    the DC bin is overwritten with the value of the kernel at the nearest
    non-DC bin so the kernel has no spectral notch at DC (standard path).

    The Fourier-space exponent is ``-H`` because fractional integration of
    order ``H`` is convolution with the Riesz potential ``|r|^(H - d)``, whose
    Fourier transform is proportional to ``|f|^(-H)`` — the spatial dimension
    cancels out. Applied as a circular convolution, i.e. periodic boundary
    conditions.

    With ``odd_axes`` set, the kernel is multiplied by the per-axis sign
    pattern ``i^k · Π_{j in odd_axes} sign(f_j)`` where ``k`` is the number
    of odd axes. This produces a real-space kernel that is antisymmetric in
    each of those axes (i.e. multiplied by ``Π sign(x_j)``); the output is
    therefore zero-mean. The complex ``i^k`` phase is real for even ``k`` and
    purely imaginary for odd ``k`` — the kernel is real-valued in the former
    case, complex in the latter, but the full-grid Hermitian symmetry holds
    in both cases so the inverse rfft yields a real output.

    Standard path remains **not mean-preserving**: the output mean is scaled
    by roughly ``outer_scale**H`` relative to the input mean. The odd-axes
    path is exactly zero-mean by construction.

    Parameters
    ----------
    signal : ndarray
        Real-valued input, 1D or N-D.
    H : float
        Hurst exponent (order of the fractional integral). Any finite nonzero
        real. ``H > 0`` reddens the spectrum (true integration); ``H < 0``
        blues it (fractional differentiation); ``H = 0`` would be the identity
        and is rejected.
    outer_scale : float, optional
        Low-frequency regularization scale. Defaults to ``max(signal.shape)``.
    odd_axes : bool or tuple of bool, optional
        Which axes the kernel should be antisymmetric along (sign-flipped on
        the negative half). ``None`` (default) or all-False gives the standard
        even kernel. Multiplies the Fourier response by
        ``i^k · Π_{j in odd_axes} sign(f_j)``. Zero-mean output.

    Returns
    -------
    ndarray
        Fractionally-integrated signal, same shape as input.

    Raises
    ------
    ValueError
        If ``H == 0`` (identity operator — use ``signal.copy()``).
    OverflowError
        If ``outer_scale**H`` would overflow the active real dtype. Call
        ``scaleinvariance.set_numerical_precision('float64')`` or reduce
        ``H`` / ``outer_scale``.
    """
    if H == 0:
        raise ValueError(
            "H == 0 is the identity operator. Use signal.copy() instead, "
            "or pass a nonzero H."
        )

    signal = B.asarray(signal)
    shape = signal.shape
    ndim = signal.ndim

    odd_tuple = _normalize_axis_flag(odd_axes, ndim, 'odd_axes')
    has_odd = any(odd_tuple)

    if outer_scale is None:
        outer_scale = max(shape)
    elif outer_scale <= 0:
        raise ValueError(f"outer_scale must be positive; got {outer_scale}.")

    _guard_single_kernel_overflow(outer_scale, H, ndim)

    rfft_shape = shape[:-1] + (shape[-1] // 2 + 1,)
    freqs = _isotropic_rfft_frequencies(shape)
    min_freq = 1.0 / float(outer_scale)
    freqs_reg = B.maximum(freqs, min_freq)
    del freqs

    magnitude = freqs_reg ** (-H)
    del freqs_reg

    if magnitude.shape != rfft_shape:
        magnitude = magnitude + B.zeros(rfft_shape, dtype=magnitude.dtype)

    if not has_odd:
        # Standard (even) path: fill DC with nearest non-DC value to avoid a
        # spectral notch.
        kernel = magnitude
        dc_index = (0,) * ndim
        if B.numel(kernel) == 1:
            kernel[dc_index] = 1.0
        else:
            neighbor_index = (1,) + (0,) * (ndim - 1)
            kernel[dc_index] = kernel[neighbor_index]
    else:
        # Odd path: assemble K = i^k · Π sign(f_j) · |f|^(-H)
        sign_product = _build_odd_phase_factor(shape, odd_tuple)
        k_odd = sum(odd_tuple)
        # i^k splits into a real sign (for even k) and a purely imaginary
        # factor (for odd k). For k mod 4 = 0: +1; 1: +i; 2: -1; 3: -i.
        phase_sign = 1.0 if (k_odd // 2) % 2 == 0 else -1.0
        signed_magnitude = phase_sign * sign_product * magnitude
        del magnitude, sign_product

        if k_odd % 2 == 0:
            kernel = signed_magnitude
        else:
            # Purely imaginary kernel: store in a complex array's imag part.
            complex_dtype = B._active_complex_dtype_np()
            kernel = B.zeros(signed_magnitude.shape, dtype=complex_dtype)
            kernel.imag[:] = signed_magnitude
            del signed_magnitude

        # Nyquist along each odd even-sized axis must be zero. DC is already
        # zero because sign(0) = 0.
        _zero_odd_axis_nyquist(kernel, shape, odd_tuple)

    spectrum = B.rfftn(signal) * kernel
    del kernel
    result = B.irfftn(spectrum, s=shape)
    del spectrum
    return result


def broken_fractional_integral_spectral(
    signal,
    H_small_scale,
    H_large_scale,
    transition_wavelength,
    outer_scale=None,
):
    """
    Apply a "broken" spectral fractional integral with two Hurst exponents.

    Like :func:`fractional_integral_spectral`, but the Fourier-space kernel
    is a piecewise power law with slope ``-H_large_scale`` at large scales
    (low ``|f|``) and slope ``-H_small_scale`` at small scales (high ``|f|``).
    The two branches meet at the transition wavenumber ``k_t =
    1/transition_wavelength``, with a continuity prefactor applied to the
    small-scale branch so the kernel magnitude is continuous at ``k_t`` (the
    slope changes abruptly; the amplitude does not jump).

    Kernel (with ``f_reg = max(|f|, 1/outer_scale)``):

    - ``f_reg <= k_t``:  ``kernel = f_reg ^ (-H_large_scale)``
    - ``f_reg >  k_t``:  ``kernel = k_t^(H_small_scale - H_large_scale)
      * f_reg^(-H_small_scale)``

    The DC bin is overwritten with the kernel value at the nearest non-DC
    bin (no spectral notch). Applied as a circular convolution (periodic
    boundaries). This is **not mean-preserving** — the output mean is
    scaled relative to the input mean by the kernel magnitude at the lowest
    non-DC bin.

    Degenerate cases: if ``transition_wavelength > outer_scale`` the entire
    kernel collapses to the small-scale branch; if ``transition_wavelength
    < 2`` (below Nyquist), the entire kernel collapses to the large-scale
    branch. Neither is an error.

    Parameters
    ----------
    signal : ndarray
        Real-valued input, 1D or N-D.
    H_small_scale : float
        Hurst exponent at small scales (``|f| > k_t``).
    H_large_scale : float
        Hurst exponent at large scales (``|f| <= k_t``).
    transition_wavelength : float
        Wavelength (in grid units) at which the slope changes. Corresponds
        to Fourier-space transition ``k_t = 1/transition_wavelength``.
    outer_scale : float, optional
        Low-frequency regularization scale. Defaults to ``max(signal.shape)``.

    Returns
    -------
    ndarray
        Fractionally-integrated signal, same shape as input.

    Raises
    ------
    ValueError
        If both ``H_small_scale == 0`` and ``H_large_scale == 0`` (identity
        operator).
    OverflowError
        If ``outer_scale**max(H_small_scale, H_large_scale)`` would overflow
        the active real dtype.
    """
    if H_small_scale == 0 and H_large_scale == 0:
        raise ValueError(
            "Both H_small_scale and H_large_scale are zero — this is the "
            "identity operator. Use signal.copy() if that's what you want, "
            "or set at least one exponent to a nonzero value."
        )
    if transition_wavelength <= 0:
        raise ValueError(
            f"transition_wavelength must be positive; got {transition_wavelength}."
        )

    signal = B.asarray(signal)
    shape = signal.shape
    ndim = signal.ndim

    if outer_scale is None:
        outer_scale = max(shape)
    elif outer_scale <= 0:
        raise ValueError(f"outer_scale must be positive; got {outer_scale}.")

    _guard_broken_kernel_overflow(
        outer_scale, H_small_scale, H_large_scale, transition_wavelength, ndim
    )

    rfft_shape = shape[:-1] + (shape[-1] // 2 + 1,)
    freqs = _isotropic_rfft_frequencies(shape)
    min_freq = 1.0 / float(outer_scale)
    freqs_reg = B.maximum(freqs, min_freq)
    del freqs

    k_t = 1.0 / float(transition_wavelength)
    small_scale_prefactor = k_t ** (H_small_scale - H_large_scale)

    large_branch = freqs_reg ** (-H_large_scale)
    small_branch = small_scale_prefactor * freqs_reg ** (-H_small_scale)
    kernel = B.where(freqs_reg <= k_t, large_branch, small_branch)
    del freqs_reg, large_branch, small_branch

    if kernel.shape != rfft_shape:
        kernel = kernel + B.zeros(rfft_shape, dtype=kernel.dtype)

    # DC bin: use kernel value at the nearest non-DC bin (no spectral notch).
    dc_index = (0,) * ndim
    if B.numel(kernel) == 1:
        kernel[dc_index] = 1.0
    else:
        neighbor_index = (1,) + (0,) * (ndim - 1)
        kernel[dc_index] = kernel[neighbor_index]

    spectrum = B.rfftn(signal) * kernel
    del kernel
    result = B.irfftn(spectrum, s=shape)
    del spectrum
    return result
