import numpy as np
from .. import backend as B
from .fractional_integration import fractional_integral_spectral


def fBm_1D_circulant(size, H, periodic=True):
    """
    Generate fractional Brownian motion by spectral synthesis.

    Creates fBm with power spectrum |k|^(-β) where β = 2H + 1.
    Uses random phases and inverse FFT.

    Parameters
    ----------
    size : int
        Number of points (must be power of 2)
    H : float
        Hurst parameter. Typical range is (0, 1) for fBm, but negative values
        are also supported (H < 0 gives blue noise with β < 1).
    periodic : bool, optional
        If True (default), returns full periodic simulation suitable for periodic
        boundary conditions. If False, doubles simulation size internally then
        returns only the first half to eliminate periodicity artifacts.

    Returns
    -------
    numpy.ndarray
        Generated 1D fBm, zero mean, unit std
    """
    if size & (size - 1) != 0:
        raise ValueError("Size must be power of 2")

    output_size = size
    if not periodic:
        size *= 2  # Double size to eliminate periodicity artifacts

    # Build the half-spectrum (DC + positive freqs + Nyquist) directly and
    # use irfft instead of materialising a full-size Hermitian array.
    k_half = B.rfftfreq(size, d=1.0) * 2 * B.pi
    k_half[0] = 1.0  # avoid division by zero at DC

    beta = 2 * H + 1
    power_half = B.abs(k_half) ** (-beta)
    del k_half

    phases = B.rand(power_half.size) * 2 * B.pi
    amplitude = B.sqrt(power_half)
    del power_half

    complex_half = (amplitude * B.cos(phases)) + 1j * (amplitude * B.sin(phases))
    del amplitude, phases
    # Imaginary part of DC and (if size is even) Nyquist must be zero for
    # the resulting real sequence; irfft enforces this by construction,
    # but explicitly zeroing keeps behaviour identical to the legacy path.
    complex_half[0] = complex_half[0].real + 0j
    if size % 2 == 0:
        complex_half[-1] = complex_half[-1].real + 0j

    fBm = B.irfft(complex_half, n=size)
    del complex_half

    # Return first half if non-periodic mode
    if not periodic:
        fBm = fBm[:output_size]

    # Normalize to unit std
    fBm = fBm - B.mean(fBm)  # Zero mean first
    fBm_std = B.std(fBm)
    if fBm_std > 1e-10:
        fBm = fBm / fBm_std

    return fBm


def fBm_ND_circulant(size, H, periodic=True):
    """
    Generate N-dimensional fractional Brownian motion by spectral synthesis.

    Creates N-D fBm with rotationally symmetric power spectrum |k|^(-β) where β = 2H + D.
    Uses random phases and inverse FFT to generate N-D random field.

    Parameters
    ----------
    size : tuple of ints
        Size of N-D array. Each dimension must be power of 2.
    H : float
        Hurst parameter. Typical range is (0, 1) for fBm, but negative values
        are also supported (H < 0 gives blue noise with β < D).
    periodic : bool or tuple of bool, optional
        If True (default), returns full periodic simulation suitable for periodic
        boundary conditions. If False, doubles simulation size internally along
        each dimension then returns only the first octant to eliminate
        periodicity artifacts. Can also be a tuple of bool for per-axis control.

    Returns
    -------
    numpy.ndarray
        Generated N-D fBm field, zero mean, unit std
    """
    size = tuple(int(s) for s in size)
    ndim = len(size)

    # Handle periodic as bool or tuple
    if isinstance(periodic, bool):
        periodic = (periodic,) * ndim
    else:
        periodic = tuple(periodic)
        if len(periodic) != ndim:
            raise ValueError(f"periodic must be bool or tuple of length {ndim}")

    # Validate dimensions are powers of 2
    for i, s in enumerate(size):
        if (s & (s - 1)) != 0:
            raise ValueError(f"Dimension {i} size {s} must be power of 2")

    # Compute output and working sizes
    output_size = size
    working_size = tuple(s * 2 if not p else s for s, p in zip(size, periodic))

    # Build the half-size (rfft-packed) squared frequency volume via
    # broadcasted 1D axes — never materialise D full-size coordinate grids.
    last_dim_half = working_size[-1] // 2 + 1
    rfft_shape = working_size[:-1] + (last_dim_half,)

    K_squared = None
    for axis_idx in range(ndim):
        if axis_idx == ndim - 1:
            axis = B.rfftfreq(working_size[-1], d=1.0) * 2 * B.pi
        else:
            axis = B.fftfreq(working_size[axis_idx], d=1.0) * 2 * B.pi
        broadcast_shape = [1] * ndim
        broadcast_shape[axis_idx] = axis.size
        term = axis.reshape(broadcast_shape) ** 2
        if K_squared is None:
            K_squared = term
        else:
            K_squared = K_squared + term
        del axis, term

    K = B.sqrt(K_squared)
    del K_squared

    # Materialise so in-place writes below touch a real allocation, not a view.
    if K.shape != rfft_shape:
        K = K + B.zeros(rfft_shape, dtype=K.dtype)

    # Avoid division by zero at k=0
    zero_idx = (0,) * ndim
    K[zero_idx] = 1.0

    # Power spectrum: |k|^(-β) where β = 2H + D for D dimensions
    beta = 2 * H + ndim
    power_spectrum_half = K ** (-beta)
    del K

    # Random phases for the non-redundant portion
    phases = B.rand(*rfft_shape) * 2 * B.pi

    # Create complex amplitudes
    amplitude = B.sqrt(power_spectrum_half)
    del power_spectrum_half
    complex_amplitudes = (amplitude * B.cos(phases)) + 1j * (amplitude * B.sin(phases))
    del amplitude, phases

    # Handle Nyquist frequencies for real output
    # Last dimension Nyquist (if even)
    if working_size[-1] % 2 == 0:
        slices_nyq = tuple(slice(None) for _ in range(ndim - 1)) + (-1,)
        complex_amplitudes[slices_nyq] = B.real(complex_amplitudes[slices_nyq]) + 0j

    # Other dimensions Nyquist planes (if even)
    for d in range(ndim - 1):
        if working_size[d] % 2 == 0:
            slices_nyq = tuple(slice(None) if i != d else working_size[d] // 2
                              for i in range(ndim - 1)) + (slice(None),)
            complex_amplitudes[slices_nyq] = B.real(complex_amplitudes[slices_nyq]) + 0j

    # Inverse FFT to get real space
    fBm_ND = B.irfftn(complex_amplitudes, s=working_size)

    # Return first octant if any dimension is non-periodic
    if not all(periodic):
        slices_output = tuple(slice(None, os) for os in output_size)
        fBm_ND = fBm_ND[slices_output]

    # Normalize to unit std
    fBm_ND = fBm_ND - B.mean(fBm_ND)
    fBm_std = B.std(fBm_ND)
    if fBm_std > 1e-10:
        fBm_ND = fBm_ND / fBm_std

    return fBm_ND


def fBm_1D(size, H, causal=False, outer_scale=None, outer_scale_width_factor=2.0,
           periodic=True, gaussian_noise=None, kernel_construction_method='LS2010'):
    """
    Generate 1D fractional Brownian motion by fractional integration of Gaussian noise.

    Supports extended Hurst range H in (-0.5, 1.5).

    - ``causal=False`` (default): spectral fractional integration of order
      ``H + 1/2``. White noise has flat PSD, so multiplying its spectrum by
      ``|f|^(-(H+1/2))`` gives fBm with target power spectrum
      ``|f|^(-(2H+1))``. Handles every H in one pass — no branching on
      ``H < 0.5`` / ``H == 0.5`` / ``H > 0.5``.
    - ``causal=True``: real-space convolution with a corrected causal
      power-law kernel (LS2010 or naive). Branches on H:
      * ``H in (-0.5, 0.5)``: convolve noise with ``|x|^(H - 1/2)``
      * ``H == 0.5``: cumsum of noise (standard Brownian motion)
      * ``H in (0.5, 1.5)``: cumsum then convolve with ``|x|^(H - 3/2)``

    Parameters
    ----------
    size : int
        Length of simulation (must be even, power of 2 recommended)
    H : float
        Hurst exponent in (-0.5, 1.5). Controls correlation structure.
    causal : bool, optional
        Use causal kernels (future doesn't affect past). Default False.
        When False, the fast spectral fractional-integral path is used and
        ``kernel_construction_method`` is ignored.
    outer_scale : int, optional
        Large-scale cutoff. Defaults to size.
    outer_scale_width_factor : float, optional
        Controls transition width for outer scale (causal path only).
        Default is 2.0.
    periodic : bool, optional
        If True (default), returns full periodic simulation. If False,
        doubles the simulation size internally then returns the first half
        to eliminate periodicity artifacts.
    gaussian_noise : ndarray, optional
        Pre-generated Gaussian noise for reproducibility. Must have same
        size as the requested output.
    kernel_construction_method : {'LS2010', 'naive'}, optional
        Kernel construction method for the causal path. Ignored when
        ``causal=False``.

    Returns
    -------
    numpy.ndarray
        1D array of simulated fBm values (zero mean, unit std).

    Raises
    ------
    ValueError
        If size is not even, H is outside (-0.5, 1.5), or an unknown causal
        kernel_construction_method is given.

    Examples
    --------
    >>> # Standard acausal fBm with H=0.7
    >>> fbm = fBm_1D(2048, H=0.7)
    >>>
    >>> # Anti-correlated process with H=-0.3
    >>> fbm_blue = fBm_1D(2048, H=-0.3)
    >>>
    >>> # Causal LS2010 path
    >>> fbm_causal = fBm_1D(2048, H=0.7, causal=True)

    Notes
    -----
    - Computational complexity is O(N log N).
    - For the acausal path, the fractional-integration kernel is
      ``|f|^(-(H+1/2))``; for causal it is the real-space Riesz kernel.
    - ``fBm_1D_circulant`` is an alternative (phases + power spectrum) that
      is only slightly cheaper but produces statistically equivalent output.
    """
    if size % 2 != 0:
        raise ValueError("size must be an even number; a power of 2 is recommended.")

    if not isinstance(H, (int, float)) or H <= -0.5 or H >= 1.5:
        raise ValueError("H must be a number in (-0.5, 1.5).")

    output_size = size
    if not periodic:
        size *= 2  # Double size to eliminate periodicity artifacts

    if outer_scale is None:
        outer_scale = output_size

    # Generate or use provided Gaussian white noise
    if gaussian_noise is None:
        noise = B.randn(size)
    else:
        noise = B.asarray(gaussian_noise)
        if noise.size != output_size:
            raise ValueError("Provided gaussian_noise must match the specified size.")
        if not periodic:
            noise = B.concatenate([noise, B.randn(output_size)])

    if not causal:
        # White noise → fBm with Hurst H is a Riesz fractional integral of
        # order (H + 1/2): |f|^(-(H+1/2)) on a flat spectrum gives |f|^(-(2H+1)).
        result = fractional_integral_spectral(noise, H + 0.5, outer_scale=outer_scale)
    else:
        # Causal path: real-space convolution with LS2010/naive kernels.
        from .kernels import create_kernel_naive, create_kernel_LS2010
        from .FIF import periodic_convolve

        def _build_causal_kernel(kernel_exponent):
            if kernel_construction_method == 'LS2010':
                norm_ratio_exponent = -kernel_exponent - 1
                return create_kernel_LS2010(size, kernel_exponent, norm_ratio_exponent,
                                            causal=True, outer_scale=outer_scale,
                                            outer_scale_width_factor=outer_scale_width_factor,
                                            final_power=None)
            if kernel_construction_method == 'naive':
                return create_kernel_naive(size, kernel_exponent, causal=True,
                                           outer_scale=outer_scale,
                                           outer_scale_width_factor=outer_scale_width_factor)
            raise ValueError(
                f"Unknown kernel_construction_method: {kernel_construction_method}. "
                f"Causal fBm_1D supports 'LS2010' and 'naive'."
            )

        if H < 0.5:
            kernel = _build_causal_kernel(H - 0.5)
            result = periodic_convolve(noise, kernel) * 2.0
        elif H == 0.5:
            result = B.cumsum(noise, axis=0)
        else:  # H > 0.5
            integrated = B.cumsum(noise, axis=0)
            kernel = _build_causal_kernel(H - 1.5)
            result = periodic_convolve(integrated, kernel) * 2.0

    # Return first half if non-periodic mode
    if not periodic:
        result = result[:output_size]

    # Normalize to unit std
    result = result - B.mean(result)
    result_std = B.std(result)
    if result_std > 1e-10:
        result = result / result_std

    return result
