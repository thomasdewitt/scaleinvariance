import numpy as np
from .. import backend as B


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

    # Frequency array with proper normalization for physical wavelengths
    k = B.fftfreq(size, d=1.0) * 2 * B.pi
    k[0] = 1  # Avoid division by zero at k=0

    # Power spectrum: |k|^(-β) where β = 2H + 1
    beta = 2 * H + 1
    power_spectrum = B.abs(k) ** (-beta)

    # Random complex phases
    phases = B.rand(size) * 2 * B.pi
    complex_amplitudes = B.sqrt(power_spectrum) * B.exp(1j * phases)

    # Ensure Hermitian symmetry for real output
    if size % 2 == 0:
        complex_amplitudes[size//2] = B.real(complex_amplitudes[size//2]) + 0j
    complex_amplitudes[size//2+1:] = B.conj(B.flip(complex_amplitudes[1:size//2], axis=0))

    # Inverse FFT to get real space
    fBm = B.real(B.ifft(complex_amplitudes))

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

    # Create frequency arrays for each dimension
    freq_arrays = [B.fftfreq(n, d=1.0) * 2 * B.pi for n in working_size]

    # Create meshgrid of frequencies
    freq_grids = B.meshgrid(*freq_arrays, indexing='ij')

    # Compute radial frequency magnitude
    K_squared = sum(kg**2 for kg in freq_grids)
    K = B.sqrt(K_squared)

    # Avoid division by zero at k=0
    zero_idx = (0,) * ndim
    K[zero_idx] = 1.0

    # Power spectrum: |k|^(-β) where β = 2H + D for D dimensions
    beta = 2 * H + ndim
    power_spectrum = K ** (-beta)

    # Use rfftn approach - only need half the last dimension
    last_dim_half = working_size[-1] // 2 + 1
    rfft_shape = working_size[:-1] + (last_dim_half,)

    # Create random phases for the non-redundant portion
    phases = B.rand(*rfft_shape) * 2 * B.pi

    # Extract corresponding power spectrum slice
    slices = tuple(slice(None) for _ in range(ndim - 1)) + (slice(None, last_dim_half),)
    power_spectrum_half = power_spectrum[slices]

    # Create complex amplitudes
    complex_amplitudes = B.sqrt(power_spectrum_half) * B.exp(1j * phases)

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


def fBm_1D(size, H, causal=True, outer_scale=None, outer_scale_width_factor=2.0,
           periodic=True, gaussian_noise=None, kernel_construction_method='naive'):
    """
    Generate 1D fractional Brownian motion by fractional integration of Gaussian noise.

    This method generates fBm by convolving Gaussian white noise with a power-law kernel.
    Supports extended Hurst range H in (-0.5, 1.5).

    Algorithm:
        - For H in (-0.5, 0.5): Convolve noise with kernel |x|^(H - 1/2)
        - For H = 0.5: Integrate noise (standard Brownian motion)
        - For H in (0.5, 1.5): Integrate noise, then convolve with kernel |x|^(H - 3/2)

    Parameters
    ----------
    size : int
        Length of simulation (must be even, power of 2 recommended)
    H : float
        Hurst exponent in (-0.5, 1.5). Controls correlation structure.
        - H in (-0.5, 0.5): Fractional integration with anti-correlation to mild correlation
        - H = 0.5: Standard Brownian motion
        - H in (0.5, 1.5): Integration + fractional integration for smooth processes
    causal : bool, optional
        Use causal kernels (future doesn't affect past). Default True.
    outer_scale : int, optional
        Large-scale cutoff. Defaults to size.
    outer_scale_width_factor : float, optional
        Controls transition width for outer scale. Default is 2.0.
    periodic : bool, optional
        If True (default), returns full periodic simulation suitable for periodic
        boundary conditions. If False, doubles simulation size internally then
        returns only the first half to eliminate periodicity artifacts.
    gaussian_noise : torch.Tensor or numpy.ndarray, optional
        Pre-generated Gaussian noise for reproducibility. Must have same size as simulation.
    kernel_construction_method : str, optional
        Method for constructing convolution kernels. Options:
        - 'naive': Simple power-law kernels without corrections (default)
        - 'LS2010': Lovejoy & Schertzer 2010 finite-size corrections

    Returns
    -------
    numpy.ndarray
        1D array of simulated fBm values (zero mean)

    Raises
    ------
    ValueError
        If size is not even or H is outside valid range (-0.5, 1.5)

    Examples
    --------
    >>> # Standard fBm with H=0.7
    >>> fbm = fBm_1D(2048, H=0.7)
    >>>
    >>> # Anti-correlated process with H=-0.3
    >>> fbm_blue = fBm_1D(2048, H=-0.3)
    >>>
    >>> # Super-smooth process with H=1.2
    >>> fbm_smooth = fBm_1D(2048, H=1.2)

    Notes
    -----
    - Computational complexity is O(N log N) due to FFT-based convolutions
    - Uses naive power-law kernel without finite-size corrections
    - For H in standard range (0, 1), fBm_1D_circulant may be more efficient
    """
    # Import here to avoid circular import
    from .FIF import create_kernel_naive, create_kernel_LS2010, periodic_convolve

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
            # Pad with additional noise for non-periodic mode
            noise = B.concatenate([noise, B.randn(output_size)])

    # Determine processing based on H value
    if H < 0.5:
        # For H in (-0.5, 0.5): convolve with kernel exponent (H + 1/2) - 1 = H - 1/2
        kernel_exponent = H - 0.5

        if kernel_construction_method == 'LS2010':
            # For LS2010, need to calculate norm_ratio_exponent
            norm_ratio_exponent = -kernel_exponent - 1  # This gives the H value
            kernel = create_kernel_LS2010(size, kernel_exponent, norm_ratio_exponent,
                                        causal=causal, outer_scale=outer_scale,
                                        outer_scale_width_factor=outer_scale_width_factor,
                                        final_power=None)
        elif kernel_construction_method == 'naive':
            kernel = create_kernel_naive(size, kernel_exponent, causal=causal,
                                        outer_scale=outer_scale,
                                        outer_scale_width_factor=outer_scale_width_factor)
        else:
            raise ValueError(f"Unknown kernel_construction_method: {kernel_construction_method}")

        result = periodic_convolve(noise, kernel)

        # Apply causality normalization if needed
        if causal:
            result *= 2.0

    elif H == 0.5:
        # For H = 0.5: just integrate (standard Brownian motion)
        result = B.cumsum(noise, axis=0)

    else:  # H > 0.5
        # For H in (0.5, 1.5): integrate, then convolve with kernel exponent (H - 1/2) - 1 = H - 3/2
        integrated = B.cumsum(noise, axis=0)

        kernel_exponent = H - 1.5

        if kernel_construction_method == 'LS2010':
            # For LS2010, need to calculate norm_ratio_exponent
            norm_ratio_exponent = -kernel_exponent - 1  # This gives the H value
            kernel = create_kernel_LS2010(size, kernel_exponent, norm_ratio_exponent,
                                        causal=causal, outer_scale=outer_scale,
                                        outer_scale_width_factor=outer_scale_width_factor,
                                        final_power=None)
        elif kernel_construction_method == 'naive':
            kernel = create_kernel_naive(size, kernel_exponent, causal=causal,
                                        outer_scale=outer_scale,
                                        outer_scale_width_factor=outer_scale_width_factor)
        else:
            raise ValueError(f"Unknown kernel_construction_method: {kernel_construction_method}")

        result = periodic_convolve(integrated, kernel)

        # Apply causality normalization if needed
        if causal:
            result *= 2.0

    # Return first half if non-periodic mode
    if not periodic:
        result = result[:output_size]

    # Normalize to unit std
    result = result - B.mean(result)  # Zero mean first
    result_std = B.std(result)
    if result_std > 1e-10:
        result = result / result_std

    return result
