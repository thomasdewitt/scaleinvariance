import torch
import numpy as np

from .. import device


def fBm_1D_circulant(size, H):
    """
    Generate fractional Brownian motion by spectral synthesis.

    Creates fBm with power spectrum |k|^(-β) where β = 2H + 1.
    Uses random phases and inverse FFT to generate periodic time series.

    Parameters
    ----------
    size : int
        Number of points (must be power of 2)
    H : float
        Hurst parameter. Typical range is (0, 1) for fBm, but negative values
        are also supported (H < 0 gives blue noise with β < 1).

    Returns
    -------
    numpy.ndarray
        Generated fBm time series, zero mean, periodic
    """
    if size & (size - 1) != 0:
        raise ValueError("Size must be power of 2")
    
    # Frequency array with proper normalization for physical wavelengths
    k = torch.fft.fftfreq(size, d=1.0, device=device) * 2 * torch.pi
    k[0] = 1  # Avoid division by zero at k=0
    
    # Power spectrum: |k|^(-β) where β = 2H + 1
    beta = 2 * H + 1
    power_spectrum = torch.abs(k) ** (-beta)

    # Random complex phases
    phases = torch.rand(size, device=device) * 2 * torch.pi
    complex_amplitudes = torch.sqrt(power_spectrum) * torch.exp(1j * phases)

    # Ensure Hermitian symmetry for real output
    if size % 2 == 0:
        complex_amplitudes[size//2] = complex_amplitudes[size//2].real + 0j
    complex_amplitudes[size//2+1:] = torch.conj(torch.flip(complex_amplitudes[1:size//2], [0]))

    # Inverse FFT to get real space
    fBm = torch.fft.ifft(complex_amplitudes).real

    # Normalize to unit std
    fBm = fBm - torch.mean(fBm)  # Zero mean first
    fBm_std = torch.std(fBm)
    if fBm_std > 1e-10:
        fBm = fBm / fBm_std

    return fBm.cpu().numpy()


def fBm_2D_circulant(size, H):
    """
    Generate 2D fractional Brownian motion by spectral synthesis.

    Creates 2D fBm with rotationally symmetric power spectrum |k|^(-β) where β = 2H + 2.
    Uses random phases and inverse FFT to generate periodic 2D random field.

    Parameters
    ----------
    size : tuple of ints or int
        Size of 2D array. If int, creates square array. Each dimension must be power of 2.
    H : float
        Hurst parameter. Typical range is (0, 1) for fBm, but negative values
        are also supported (H < 0 gives blue noise with β < 2).

    Returns
    -------
    numpy.ndarray
        Generated 2D fBm field, zero mean, periodic
    """
    if isinstance(size, int):
        nx, ny = size, size
    else:
        nx, ny = size
    
    # Convert to int if needed
    nx, ny = int(nx), int(ny)
        
    if (nx & (nx - 1)) != 0 or (ny & (ny - 1)) != 0:
        raise ValueError("All dimensions must be powers of 2")
    
    # 2D frequency arrays with proper normalization for physical wavelengths
    # Assume unit grid spacing, so frequencies are in cycles per unit length
    kx = torch.fft.fftfreq(nx, d=1.0, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(ny, d=1.0, device=device) * 2 * torch.pi
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    
    # Radial frequency magnitude in physical units
    K = torch.sqrt(KX**2 + KY**2)
    K[0, 0] = 1  # Avoid division by zero at k=0
    
    # Power spectrum: |k|^(-β) where β = 2H + 2 for 2D
    beta = 2 * H + 2
    power_spectrum = K ** (-beta)

    # Use rfft2 approach - only need half the frequency domain
    # Create random phases for the non-redundant half
    phases = torch.rand(nx, ny//2 + 1, device=device) * 2 * torch.pi

    # Extract corresponding power spectrum slice
    power_spectrum_half = power_spectrum[:, :ny//2 + 1]

    # Create complex amplitudes
    complex_amplitudes = torch.sqrt(power_spectrum_half) * torch.exp(1j * phases)

    # Handle special cases for real output
    if ny % 2 == 0:
        complex_amplitudes[:, -1] = complex_amplitudes[:, -1].real + 0j  # Nyquist freq real
    if nx % 2 == 0:
        complex_amplitudes[nx//2, :] = complex_amplitudes[nx//2, :].real + 0j  # Nyquist freq real
        if ny % 2 == 0:
            complex_amplitudes[nx//2, -1] = complex_amplitudes[nx//2, -1].real + 0j

    # Inverse FFT to get real space (irfft2 handles Hermitian symmetry automatically)
    fBm_2D = torch.fft.irfft2(complex_amplitudes, s=(nx, ny))

    # Normalize to unit std
    fBm_2D = fBm_2D - torch.mean(fBm_2D)  # Zero mean first
    fBm_std = torch.std(fBm_2D)
    if fBm_std > 1e-10:
        fBm_2D = fBm_2D / fBm_std

    return fBm_2D.cpu().numpy()


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
        If True, doubles simulation size then returns only first half to eliminate
        periodicity artifacts. If False, returns full periodic simulation. Default True.
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
    if periodic:
        size *= 2  # Double size to eliminate periodicity

    if outer_scale is None:
        outer_scale = output_size

    # Generate or use provided Gaussian white noise
    if gaussian_noise is None:
        noise = torch.randn(size, dtype=torch.float64, device=device)
    else:
        noise_tensor = torch.as_tensor(gaussian_noise, dtype=torch.float64, device=device)
        if noise_tensor.numel() != output_size:
            raise ValueError("Provided gaussian_noise must match the specified size.")
        if periodic:
            # Pad with additional noise
            noise = torch.cat([noise_tensor, torch.randn(output_size, dtype=torch.float64, device=device)])
        else:
            noise = noise_tensor

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
        result = torch.cumsum(noise, dim=0)

    else:  # H > 0.5
        # For H in (0.5, 1.5): integrate, then convolve with kernel exponent (H - 1/2) - 1 = H - 3/2
        integrated = torch.cumsum(noise, dim=0)

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

    # Return first half if periodic mode
    if periodic:
        result = result[:size//2]

    # Normalize to unit std
    result = result - torch.mean(result)  # Zero mean first
    result_std = torch.std(result)
    if result_std > 1e-10:
        result = result / result_std

    return result.cpu().numpy()
