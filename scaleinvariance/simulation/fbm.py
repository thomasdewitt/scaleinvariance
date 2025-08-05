import torch
import numpy as np

from .. import device


def acausal_fBm_1D(size, H):
    """
    Generate fractional Brownian motion by spectral synthesis.
    
    Creates fBm with power spectrum |k|^(-β) where β = 2H + 1.
    Uses random phases and inverse FFT to generate periodic time series.
    
    Parameters
    ----------
    size : int
        Number of points (must be power of 2)
    H : float
        Hurst parameter (0 < H < 1)
        
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
    power_spectrum[0] = 0  # Zero DC component for zero mean
    
    # Random complex phases
    phases = torch.rand(size, device=device) * 2 * torch.pi
    complex_amplitudes = torch.sqrt(power_spectrum) * torch.exp(1j * phases)
    
    # Ensure Hermitian symmetry for real output
    if size % 2 == 0:
        complex_amplitudes[size//2] = complex_amplitudes[size//2].real + 0j
    complex_amplitudes[size//2+1:] = torch.conj(torch.flip(complex_amplitudes[1:size//2], [0]))
    
    # Inverse FFT to get real space
    fBm = torch.fft.ifft(complex_amplitudes).real
    
    return (fBm - torch.mean(fBm)).cpu().numpy()


def acausal_fBm_2D(size, H):
    """
    Generate 2D fractional Brownian motion by spectral synthesis.
    
    Creates 2D fBm with rotationally symmetric power spectrum |k|^(-β) where β = 2H + 2.
    Uses random phases and inverse FFT to generate periodic 2D random field.
    
    Parameters
    ----------
    size : tuple of ints or int
        Size of 2D array. If int, creates square array. Each dimension must be power of 2.
    H : float
        Hurst parameter (0 < H < 1)
        
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
    power_spectrum[0, 0] = 0  # Zero DC component for zero mean
    
    # Use rfft2 approach - only need half the frequency domain
    # Create random phases for the non-redundant half
    phases = torch.rand(nx, ny//2 + 1, device=device) * 2 * torch.pi
    
    # Extract corresponding power spectrum slice
    power_spectrum_half = power_spectrum[:, :ny//2 + 1]
    
    # Create complex amplitudes
    complex_amplitudes = torch.sqrt(power_spectrum_half) * torch.exp(1j * phases)
    
    # Handle special cases for real output
    complex_amplitudes[0, 0] = 0  # DC component = 0 for zero mean
    if ny % 2 == 0:
        complex_amplitudes[:, -1] = complex_amplitudes[:, -1].real + 0j  # Nyquist freq real
    if nx % 2 == 0:
        complex_amplitudes[nx//2, :] = complex_amplitudes[nx//2, :].real + 0j  # Nyquist freq real
        if ny % 2 == 0:
            complex_amplitudes[nx//2, -1] = complex_amplitudes[nx//2, -1].real + 0j
    
    # Inverse FFT to get real space (irfft2 handles Hermitian symmetry automatically)
    fBm_2D = torch.fft.irfft2(complex_amplitudes, s=(nx, ny))
    
    return (fBm_2D - torch.mean(fBm_2D)).cpu().numpy()
