import torch
import numpy as np

from .. import device


def acausal_fBm(size, H):
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
    
    # Frequency array
    k = torch.fft.fftfreq(size, device=device) * size
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


def fractional_brownian_motion(n_points, hurst, **kwargs):
    raise NotImplementedError