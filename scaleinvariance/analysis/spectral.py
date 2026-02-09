import numpy as np
import warnings

from ..utils import estimate_hurst_from_scaling


def spectral_analysis(data, max_wavelength=None, min_wavelength=None, nbins=50, axis=0):
    """
    Compute binned power spectral density for data along specified axis.
    
    Parameters
    ----------
    data : array-like
        N-dimensional input data array
    max_wavelength : float, optional
        Maximum wavelength to include in analysis
    min_wavelength : float, optional
        Minimum wavelength to include in analysis
    nbins : int, optional
        Number of logarithmic bins for PSD (default: 50)
    axis : int, optional
        Axis along which to compute FFT (default: 0)
        
    Returns
    -------
    tuple (np.ndarray, np.ndarray)
        binned_freq : 1-D array of binned frequencies
        binned_psd : 1-D array of binned power spectral density values
    """
    data = np.asarray(data)
    if data.size == 0:
        raise ValueError("Input data must not be empty")
    if max_wavelength is not None and max_wavelength <= 0:
        raise ValueError("max_wavelength must be positive")
        
    n = data.shape[axis]
    # Compute the one-sided FFT and corresponding power spectral density along specified axis
    fft_vals = np.fft.rfft(data, axis=axis)
    psd = np.abs(fft_vals)**2 / n
    
    # Average PSD across other dimensions
    avg_axes = tuple(i for i in range(data.ndim) if i != axis)
    if avg_axes:
        psd = np.mean(psd, axis=avg_axes)
    
    freqs = np.fft.rfftfreq(n, d=1)  # assuming unit spacing

    # Filter frequencies based on wavelength constraints
    if max_wavelength is not None:
        valid = (freqs > 0) & (freqs >= 1 / max_wavelength)
    else:
        valid = (freqs > 0)
    if min_wavelength is not None:
        valid = valid & (freqs <= 1 / min_wavelength)
    if np.sum(valid) < 2:
        raise ValueError("Not enough frequency points available for the given wavelength range")

    f_valid = freqs[valid]
    p_valid = psd[valid]

    # Create logarithmic bins
    bins = np.logspace(np.log10(f_valid.min()), np.log10(f_valid.max()), nbins + 1, dtype=np.float64)
    
    # Bin the data
    binned_freq = []
    binned_psd = []
    for i in range(nbins):
        in_bin = (f_valid >= bins[i]) & (f_valid < bins[i+1])
        if np.any(in_bin):
            binned_freq.append(f_valid[in_bin].mean())
            binned_psd.append(p_valid[in_bin].mean())

    return np.array(binned_freq, dtype=np.float64), np.array(binned_psd, dtype=np.float64)


def spectral_hurst(data, max_wavelength=None, min_wavelength=None, nbins=50, axis=0, return_fit=False):
    """
    Estimate the Hurst exponent from the power spectral density along specified axis.
    
    For fractional Brownian motion, PSD ~ f^(-β) with β = 2H + 1.
    This function estimates H by performing linear regression on log(PSD) vs log(frequency).
    
    Parameters
    ----------
    data : array-like
        N-dimensional input data array
    max_wavelength : float, optional
        Maximum wavelength to include in analysis. If None, defaults to domain_size/8
        or domain_size if domain_size < 512 (with warning).
    min_wavelength : float, optional  
        Minimum wavelength to include in analysis. If None, defaults to 4*Nyquist
        or Nyquist if domain_size < 512.
    nbins : int, optional
        Number of logarithmic bins for PSD (default: 50)
    axis : int, optional
        Axis along which to compute FFT (default: 0)
    return_fit : bool, optional
        If True, return frequencies, PSD values, and fitted line
        
    Returns
    -------
    If return_fit is False:
        H : float
            Estimated Hurst exponent
        uncertainty : float
            Standard error of the Hurst exponent estimate
    If return_fit is True:
        H : float
            Estimated Hurst exponent
        uncertainty : float
            Standard error of the Hurst exponent estimate
        freqs : np.ndarray
            The frequencies used in the calculation
        psd_values : np.ndarray
            The power spectral density values
        fit_line : np.ndarray
            The fitted line used to estimate H
    """
    data = np.asarray(data)
    array_size = data.shape[axis]
    
    # Check minimum array size for reliable Hurst estimation
    if array_size < 16:
        raise ValueError(f"Array size along axis {axis} is {array_size}, but minimum 16 points required for Hurst estimation")
    
    # Set default wavelengths to isolate scaling range >> grid size and << domain size
    domain_size = array_size  # In grid units
    nyquist_wavelength = 2    # Nyquist wavelength in grid units
    
    if max_wavelength is None:
        if array_size <= 512:
            max_wavelength = domain_size
            warnings.warn(f"Array size ({array_size}) is small. Using max_wavelength={max_wavelength}. "
                         "Consider using larger arrays for more reliable Hurst estimation.",
                         UserWarning)
        else:
            max_wavelength = domain_size / 8
    
    if min_wavelength is None:
        if array_size <= 512:
            min_wavelength = nyquist_wavelength
        else:
            min_wavelength = 4 * nyquist_wavelength
    
    # Compute binned power spectral density
    binned_freq, binned_psd = spectral_analysis(data, max_wavelength, min_wavelength, nbins, axis)
    
    if len(binned_freq) < 2:
        raise ValueError("Not enough frequency bins for Hurst estimation")
    
    # For fBm: PSD ~ f^(-β) where β = 2H + 1
    # So H = ((-slope) - 1) / 2
    # We use the common regression utility, then transform the slope
    slope, slope_err, freqs_fit, psd_fit, fit_line = estimate_hurst_from_scaling(
        binned_freq, binned_psd, return_fit=True
    )
    
    # Transform slope to Hurst exponent: H = ((-slope) - 1) / 2
    H = ((-slope) - 1) / 2
    H_uncertainty = slope_err / 2  # Error propagation
    
    if return_fit:
        return H, H_uncertainty, freqs_fit, psd_fit, fit_line
    else:
        return H, H_uncertainty