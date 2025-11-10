import torch
from torch.fft import fft as torch_fft, ifft as torch_ifft
import numpy as np
from .fbm import fBm_1D_circulant, fBm_2D_circulant

device = torch.device("cpu")
dtype = torch.float64

def extremal_levy(alpha, size=1):
    """
    Generate random samples from an extremal Lévy distribution using a modified 
    Chambers–Mallows–Stuck method.

    Translated from Mathematica code provided by Lovejoy: 
    https://www.physics.mcgill.ca/~gang/multifrac/multifractals/software.htm

    Parameters:
        alpha (float): Stability parameter in (0, 2)
        size (int): Number of samples to generate (default is 1)

    Returns:
        ndarray: Array of generated extremal Lévy random variables
    """
    phi = (torch.rand(size, dtype=dtype, device=device) - 0.5) * torch.pi
    alpha_t = torch.tensor(alpha, dtype=dtype, device=device)
    phi0 = -(torch.pi/2) * (1 - torch.abs(1 - alpha_t)) / alpha_t
    R = torch.distributions.Exponential(1).sample((size,)).to(device)
    eps = 1e-12
    cos_phi = torch.cos(phi)
    cos_phi = torch.where(torch.abs(cos_phi) < eps, torch.full_like(cos_phi, eps), cos_phi)
    abs_alpha1 = torch.abs(alpha_t - 1)
    abs_alpha1 = torch.where(abs_alpha1 < eps, torch.full_like(abs_alpha1, eps), abs_alpha1)
    denom = torch.cos(phi - alpha_t * (phi - phi0))
    denom = torch.where(torch.abs(denom) < eps, torch.full_like(denom, eps), denom)
    R = torch.where(R < eps, torch.full_like(R, eps), R)
    sample = (
        torch.sign(alpha_t - 1) *
        torch.sin(alpha_t * (phi - phi0)) *
        (cos_phi * abs_alpha1) ** (-1/alpha_t) *
        (denom / R) ** ((1 - alpha_t) / alpha_t)
    )
    return sample

# LS 2010 kernels

def _apply_outer_scale(kernel, distance, outer_scale, width_factor=2.0):
    """
    Apply outer scale cutoff using a Hanning window transition.

    Parameters
    ----------
    kernel : torch.Tensor
        The kernel to apply outer scale to
    distance : torch.Tensor
        Distance array (same shape as kernel)
    outer_scale : float
        Center of the transition region
    width_factor : float, optional
        Controls transition width. Transition width = outer_scale * width_factor.
        Default is 2.0.

    Returns
    -------
    torch.Tensor
        Kernel with outer scale applied
    """
    if not isinstance(distance, torch.Tensor):
        distance = torch.as_tensor(distance, device=device, dtype=dtype)

    # Transition region centered at outer_scale with width = outer_scale * width_factor
    transition_width = outer_scale * width_factor
    lower_edge = outer_scale - transition_width / 2.0
    upper_edge = outer_scale + transition_width / 2.0

    # Compute normalized distance: 0 at lower_edge, 1 at upper_edge
    # Clamp to [0, 1] so values below lower_edge → 0, above upper_edge → 1
    normalized_dist = torch.clamp((distance - lower_edge) / transition_width, 0.0, 1.0)

    # Hanning window: 1 when normalized_dist=0, 0 when normalized_dist=1
    window = 0.5 * (1.0 + torch.cos(torch.pi * normalized_dist))

    return kernel * window

def _apply_LS2010_correction(distance, exponent, norm_ratio_exponent, final_power=None):
    """
    Apply Lovejoy & Schertzer 2010 finite-size correction to a power-law kernel.

    This implements the correction method from Lovejoy & Schertzer (2010)
    "On the simulation of continuous in scale universal multifractals, Part II",
    following Lovejoy's Frac.m implementation.

    The method reduces leading-order finite-size correction terms by applying
    exponential cutoffs and normalization corrections to reduce boundary effects.

    Works for both 1D and 2D distance arrays with dx=2 spacing.

    Parameters
    ----------
    distance : torch.Tensor
        Distance array (1D or 2D), expects dx=2 spacing
    exponent : float
        Power-law exponent for base kernel (e.g., -1/α' for flux, -1+H for H kernel)
    norm_ratio_exponent : float
        Exponent for ratio in normalization factor (e.g., -1/α for flux, -H for H kernel)
    final_power : float, optional
        Final power transformation to apply (e.g., 1/(α-1) for flux kernel).
        If None, no transformation is applied (kernel remains as corrected).

    Returns
    -------
    torch.Tensor
        Corrected kernel
    """
    if not isinstance(distance, torch.Tensor):
        distance = torch.as_tensor(distance, device=device, dtype=dtype)

    # Determine domain size for cutoff calculation
    if distance.ndim == 1:
        domain_size = distance.numel()
    elif distance.ndim == 2:
        domain_size = min(distance.shape)
    else:
        raise ValueError("Distance array must be 1D or 2D")

    ratio = 2.0
    cutoff_length = domain_size / 2.0
    cutoff_length2 = cutoff_length / ratio

    # Calculate exponential cutoffs
    exponential_cutoff = torch.exp(torch.clamp(-(distance/cutoff_length)**4, max=0, min=-200))
    exponential_cutoff2 = torch.exp(torch.clamp(-(distance/cutoff_length2)**4, max=0, min=-200))

    # Base singularity kernel
    base_kernel = distance**exponent

    # Calculate normalization constants
    smoothed_kernel1 = base_kernel * exponential_cutoff
    norm_constant1 = torch.sum(smoothed_kernel1)

    smoothed_kernel2 = base_kernel * exponential_cutoff2
    norm_constant2 = torch.sum(smoothed_kernel2)

    # Normalization factor
    ratio_factor = ratio**norm_ratio_exponent
    normalization_factor = (ratio_factor * norm_constant1 - norm_constant2) / (ratio_factor - 1)

    # Final smoothing filter
    final_filter = torch.exp(torch.clamp(-distance/3.0, max=0, min=-200))
    smoothed_kernel = base_kernel * final_filter
    filter_integral = torch.sum(smoothed_kernel)

    # Apply correction
    correction_factor = -normalization_factor / filter_integral
    corrected_kernel = base_kernel * (1 + correction_factor * final_filter)

    # Apply final power transformation if specified
    if final_power is not None:
        corrected_kernel = corrected_kernel ** final_power

    return corrected_kernel

def create_kernel_LS2010(size, exponent, norm_ratio_exponent, causal=False, outer_scale=None,
                         outer_scale_width_factor=2.0, final_power=None):
    """
    Create kernel using Lovejoy & Schertzer 2010 finite-size corrections.

    Unified function for creating both flux and H kernels with LS2010 corrections.
    Handles both 1D and 2D cases automatically.

    Parameters
    ----------
    size : int or tuple
        For 1D: int specifying kernel size
        For 2D: tuple (height, width)
    exponent : float
        Power-law exponent for base kernel
        Examples: -1/α' for 1D flux, -2/α' for 2D flux, -1+H for 1D H kernel, -2+H for 2D H kernel
    norm_ratio_exponent : float
        Exponent for ratio in normalization factor
        Examples: -1/α for 1D flux, -2/α for 2D flux, -H for H kernels
    causal : bool, optional
        Whether to make kernel causal (1D only). Default False.
    outer_scale : int, optional
        Large-scale cutoff. If None, no outer scale cutoff is applied.
    outer_scale_width_factor : float, optional
        Controls transition width for outer scale. Transition width = outer_scale * width_factor.
        Default is 2.0.
    final_power : float, optional
        Final power transformation (e.g., 1/(α-1) for flux kernel). Default None.

    Returns
    -------
    torch.Tensor
        Corrected kernel ready for convolution
    """
    # Handle 1D vs 2D
    if isinstance(size, int):
        # 1D case
        position_range = torch.arange(-(size - 1), size, 2, dtype=dtype, device=device)
        distance = torch.abs(position_range)
        kernel = _apply_LS2010_correction(distance, exponent, norm_ratio_exponent, final_power)

        # Apply outer scale cutoff with Hanning window (convert distance to units of dx=1)
        if outer_scale is not None:
            kernel = _apply_outer_scale(kernel, distance / 2, outer_scale, outer_scale_width_factor)

        # Apply causality
        if causal:
            kernel[:size//2] = 0

    else:
        # 2D case
        height, width = size
        y_coords = torch.arange(-(height - 1), height, 2, dtype=dtype, device=device)
        x_coords = torch.arange(-(width - 1), width, 2, dtype=dtype, device=device)
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        distance = torch.sqrt(X**2 + Y**2)
        kernel = _apply_LS2010_correction(distance, exponent, norm_ratio_exponent, final_power)

        # Apply outer scale cutoff with Hanning window (convert distance to units of dx=1)
        if outer_scale is not None:
            kernel = _apply_outer_scale(kernel, distance / 2, outer_scale, outer_scale_width_factor)

    return kernel

# Naive kernel construction methods

def create_kernel_naive(size, exponent, causal=False, outer_scale=None, outer_scale_width_factor=2.0):
    """
    Create kernel using simple power-law (no finite-size corrections).

    Unified function for creating both flux and H kernels with naive power-law.
    Currently only supports 1D.

    Parameters
    ----------
    size : int
        Size of kernel array (1D only for naive method)
    exponent : float
        Power-law exponent for kernel
        Examples: -1/α for flux, -1+H for H kernel
    causal : bool, optional
        Whether to make kernel causal. Default False.
    outer_scale : int, optional
        Large-scale cutoff. If None, no outer scale cutoff is applied.
    outer_scale_width_factor : float, optional
        Controls transition width for outer scale. Transition width = outer_scale * width_factor.
        Default is 2.0.

    Returns
    -------
    torch.Tensor
        Power-law kernel ready for convolution
    """
    # Create distance array (dx=1 spacing; avoid singularity by adding 1/2)
    distance = torch.abs(torch.arange(-size//2, size//2, dtype=dtype, device=device)+0.5)

    kernel = distance ** exponent

    # Apply outer scale cutoff with Hanning window
    if outer_scale is not None:
        kernel = _apply_outer_scale(kernel, distance, outer_scale, outer_scale_width_factor)

    # Apply causality
    if causal:
        kernel[:size//2] = 0

    return kernel

# Convolutions
def periodic_convolve(signal, kernel):
    """
    Performs periodic convolution of two 1D arrays using Fourier methods.
    Both arrays must have the same length.

    Parameters:
        signal (array-like): Input signal array.
        kernel (array-like): Convolution kernel array.

    Returns:
        ndarray: The result of the periodic convolution.

    Raises:
        ValueError: If signal and kernel lengths do not match.
    """
    if not isinstance(signal, torch.Tensor):
        signal = torch.as_tensor(signal, device=device, dtype=dtype)
    if not isinstance(kernel, torch.Tensor):
        kernel = torch.as_tensor(kernel, device=device, dtype=dtype)

    if signal.numel() != kernel.numel():
        raise ValueError("Signal and kernel must have the same length for periodic convolution.")

    # Shift kernel so zero-lag is at index 0 (required for FFT convolution)
    kernel = torch.fft.ifftshift(kernel)

    fft_signal = torch_fft(signal)
    fft_kernel = torch_fft(kernel)

    convolved = torch_ifft(fft_signal * fft_kernel)
    return convolved.real

def periodic_convolve_2d(signal, kernel):
    """
    Performs periodic convolution of two 2D arrays using Fourier methods.
    Both arrays must have the same shape.

    Parameters:
        signal (torch.Tensor): Input signal array (2D).
        kernel (torch.Tensor): Convolution kernel array (2D).

    Returns:
        torch.Tensor: The result of the periodic convolution.

    Raises:
        ValueError: If signal and kernel shapes do not match.
    """
    if not isinstance(signal, torch.Tensor):
        signal = torch.as_tensor(signal, device=device, dtype=dtype)
    if not isinstance(kernel, torch.Tensor):
        kernel = torch.as_tensor(kernel, device=device, dtype=dtype)

    if signal.shape != kernel.shape:
        raise ValueError("Signal and kernel must have the same shape for periodic convolution.")

    # Shift kernel so zero-lag is at index (0,0) (required for FFT convolution)
    kernel = torch.fft.ifftshift(kernel)

    fft_signal = torch.fft.fft2(signal)
    fft_kernel = torch.fft.fft2(kernel)

    convolved = torch.fft.ifft2(fft_signal * fft_kernel)
    return convolved.real

# FIF

def FIF_1D(size, alpha, C1, H, levy_noise=None, causal=True, outer_scale=None,
           outer_scale_width_factor=2.0, kernel_construction_method='LS2010', periodic=True):
    """
    Generate a 1D Fractionally Integrated Flux (FIF) multifractal simulation.

    FIF is a multiplicative cascade model that generates multifractal fields with
    specified Hurst exponent H, intermittency parameter C1, and Levy index alpha.
    The method follows Lovejoy & Schertzer 2010, including finite-size corrections.

    Returns field normalized by mean.

    Algorithm:
        1. Generate extremal Lévy noise with stability parameter alpha
        2. Convolve with corrected kernel (finite-size corrected |k|^(-1/alpha)) to create log-flux
        3. Scale by (C1)^(1/alpha) and exponentiate to get conserved flux
        4. For H ≠ 0: Convolve flux with kernel |k|^(-1+H) to get observable
        5. For H < 0: Apply differencing to handle negative Hurst exponents

    Parameters
    ----------
    size : int
        Length of simulation (must be even, power of 2 recommended)
    alpha : float
        Lévy stability parameter in (0, 2) and != 1. Controls noise distribution.
    C1 : float
        Codimension of the mean, controls intermittency strength.
        Must be > 0 for multifractal behavior.
    H : float
        Hurst exponent in (-1, 1). Controls correlation structure.
    levy_noise : torch.Tensor, optional
        Pre-generated Lévy noise for reproducibility. Must have same size as simulation.
    causal : bool, optional
        Use causal kernels (future doesn't affect past). Default True.
        False gives symmetric (non-causal) kernels.
    outer_scale : int, optional
        Large-scale cutoff. Defaults to size.
    outer_scale_width_factor : float, optional
        Controls transition width for outer scale. Transition width = outer_scale * width_factor.
        Default is 2.0.
    kernel_construction_method : str, optional
        Method for constructing convolution kernels. Options:
        - 'LS2010': Lovejoy & Schertzer 2010 finite-size corrections (default)
        - 'naive': Simple power-law kernels without corrections
    periodic : bool, optional
        If True, doubles simulation size then returns only first half to eliminate periodicity artifacts.
        If False, returns full periodic simulation. Default True.

    Returns
    -------
    numpy.ndarray
        1D array of simulated multifractal field values

    Raises
    ------
    ValueError
        If size is not even or H is outside valid range (-1, 1)
    ValueError
        If C1 <= 0 (must be positive for multifractal behavior)
    ValueError
        If provided levy_noise doesn't match specified size

    Examples
    --------
    >>> # Basic multifractal with strong intermittency
    >>> fif = FIF_1D(1024, alpha=1.8, C1=0.1, H=0.3)

    Notes
    -----
    - Computational complexity is O(N log N) due to FFT-based convolutions
    - Large C1 values (> 0.5) can produce extreme values requiring careful handling
    """
    if size % 2 != 0:
        raise ValueError("size must be an even number; a power of 2 is recommended.")

    output_size = size
    if not periodic:
        size *= 2   # duplicate to eliminate periodicity

    if C1 == 0 and not causal:
        if levy_noise is not None:
            raise ValueError('noise argument not supported for C1=0')
        result = fBm_1D_circulant(size, H)
        return result[:output_size] if periodic else result
    
    if not isinstance(C1, (int, float)) or C1 <= 0:
        raise ValueError("C1 must be a positive number.")
    
    H_int = 0
    if H < 0:
        H_int = -1
        H += 1

    if not isinstance(H, (int, float)) or H < 0 or H > 1:
        raise ValueError("H must be a number between -1 and 1.")
    

    if not isinstance(alpha, (int, float)) or alpha <= 0 or alpha > 2:
        raise ValueError("alpha must be a number > 0 and <= 2.")
    
    if alpha == 1: 
        raise ValueError("alpha=1 not supported")   # requires special treatment which is not implemented

    if levy_noise is None:
        noise = extremal_levy(alpha, size=size)
    else:
        if levy_noise.size()[0] != output_size:
            raise ValueError("Provided levy_noise must match the specified size.")
        # if aperiodic is requested, need to pad noise with more noise 
        if not periodic:
            noise = torch.cat([torch.as_tensor(levy_noise, device=device, dtype=dtype),extremal_levy(alpha, size=output_size)])
        else:
            noise = torch.as_tensor(levy_noise, device=device, dtype=dtype)
    if outer_scale is None:
        outer_scale = output_size

    # Create flux kernel (kernel 1)
    # Calculate exponent and normalization parameters for flux kernel
    alpha_prime = 1.0 / (1.0 - 1.0/alpha)
    flux_exponent = -1.0 / alpha_prime
    flux_norm_ratio_exp = -1.0 / alpha
    flux_final_power = 1.0 / (alpha - 1.0)

    if kernel_construction_method == 'LS2010':
        kernel1 = create_kernel_LS2010(size, flux_exponent, flux_norm_ratio_exp,
                                      causal=causal, outer_scale=outer_scale,
                                      outer_scale_width_factor=outer_scale_width_factor,
                                      final_power=flux_final_power)
    elif kernel_construction_method == 'naive':
        kernel1 = create_kernel_naive(size, flux_exponent, causal=causal, outer_scale=outer_scale,
                                     outer_scale_width_factor=outer_scale_width_factor)
    else:
        raise ValueError(f"Unknown kernel_construction_method: {kernel_construction_method}")

    integrated = periodic_convolve(noise, kernel1)
    del noise, kernel1  # Clean memory

    # If causal, adjust for the fact that half the kernel is being deleted
    if causal:
        causality_factor = 2.0
    else:
        causality_factor = 1.0


    scaled = integrated * ((causality_factor * C1) ** (1/alpha))
    del integrated
    flux = torch.exp(scaled)
    del scaled

    if H == 0:
        # Normalize - slice first (if periodic), then normalize by mean
        if not periodic:
            flux = flux[:size//2]     # eliminate periodicity by removing the part corresponding to the appended noise
        flux = flux / torch.mean(flux)
        return flux.cpu().numpy()

    # Create H kernel (kernel 2)
    # Calculate exponent and normalization parameters for H kernel
    H_exponent = -1.0 + H
    H_norm_ratio_exp = -H

    if kernel_construction_method == 'LS2010':
        kernel2 = create_kernel_LS2010(size, H_exponent, H_norm_ratio_exp,
                                      causal=causal, outer_scale=outer_scale,
                                      outer_scale_width_factor=outer_scale_width_factor,
                                      final_power=None)
    elif kernel_construction_method == 'naive':
        kernel2 = create_kernel_naive(size, H_exponent, causal=causal, outer_scale=outer_scale,
                                     outer_scale_width_factor=outer_scale_width_factor)
    else:
        raise ValueError(f"Unknown kernel_construction_method: {kernel_construction_method}")

    observable = periodic_convolve(flux, kernel2)
    if not periodic:
        observable = observable[:size//2]     # eliminate periodicity by removing the part corresponding to the appended noise
    del flux, kernel2

    if H_int == -1:
        # Apply differencing for H<0
        observable = torch.diff(observable)
        # Duplicate last value to maintain original size
        observable = torch.cat([observable, observable[-1:]])
        # Normalize to zero mean (increments should have zero mean)
        observable = observable - torch.mean(observable)
    else:
        # Normalize to unit mean (levels should have unit mean)
        observable = observable / torch.mean(observable)

    return observable.cpu().numpy()

def FIF_2D(size, alpha, C1, H, levy_noise=None, outer_scale=None, outer_scale_width_factor=2.0,
           kernel_construction_method='LS2010', periodic=False):
    """
    Generate a 2D Fractionally Integrated Flux (FIF) multifractal simulation.
    
    FIF is a multiplicative cascade model that generates multifractal fields with
    specified Hurst exponent H, intermittency parameter C1, and Levy index alpha. 
    The method follows Lovejoy & Schertzer 2010, including finite-size corrections.
    Returns field normalized by mean.
    
    Algorithm:
        1. Generate 2D extremal Lévy noise with stability parameter alpha
        2. Convolve with corrected kernel (finite-size corrected |r|^(-2/alpha)) to create log-flux
        3. Scale by (C1)^(1/alpha) and exponentiate to get conserved flux  
        4. For H ≠ 0: Convolve flux with kernel |r|^(-2+H) to get observable
    
    Parameters
    ----------
    size : int or tuple of ints
        Size of simulation. If int, creates square array (size x size).
        If tuple, specifies (height, width). Must be even numbers.
    alpha : float
        Lévy stability parameter in (0, 2) and != 1. Controls noise distribution.
    C1 : float
        Codimension of the mean, controls intermittency strength. 
        Must be > 0 for multifractal behavior.
    H : float
        Hurst exponent in (0, 1). Controls correlation structure.
    levy_noise : torch.Tensor, optional
        Pre-generated 2D Lévy noise for reproducibility. Must have same shape as simulation.
    outer_scale : int, optional
        Large-scale cutoff. Defaults to max(height, width).
    outer_scale_width_factor : float, optional
        Controls transition width for outer scale. Transition width = outer_scale * width_factor.
        Default is 2.0.
    kernel_construction_method : str, optional
        Method for constructing convolution kernels. Options:
        - 'LS2010': Lovejoy & Schertzer 2010 finite-size corrections (default)
        Note: 'naive' method is not yet implemented for 2D.
    periodic : bool, optional
        If False (default), doubles domain size internally and returns one quadrant to
        suppress periodic artifacts. If True, keeps the simulation strictly periodic
        with the provided size.
    
    Returns
    -------
    numpy.ndarray
        2D array of simulated multifractal field values
        
    Raises
    ------
    ValueError
        If size dimensions are not even or H is outside valid range (0, 1)
    ValueError
        If C1 <= 0 (must be positive for multifractal behavior)
    ValueError
        If provided levy_noise doesn't match specified size
        
    Examples
    --------
    >>> # Basic 2D multifractal with strong intermittency  
    >>> fif = FIF_2D(512, alpha=1.8, C1=0.1, H=0.3)
    >>> 
    >>> # Rectangular domain
    >>> fif = FIF_2D((256, 512), alpha=1.5, C1=0.05, H=0.2)
    
    Notes
    -----
    - Computational complexity is O(N log N) due to FFT-based convolutions
    - Large C1 values (> 0.5) can produce extreme values requiring careful handling
    - Always uses finite-size corrections and non-causal kernels for 2D
    - Does not support negative H values (use FIF_1D for H < 0)
    """
    # Handle size parameter
    if isinstance(size, int):
        output_height, output_width = size, size
    else:
        output_height, output_width = size

    # Determine simulation domain size
    sim_height = output_height if periodic else output_height * 2
    sim_width = output_width if periodic else output_width * 2

    if C1 == 0:
        fbm = fBm_2D_circulant((sim_height, sim_width), H)
        return fbm if periodic else fbm[:output_height, :output_width]

    if outer_scale is None:
        outer_scale = max(sim_height, sim_width)
        
    if sim_height % 2 != 0 or sim_width % 2 != 0:
        raise ValueError("Height and width must be even numbers; powers of 2 are recommended.")

    if not isinstance(C1, (int, float)) or C1 <= 0:
        raise ValueError("C1 must be a positive number.")

    if not isinstance(alpha, (int, float)) or alpha <= 0 or alpha > 2:
        raise ValueError("alpha must be a number > 0 and <= 2.")
    
    if alpha == 1: 
        raise ValueError("alpha=1 not supported")   # requires special treatment which is not implemented


    if not isinstance(H, (int, float)) or H < 0 or H > 1:
        raise ValueError("H must be a number between 0 and 1.")

    if levy_noise is None:
        noise = extremal_levy(alpha, size=sim_height * sim_width).reshape(sim_height, sim_width)
    else:
        levy_tensor = torch.as_tensor(levy_noise, device=device, dtype=dtype)
        if periodic:
            if levy_tensor.shape != (sim_height, sim_width):
                raise ValueError("Provided levy_noise must match the specified size.")
            noise = levy_tensor
        else:
            if levy_tensor.shape != (output_height, output_width):
                raise ValueError("Provided levy_noise must match the specified size.")
            noise = extremal_levy(alpha, size=sim_height * sim_width).reshape(sim_height, sim_width)
            noise[:output_height, :output_width] = levy_tensor


    # Create flux kernel (kernel 1)
    # Calculate exponent and normalization parameters for 2D flux kernel
    alpha_prime = 1.0 / (1.0 - 1.0/alpha)
    flux_exponent = -2.0 / alpha_prime
    flux_norm_ratio_exp = -2.0 / alpha
    flux_final_power = 1.0 / (alpha - 1.0)

    if kernel_construction_method == 'LS2010':
        kernel1 = create_kernel_LS2010((sim_height, sim_width), flux_exponent, flux_norm_ratio_exp,
                                      causal=False, outer_scale=outer_scale,
                                      outer_scale_width_factor=outer_scale_width_factor,
                                      final_power=flux_final_power)
    else:
        raise ValueError(f"Unknown kernel_construction_method for 2D: {kernel_construction_method}")

    # Perform first convolution
    integrated = periodic_convolve_2d(noise, kernel1)

    # Scale and exponentiate to get flux
    scaled = integrated * (C1 ** (1/alpha))
    del integrated
    flux = torch.exp(scaled)
    del scaled

    if H == 0:
        # Normalize and return first quadrant only to eliminate periodicity
        flux = flux / torch.mean(flux)
        return flux.cpu().numpy() if periodic else flux[:output_height, :output_width].cpu().numpy()

    # Create H kernel (kernel 2)
    # Calculate exponent and normalization parameters for 2D H kernel
    H_exponent = -2.0 + H
    H_norm_ratio_exp = -H

    if kernel_construction_method == 'LS2010':
        kernel2 = create_kernel_LS2010((sim_height, sim_width), H_exponent, H_norm_ratio_exp,
                                      causal=False, outer_scale=outer_scale,
                                      outer_scale_width_factor=outer_scale_width_factor,
                                      final_power=None)
    else:
        raise ValueError(f"Unknown kernel_construction_method for 2D: {kernel_construction_method}")
    
    # Perform second convolution
    observable = periodic_convolve_2d(flux, kernel2)

    # Normalize by mean
    observable = observable / torch.mean(observable)

    # Return full periodic field or first quadrant to eliminate periodicity
    return observable.cpu().numpy() if periodic else observable[:output_height, :output_width].cpu().numpy()
