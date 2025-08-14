import torch
from torch.fft import fft as torch_fft, ifft as torch_ifft
import numpy as np

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

def create_corrected_kernel(distance, alpha):
    """
    Create a finite-size corrected kernel for fractionally integrated flux simulations.
    
    This function implements the correction method from Lovejoy & Schertzer (2010) 
    "On the simulation of continuous in scale universal multifractals, Part II".
    
    The method reduces leading-order finite-size correction terms (Δx^(-D/α)) that 
    cause deviations from pure scaling behavior in fractionally integrated flux model.
    It modifies the basic power-law singularity |x|^(-1/α') by applying exponential 
    cutoffs and normalization corrections to reduce boundary effects.
    
    This function expects the following distance array format:
    odd integers with dx=2 spacing (e.g., [..., 3, 1, 3, 5, 7, 9, ...]).
    
    Parameters:
    -----------
    distance : torch.Tensor
        Distance array with dx=2 spacing of odd integers
    alpha : float  
        Lévy index parameter (1 < alpha <= 2)
        
    Returns:
    --------
    torch.Tensor
        Corrected singularity kernel for convolution
    """
    if not isinstance(distance, torch.Tensor):
        distance = torch.as_tensor(distance, device=device, dtype=dtype)
    
    grid_size = distance.numel()
    ratio = 2.0
    alpha_t = torch.tensor(alpha, dtype=dtype, device=device)
    alpha_prime = 1.0 / (1 - 1/alpha_t)
    cutoff_length = grid_size / 2.0
    cutoff_length2 = cutoff_length / ratio
    
    # Calculate exponential cutoffs
    exponential_cutoff = torch.exp(torch.clamp(-(distance/cutoff_length)**4, max=0, min=-200))
    exponential_cutoff2 = torch.exp(torch.clamp(-(distance/cutoff_length2)**4, max=0, min=-200))
    
    # Base singularity
    convolution_kernel = distance**(-1/alpha_prime)
    
    # Calculate normalization constants
    smoothed_kernel = convolution_kernel * exponential_cutoff
    norm_constant1 = torch.sum(smoothed_kernel)
    
    smoothed_kernel = convolution_kernel * exponential_cutoff2
    norm_constant2 = torch.sum(smoothed_kernel)
    
    
    normalization_factor = (ratio**(-1/alpha_t) * norm_constant1 - norm_constant2) / (ratio**(-1/alpha_t) - 1)
    
    # Final smoothing
    final_filter = torch.exp(torch.clamp(-distance/3.0, max=0, min=-200))
    smoothed_kernel = convolution_kernel * final_filter
    filter_integral = torch.sum(smoothed_kernel)
    
    correction_factor = -normalization_factor / filter_integral
    convolution_kernel = (convolution_kernel * (1 + correction_factor * final_filter))**(1/(alpha_t - 1))
    
    return convolution_kernel

def create_corrected_H_kernel(distance, H, size):
    """
    Create a finite-size corrected kernel for the second (H) fractional integral.
    
    This function implements the correction method from Lovejoy & Schertzer (2010)
    for the second integral kernel |x|^(-1+H). It reduces finite-size effects
    by applying exponential cutoffs and normalization corrections.
    
    Note: This function expects distance array with dx=1 spacing 
    (e.g., [..., -2, -1, 0, 1, 2, ...]) unlike Lovejoy's dx=2.
    
    Parameters:
    -----------
    distance : torch.Tensor
        Distance array with dx=1 spacing
    H : float
        Hurst exponent parameter (-1 < H < 1)
    size : int
        Size of the simulation domain
        
    Returns:
    --------
    torch.Tensor
        Corrected H-integral kernel for convolution
    """
    if not isinstance(distance, torch.Tensor):
        distance = torch.as_tensor(distance, device=device, dtype=dtype)
    
    # Adjustment for dx=1 vs Lovejoy's dx=2
    # In Lovejoy's code, lambda is the grid size with dx=2
    # In our code with dx=1, the equivalent lambda is size
    lambda_equiv = size
    ratio = 2.0
    
    # Cutoff lengths (following Lovejoy's FracDiffH)
    outer_cutoff = lambda_equiv / 2.0
    outer_cutoff2 = outer_cutoff / ratio
    inner_cutoff = 1.0  # For acausal; use 0.01 for causal as in Lovejoy
    smoothing_cutoff = lambda_equiv / 4.0
    
    # Base singularity kernel: |x|^(-1+H)
    # Handle zero by setting to 1 (will be zeroed by cutoffs anyway)
    abs_distance = torch.abs(distance)
    abs_distance[abs_distance == 0] = 1
    base_kernel = abs_distance ** (-1 + H)
    
    # Inner cutoff to remove singularity at origin
    # Using smooth cutoff: 1 - exp(-(|x|/inner_cutoff)^2)
    inner_cutoff_factor = 1.0 - torch.exp(
        torch.clamp(-(abs_distance/inner_cutoff)**2, max=0, min=-200)
    )
    
    # Outer exponential cutoffs
    outer_exp_cutoff = torch.exp(
        torch.clamp(-(abs_distance/outer_cutoff)**4, max=0, min=-200)
    )
    outer_exp_cutoff2 = torch.exp(
        torch.clamp(-(abs_distance/outer_cutoff2)**4, max=0, min=-200)
    )
    
    # Calculate normalization constants (following Lovejoy's approach)
    # Note: Mean instead of sum because of different normalization convention
    smoothed_kernel1 = base_kernel * outer_exp_cutoff * inner_cutoff_factor
    t1 = torch.sum(smoothed_kernel1)
    
    smoothed_kernel2 = base_kernel * outer_exp_cutoff2 * inner_cutoff_factor
    t2 = torch.sum(smoothed_kernel2)
    
    # Compute normalization factor EEH (equivalent to Lovejoy's EEH)
    # Adjusted for our dx=1 convention
    norm_factor = (t1 * ratio**(-H) - t2) / (ratio**(-H) - 1)
    
    # Final smoothing function for correction
    final_smooth = inner_cutoff_factor * torch.exp(
        torch.clamp(-abs_distance/smoothing_cutoff, max=0, min=-200)
    )
    
    # Calculate correction coefficient
    smoothed_final = base_kernel * inner_cutoff_factor * outer_exp_cutoff2 * final_smooth
    GH = torch.sum(smoothed_final)
    
    # Avoid division by zero
    if torch.abs(GH) < 1e-12:
        correction_coeff = 0.0
    else:
        correction_coeff = -norm_factor / GH
    
    # Apply correction to kernel
    corrected_kernel = base_kernel * (1 + correction_coeff * final_smooth)
    
    # For numerical stability, ensure kernel is not negative
    # This shouldn't happen with proper parameters, but safeguard
    corrected_kernel = torch.where(
        corrected_kernel > 0, 
        corrected_kernel, 
        torch.zeros_like(corrected_kernel)
    )
    
    return corrected_kernel

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
    
    fft_signal = torch_fft(signal)
    fft_kernel = torch_fft(kernel)
    
    convolved = torch_ifft(fft_signal * fft_kernel)
    return convolved.real

def FIF_1D(size, alpha, C1, H, levy_noise=None, causal=True, outer_scale=None, correct_for_finite_size_effects=True):
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
        Lévy stability parameter in (0, 2). Controls noise distribution.
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
    correct_for_finite_size_effects : bool, optional
        Whether to use corrections for finite-size effects as described in Lovejoy & Schertzer 2010.
        Default True. Essentially the only reason for False is illustration of the method.
    
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
    
    if not isinstance(C1, (int, float)) or C1 <= 0:
        raise ValueError("C1 must be a positive number.")
    
    H_int = 0
    if H < 0:
        H_int = -1
        H += 1

    if not isinstance(H, (int, float)) or H < 0 or H > 1:
        raise ValueError("H must be a number between -1 and 1.")

    if levy_noise is None:
        noise = extremal_levy(alpha, size=size)
    else: 
        if levy_noise.size()[0] != size:
            raise ValueError("Provided levy_noise must match the specified size.")
        noise = torch.as_tensor(levy_noise, device=device, dtype=dtype)

    if outer_scale is None: outer_scale = size

    # Create kernel 1
    if correct_for_finite_size_effects:
        # Create distance array for corrected kernel (odd numbers, dx=2)
        position_range = torch.arange(-(size - 1), size, 2, dtype=dtype, device=device)
        distance_corrected = torch.abs(position_range)
        # Use corrected kernel with distance array
        kernel1 = create_corrected_kernel(distance_corrected, alpha)
    else:
        t = torch.abs(torch.arange(-size//2, size//2, dtype=dtype, device=device))
        t[t==0] = 1
        kernel1 = t ** (-1/alpha)

    kernel1[size//2+outer_scale//2:] = 0
    kernel1[size//2-outer_scale//2] = 0
    

    if causal: 
        kernel1[:size//2] = 0

    kernel1 = torch.cat([torch.zeros(size//2, device=device), kernel1, torch.zeros(size//2, device=device)]) 
    noise = torch.cat([noise, torch.zeros_like(noise, device=device)]) 
    
    integrated = periodic_convolve(noise, kernel1)[size:]

    # If causal, adjust for the fact that half the kernel is being deleted
    # TODO: Check this.. lovejoy does it but idk if it is correct from empirical testing
    if causal:
        causality_factor = 2.0
    else:
        causality_factor = 1.0
    

    scaled = integrated * ((causality_factor * C1) ** (1/alpha))
    flux = torch.exp(scaled)

    if H == 0:
        # Normalize
        flux = flux/torch.mean(flux)
        return flux.cpu().numpy()
    
    # Create kernel 2 only for H \ne 0 case
    if correct_for_finite_size_effects:
        # Use prior created distance array for corrected kernels with dx=2
        kernel2 = create_corrected_H_kernel(distance_corrected, H, size)
    else:
        distance_standard = torch.abs(torch.arange(-size//2, size//2, dtype=dtype, device=device))
        distance_standard[distance_standard==0] = 1
        kernel2 = distance_standard ** (-1 + H)
    
    if causal: 
        kernel2[:size//2] = 0

    kernel2[size//2+outer_scale//2:] = 0
    kernel2[size//2-outer_scale//2] = 0

    kernel2 = torch.cat([torch.zeros(size//2, device=device), kernel2, torch.zeros(size//2, device=device)]) 
    flux = torch.cat([flux, torch.ones_like(flux, device=device)])  
    observable = periodic_convolve(flux, kernel2)[size:] 

    if H_int == -1:
        observable = torch.diff(observable)

    # Normalize by mean
    observable = observable/torch.mean(observable)

    return observable.cpu().numpy()
