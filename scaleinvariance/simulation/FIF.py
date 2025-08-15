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
    
    The method reduces leading-order finite-size correction terms that 
    cause deviations from pure scaling behavior in fractionally integrated flux model.
    It modifies the basic power-law singularity by applying exponential 
    cutoffs and normalization corrections to reduce boundary effects.
    
    Works for both 1D and 2D distance arrays.
    For 1D: expects dx=2 spacing (e.g., [..., 3, 1, 3, 5, 7, 9, ...])
    For 2D: expects circularly symmetric distance array
    
    Parameters:
    -----------
    distance : torch.Tensor
        Distance array (1D or 2D)
    alpha : float  
        Lévy index parameter (1 < alpha <= 2)
    causal : bool
        Whether the resulting kernel should be causal.
        
    Returns:
    --------
    torch.Tensor
        Corrected singularity kernel for convolution
    """
    if not isinstance(distance, torch.Tensor):
        distance = torch.as_tensor(distance, device=device, dtype=dtype)

    
    # Determine dimensionality and set appropriate parameters
    if distance.ndim == 1:
        # 1D case - use grid size for cutoff calculation
        domain_size = distance.numel()
        norm_exponent = 1
        singularity_exponent = -1  # |x|^(-1/α') for 1D
    elif distance.ndim == 2:
        # 2D case - use domain size (not total elements) for cutoff calculation
        domain_size = min(distance.shape)
        norm_exponent = 2
        singularity_exponent = -2  # |r|^(-2/α') for 2D
    else:
        raise ValueError("Distance array must be 1D or 2D")
    
    ratio = 2.0
    alpha_t = torch.tensor(alpha, dtype=dtype, device=device)
    alpha_prime = 1.0 / (1 - 1/alpha_t)
    cutoff_length = domain_size / 2.0
    cutoff_length2 = cutoff_length / ratio
    
    # Calculate exponential cutoffs
    exponential_cutoff = torch.exp(torch.clamp(-(distance/cutoff_length)**4, max=0, min=-200))
    exponential_cutoff2 = torch.exp(torch.clamp(-(distance/cutoff_length2)**4, max=0, min=-200))
    
    # Base singularity: |x|^(-1/α') for 1D, |r|^(-2/α') for 2D
    convolution_kernel = distance**(singularity_exponent/alpha_prime)
    
    # Calculate normalization constants
    smoothed_kernel = convolution_kernel * exponential_cutoff
    norm_constant1 = torch.sum(smoothed_kernel)
    
    smoothed_kernel = convolution_kernel * exponential_cutoff2
    norm_constant2 = torch.sum(smoothed_kernel)
    
    # Normalization factor
    normalization_factor = (ratio**(-norm_exponent/alpha_t) * norm_constant1 - norm_constant2) / (ratio**(-norm_exponent/alpha_t) - 1)
    
    # Final smoothing
    final_filter = torch.exp(torch.clamp(-distance/3.0, max=0, min=-200))
    smoothed_kernel = convolution_kernel * final_filter
    filter_integral = torch.sum(smoothed_kernel)
    
    correction_factor = -normalization_factor / filter_integral
    convolution_kernel = (convolution_kernel * (1 + correction_factor * final_filter))**(1/(alpha_t - 1))
    
    return convolution_kernel

def create_corrected_H_kernel(distance, H, size, causal):
    """
    Create a finite-size corrected kernel for the second (H) fractional integral.
    
    This function implements the correction method from Lovejoy & Schertzer (2010)
    for the second integral kernel. It reduces finite-size effects
    by applying exponential cutoffs and normalization corrections.
    
    Works for both 1D and 2D distance arrays.
    For 1D: kernel is |x|^(-1+H)
    For 2D: kernel is |r|^(-2+H)
    
    Parameters:
    -----------
    distance : torch.Tensor
        Distance array (1D or 2D)
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
    
    # Determine dimensionality and set appropriate parameters
    if distance.ndim == 1:
        singularity_exponent = -1 + H  # |x|^(-1+H) for 1D
        ratio_exponent = -H  # For 1D normalization
    elif distance.ndim == 2:
        singularity_exponent = -2 + H  # |r|^(-2+H) for 2D
        ratio_exponent = -H  # For 2D normalization (same as 1D for H)
    else:
        raise ValueError("Distance array must be 1D or 2D")
    
    lambda_equiv = size
    ratio = 2.0
    
    # Cutoff lengths (following Lovejoy's FracDiffH)
    outer_cutoff = lambda_equiv / 2.0
    outer_cutoff2 = outer_cutoff / ratio
    inner_cutoff = 0.01 if causal else 1.0
    smoothing_cutoff = lambda_equiv / 4.0
    
    # Base singularity kernel
    # Handle zero by setting to 1 (will be zeroed by cutoffs anyway)
    if distance.ndim == 1:
        abs_distance = torch.abs(distance)
    else:  # 2D case
        abs_distance = distance  # Already positive radial distance
    
    abs_distance_safe = torch.where(abs_distance == 0, torch.ones_like(abs_distance), abs_distance)
    base_kernel = abs_distance_safe ** singularity_exponent
    
    # Inner cutoff to remove singularity at origin
    # Using smooth cutoff: 1 - exp(-(|r|/inner_cutoff)^2)
    inner_cutoff_factor = 1.0 - torch.exp(
        torch.clamp(-(abs_distance_safe/inner_cutoff)**2, max=0, min=-200)
    )
    
    # Outer exponential cutoffs
    outer_exp_cutoff = torch.exp(
        torch.clamp(-(abs_distance_safe/outer_cutoff)**4, max=0, min=-200)
    )
    outer_exp_cutoff2 = torch.exp(
        torch.clamp(-(abs_distance_safe/outer_cutoff2)**4, max=0, min=-200)
    )
    
    # Calculate normalization constants
    smoothed_kernel1 = base_kernel * outer_exp_cutoff * inner_cutoff_factor
    t1 = torch.sum(smoothed_kernel1)
    
    smoothed_kernel2 = base_kernel * outer_exp_cutoff2 * inner_cutoff_factor
    t2 = torch.sum(smoothed_kernel2)
    
    # Compute normalization factor
    norm_factor = (t1 * ratio**ratio_exponent - t2) / (ratio**ratio_exponent - 1)
    
    # Final smoothing function for correction
    final_smooth = inner_cutoff_factor * torch.exp(
        torch.clamp(-abs_distance_safe/smoothing_cutoff, max=0, min=-200)
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
    
    fft_signal = torch.fft.fft2(signal)
    fft_kernel = torch.fft.fft2(kernel)
    
    convolved = torch.fft.ifft2(fft_signal * fft_kernel)
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

    lo = size//2 - outer_scale//2
    hi = size//2 + outer_scale//2
    kernel1[:lo] = 0
    kernel1[hi:] = 0
    

    if causal: 
        kernel1[:size//2] = 0

    kernel1 = torch.cat([torch.zeros(size//2, device=device), kernel1, torch.zeros(size//2, device=device)]) 
    noise = torch.cat([noise, torch.zeros_like(noise, device=device)]) 
    
    integrated = periodic_convolve(noise, kernel1)[size:]
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
        # Normalize
        flux = flux/torch.mean(flux)
        return flux.cpu().numpy()
    
    # Create kernel 2 only for H \ne 0 case
    if correct_for_finite_size_effects:
        # Use prior created distance array for corrected kernels with dx=2
        kernel2 = create_corrected_H_kernel(distance_corrected, H, size, causal)
        del distance_corrected
    else:
        distance_standard = torch.abs(torch.arange(-size//2, size//2, dtype=dtype, device=device))
        distance_standard[distance_standard==0] = 1
        kernel2 = distance_standard ** (-1 + H)
        del distance_standard
    
    if causal: 
        kernel2[:size//2] = 0

    lo = size//2 - outer_scale//2
    hi = size//2 + outer_scale//2
    kernel2[:lo] = 0
    kernel2[hi:] = 0

    kernel2 = torch.cat([torch.zeros(size//2, device=device), kernel2, torch.zeros(size//2, device=device)]) 
    flux = torch.cat([flux, torch.ones_like(flux, device=device)])        # Pad with statistical mean of process to eliminate boundary artifacts
    observable = periodic_convolve(flux, kernel2)[size:] 
    del flux, kernel2

    if H_int == -1:
        observable = torch.diff(observable)

    # Normalize by mean
    observable = observable/torch.mean(observable)

    return observable.cpu().numpy()

def FIF_2D(size, alpha, C1, H, levy_noise=None, outer_scale=None):
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
        5. For H < 0: Apply differencing to handle negative Hurst exponents
    
    Parameters
    ----------
    size : int or tuple of ints
        Size of simulation. If int, creates square array (size x size).
        If tuple, specifies (height, width). Must be even numbers.
    alpha : float
        Lévy stability parameter in (0, 2). Controls noise distribution.
    C1 : float
        Codimension of the mean, controls intermittency strength. 
        Must be > 0 for multifractal behavior.
    H : float
        Hurst exponent in (0, 1). Controls correlation structure.
    levy_noise : torch.Tensor, optional
        Pre-generated 2D Lévy noise for reproducibility. Must have same shape as simulation.
    outer_scale : int, optional
        Large-scale cutoff. Defaults to min(height, width).
    
    Returns
    -------
    numpy.ndarray
        2D array of simulated multifractal field values
        
    Raises
    ------
    ValueError
        If size dimensions are not even or H is outside valid range (-1, 1)
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
    >>> fif = FIF_2D((256, 512), alpha=1.5, C1=0.05, H=-0.2)
    
    Notes
    -----
    - Computational complexity is O(N log N) due to FFT-based convolutions
    - Large C1 values (> 0.5) can produce extreme values requiring careful handling
    - Always uses finite-size corrections (no causal option for 2D)
    """
    # Handle size parameter
    if isinstance(size, int):
        height, width = size, size
    else:
        height, width = size
        
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("Height and width must be even numbers; powers of 2 are recommended.")
    
    if not isinstance(C1, (int, float)) or C1 <= 0:
        raise ValueError("C1 must be a positive number.")


    if not isinstance(H, (int, float)) or H < 0 or H > 1:
        raise ValueError("H must be a number between 0 and 1.")

    if levy_noise is None:
        noise = extremal_levy(alpha, size=height * width).reshape(height, width)
    else: 
        if levy_noise.shape != (height, width):
            raise ValueError("Provided levy_noise must match the specified size.")
        noise = torch.as_tensor(levy_noise, device=device, dtype=dtype)

    if outer_scale is None: 
        outer_scale = min(height, width)

    # Create 2D distance array for corrected kernels
    # Use the same spacing convention as 1D: dx=2 for first kernel, dx=1 for second
    y_coords = torch.arange(-(height - 1), height, 2, dtype=dtype, device=device)
    x_coords = torch.arange(-(width - 1), width, 2, dtype=dtype, device=device)
    Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
    distance_corrected = torch.sqrt(X**2 + Y**2)
    del Y, X    # Clean memory
    
    # Create kernel 1 using corrected method
    kernel1 = create_corrected_kernel(distance_corrected, alpha)
    
    # Apply outer scale cutoff
    kernel1[distance_corrected*2 > outer_scale] = 0     # convert to units of dx=1 here
    
    # Pad arrays to double size for periodic convolution
    kernel1_padded = torch.zeros((2*height, 2*width), device=device, dtype=dtype)
    noise_padded = torch.zeros((2*height, 2*width), device=device, dtype=dtype)
    
    # Center the kernel (like 1D) and place noise at top-left
    kernel1_padded[height//2:height//2+height, width//2:width//2+width] = kernel1
    noise_padded[:height, :width] = noise
    del kernel1, noise          # Clean memory
    
    # Perform first convolution
    integrated = periodic_convolve_2d(noise_padded, kernel1_padded)[height:, width:]
    del noise_padded, kernel1_padded    # Clean memory
    
    # Scale and exponentiate to get flux
    scaled = integrated * (C1 ** (1/alpha))
    del integrated
    flux = torch.exp(scaled)
    del scaled

    if H == 0:
        # Normalize and return
        flux = flux / torch.mean(flux)
        return flux.cpu().numpy()
    
    # Create kernel 2 for H != 0 case
    
    # Use the corrected H-kernel
    kernel2 = create_corrected_H_kernel(distance_corrected, H, min(height, width), False)
    
    # Apply outer scale cutoff
    kernel2[distance_corrected*2 > outer_scale] = 0
    del distance_corrected
    
    # Pad for convolution
    kernel2_padded = torch.zeros((2*height, 2*width), device=device, dtype=dtype)
    flux_padded = torch.ones((2*height, 2*width), device=device, dtype=dtype)       # Pad with statistical mean of process to eliminate boundary artifacts
    
    # Center the kernel (like 1D) and place flux at top-left
    kernel2_padded[height//2:height//2+height, width//2:width//2+width] = kernel2
    flux_padded[:height, :width] = flux
    del flux, kernel2
    
    # Perform second convolution
    observable = periodic_convolve_2d(flux_padded, kernel2_padded)[height:, width:]
    del flux_padded, kernel2_padded

    # Normalize by mean
    observable = observable / torch.mean(observable)

    return observable.cpu().numpy()
