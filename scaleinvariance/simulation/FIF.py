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

def FIF_1D(size, alpha, C1, H, levy_noise=None, causal=True, outer_scale=None):
    """
    Generate a 1D Fractionally Integrated Flux (FIF) multifractal simulation.
    
    FIF is a multiplicative cascade model that generates multifractal fields with
    specified Hurst exponent H and intermittency parameter C1. The method follows
    the theory of Schertzer & Lovejoy for multifractal fields.

    Note: Lovejoy & Schertzer singularity corrections are TODO
    
    Algorithm:
        1. Generate extremal Lévy noise with stability parameter alpha
        2. Convolve with power-law kernel |k|^(-1/alpha) to create log-flux
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
    - The causality factor adjustment follows Lovejoy's implementation
      but may require further empirical validation 
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

    t = torch.abs(torch.arange(-size//2, size//2, dtype=dtype, device=device))
    t[t==0] = 1
    kernel1 = t ** (-1/alpha)
    kernel2 = t ** (-1 + H)

    kernel1[size//2+outer_scale//2:] = 0
    kernel1[size//2-outer_scale//2] = 0
    kernel2[size//2+outer_scale//2:] = 0
    kernel2[size//2-outer_scale//2] = 0

    if causal: 
        kernel1[:size//2] = 0
        kernel2[:size//2] = 0

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
        return flux.cpu().numpy()

    kernel2 = torch.cat([torch.zeros(size//2, device=device), kernel2, torch.zeros(size//2, device=device)]) 
    flux = torch.cat([flux, torch.ones_like(flux, device=device)])  
    observable = periodic_convolve(flux, kernel2)[size:] 

    if H_int == -1:
        observable = torch.diff(observable)

    return observable.cpu().numpy()
