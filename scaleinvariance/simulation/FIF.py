import numpy as np
from .. import backend as B
from .fbm import fBm_1D_circulant, fBm_ND_circulant

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
    phi = (B.rand(size) - 0.5) * B.pi
    alpha_t = alpha
    phi0 = -(B.pi/2) * (1 - B.abs(1 - alpha_t)) / alpha_t
    R = B.exponential(scale=1.0, size=size)
    eps = 1e-12
    cos_phi = B.cos(phi)
    cos_phi = B.where(B.abs(cos_phi) < eps, B.full_like(cos_phi, eps), cos_phi)
    abs_alpha1 = B.abs(alpha_t - 1)
    abs_alpha1 = B.where(abs_alpha1 < eps, B.full_like(abs_alpha1, eps), abs_alpha1)
    denom = B.cos(phi - alpha_t * (phi - phi0))
    denom = B.where(B.abs(denom) < eps, B.full_like(denom, eps), denom)
    R = B.where(R < eps, B.full_like(R, eps), R)
    sample = (
        B.sign(alpha_t - 1) *
        B.sin(alpha_t * (phi - phi0)) *
        (cos_phi * abs_alpha1) ** (-1/alpha_t) *
        (denom / R) ** ((1 - alpha_t) / alpha_t)
    )
    return sample

def _apply_outer_scale(kernel, distance, outer_scale, width_factor=2.0):
    """
    Apply outer scale cutoff using a Hanning window transition.

    Parameters
    ----------
    kernel : ndarray
        The kernel to apply outer scale to
    distance : ndarray
        Distance array (same shape as kernel)
    outer_scale : float
        Center of the transition region
    width_factor : float, optional
        Controls transition width. Transition width = outer_scale * width_factor.
        Default is 2.0.

    Returns
    -------
    ndarray
        Kernel with outer scale applied
    """
    distance = B.asarray(distance)

    # Transition region centered at outer_scale with width = outer_scale * width_factor
    transition_width = outer_scale * width_factor
    lower_edge = outer_scale - transition_width / 2.0
    upper_edge = outer_scale + transition_width / 2.0

    # Compute normalized distance: 0 at lower_edge, 1 at upper_edge
    # Clamp to [0, 1] so values below lower_edge → 0, above upper_edge → 1
    normalized_dist = B.clip((distance - lower_edge) / transition_width, 0.0, 1.0)

    # Hanning window: 1 when normalized_dist=0, 0 when normalized_dist=1
    window = 0.5 * (1.0 + B.cos(B.pi * normalized_dist))

    return kernel * window

# LS 2010 kernels

def _apply_LS2010_correction(distance, exponent, norm_ratio_exponent, final_power=None):
    """
    Apply Lovejoy & Schertzer 2010 finite-size correction to a power-law kernel.

    This implements the correction method from Lovejoy & Schertzer (2010)
    "On the simulation of continuous in scale universal multifractals, Part II",
    following Lovejoy's Frac.m implementation.

    The method reduces leading-order finite-size correction terms by applying
    exponential cutoffs and normalization corrections to reduce boundary effects.

    Works for 1D and N-D distance arrays with dx=2 spacing.

    Parameters
    ----------
    distance : ndarray
        Distance array (1D or N-D), expects dx=2 spacing
    exponent : float
        Power-law exponent for base kernel (e.g., -1/α' for flux, -1+H for H kernel)
    norm_ratio_exponent : float
        Exponent for ratio in normalization factor (e.g., -1/α for flux, -H for H kernel)
    final_power : float, optional
        Final power transformation to apply (e.g., 1/(α-1) for flux kernel).
        If None, no transformation is applied (kernel remains as corrected).

    Returns
    -------
    ndarray
        Corrected kernel
    """
    distance = B.asarray(distance)

    # Determine domain size for cutoff calculation
    if distance.ndim == 1:
        domain_size = distance.size
    else:
        # For N-D arrays, use minimum dimension size
        domain_size = min(distance.shape)

    ratio = 2.0
    cutoff_length = domain_size / 2.0
    cutoff_length2 = cutoff_length / ratio

    # Calculate exponential cutoffs
    exponential_cutoff = B.exp(B.clip(-(distance/cutoff_length)**4, -200, 0))
    exponential_cutoff2 = B.exp(B.clip(-(distance/cutoff_length2)**4, -200, 0))

    # Base singularity kernel
    base_kernel = distance**exponent

    # Calculate normalization constants
    smoothed_kernel1 = base_kernel * exponential_cutoff
    norm_constant1 = B.sum(smoothed_kernel1)

    smoothed_kernel2 = base_kernel * exponential_cutoff2
    norm_constant2 = B.sum(smoothed_kernel2)

    # Normalization factor
    ratio_factor = ratio**norm_ratio_exponent
    normalization_factor = (ratio_factor * norm_constant1 - norm_constant2) / (ratio_factor - 1)

    # Final smoothing filter
    final_filter = B.exp(B.clip(-distance/3.0, -200, 0))
    smoothed_kernel = base_kernel * final_filter
    filter_integral = B.sum(smoothed_kernel)

    # Apply correction
    correction_factor = -normalization_factor / filter_integral
    corrected_kernel = base_kernel * (1 + correction_factor * final_filter)

    # Apply final power transformation if specified
    if final_power is not None:
        corrected_kernel = corrected_kernel ** final_power

    return corrected_kernel

def create_kernel_LS2010(size, exponent, norm_ratio_exponent, causal=False, outer_scale=None,
                         outer_scale_width_factor=2.0, final_power=None, scale_metric=None):
    """
    Create kernel using Lovejoy & Schertzer 2010 finite-size corrections.

    Unified function for creating both flux and H kernels with LS2010 corrections.
    Handles 1D and N-D cases automatically.

    **IMPORTANT**: This method uses dx=2 grid spacing for kernel construction.
    Distance coordinates are defined as arange(-(size-1), size, 2) for each dimension.

    Parameters
    ----------
    size : int or tuple
        For 1D: int specifying kernel size
        For N-D: tuple of dimensions (e.g., (height, width) for 2D, (depth, height, width) for 3D)
    exponent : float
        Power-law exponent for base kernel
        Examples: -1/α' for 1D flux, -d/α' for d-D flux, -1+H for 1D H kernel, -d+H for d-D H kernel
    norm_ratio_exponent : float
        Exponent for ratio in normalization factor
        Examples: -1/α for 1D flux, -d/α for d-D flux, -H for H kernels
    causal : bool, optional
        Whether to make kernel causal (1D only). Default False.
    outer_scale : int, optional
        Large-scale cutoff. If None, no outer scale cutoff is applied.
    outer_scale_width_factor : float, optional
        Controls transition width for outer scale. Transition width = outer_scale * width_factor.
        Default is 2.0.
    final_power : float, optional
        Final power transformation (e.g., 1/(α-1) for flux kernel). Default None.
    scale_metric : ndarray, optional
        Custom N-D array defining the scale metric (generalized distance) for GSI norms.
        If None (default), standard Euclidean distance is computed with dx=2 spacing.
        **Shape requirement**: Must match the kernel size (same shape as size parameter).
        **Note**: Only applicable for N-D case; ignored for 1D.

    Returns
    -------
    ndarray
        Corrected kernel ready for convolution
    """
    # Handle 1D vs N-D
    if isinstance(size, int):
        # 1D case
        position_range = B.arange(-(size - 1), size, 2)
        distance = B.abs(position_range)
        kernel = _apply_LS2010_correction(distance, exponent, norm_ratio_exponent, final_power)

        # Apply outer scale cutoff with Hanning window (convert distance to units of dx=1)
        if outer_scale is not None:
            kernel = _apply_outer_scale(kernel, distance / 2, outer_scale, outer_scale_width_factor)

        # Apply causality
        if causal:
            kernel[:size//2] = 0

    else:
        # N-D case
        if scale_metric is None:
            # Create coordinate arrays for each dimension (dx=2 spacing)
            coord_arrays = [B.arange(-(dim - 1), dim, 2) for dim in size]

            # Create N-D meshgrid
            coord_grids = B.meshgrid(*coord_arrays, indexing='ij')

            # Calculate Euclidean distance: sqrt(x1^2 + x2^2 + ... + xn^2)
            distance = B.sqrt(sum(grid**2 for grid in coord_grids))
        else:
            # Use provided scale metric
            distance = scale_metric

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
    distance = B.abs(B.arange(-size//2, size//2)+0.5)

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
    signal = B.asarray(signal)
    kernel = B.asarray(kernel)

    if signal.size != kernel.size:
        raise ValueError("Signal and kernel must have the same length for periodic convolution.")

    # Shift kernel so zero-lag is at index 0 (required for FFT convolution)
    kernel = B.ifftshift(kernel)

    fft_signal = B.fft(signal)
    fft_kernel = B.fft(kernel)

    convolved = B.real(B.ifft(fft_signal * fft_kernel))
    return convolved

def periodic_convolve_nd(signal, kernel):
    """
    Performs periodic convolution of two N-D arrays using Fourier methods.
    Both arrays must have the same shape.

    Parameters:
        signal (ndarray): Input signal array (N-D).
        kernel (ndarray): Convolution kernel array (N-D).

    Returns:
        ndarray: The result of the periodic convolution.

    Raises:
        ValueError: If signal and kernel shapes do not match.
    """
    signal = B.asarray(signal)
    kernel = B.asarray(kernel)

    if signal.shape != kernel.shape:
        raise ValueError("Signal and kernel must have the same shape for periodic convolution.")

    # Shift kernel so zero-lag is at index (0,0,...,0) (required for FFT convolution)
    kernel = B.ifftshift(kernel)

    fft_signal = B.fftn(signal)
    fft_kernel = B.fftn(kernel)

    convolved = B.real(B.ifftn(fft_signal * fft_kernel))
    return convolved

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
        Special case: C1 = 0 routes to fBm (no intermittency), but requires
        causal=False since fBm cannot be causal.
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
        If True (default), returns full periodic simulation suitable for periodic
        boundary conditions. If False, doubles simulation size internally then
        returns only the first half to eliminate periodicity artifacts.

    Returns
    -------
    numpy.ndarray
        1D array of simulated multifractal field values

    Raises
    ------
    ValueError
        If size is not even or H is outside valid range (-1, 1)
    ValueError
        If C1 < 0 or if C1 = 0 with causal=True (fBm cannot be causal)
    ValueError
        If provided levy_noise doesn't match specified size

    Examples
    --------
    >>> # Basic multifractal with strong intermittency
    >>> fif = FIF_1D(1024, alpha=1.8, C1=0.1, H=0.3)
    >>>
    >>> # Monofractal case (C1=0 routes to fBm, requires causal=False)
    >>> fbm = FIF_1D(1024, alpha=1.8, C1=0, H=0.3, causal=False)

    Notes
    -----
    - Computational complexity is O(N log N) due to FFT-based convolutions
    - Large C1 values (> 0.5) can produce extreme values requiring careful handling
    - C1 = 0: Routes internally to fBm_1D_circulant() for monofractal case
      (requires causal=False since fBm cannot be causal)
    """
    if size % 2 != 0:
        raise ValueError("size must be an even number; a power of 2 is recommended.")

    output_size = size
    if not periodic:
        size *= 2   # duplicate to eliminate periodicity

    if C1 == 0 and not causal:
        if levy_noise is not None:
            raise ValueError('noise argument not supported for C1=0')
        result = fBm_1D_circulant(output_size, H, periodic=periodic)
        return result

    if not isinstance(C1, (int, float)) or C1 < 0:
        raise ValueError("C1 must be a non-negative number.")

    if C1 == 0 and causal:
        raise ValueError("C1=0 requires causal=False (fBm cannot be causal)")

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
        levy_noise_arr = B.asarray(levy_noise)
        if levy_noise_arr.size != output_size:
            raise ValueError("Provided levy_noise must match the specified size.")
        # if aperiodic is requested, need to pad noise with more noise
        if not periodic:
            noise = B.concatenate([levy_noise_arr, extremal_levy(alpha, size=output_size)])
        else:
            noise = levy_noise_arr
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
    flux = B.exp(scaled)
    del scaled

    if H == 0:
        # Normalize - slice first (if periodic), then normalize by mean
        if not periodic:
            flux = flux[:size//2]     # eliminate periodicity by removing the part corresponding to the appended noise
        flux = flux / B.mean(flux)
        return flux

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
        observable = B.diff(observable)
        # Duplicate last value to maintain original size
        observable = B.concatenate([observable, observable[-1:]])
        # Normalize to zero mean (increments should have zero mean)
        observable = observable - B.mean(observable)
    else:
        # Normalize to unit mean (levels should have unit mean)
        observable = observable / B.mean(observable)

    return observable

def FIF_ND(size, alpha, C1, H, levy_noise=None, outer_scale=None, outer_scale_width_factor=2.0,
           kernel_construction_method='LS2010', periodic=False, scale_metric=None, scale_metric_dim=None):
    """
    Generate an N-D Fractionally Integrated Flux (FIF) multifractal simulation.

    FIF is a multiplicative cascade model that generates multifractal fields with
    specified Hurst exponent H, intermittency parameter C1, and Levy index alpha.
    The method follows Lovejoy & Schertzer 2010, including finite-size corrections.
    Returns field normalized by mean.

    Algorithm:
        1. Generate N-D extremal Lévy noise with stability parameter alpha
        2. Convolve with corrected kernel (finite-size corrected |r|^(-d/alpha)) to create log-flux
        3. Scale by (C1)^(1/alpha) and exponentiate to get conserved flux
        4. For H ≠ 0: Convolve flux with kernel |r|^(-d+H) to get observable

    Parameters
    ----------
    size : tuple of ints
        Size of simulation as tuple specifying dimensions (e.g., (height, width) for 2D,
        (depth, height, width) for 3D). All dimensions must be even numbers.
    alpha : float
        Lévy stability parameter in (0, 2) and != 1. Controls noise distribution.
    C1 : float
        Codimension of the mean, controls intermittency strength.
        Must be > 0 for multifractal behavior.
        Special case: C1 = 0 routes to fBm_ND_circulant() (no intermittency).
    H : float
        Hurst exponent in (0, 1). Controls correlation structure.
    levy_noise : ndarray, optional
        Pre-generated N-D Lévy noise for reproducibility. Must have same shape as simulation.
    outer_scale : int, optional
        Large-scale cutoff. Defaults to max(dimensions).
    outer_scale_width_factor : float, optional
        Controls transition width for outer scale. Transition width = outer_scale * width_factor.
        Default is 2.0.
    kernel_construction_method : str, optional
        Method for constructing convolution kernels. Options:
        - 'LS2010': Lovejoy & Schertzer 2010 finite-size corrections (default)
          **IMPORTANT**: LS2010 kernels use dx=2 grid spacing
    periodic : bool or tuple of bool, optional
        Controls periodicity behavior for each axis.
        - If bool: applies same periodicity to all axes (default False)
        - If tuple: specifies per-axis periodicity, must have length matching size tuple
        When an axis is non-periodic (False), the domain is doubled along that axis
        internally and one hyper-octant is returned to suppress periodic artifacts.
    scale_metric : ndarray, optional
        Custom N-D array defining the scale metric (generalized distance) for GSI norms.
        If None (default), standard Euclidean distance is computed.
        **Shape requirement**: Must match simulation domain shape after accounting for
        periodicity (doubled along non-periodic axes). For example, if size=(256, 256)
        and periodic=False, scale_metric.shape must be (512, 512).
        **Note**: For LS2010 method, the metric should use dx=2 spacing to match
        the kernel construction grid.
    scale_metric_dim : float, optional
        Dimension for scaling exponents in kernel calculations. This may differ from
        the spatial dimension when using GSI norms with non-integer dimensions.
        If None (default), uses the spatial dimension (inferred from size tuple length).
        **Use case**: In Generalized Scale Invariance (GSI), the scaling dimension
        may be non-integer and different from the embedding space dimension.

    Returns
    -------
    numpy.ndarray
        N-D array of simulated multifractal field values

    Raises
    ------
    ValueError
        If size dimensions are not even or H is outside valid range (0, 1)
    ValueError
        If C1 < 0 (must be non-negative; C1 = 0 routes to fBm)
    ValueError
        If provided levy_noise doesn't match specified size

    Examples
    --------
    >>> # 2D multifractal with strong intermittency
    >>> fif = FIF_ND((512, 512), alpha=1.8, C1=0.1, H=0.3)
    >>>
    >>> # 3D multifractal
    >>> fif = FIF_ND((128, 128, 128), alpha=1.5, C1=0.05, H=0.2)
    >>>
    >>> # 2D with per-axis periodicity (periodic in x, non-periodic in y)
    >>> fif = FIF_ND((256, 256), alpha=1.6, C1=0.08, H=0.4, periodic=(True, False))
    >>>
    >>> # Monofractal case (C1=0 routes to fBm_ND_circulant)
    >>> fbm = FIF_ND((512, 512), alpha=1.8, C1=0, H=0.3)
    >>>
    >>> # Using custom GSI norm (example with anisotropic scaling)
    >>> import numpy as np
    >>> size = (256, 256)
    >>> sim_size = (512, 512)  # doubled for non-periodic
    >>> # Create coordinate arrays with dx=2 spacing (to match LS2010)
    >>> x = np.arange(-(sim_size[0]-1), sim_size[0], 2)
    >>> y = np.arange(-(sim_size[1]-1), sim_size[1], 2)
    >>> X, Y = np.meshgrid(x, y, indexing='ij')
    >>> # Define anisotropic GSI metric with different scaling in x and y
    >>> scale_metric = (X**2 + (Y/2)**2)**0.5  # y-direction scales faster
    >>> # Use non-integer dimension for GSI
    >>> fif = FIF_ND(size, alpha=1.7, C1=0.1, H=0.3, periodic=False,
    ...              scale_metric=scale_metric, scale_metric_dim=2.3)

    Notes
    -----
    - Computational complexity is O(N log N) due to FFT-based convolutions
    - Large C1 values (> 0.5) can produce extreme values requiring careful handling
    - Always uses finite-size corrections and non-causal kernels for N-D
    - Does not support negative H values (use FIF_1D for H < 0)
    - C1 = 0: Routes internally to fBm_ND_circulant() for monofractal case
    """
    # Handle size parameter and infer dimension
    if not isinstance(size, tuple):
        raise ValueError("size must be a tuple of dimensions (e.g., (512, 512) for 2D)")

    output_size = size
    ndim = len(size)

    # Handle periodic parameter (bool or tuple of bool)
    if isinstance(periodic, bool):
        periodic_tuple = tuple(periodic for _ in range(ndim))
    elif isinstance(periodic, tuple):
        if len(periodic) != ndim:
            raise ValueError(f"periodic tuple length ({len(periodic)}) must match size tuple length ({ndim})")
        if not all(isinstance(p, bool) for p in periodic):
            raise ValueError("All elements in periodic tuple must be boolean")
        periodic_tuple = periodic
    else:
        raise ValueError("periodic must be either a bool or a tuple of bool")

    # Determine simulation domain size (double each dimension if not periodic)
    sim_size = tuple(s if periodic_tuple[i] else s * 2 for i, s in enumerate(output_size))

    # Validate scale_metric shape if provided
    if scale_metric is not None:
        scale_metric = B.asarray(scale_metric)
        if scale_metric.shape != sim_size:
            raise ValueError(
                f"scale_metric shape {scale_metric.shape} must match simulation domain shape {sim_size}. "
                f"For periodic={periodic}, with size={output_size}, the simulation domain is "
                f"{sim_size} (doubled along non-periodic axes)."
            )

    # Set dimension for scaling exponents (defaults to spatial dimension)
    if scale_metric_dim is None:
        scale_metric_dim = float(ndim)
    else:
        scale_metric_dim = float(scale_metric_dim)

    # C1==0 shortcut: use fBm (no intermittency)
    if C1 == 0:
        fbm = fBm_ND_circulant(sim_size, H, periodic=True)
        # Extract output based on per-axis periodicity
        output_slices = tuple(slice(None) if periodic_tuple[i] else slice(output_size[i]) for i in range(ndim))
        return fbm[output_slices]

    if outer_scale is None:
        outer_scale = max(sim_size)

    if any(s % 2 != 0 for s in sim_size):
        raise ValueError("All dimensions must be even numbers; powers of 2 are recommended.")

    if not isinstance(C1, (int, float)) or C1 < 0:
        raise ValueError("C1 must be a non-negative number.")

    if not isinstance(alpha, (int, float)) or alpha <= 0 or alpha > 2:
        raise ValueError("alpha must be a number > 0 and <= 2.")

    if alpha == 1:
        raise ValueError("alpha=1 not supported")   # requires special treatment which is not implemented

    if not isinstance(H, (int, float)) or H < 0 or H > 1:
        raise ValueError("H must be a number between 0 and 1.")

    # Generate or validate Lévy noise
    if levy_noise is None:
        total_size = np.prod(sim_size)
        noise = extremal_levy(alpha, size=total_size).reshape(sim_size)
    else:
        levy_tensor = B.asarray(levy_noise)
        if all(periodic_tuple):
            if levy_tensor.shape != sim_size:
                raise ValueError("Provided levy_noise must match the specified size.")
            noise = levy_tensor
        else:
            if levy_tensor.shape != output_size:
                raise ValueError("Provided levy_noise must match the specified size.")
            total_size = np.prod(sim_size)
            noise = extremal_levy(alpha, size=total_size).reshape(sim_size)
            # Insert provided noise at the beginning
            noise[tuple(slice(s) for s in output_size)] = levy_tensor

    # Create flux kernel (kernel 1)
    # Calculate exponent and normalization parameters for N-D flux kernel
    alpha_prime = 1.0 / (1.0 - 1.0/alpha)
    flux_exponent = -scale_metric_dim / alpha_prime
    flux_norm_ratio_exp = -scale_metric_dim / alpha
    flux_final_power = 1.0 / (alpha - 1.0)

    if kernel_construction_method == 'LS2010':
        kernel1 = create_kernel_LS2010(sim_size, flux_exponent, flux_norm_ratio_exp,
                                      causal=False, outer_scale=outer_scale,
                                      outer_scale_width_factor=outer_scale_width_factor,
                                      final_power=flux_final_power, scale_metric=scale_metric)
    else:
        raise ValueError(f"Unknown kernel_construction_method for N-D: {kernel_construction_method}")

    # Perform first convolution
    integrated = periodic_convolve_nd(noise, kernel1)

    # Scale and exponentiate to get flux
    scaled = integrated * (C1 ** (1/alpha))
    del integrated
    flux = B.exp(scaled)
    del scaled

    if H == 0:
        # Normalize and extract output based on per-axis periodicity
        flux = flux / B.mean(flux)
        output_slices = tuple(slice(None) if periodic_tuple[i] else slice(output_size[i]) for i in range(ndim))
        return flux[output_slices]

    # Create H kernel (kernel 2)
    # Calculate exponent and normalization parameters for N-D H kernel
    H_exponent = -scale_metric_dim + H
    H_norm_ratio_exp = -H

    if kernel_construction_method == 'LS2010':
        kernel2 = create_kernel_LS2010(sim_size, H_exponent, H_norm_ratio_exp,
                                      causal=False, outer_scale=outer_scale,
                                      outer_scale_width_factor=outer_scale_width_factor,
                                      final_power=None, scale_metric=scale_metric)
    else:
        raise ValueError(f"Unknown kernel_construction_method for N-D: {kernel_construction_method}")

    # Perform second convolution
    observable = periodic_convolve_nd(flux, kernel2)

    # Normalize by mean
    observable = observable / B.mean(observable)

    # Extract output based on per-axis periodicity
    output_slices = tuple(slice(None) if periodic_tuple[i] else slice(output_size[i]) for i in range(ndim))
    return observable[output_slices]
