import numpy as np
from .. import backend as B


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
    ndarray
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

# Spectral kernel construction

def create_kernel_spectral(size, exponent, causal=False, outer_scale=None, outer_scale_width_factor=2.0,
                           final_power=None, scaling_dimension=None):
    """
    Create a periodic kernel from an exact spectral-space power law.

    The returned real-space kernel is the impulse response whose circular FFT is
    the requested power-law transfer function. This is only valid for non-causal
    periodic filtering.
    """
    if causal:
        raise ValueError("Spectral kernel construction does not support causal kernels")

    if isinstance(size, int):
        shape = (size,)
        if scaling_dimension is None:
            scaling_dimension = 1.0
    else:
        shape = tuple(size)
        if scaling_dimension is None:
            scaling_dimension = float(len(shape))

    effective_exponent = exponent if final_power is None else exponent * final_power
    response_exponent = -(float(scaling_dimension) + effective_exponent)

    freq_axes = [np.fft.fftfreq(n, d=1.0) for n in shape]
    if len(shape) == 1:
        freqs = np.abs(freq_axes[0])
    else:
        freq_grids = np.meshgrid(*freq_axes, indexing='ij')
        freqs = np.sqrt(sum(grid**2 for grid in freq_grids))

    if outer_scale is None:
        outer_scale = max(shape)
    min_freq = 1.0 / float(outer_scale)
    freqs_regularized = np.maximum(freqs, min_freq)

    response = freqs_regularized ** response_exponent
    if shape == (1,):
        response[(0,)] = 1.0
    elif len(shape) == 1:
        response[0] = response[1]
    else:
        response[(0,) * len(shape)] = response[(1,) + (0,) * (len(shape) - 1)]

    impulse_response = np.real(np.fft.ifftn(response))
    kernel = np.fft.fftshift(impulse_response)
    return kernel.astype(np.float64, copy=False)


def create_kernel_spectral_odd(size, exponent, causal=False, outer_scale=None, outer_scale_width_factor=2.0,
                               final_power=None, scaling_dimension=None):
    """
    Create an odd (antisymmetric) periodic kernel from a spectral-space power law.

    Like create_kernel_spectral, but the resulting real-space kernel is made odd
    by negating its left half. The spectral transfer function is multiplied by
    i*sign(f), producing a purely imaginary spectrum whose inverse FFT is
    antisymmetric.

    Only valid for 1D, non-causal, periodic filtering.

    Parameters
    ----------
    size : int
        Size of kernel array (1D only).
    exponent : float
        Power-law exponent for the base kernel.
    causal : bool, optional
        Must be False; causal not supported. Default False.
    outer_scale : float, optional
        Large-scale cutoff. If None, defaults to size.
    outer_scale_width_factor : float, optional
        Controls transition width for outer scale. Default is 2.0.
    final_power : float, optional
        Final power transformation. Default None.
    scaling_dimension : float, optional
        Scaling dimension override. Default 1.0 for 1D.

    Returns
    -------
    ndarray
        Odd (antisymmetric) real-space kernel.
    """
    if causal:
        raise ValueError("Spectral odd kernel does not support causal kernels")

    if not isinstance(size, int):
        raise ValueError("Spectral odd kernel currently only supports 1D (size must be int)")

    if scaling_dimension is None:
        scaling_dimension = 1.0

    effective_exponent = exponent if final_power is None else exponent * final_power
    response_exponent = -(float(scaling_dimension) + effective_exponent)

    freqs = np.fft.fftfreq(size, d=1.0)
    freqs_abs = np.abs(freqs)

    if outer_scale is None:
        outer_scale = size
    min_freq = 1.0 / float(outer_scale)
    freqs_regularized = np.maximum(freqs_abs, min_freq)

    # Even magnitude response (same as standard spectral kernel)
    magnitude = freqs_regularized ** response_exponent
    magnitude[0] = magnitude[1]

    # Make odd: multiply by i*sign(f)
    # This gives a purely imaginary spectrum whose IFFT is real and antisymmetric
    sign_f = np.sign(freqs)
    sign_f[0] = 0.0  # DC component is zero for odd function
    # Nyquist component must also be zero for real odd sequences
    if size % 2 == 0:
        sign_f[size // 2] = 0.0
    response = 1j * sign_f * magnitude

    impulse_response = np.real(np.fft.ifft(response))
    kernel = np.fft.fftshift(impulse_response)
    return kernel.astype(np.float64, copy=False)
