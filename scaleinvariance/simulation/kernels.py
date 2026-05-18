from .. import backend as B


def _normalize_axis_flag(flag, ndim, name):
    """Normalize a bool|tuple[bool]|None flag to a length-``ndim`` tuple of bools.

    Used for ``causal`` and ``odd_axes`` parameters that may be specified
    globally (``True``/``False``) or per-axis (``tuple[bool]``). ``None`` is
    treated as all-False.
    """
    if flag is None:
        return tuple(False for _ in range(ndim))
    if isinstance(flag, bool):
        return tuple(flag for _ in range(ndim))
    if isinstance(flag, tuple):
        if len(flag) != ndim:
            raise ValueError(
                f"{name} tuple length ({len(flag)}) must match ndim ({ndim})."
            )
        if not all(isinstance(x, bool) for x in flag):
            raise ValueError(f"All elements of {name} tuple must be bool.")
        return flag
    raise ValueError(
        f"{name} must be bool, tuple[bool], or None; got {type(flag).__name__}."
    )


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
    del normalized_dist

    result = kernel * window
    del window
    return result

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
        domain_size = len(distance)
    else:
        # For N-D arrays, use minimum dimension size
        domain_size = min(distance.shape)

    ratio = 2.0
    cutoff_length = domain_size / 2.0
    cutoff_length2 = cutoff_length / ratio

    # Base singularity kernel (kept live; everything else below is temporary).
    base_kernel = distance ** exponent

    # First normalization constant — use exp cutoff then drop the smoothed
    # copy immediately so we never hold two full-volume copies at peak.
    exp_cutoff1 = B.exp(B.clip(-(distance / cutoff_length) ** 4, -200, 0))
    smoothed = base_kernel * exp_cutoff1
    del exp_cutoff1
    norm_constant1 = B.sum(smoothed)
    del smoothed

    # Second normalization constant (narrower cutoff).
    exp_cutoff2 = B.exp(B.clip(-(distance / cutoff_length2) ** 4, -200, 0))
    smoothed = base_kernel * exp_cutoff2
    del exp_cutoff2
    norm_constant2 = B.sum(smoothed)
    del smoothed

    ratio_factor = ratio ** norm_ratio_exponent
    normalization_factor = (ratio_factor * norm_constant1 - norm_constant2) / (ratio_factor - 1)

    # Final smoothing filter; reused once more for the correction below.
    final_filter = B.exp(B.clip(-distance / 3.0, -200, 0))
    filter_integral = B.sum(base_kernel * final_filter)

    correction_factor = -normalization_factor / filter_integral
    corrected_kernel = base_kernel * (1 + correction_factor * final_filter)
    del final_filter, base_kernel

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
    causal : bool or tuple of bool, optional
        Whether to make kernel causal. Default False.
        - bool: applied uniformly to all axes (all causal or all acausal)
        - tuple of bool: per-axis causality, length must match the number of
          spatial dimensions. ``True`` along an axis zeros out the half of
          the kernel where that axis coordinate is negative.
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
        ndim = 1
        shape = (size,)
    else:
        ndim = len(size)
        shape = tuple(size)

    causal_tuple = _normalize_axis_flag(causal, ndim, 'causal')

    # Coordinate arrays per axis. Built unconditionally even when the user
    # supplies a custom scale_metric, because per-axis causal masks need
    # signed coordinates regardless of the metric used for the radial part.
    coord_arrays = [B.arange(-(dim - 1), dim, 2) for dim in shape]

    # Build distance / scale metric.
    if scale_metric is None:
        if ndim == 1:
            distance = B.abs(coord_arrays[0])
        else:
            # |r|^2 via broadcasted 1D axes (no full meshgrid).
            dist_sq = None
            for axis_idx, axis in enumerate(coord_arrays):
                broadcast_shape = [1] * ndim
                broadcast_shape[axis_idx] = axis.shape[0]
                term = axis.reshape(broadcast_shape) ** 2
                dist_sq = term if dist_sq is None else dist_sq + term
                del term
            distance = B.sqrt(dist_sq)
            del dist_sq
    else:
        if ndim == 1:
            raise ValueError("scale_metric is not supported for 1D kernels.")
        distance = scale_metric

    kernel = _apply_LS2010_correction(distance, exponent, norm_ratio_exponent, final_power)

    # Outer scale cutoff with Hanning window (convert distance to units of dx=1).
    if outer_scale is not None:
        kernel = _apply_outer_scale(kernel, distance / 2, outer_scale, outer_scale_width_factor)

    # Per-axis causal masking: zero out coord_j < 0.
    # Coordinates are arange(-(dim-1), dim, 2) → never zero, so sign(coord) ∈ {-1, +1}
    # and (sign + 1)/2 is the indicator of the positive half (∈ {0, 1}).
    for axis_idx, is_causal in enumerate(causal_tuple):
        if not is_causal:
            continue
        coord = coord_arrays[axis_idx]
        mask_1d = (B.sign(coord) + 1) / 2
        broadcast_shape = [1] * ndim
        broadcast_shape[axis_idx] = coord.shape[0]
        kernel = kernel * mask_1d.reshape(broadcast_shape)

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

def create_kernel_spectral_odd(size, exponent, causal=False, outer_scale=None, outer_scale_width_factor=2.0,
                               final_power=None, scaling_dimension=None):
    """
    Create an odd (antisymmetric) Fourier-space transfer function from a power law.

    Like the even spectral kernel, but the transfer function is multiplied by
    i*sign(f), producing a purely imaginary spectrum whose inverse FFT is
    real and antisymmetric.

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
        Fourier-space transfer function (complex). Pass to ``periodic_convolve``
        with ``kernel_is_fourier=True``.
    """
    if causal:
        raise ValueError("Spectral odd kernel does not support causal kernels")

    if not isinstance(size, int):
        raise ValueError("Spectral odd kernel currently only supports 1D (size must be int)")

    if scaling_dimension is None:
        scaling_dimension = 1.0

    effective_exponent = exponent if final_power is None else exponent * final_power
    response_exponent = -(float(scaling_dimension) + effective_exponent)

    freqs = B.fftfreq(size, d=1.0)
    freqs_abs = B.abs(freqs)

    if outer_scale is None:
        outer_scale = size
    min_freq = 1.0 / float(outer_scale)
    freqs_regularized = B.maximum(freqs_abs, min_freq)
    del freqs_abs

    # Even magnitude response (same as standard spectral kernel)
    magnitude = freqs_regularized ** response_exponent
    del freqs_regularized
    magnitude[0] = magnitude[1]

    sign_f = B.sign(freqs)
    del freqs
    sign_f[0] = 0.0  # DC component is zero for odd function
    # Nyquist component must also be zero for real odd sequences
    if size % 2 == 0:
        sign_f[size // 2] = 0.0
    # Build the purely imaginary response at the active complex precision.
    # Assembling via real/imag avoids materialising an intermediate
    # complex128 literal from ``1j`` in float32 mode.
    response = B.zeros(magnitude.shape, dtype=B._active_complex_dtype_np())
    response.imag[:] = sign_f * magnitude
    return response
