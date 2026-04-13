import numpy as np
import warnings
from .. import backend as B
from .fbm import fBm_1D_circulant, fBm_ND_circulant
from .kernels import (
    create_kernel_LS2010,
    create_kernel_naive,
    create_kernel_spectral_odd,
)
from .fractional_integration import fractional_integral_spectral

def _clip_and_exp_flux(scaled, caller_name):
    """Clip the log-flux to safe exponent range for the active precision,
    warn if saturation occurred, and return ``exp(scaled)``.

    Prevents unbounded `exp(scaled)` from producing ``inf`` for extreme
    cascade amplitudes, and lets users know their simulation hit the
    ceiling instead of silently saturating.
    """
    lo, hi = B.exp_clip_limits()
    saturated = int(B.sum(scaled < lo)) + int(B.sum(scaled > hi))
    scaled_clipped = B.clip(scaled, lo, hi)
    flux = B.exp(scaled_clipped)
    if saturated > 0:
        warnings.warn(
            f"{caller_name}: {saturated} flux samples clipped at exp limits "
            f"for precision={B.get_numerical_precision()} "
            "(cascade saturated; consider smaller C1 or higher precision).",
            RuntimeWarning,
            stacklevel=3,
        )
    return flux


def _extremal_levy_core(alpha, size=1):
    """Core extremal Lévy generation. Returns native backend type (tensor or ndarray)."""
    phi = (B.rand(size) - 0.5) * B.pi
    R = B.exponential(scale=1.0, size=size)
    eps = B.eps()

    # Keep all alpha-dependent constants as Python floats — passing them
    # through numpy scalar ops would produce float64 scalars that then
    # promote the array results back to float64 under numpy's scalar
    # promotion rules.
    alpha_t = float(alpha)
    phi0 = -(np.pi / 2.0) * (1.0 - abs(1.0 - alpha_t)) / alpha_t
    abs_alpha1 = abs(alpha_t - 1.0)
    if abs_alpha1 < eps:
        abs_alpha1 = eps
    sign_alpha_minus_1 = 1.0 if alpha_t > 1.0 else -1.0
    inv_alpha_neg = -1.0 / alpha_t
    pow_denom_R = (1.0 - alpha_t) / alpha_t

    # phi ∈ [-pi/2, pi/2] so cos(phi) ≥ 0; clip protects the negative-power
    # below against underflow (and floating-point drift at the endpoints).
    cos_phi = B.clip(B.cos(phi), eps, None)
    denom = B.clip(B.cos(phi - alpha_t * (phi - phi0)), eps, None)
    R = B.clip(R, eps, None)

    sample = (
        sign_alpha_minus_1
        * B.sin(alpha_t * (phi - phi0))
        * (cos_phi * abs_alpha1) ** inv_alpha_neg
        * (denom / R) ** pow_denom_R
    )
    del phi, cos_phi, denom, R
    return sample


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
        numpy.ndarray: Array of generated extremal Lévy random variables
    """
    return B.to_numpy(_extremal_levy_core(alpha, size))

# Convolutions
def periodic_convolve(signal, kernel, kernel_is_fourier=False):
    """
    Performs periodic convolution of two 1D arrays using Fourier methods.

    When the kernel is real (either a real-space kernel or a real-valued
    Fourier-space response), this uses rfft/irfft, roughly halving the
    complex buffer memory. When the kernel is a complex Fourier-space
    response (e.g. from ``create_kernel_spectral_odd``), it falls back to
    the full fft/ifft path.

    Fourier-space kernels may be supplied as either full-size ``N`` or
    packed rfft half-size ``N//2+1``.

    Parameters:
        signal (array-like): Input signal array.
        kernel (array-like): Convolution kernel array (real-space) or
            Fourier-space transfer function if *kernel_is_fourier* is True.
        kernel_is_fourier (bool): If True, *kernel* is already in Fourier space.

    Returns:
        ndarray: The result of the periodic convolution.

    Raises:
        ValueError: If signal and kernel lengths do not match.
    """
    signal = B.asarray(signal)
    n = len(signal)

    # Full-size complex Fourier kernel (e.g. spectral_odd): fall back to fft.
    if kernel_is_fourier and B.iscomplexobj(kernel):
        fft_kernel = kernel
        if n != len(fft_kernel):
            raise ValueError(
                "Signal and kernel must have the same length for periodic convolution."
            )
        fft_signal = B.fft(signal)
        return B.real(B.ifft(fft_signal * fft_kernel))

    # Real-kernel path: use rfft/irfft.
    if kernel_is_fourier:
        kernel_arr = B.asarray(kernel)
        expected_half = n // 2 + 1
        if len(kernel_arr) == expected_half:
            rfft_kernel = kernel_arr
        elif len(kernel_arr) == n:
            # Legacy: full-size real Fourier response — the first half is
            # the non-redundant rfft packing for Hermitian-symmetric data.
            rfft_kernel = kernel_arr[:expected_half]
        else:
            raise ValueError(
                "Fourier-space kernel size must be N or N//2+1 to match signal "
                f"length {n}, got {len(kernel_arr)}."
            )
    else:
        kernel_arr = B.asarray(kernel)
        if len(kernel_arr) != n:
            raise ValueError(
                "Signal and kernel must have the same length for periodic convolution."
            )
        rfft_kernel = B.rfft(B.ifftshift(kernel_arr))

    rfft_signal = B.rfft(signal)
    return B.irfft(rfft_signal * rfft_kernel, n=n)


def periodic_convolve_nd(signal, kernel, kernel_is_fourier=False):
    """
    Performs periodic convolution of two N-D arrays using Fourier methods.

    Uses rfftn/irfftn for real kernels (halving the complex buffer). When
    the kernel is a full-size complex Fourier-space array, falls back to
    fftn/ifftn like the 1D version.

    Fourier-space kernels may be supplied as either full-size or packed
    rfft half-size (``shape[:-1] + (shape[-1]//2+1,)``).

    Parameters:
        signal (ndarray): Input signal array (N-D).
        kernel (ndarray): Convolution kernel array (N-D) or Fourier-space
            transfer function if *kernel_is_fourier* is True.
        kernel_is_fourier (bool): If True, *kernel* is already in Fourier space.

    Returns:
        ndarray: The result of the periodic convolution.

    Raises:
        ValueError: If signal and kernel shapes do not match.
    """
    signal = B.asarray(signal)
    shape = signal.shape

    # Full-size complex Fourier kernel: fall back to fftn/ifftn.
    if kernel_is_fourier and B.iscomplexobj(kernel):
        fft_kernel = kernel
        if shape != fft_kernel.shape:
            raise ValueError(
                "Signal and kernel must have the same shape for periodic convolution."
            )
        fft_signal = B.fftn(signal)
        return B.real(B.ifftn(fft_signal * fft_kernel))

    # Real-kernel path: rfftn/irfftn.
    expected_half_shape = shape[:-1] + (shape[-1] // 2 + 1,)
    kernel_arr = B.asarray(kernel)
    if kernel_is_fourier:
        if kernel_arr.shape == expected_half_shape:
            rfftn_kernel = kernel_arr
        elif kernel_arr.shape == shape:
            # Legacy full-size real Fourier response → take rfft packing.
            slices = tuple(slice(None) for _ in range(len(shape) - 1)) + (
                slice(None, shape[-1] // 2 + 1),
            )
            rfftn_kernel = kernel_arr[slices]
        else:
            raise ValueError(
                f"Fourier-space kernel shape {kernel_arr.shape} must be "
                f"{shape} (full) or {expected_half_shape} (rfft) to match signal."
            )
    else:
        if kernel_arr.shape != shape:
            raise ValueError(
                "Signal and kernel must have the same shape for periodic convolution."
            )
        rfftn_kernel = B.rfftn(B.ifftshift(kernel_arr))

    rfftn_signal = B.rfftn(signal)
    return B.irfftn(rfftn_signal * rfftn_kernel, s=shape)


def _warn_if_naive_kernel_selected(kernel_construction_method_flux, kernel_construction_method_observable):
    """Warn when legacy naive FIF kernels are selected."""
    if 'naive' in (kernel_construction_method_flux, kernel_construction_method_observable):
        warnings.warn(
            "Naive FIF kernels are deprecated because their outputs are not remotely accurate. "
            "Use kernel_construction_method_flux='LS2010' or "
            "kernel_construction_method_observable='LS2010'/'spectral' instead.",
            FutureWarning,
            stacklevel=2,
        )

# FIF

def FIF_1D(size, alpha, C1, H, levy_noise=None, causal=False, outer_scale=None,
           outer_scale_width_factor=2.0, kernel_construction_method_flux='LS2010',
           kernel_construction_method_observable='spectral', periodic=True):
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
        Lévy stability parameter in [0.5, 2] and != 1. Controls noise distribution.
        Values < 0.5 are currently not implemented for FIF simulation.
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
        Use causal kernels (future doesn't affect past). Default False.
        False gives symmetric (non-causal) kernels.
    outer_scale : int, optional
        Large-scale cutoff. Defaults to size.
    outer_scale_width_factor : float, optional
        Controls transition width for outer scale. Transition width = outer_scale * width_factor.
        Default is 2.0.
    kernel_construction_method_flux : str, optional
        Method for constructing flux (cascade) kernel. Options:
        - 'LS2010': Lovejoy & Schertzer 2010 finite-size corrections (default)
        - 'naive': Simple power-law kernels without corrections
    kernel_construction_method_observable : str, optional
        Method for constructing observable (H) kernel. Options:
        - 'spectral': Fourier-space power-law transfer function (default, non-causal only)
        - 'LS2010': Lovejoy & Schertzer 2010 finite-size corrections
        - 'naive': Simple power-law kernels without corrections
        - 'spectral_odd': Odd (antisymmetric) spectral kernel (non-causal only)
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
    ValueError
        If alpha < 0.5 or alpha == 1 (not implemented)

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
    - alpha < 0.5 and alpha = 1 are not currently implemented
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
        return B.to_numpy(result)

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

    if alpha < 0.5:
        raise ValueError("alpha < 0.5 is not implemented for FIF simulation")

    if alpha == 1:
        raise ValueError("alpha=1 not supported")   # requires special treatment which is not implemented

    _warn_if_naive_kernel_selected(kernel_construction_method_flux, kernel_construction_method_observable)

    if levy_noise is None:
        noise = _extremal_levy_core(alpha, size=size)
    else:
        levy_noise_arr = B.asarray(levy_noise)
        if B.numel(levy_noise_arr) != output_size:
            raise ValueError("Provided levy_noise must match the specified size.")
        # if aperiodic is requested, need to pad noise with more noise
        if not periodic:
            noise = B.concatenate([levy_noise_arr, _extremal_levy_core(alpha, size=output_size)])
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

    if kernel_construction_method_flux == 'LS2010':
        kernel1 = create_kernel_LS2010(size, flux_exponent, flux_norm_ratio_exp,
                                      causal=causal, outer_scale=outer_scale,
                                      outer_scale_width_factor=outer_scale_width_factor,
                                      final_power=flux_final_power)
    elif kernel_construction_method_flux == 'naive':
        kernel1 = create_kernel_naive(size, flux_exponent, causal=causal, outer_scale=outer_scale,
                                     outer_scale_width_factor=outer_scale_width_factor)
    else:
        raise ValueError(f"Unknown kernel_construction_method_flux: {kernel_construction_method_flux}. "
                         "Supported: 'LS2010', 'naive'.")

    integrated = periodic_convolve(noise, kernel1)
    del noise, kernel1  # Clean memory

    # If causal, adjust for the fact that half the kernel is being deleted
    if causal:
        causality_factor = 2.0
    else:
        causality_factor = 1.0


    scaled = integrated * ((causality_factor * C1) ** (1/alpha))
    del integrated
    flux = _clip_and_exp_flux(scaled, "FIF_1D")
    del scaled

    if H == 0:
        # Normalize - slice first (if periodic), then normalize by mean
        if not periodic:
            flux = flux[:size//2]     # eliminate periodicity by removing the part corresponding to the appended noise
        flux = flux / B.mean(flux)
        return B.to_numpy(flux)

    # Pre-normalize flux before the observable convolution to prevent float32
    # overflow in the FFT.  We divide by max (not mean) because the sum used
    # by mean can itself overflow at float32.  The final result is always
    # re-normalized, and convolution is linear, so this is lossless.
    flux = flux / B.max(flux)

    # Observable kernel (kernel 2) real-space power-law parameters.
    # obs_kernel_exponent is the exponent of |r|^p in the real-space kernel;
    # obs_kernel_norm_ratio_exp is the exponent used in the LS2010 ratio
    # normalization. Both are derived from the Hurst exponent H.
    obs_kernel_exponent = -1.0 + H
    obs_kernel_norm_ratio_exp = -H

    if kernel_construction_method_observable == 'spectral':
        if causal:
            raise ValueError("Spectral observable kernels do not support causal mode. "
                             "Use kernel_construction_method_observable='LS2010' with causal=True.")
        observable = fractional_integral_spectral(flux, H, outer_scale=outer_scale)
        del flux
    else:
        if kernel_construction_method_observable == 'LS2010':
            kernel2 = create_kernel_LS2010(size, obs_kernel_exponent, obs_kernel_norm_ratio_exp,
                                          causal=causal, outer_scale=outer_scale,
                                          outer_scale_width_factor=outer_scale_width_factor,
                                          final_power=None)
        elif kernel_construction_method_observable == 'naive':
            kernel2 = create_kernel_naive(size, obs_kernel_exponent, causal=causal, outer_scale=outer_scale,
                                         outer_scale_width_factor=outer_scale_width_factor)
        elif kernel_construction_method_observable == 'spectral_odd':
            if causal:
                raise ValueError("Spectral odd observable kernels do not support causal mode. "
                                 "Use kernel_construction_method_observable='LS2010' with causal=True.")
            kernel2 = create_kernel_spectral_odd(size, obs_kernel_exponent, causal=False, outer_scale=outer_scale,
                                                 outer_scale_width_factor=outer_scale_width_factor)
        else:
            raise ValueError(f"Unknown kernel_construction_method_observable: {kernel_construction_method_observable}")

        kernel2_is_fourier = kernel_construction_method_observable == 'spectral_odd'
        observable = periodic_convolve(flux, kernel2, kernel_is_fourier=kernel2_is_fourier)
        del flux, kernel2

    if not periodic:
        observable = observable[:size//2]     # eliminate periodicity by removing the part corresponding to the appended noise

    if H_int == -1:
        # Apply differencing for H<0
        observable = B.diff(observable)
        # Duplicate last value to maintain original size
        observable = B.concatenate([observable, observable[-1:]])
        # Normalize to zero mean (increments should have zero mean)
        observable = observable - B.mean(observable)
    elif kernel_construction_method_observable == 'spectral_odd':
        # Odd kernel produces zero-mean output (antisymmetric kernel sums to zero).
        # Normalize to zero mean and unit variance instead of unit mean.
        observable = observable - B.mean(observable)
        observable = observable / B.std(observable)
    else:
        # Normalize to unit mean (levels should have unit mean)
        observable = observable / B.mean(observable)

    return B.to_numpy(observable)

def FIF_ND(size, alpha, C1, H, levy_noise=None, outer_scale=None, outer_scale_width_factor=2.0,
           kernel_construction_method_flux='LS2010', kernel_construction_method_observable='spectral',
           periodic=False, scale_metric=None, scale_metric_dim=None):
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
        Lévy stability parameter in [0.5, 2] and != 1. Controls noise distribution.
        Values < 0.5 are currently not implemented for FIF simulation.
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
    kernel_construction_method_flux : str, optional
        Method for constructing flux (cascade) kernel. Options:
        - 'LS2010': Lovejoy & Schertzer 2010 finite-size corrections (default)
    kernel_construction_method_observable : str, optional
        Method for constructing observable (H) kernel. Options:
        - 'spectral': Fourier-space power-law transfer function (default, requires no scale_metric)
        - 'LS2010': Lovejoy & Schertzer 2010 finite-size corrections
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
    ValueError
        If alpha < 0.5 or alpha == 1 (not implemented)

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
    ...              scale_metric=scale_metric, scale_metric_dim=2.3,
    ...              kernel_construction_method_observable='LS2010')

    Notes
    -----
    - Computational complexity is O(N log N) due to FFT-based convolutions
    - Large C1 values (> 0.5) can produce extreme values requiring careful handling
    - alpha < 0.5 and alpha = 1 are not currently implemented
    - N-D kernels are always non-causal
    - Spectral N-D kernels currently assume the default isotropic Fourier metric and
      do not support custom `scale_metric`
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
        # The metric is raised to a negative power inside the LS2010 kernel
        # construction, so zeros or negatives produce inf/nan silently.
        if not B.all(B.asarray(np.isfinite(B.to_numpy(scale_metric)))) or B.any(scale_metric <= 0):
            raise ValueError(
                "scale_metric must be finite and strictly positive everywhere"
            )

    # Set dimension for scaling exponents (defaults to spatial dimension)
    if scale_metric_dim is None:
        scale_metric_dim = float(ndim)
    else:
        scale_metric_dim = float(scale_metric_dim)

    # C1==0 shortcut: use fBm (no intermittency)
    if C1 == 0:
        if levy_noise is not None:
            raise ValueError('levy_noise argument not supported for C1=0 (fBm shortcut)')
        return B.to_numpy(fBm_ND_circulant(output_size, H, periodic=periodic_tuple))

    if outer_scale is None:
        outer_scale = max(sim_size)

    if any(s % 2 != 0 for s in sim_size):
        raise ValueError("All dimensions must be even numbers; powers of 2 are recommended.")

    if not isinstance(C1, (int, float)) or C1 < 0:
        raise ValueError("C1 must be a non-negative number.")

    if not isinstance(alpha, (int, float)) or alpha <= 0 or alpha > 2:
        raise ValueError("alpha must be a number > 0 and <= 2.")

    if alpha < 0.5:
        raise ValueError("alpha < 0.5 is not implemented for FIF simulation")

    if alpha == 1:
        raise ValueError("alpha=1 not supported")   # requires special treatment which is not implemented

    if not isinstance(H, (int, float)) or H < 0 or H > 1:
        raise ValueError("H must be a number between 0 and 1.")

    # Generate or validate Lévy noise
    if levy_noise is None:
        total_size = np.prod(sim_size)
        noise = _extremal_levy_core(alpha, size=total_size).reshape(sim_size)
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
            noise = _extremal_levy_core(alpha, size=total_size).reshape(sim_size)
            # Insert provided noise at the beginning
            noise[tuple(slice(s) for s in output_size)] = levy_tensor

    # Create flux kernel (kernel 1)
    # Calculate exponent and normalization parameters for N-D flux kernel
    alpha_prime = 1.0 / (1.0 - 1.0/alpha)
    flux_exponent = -scale_metric_dim / alpha_prime
    flux_norm_ratio_exp = -scale_metric_dim / alpha
    flux_final_power = 1.0 / (alpha - 1.0)

    if kernel_construction_method_flux == 'LS2010':
        kernel1 = create_kernel_LS2010(sim_size, flux_exponent, flux_norm_ratio_exp,
                                      causal=False, outer_scale=outer_scale,
                                      outer_scale_width_factor=outer_scale_width_factor,
                                      final_power=flux_final_power, scale_metric=scale_metric)
    else:
        raise ValueError(f"Unknown kernel_construction_method_flux for N-D: {kernel_construction_method_flux}. "
                         "Supported: 'LS2010'.")

    # Perform first convolution
    integrated = periodic_convolve_nd(noise, kernel1)
    del noise, kernel1

    # Scale and exponentiate to get flux
    scaled = integrated * (C1 ** (1/alpha))
    del integrated
    flux = _clip_and_exp_flux(scaled, "FIF_ND")
    del scaled

    if H == 0:
        output_slices = tuple(slice(None) if periodic_tuple[i] else slice(output_size[i]) for i in range(ndim))
        flux = flux[output_slices]
        return B.to_numpy(flux / B.mean(flux))

    # Pre-normalize flux before the observable convolution to prevent float32
    # overflow in the FFT.  We divide by max (not mean) because the sum used
    # by mean can itself overflow at float32.  The final result is always
    # re-normalized, and convolution is linear, so this is lossless.
    flux = flux / B.max(flux)

    # Observable kernel (kernel 2) real-space power-law parameters, derived
    # from the Hurst exponent H and the scaling dimension.
    obs_kernel_exponent = -scale_metric_dim + H
    obs_kernel_norm_ratio_exp = -H

    if kernel_construction_method_observable == 'spectral':
        if scale_metric is not None:
            raise ValueError("Spectral observable kernels do not support custom scale_metric. "
                             "Use kernel_construction_method_observable='LS2010' with scale_metric.")
        observable = fractional_integral_spectral(flux, H, outer_scale=outer_scale)
        del flux
    elif kernel_construction_method_observable == 'LS2010':
        kernel2 = create_kernel_LS2010(sim_size, obs_kernel_exponent, obs_kernel_norm_ratio_exp,
                                      causal=False, outer_scale=outer_scale,
                                      outer_scale_width_factor=outer_scale_width_factor,
                                      final_power=None, scale_metric=scale_metric)
        observable = periodic_convolve_nd(flux, kernel2)
        del flux, kernel2
    else:
        raise ValueError(f"Unknown kernel_construction_method_observable for N-D: {kernel_construction_method_observable}")

    # Extract output based on per-axis periodicity, then normalize by mean of returned portion
    output_slices = tuple(slice(None) if periodic_tuple[i] else slice(output_size[i]) for i in range(ndim))
    observable = observable[output_slices]
    observable = observable / B.mean(observable)
    return B.to_numpy(observable)
