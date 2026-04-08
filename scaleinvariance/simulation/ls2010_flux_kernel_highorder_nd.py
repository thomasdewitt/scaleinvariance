import numpy as np
import warnings

try:
    B  # noqa: F821
except NameError:  # standalone use
    B = np


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def _apply_outer_scale_highorder(kernel, distance, outer_scale, width_factor=2.0):
    """
    Same Hanning outer-scale taper used in the existing kernels.py.
    `distance` is expected in dx=1 units here.
    """
    if outer_scale is None:
        return kernel

    distance = np.asarray(distance, dtype=float)
    kernel = np.asarray(kernel, dtype=float)

    transition_width = float(outer_scale) * float(width_factor)
    lower_edge = float(outer_scale) - transition_width / 2.0
    upper_edge = float(outer_scale) + transition_width / 2.0

    normalized = np.clip((distance - lower_edge) / transition_width, 0.0, 1.0)
    window = 0.5 * (1.0 + np.cos(np.pi * normalized))
    return kernel * window



def _default_lambda_grid(order, lambda_base):
    return float(lambda_base) * (2.0 ** np.arange(order, dtype=float))



def _make_distance_field_nd(shape, scale_metric=None):
    """
    Create the N-D LS2010 odd-grid distance field with dx=2 spacing.

    If `scale_metric` is supplied it is used directly; otherwise the Euclidean
    odd-grid distance is generated with broadcasting rather than a dense meshgrid.
    """
    shape = tuple(int(s) for s in shape)

    if scale_metric is not None:
        distance = np.asarray(scale_metric, dtype=float)
        if distance.shape != shape:
            raise ValueError(
                f"scale_metric.shape={distance.shape} must match size={shape}."
            )
        return np.maximum(distance, 1.0e-12)

    ndim = len(shape)
    distance_sq = np.zeros(shape, dtype=float)

    for ax, n in enumerate(shape):
        coords = np.arange(-(n - 1), n, 2, dtype=float)
        reshape = [1] * ndim
        reshape[ax] = n
        distance_sq += coords.reshape(reshape) ** 2

    return np.sqrt(np.maximum(distance_sq, 1.0e-24))



def _central_crop(arr, shape):
    """Crop an N-D array symmetrically to the requested shape."""
    arr = np.asarray(arr)
    shape = tuple(int(s) for s in shape)

    if arr.ndim != len(shape):
        raise ValueError(
            f"Crop rank mismatch: arr.ndim={arr.ndim}, requested shape={shape}."
        )

    slices = []
    for n, m in zip(arr.shape, shape):
        if m > n:
            raise ValueError(f"Cannot crop axis from {n} to {m}.")
        start = (n - m) // 2
        slices.append(slice(start, start + m))

    return np.asarray(arr[tuple(slices)], dtype=float)



def _shift_with_zeros_nd(arr, offset, spatial_ndim=None):
    """
    Shift an array by integer index offsets and zero-fill the uncovered region.

    Parameters
    ----------
    arr : ndarray
        Spatial array, or spatial array with extra trailing parameter axes.
    offset : tuple of int
        Shift in index units (not physical units).  Because the LS2010 odd grid
        has spacing 2, an index shift of +1 corresponds to a physical lag of +2.
    spatial_ndim : int, optional
        Number of leading spatial axes.  Trailing axes are copied unchanged.
    """
    offset = tuple(int(o) for o in offset)
    if spatial_ndim is None:
        spatial_ndim = len(offset)

    out = np.zeros_like(arr)
    src_slices = []
    dst_slices = []

    for ax, sh in enumerate(offset[:spatial_ndim]):
        n = arr.shape[ax]
        if abs(sh) >= n:
            return out

        if sh >= 0:
            src_slices.append(slice(0, n - sh))
            dst_slices.append(slice(sh, n))
        else:
            src_slices.append(slice(-sh, n))
            dst_slices.append(slice(0, n + sh))

    extra_ndim = arr.ndim - spatial_ndim
    if extra_ndim > 0:
        src_slices.extend([slice(None)] * extra_ndim)
        dst_slices.extend([slice(None)] * extra_ndim)

    out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out



def _normalize_direction(direction):
    """
    Reduce an integer direction vector to a primitive non-negative direction.

    Examples
    --------
    (2, 2, 0) -> (1, 1, 0)
    (-3, 0)   -> (1, 0)
    """
    direction = tuple(int(v) for v in direction)
    if not any(direction):
        raise ValueError("direction may not be the zero vector")

    direction = tuple(abs(v) for v in direction)

    from math import gcd

    g = 0
    for v in direction:
        g = gcd(g, v)
    if g <= 0:
        g = 1

    return tuple(v // g for v in direction)



def _default_fit_directions(ndim, scale_metric_is_default=True):
    """
    Default set of rays used to calibrate the N-D kernel.

    For isotropic Euclidean kernels this uses all coordinate axes plus the full
    diagonal.  For custom scale metrics, the safe default is axes only.
    """
    directions = [tuple(1 if i == ax else 0 for i in range(ndim)) for ax in range(ndim)]

    if scale_metric_is_default and ndim >= 2:
        directions.append(tuple(1 for _ in range(ndim)))

    out = []
    seen = set()
    for d in directions:
        d = _normalize_direction(d)
        if d not in seen:
            seen.add(d)
            out.append(d)

    return out



def _prepare_fit_directions(fit_directions, ndim, scale_metric_is_default):
    if fit_directions in (None, "auto"):
        return _default_fit_directions(ndim, scale_metric_is_default)

    out = []
    seen = set()
    for direction in fit_directions:
        if len(direction) != ndim:
            raise ValueError(
                f"Each fit direction must have length {ndim}; got {direction}."
            )
        d = _normalize_direction(direction)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out



def _default_fit_shape(full_shape, max_fit_points=65536, min_side=16):
    """
    Choose a modest calibration domain so the fit remains cheap in 2D/3D.

    The correction is local in scale, so fitting on the full simulation grid is
    not usually necessary.
    """
    full_shape = tuple(int(s) for s in full_shape)
    ndim = len(full_shape)

    target_side = int(np.floor(float(max_fit_points) ** (1.0 / ndim)))
    target_side = max(int(min_side), target_side)
    if target_side % 2 != 0:
        target_side -= 1
    target_side = max(4, target_side)

    fit_shape = []
    for n in full_shape:
        m = min(n, target_side)
        if m % 2 != 0:
            m -= 1
        fit_shape.append(max(4, m))

    return tuple(fit_shape)


# -----------------------------------------------------------------------------
# N-D higher-order LS2010 flux kernel
# -----------------------------------------------------------------------------

def _build_flux_kernel_and_jacobian_nd(
    shape,
    alpha,
    d_eff,
    coeffs,
    lambdas,
    outer_scale=None,
    outer_scale_width_factor=2.0,
    scale_metric=None,
    need_jac=True,
):
    """
    Build the corrected N-D flux kernel and (optionally) its Jacobian.

    Radial ansatz:
        g(r) = r^{-d_eff/alpha} * (1 + sum_m c_m exp(-r/lambda_m))^{1/(alpha-1)}

    Here `r` is either the Euclidean odd-grid distance or the user-supplied
    `scale_metric`, and `d_eff` is the scaling dimension (typically len(shape),
    but it may be `scale_metric_dim` for GSI-style metrics).

    This is the direct N-D extension of the 1D five-mode LS2010 kernel.
    """
    if abs(alpha - 1.0) < 1e-12:
        raise ValueError("alpha=1 is not supported.")

    shape = tuple(int(s) for s in shape)
    coeffs = np.asarray(coeffs, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)

    distance_dx2 = _make_distance_field_nd(shape, scale_metric=scale_metric)

    inside = np.ones(shape, dtype=float)
    mode_arrays = []
    for lam, coeff in zip(lambdas, coeffs):
        mode = np.exp(-distance_dx2 / lam)
        inside += coeff * mode
        if need_jac:
            mode_arrays.append(mode)

    if np.any(inside <= 0.0):
        return None, None, None, None

    power = 1.0 / (float(alpha) - 1.0)
    kernel = distance_dx2 ** (-float(d_eff) / float(alpha))
    kernel = kernel * inside ** power

    if outer_scale is not None:
        taper = _apply_outer_scale_highorder(
            np.ones_like(kernel),
            distance_dx2 / 2.0,
            outer_scale=outer_scale,
            width_factor=outer_scale_width_factor,
        )
        kernel = kernel * taper

    if need_jac:
        prefactor = (power * kernel / inside)
        jac = np.stack([prefactor * mode for mode in mode_arrays], axis=-1)
    else:
        jac = None

    return kernel, jac, inside, distance_dx2



def _Sf_residual_and_jacobian_nd(
    shape,
    alpha,
    d_eff,
    coeffs,
    lambdas,
    max_fit_step,
    outer_scale=None,
    outer_scale_width_factor=2.0,
    scale_metric=None,
    fit_directions=None,
    weight_power=0.0,
):
    """
    Compute the projected LS2010 residual for the N-D discrete log-correlation.

    The ideal isotropic/acausal discrete flux kernel yields S_f(Delta) affine in
    log|Delta|.  We therefore evaluate S_f along a small set of rays, project out
    span{1, log|Delta|}, and fit the remaining curvature.
    """
    built = _build_flux_kernel_and_jacobian_nd(
        shape=shape,
        alpha=alpha,
        d_eff=d_eff,
        coeffs=coeffs,
        lambdas=lambdas,
        outer_scale=outer_scale,
        outer_scale_width_factor=outer_scale_width_factor,
        scale_metric=scale_metric,
        need_jac=True,
    )
    kernel, jac, inside, distance = built
    if kernel is None:
        return None, None, None, None

    ndim = len(shape)
    directions = _prepare_fit_directions(
        fit_directions,
        ndim=ndim,
        scale_metric_is_default=(scale_metric is None),
    )

    spatial_axes = tuple(range(ndim))
    Sf_list = []
    JSf_list = []
    lag_list = []

    max_fit_step = int(min(max_fit_step, max(1, (min(shape) - 2) // 2)))

    for direction in directions:
        direction_norm = float(np.linalg.norm(direction))
        for step in range(1, max_fit_step + 1):
            offset = tuple(step * d for d in direction)

            shifted_kernel = _shift_with_zeros_nd(kernel, offset, spatial_ndim=ndim)
            shifted_jac = _shift_with_zeros_nd(jac, offset, spatial_ndim=ndim)

            pair_sum = shifted_kernel + kernel
            Sf_list.append(float(np.sum(pair_sum ** alpha)))
            JSf_list.append(
                np.sum(
                    (alpha * pair_sum[..., None] ** (alpha - 1.0)) * (shifted_jac + jac),
                    axis=spatial_axes,
                )
            )
            # The constant conversion factor between index lag and physical lag is
            # absorbed by the affine fit in log-lag, so step*||direction|| is enough.
            lag_list.append(float(step) * direction_norm)

    Sf = np.asarray(Sf_list, dtype=float)
    JSf = np.asarray(JSf_list, dtype=float)
    lags = np.asarray(lag_list, dtype=float)

    # Remove the pure scaling part: best affine fit in log-lag.
    X = np.column_stack([np.ones_like(lags), np.log(lags)])
    projector = np.eye(len(lags)) - X @ np.linalg.pinv(X)

    residual = projector @ Sf
    jac_residual = projector @ JSf

    if weight_power != 0.0:
        weights = lags ** (-float(weight_power))
        residual = weights * residual
        jac_residual = weights[:, None] * jac_residual

    return lags, residual, jac_residual, kernel



def _fit_flux_mode_amplitudes_nd(
    size,
    alpha,
    d_eff,
    order=5,
    lambdas=None,
    max_fit_step=None,
    outer_scale=None,
    outer_scale_width_factor=2.0,
    fit_shape=None,
    scale_metric=None,
    fit_directions="auto",
    fit_steps=6,
    fit_reg=1.0e-4,
    weight_power=0.0,
):
    """
    Fit the N-D multi-mode LS2010 amplitudes with a damped Gauss-Newton solve.

    The fit is performed on a modest central crop (`fit_shape`) because the bias is
    a local small-scale effect and does not require the full production grid.
    """
    full_shape = tuple(int(s) for s in size)

    if lambdas is None:
        lambdas = _default_lambda_grid(order=order, lambda_base=1.5)
    else:
        lambdas = np.asarray(lambdas, dtype=float)
        order = int(lambdas.size)

    if fit_shape is None:
        fit_shape = _default_fit_shape(full_shape)
    else:
        fit_shape = tuple(int(s) for s in fit_shape)
        if len(fit_shape) != len(full_shape):
            raise ValueError(
                f"fit_shape must have length {len(full_shape)}; got {fit_shape}."
            )
        if any(m > n for m, n in zip(fit_shape, full_shape)):
            raise ValueError(
                f"fit_shape={fit_shape} must not exceed full size={full_shape}."
            )
        if any(m % 2 != 0 for m in fit_shape):
            raise ValueError("Each fit_shape dimension must be even.")

    fit_scale_metric = None
    if scale_metric is not None:
        fit_scale_metric = _central_crop(scale_metric, fit_shape)

    if max_fit_step is None:
        if outer_scale is None:
            effective_scale = min(fit_shape)
        else:
            effective_scale = min(min(fit_shape), int(max(8, round(float(outer_scale)))))
        max_fit_step = min(32, max(4, effective_scale // 8))

    coeffs = np.zeros(order, dtype=float)

    for _ in range(int(fit_steps)):
        out = _Sf_residual_and_jacobian_nd(
            shape=fit_shape,
            alpha=alpha,
            d_eff=d_eff,
            coeffs=coeffs,
            lambdas=lambdas,
            max_fit_step=max_fit_step,
            outer_scale=outer_scale,
            outer_scale_width_factor=outer_scale_width_factor,
            scale_metric=fit_scale_metric,
            fit_directions=fit_directions,
            weight_power=weight_power,
        )
        lags, residual, jac, kernel = out
        if residual is None:
            break

        current_obj = float(np.mean(residual ** 2))
        normal = jac.T @ jac + float(fit_reg) * np.eye(order)
        rhs = -jac.T @ residual

        try:
            step = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            break

        best = None
        for damping in (1.0, 0.7, 0.5, 0.25, 0.1, 0.05):
            trial = coeffs + damping * step
            out_trial = _Sf_residual_and_jacobian_nd(
                shape=fit_shape,
                alpha=alpha,
                d_eff=d_eff,
                coeffs=trial,
                lambdas=lambdas,
                max_fit_step=max_fit_step,
                outer_scale=outer_scale,
                outer_scale_width_factor=outer_scale_width_factor,
                scale_metric=fit_scale_metric,
                fit_directions=fit_directions,
                weight_power=weight_power,
            )
            if out_trial[1] is None:
                continue

            trial_obj = float(np.mean(out_trial[1] ** 2))
            if best is None or trial_obj < best[0]:
                best = (trial_obj, trial)

        if best is None or best[0] >= current_obj:
            break

        coeffs = best[1]

    return coeffs, lambdas, max_fit_step, fit_shape


# -----------------------------------------------------------------------------
# Public entry points
# -----------------------------------------------------------------------------

def create_kernel_flux_LS2010_highorder_nd(
    size,
    alpha,
    outer_scale=None,
    outer_scale_width_factor=2.0,
    order=5,
    lambdas=None,
    max_fit_step=None,
    fit_shape=None,
    scale_metric=None,
    scale_metric_dim=None,
    fit_directions="auto",
    fit_steps=6,
    fit_reg=1.0e-4,
    weight_power=0.0,
):
    """
    Higher-order acausal LS2010-style flux kernel for the N-D FIF flux kernel.

    This is the N-D analogue of the 1D five-mode kernel.  It fits several short-
    range radial correction modes so that the discrete S_f(Delta) surface is as
    close as possible to an affine function of log|Delta| along a small set of rays.

    Parameters
    ----------
    size : tuple of int
        Kernel shape (same convention as your current N-D create_kernel_LS2010).
    alpha : float
        Levy index.
    outer_scale : int or float, optional
        Same meaning as in your current code.
    outer_scale_width_factor : float, optional
        Same meaning as in your current code.
    order : int, optional
        Number of correction modes.  order=5 is the default.
    lambdas : array-like, optional
        Optional decay lengths for the correction modes.  Default is the geometric
        grid 1.5, 3, 6, 12, 24 for order=5.
    max_fit_step : int, optional
        Largest index-space lag used in the internal fit.  Physical lags are twice
        this because the LS2010 construction grid has dx=2.
    fit_shape : tuple of int, optional
        Optional calibration domain shape.  If omitted, a modest central crop is
        chosen automatically to keep the fit cheap in 2D/3D.
    scale_metric : ndarray, optional
        Optional custom N-D scale metric, exactly as in your current N-D code.
    scale_metric_dim : float, optional
        Effective scaling dimension.  Defaults to len(size).
    fit_directions : 'auto' or iterable of tuples, optional
        Rays used in the internal calibration.  'auto' means axes (and for the
        default Euclidean metric also the full diagonal).
    fit_steps : int, optional
        Number of damped Gauss-Newton refinement steps.
    fit_reg : float, optional
        Tikhonov regularization for the normal equations.
    weight_power : float, optional
        Optional extra weighting of small lags.

    Returns
    -------
    backend array
        N-D flux kernel ready to pass to periodic_convolve_nd().
    """
    if isinstance(size, int):
        raise ValueError(
            "create_kernel_flux_LS2010_highorder_nd expects an N-D tuple size. "
            "For 1D use the companion 1D function."
        )

    shape = tuple(int(s) for s in size)
    if len(shape) < 2:
        raise ValueError(
            "This N-D kernel expects at least 2 dimensions. For 1D use the 1D builder."
        )

    if alpha <= 0.0 or abs(alpha - 1.0) < 1.0e-12:
        raise ValueError("alpha must be > 0 and not equal to 1.")

    d_eff = float(len(shape) if scale_metric_dim is None else scale_metric_dim)

    coeffs, lambdas, max_fit_step, fit_shape = _fit_flux_mode_amplitudes_nd(
        size=shape,
        alpha=alpha,
        d_eff=d_eff,
        order=order,
        lambdas=lambdas,
        max_fit_step=max_fit_step,
        outer_scale=outer_scale,
        outer_scale_width_factor=outer_scale_width_factor,
        fit_shape=fit_shape,
        scale_metric=scale_metric,
        fit_directions=fit_directions,
        fit_steps=fit_steps,
        fit_reg=fit_reg,
        weight_power=weight_power,
    )

    built = _build_flux_kernel_and_jacobian_nd(
        shape=shape,
        alpha=alpha,
        d_eff=d_eff,
        coeffs=coeffs,
        lambdas=lambdas,
        outer_scale=outer_scale,
        outer_scale_width_factor=outer_scale_width_factor,
        scale_metric=scale_metric,
        need_jac=False,
    )
    kernel, _, _, _ = built
    if kernel is None:
        raise RuntimeError(
            "Higher-order LS2010 N-D flux fit produced a non-positive correction factor. "
            "Try fewer modes, larger lambdas, a slightly larger fit_reg, or a smaller order."
        )

    return B.asarray(kernel)



def create_kernel_LS2010_flux_highorder_nd_from_params(
    size,
    exponent,
    norm_ratio_exponent,
    causal=False,
    outer_scale=None,
    outer_scale_width_factor=2.0,
    final_power=None,
    scale_metric=None,
    scale_metric_dim=None,
    order=5,
    lambdas=None,
    max_fit_step=None,
    fit_shape=None,
    fit_directions="auto",
    fit_steps=6,
    fit_reg=1.0e-4,
    weight_power=0.0,
):
    """
    Thin wrapper so you can swap the N-D flux-kernel call site with minimal edits.

    It assumes the caller is passing the standard FIF N-D flux-kernel parameters:
        exponent            = -d_eff/alpha'
        norm_ratio_exponent = -d_eff/alpha
        final_power         =  1/(alpha-1)

    Exact replacement in FIF_ND:

        kernel1 = create_kernel_LS2010_flux_highorder_nd_from_params(
            sim_size, flux_exponent, flux_norm_ratio_exp,
            causal=False, outer_scale=outer_scale,
            outer_scale_width_factor=outer_scale_width_factor,
            final_power=flux_final_power,
            scale_metric=scale_metric,
            scale_metric_dim=scale_metric_dim,
            order=5,
        )
    """
    if causal:
        raise ValueError(
            "The N-D FIF flux kernel is acausal in the current implementation; "
            "causal=True is not supported here."
        )

    if isinstance(size, int):
        raise ValueError(
            "create_kernel_LS2010_flux_highorder_nd_from_params expects tuple size."
        )

    shape = tuple(int(s) for s in size)
    d_eff = float(len(shape) if scale_metric_dim is None else scale_metric_dim)

    alpha = -float(d_eff) / float(norm_ratio_exponent)

    expected_exponent = d_eff * (1.0 / alpha - 1.0)
    if abs(float(exponent) - expected_exponent) > 1.0e-8:
        warnings.warn(
            "The supplied exponent does not match the standard N-D FIF flux-kernel exponent. "
            "Proceeding by inferring alpha from norm_ratio_exponent.",
            RuntimeWarning,
            stacklevel=2,
        )

    expected_final_power = 1.0 / (alpha - 1.0)
    if final_power is not None and abs(float(final_power) - expected_final_power) > 1.0e-8:
        warnings.warn(
            "The supplied final_power does not match the standard FIF N-D flux-kernel value. "
            "Proceeding anyway, but note that this helper is intended only for the flux kernel.",
            RuntimeWarning,
            stacklevel=2,
        )

    return create_kernel_flux_LS2010_highorder_nd(
        size=shape,
        alpha=alpha,
        outer_scale=outer_scale,
        outer_scale_width_factor=outer_scale_width_factor,
        order=order,
        lambdas=lambdas,
        max_fit_step=max_fit_step,
        fit_shape=fit_shape,
        scale_metric=scale_metric,
        scale_metric_dim=d_eff,
        fit_directions=fit_directions,
        fit_steps=fit_steps,
        fit_reg=fit_reg,
        weight_power=weight_power,
    )
