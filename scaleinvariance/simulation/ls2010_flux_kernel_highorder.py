import numpy as np
import warnings
from .. import backend as B


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


def _odd_positions_1d(size):
    """Odd grid used by LS2010: -(size-1), ..., -1, 1, ..., size-1."""
    return np.arange(-(size - 1), size, 2, dtype=int)


def _default_lambda_grid(order, lambda_base):
    return float(lambda_base) * (2.0 ** np.arange(order, dtype=float))


def _build_flux_kernel_and_jacobian(
    size,
    alpha,
    coeffs,
    lambdas,
    outer_scale=None,
    outer_scale_width_factor=2.0,
):
    """
    Build the corrected flux kernel and its derivatives wrt the mode amplitudes.

    Ansatz:
        g(x) = |x|^{-1/alpha} * (1 + sum_m c_m exp(-|x|/lambda_m))^{1/(alpha-1)}

    This is the direct multi-mode extension of LS2010 Eq. (29) with
    f_m(x) = exp(-|x|/lambda_m).  Because
        g(x)^{alpha-1} = |x|^{1/alpha - 1} * (1 + sum_m c_m exp(-|x|/lambda_m)),
    the one-point log-scaling is unchanged at large scales.
    """
    if abs(alpha - 1.0) < 1e-12:
        raise ValueError("alpha=1 is not supported.")

    pos = _odd_positions_1d(int(size))
    distance_dx2 = np.abs(pos).astype(float)      # LS2010 odd grid, dx=2
    distance_dx1 = distance_dx2 / 2.0             # convert to user-facing dx=1 units

    modes = np.exp(-distance_dx2[:, None] / np.asarray(lambdas, dtype=float)[None, :])
    inside = 1.0 + modes @ np.asarray(coeffs, dtype=float)

    if np.any(inside <= 0.0):
        return None, None, None, None, None

    power = 1.0 / (float(alpha) - 1.0)
    kernel = distance_dx2 ** (-1.0 / float(alpha))
    kernel = kernel * inside ** power

    if outer_scale is not None:
        taper = _apply_outer_scale_highorder(
            np.ones_like(kernel),
            distance_dx1,
            outer_scale=outer_scale,
            width_factor=outer_scale_width_factor,
        )
        kernel = kernel * taper
    else:
        taper = np.ones_like(kernel)

    # d g / d c_m
    jac = (power * kernel / inside)[:, None] * modes

    return kernel, jac, inside, pos, distance_dx2


def _Sf_residual_and_jacobian(
    size,
    alpha,
    coeffs,
    lambdas,
    max_fit_delta,
    outer_scale=None,
    outer_scale_width_factor=2.0,
    weight_power=0.0,
):
    """
    Compute the LS2010 discrete log-correlation residual and its Jacobian.

    For a perfectly scaling acausal flux kernel, S_f(Delta) should be affine in log Delta.
    We therefore project out span{1, log Delta} and fit the correction amplitudes by
    minimizing the projected residual.
    """
    built = _build_flux_kernel_and_jacobian(
        size=size,
        alpha=alpha,
        coeffs=coeffs,
        lambdas=lambdas,
        outer_scale=outer_scale,
        outer_scale_width_factor=outer_scale_width_factor,
    )
    kernel, jac, inside, pos, _ = built
    if kernel is None:
        return None, None, None, None

    maxx = int(size) - 1
    vals = np.zeros(maxx + 1, dtype=float)
    dvals = np.zeros((maxx + 1, len(coeffs)), dtype=float)

    abspos = np.abs(pos)
    vals[abspos] = kernel
    dvals[abspos, :] = jac

    max_fit_delta = int(min(max_fit_delta, max(2, (int(size) - 2) // 2)))
    deltas = np.arange(2, 2 * max_fit_delta + 1, 2, dtype=int)

    Sf = np.empty(deltas.size, dtype=float)
    JSf = np.empty((deltas.size, len(coeffs)), dtype=float)

    vb = vals[abspos]
    dvb = dvals[abspos, :]

    for k, delta in enumerate(deltas):
        shifted = np.abs(pos - delta)

        va = np.zeros_like(vb)
        dva = np.zeros_like(dvb)

        mask = shifted <= maxx
        sh = shifted[mask]
        va[mask] = vals[sh]
        dva[mask, :] = dvals[sh, :]

        pair_sum = va + vb
        Sf[k] = np.sum(pair_sum ** alpha)
        JSf[k, :] = np.sum(
            (alpha * pair_sum[:, None] ** (alpha - 1.0)) * (dva + dvb),
            axis=0,
        )

    # Remove the pure scaling part: best affine fit in log Delta.
    x = np.log(deltas.astype(float))
    X = np.column_stack([np.ones_like(x), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    projector = np.eye(len(deltas)) - X @ XtX_inv @ X.T

    residual = projector @ Sf
    jac_residual = projector @ JSf

    if weight_power != 0.0:
        weights = deltas.astype(float) ** (-float(weight_power))
        residual = weights * residual
        jac_residual = weights[:, None] * jac_residual

    return deltas.astype(float), residual, jac_residual, kernel


def _fit_flux_mode_amplitudes(
    size,
    alpha,
    order=5,
    lambdas=None,
    max_fit_delta=None,
    outer_scale=None,
    outer_scale_width_factor=2.0,
    fit_steps=6,
    fit_reg=1.0e-4,
    weight_power=0.0,
):
    """
    Solve for the multi-mode LS2010 amplitudes with a damped Gauss-Newton iteration.
    """
    if lambdas is None:
        lambdas = _default_lambda_grid(order=order, lambda_base=1.5)
    else:
        lambdas = np.asarray(lambdas, dtype=float)
        order = int(lambdas.size)

    if max_fit_delta is None:
        # Only small-to-intermediate lags are needed; those are exactly the biased scales.
        if outer_scale is None:
            effective_scale = int(size)
        else:
            effective_scale = min(int(size), int(max(8, round(float(outer_scale)))))

        max_fit_delta = min(64, max(8, effective_scale // 8))

    coeffs = np.zeros(order, dtype=float)

    for _ in range(int(fit_steps)):
        out = _Sf_residual_and_jacobian(
            size=size,
            alpha=alpha,
            coeffs=coeffs,
            lambdas=lambdas,
            max_fit_delta=max_fit_delta,
            outer_scale=outer_scale,
            outer_scale_width_factor=outer_scale_width_factor,
            weight_power=weight_power,
        )
        deltas, residual, jac, kernel = out
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
            out_trial = _Sf_residual_and_jacobian(
                size=size,
                alpha=alpha,
                coeffs=trial,
                lambdas=lambdas,
                max_fit_delta=max_fit_delta,
                outer_scale=outer_scale,
                outer_scale_width_factor=outer_scale_width_factor,
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

    return coeffs, lambdas, max_fit_delta


def create_kernel_flux_LS2010_highorder(
    size,
    alpha,
    causal=False,
    outer_scale=None,
    outer_scale_width_factor=2.0,
    order=5,
    lambdas=None,
    max_fit_delta=None,
    fit_steps=6,
    fit_reg=1.0e-4,
    weight_power=0.0,
):
    """
    Higher-order acausal LS2010-style flux kernel for the 1D FIF flux kernel.

    This is a five-mode extension of LS2010's small-x correction idea.  Instead of
    the single correction factor used in the current implementation, it fits several
    exponentially localized modes so that the discrete S_f(Delta) curve is as close
    as possible to a straight line in log Delta over the biased small/intermediate
    scales.

    Parameters
    ----------
    size : int
        Kernel length (same convention as your current create_kernel_LS2010).
    alpha : float
        Levy index.
    causal : bool, optional
        Supported only in the limited sense that the fitted acausal kernel is built
        first and then truncated to one side.  The fit itself is calibrated for the
        acausal case, because that is where the problem is strongest.
    outer_scale : int or float, optional
        Same meaning as in your current code.
    outer_scale_width_factor : float, optional
        Same meaning as in your current code.
    order : int, optional
        Number of correction modes.  order=5 is the default.
    lambdas : array-like, optional
        Optional decay lengths for the correction modes.  Default is a geometric
        grid: 1.5, 3, 6, 12, 24 for order=5.
    max_fit_delta : int, optional
        Largest even lag used in the internal LS2010 residual fit.
    fit_steps : int, optional
        Number of damped Gauss-Newton refinement steps.
    fit_reg : float, optional
        Tikhonov regularization for the normal equations.
    weight_power : float, optional
        Optional small-scale weighting; 0.0 means unweighted.

    Returns
    -------
    backend array
        Flux kernel ready to pass to periodic_convolve().
    """
    if not isinstance(size, int):
        raise ValueError("create_kernel_flux_LS2010_highorder currently supports only 1D integer sizes.")

    if alpha <= 0.0 or abs(alpha - 1.0) < 1e-12:
        raise ValueError("alpha must be > 0 and not equal to 1.")

    coeffs, lambdas, max_fit_delta = _fit_flux_mode_amplitudes(
        size=size,
        alpha=alpha,
        order=order,
        lambdas=lambdas,
        max_fit_delta=max_fit_delta,
        outer_scale=outer_scale,
        outer_scale_width_factor=outer_scale_width_factor,
        fit_steps=fit_steps,
        fit_reg=fit_reg,
        weight_power=weight_power,
    )

    built = _build_flux_kernel_and_jacobian(
        size=size,
        alpha=alpha,
        coeffs=coeffs,
        lambdas=lambdas,
        outer_scale=outer_scale,
        outer_scale_width_factor=outer_scale_width_factor,
    )
    kernel, _, inside, _, _ = built
    if kernel is None:
        raise RuntimeError(
            "Higher-order LS2010 flux fit produced a non-positive correction factor. "
            "Try fewer modes, larger lambdas, or a slightly larger fit_reg."
        )

    if causal:
        warnings.warn(
            "create_kernel_flux_LS2010_highorder is calibrated for the acausal case. "
            "For causal=True it simply truncates the fitted acausal kernel.",
            RuntimeWarning,
            stacklevel=2,
        )
        kernel = kernel.copy()
        kernel[: size // 2] = 0.0

    return B.asarray(kernel)


def create_kernel_LS2010_flux_highorder_from_params(
    size,
    exponent,
    norm_ratio_exponent,
    causal=False,
    outer_scale=None,
    outer_scale_width_factor=2.0,
    final_power=None,
    order=5,
    lambdas=None,
    max_fit_delta=None,
    fit_steps=6,
    fit_reg=1.0e-4,
    weight_power=0.0,
):
    """
    Thin wrapper so you can swap the flux-kernel call site with minimal edits.

    It assumes the caller is passing the standard FIF flux-kernel parameters:
        exponent            = -1/alpha'
        norm_ratio_exponent = -1/alpha
        final_power         =  1/(alpha-1)

    Exact replacement in FIF_1D:

        kernel1 = create_kernel_LS2010_flux_highorder_from_params(
            size, flux_exponent, flux_norm_ratio_exp,
            causal=causal, outer_scale=outer_scale,
            outer_scale_width_factor=outer_scale_width_factor,
            final_power=flux_final_power,
            order=5,
        )
    """
    alpha = -1.0 / float(norm_ratio_exponent)

    expected_exponent = -1.0 + 1.0 / alpha
    if abs(float(exponent) - expected_exponent) > 1.0e-8:
        warnings.warn(
            "The supplied exponent does not match the standard FIF flux-kernel exponent. "
            "Proceeding by inferring alpha from norm_ratio_exponent.",
            RuntimeWarning,
            stacklevel=2,
        )

    expected_final_power = 1.0 / (alpha - 1.0)
    if final_power is not None and abs(float(final_power) - expected_final_power) > 1.0e-8:
        warnings.warn(
            "The supplied final_power does not match the standard FIF flux-kernel value. "
            "Proceeding anyway, but note that this helper is intended only for the flux kernel.",
            RuntimeWarning,
            stacklevel=2,
        )

    return create_kernel_flux_LS2010_highorder(
        size=size,
        alpha=alpha,
        causal=causal,
        outer_scale=outer_scale,
        outer_scale_width_factor=outer_scale_width_factor,
        order=order,
        lambdas=lambdas,
        max_fit_delta=max_fit_delta,
        fit_steps=fit_steps,
        fit_reg=fit_reg,
        weight_power=weight_power,
    )
