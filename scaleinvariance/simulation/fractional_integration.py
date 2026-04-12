from .. import backend as B


def fractional_integral_spectral(signal, H, outer_scale=None):
    """
    Apply an acausal spectral fractional integral of order H to a real signal.

    Computes ``irfftn(rfftn(signal) * |f|^(-H))`` where |f| is the isotropic
    Fourier wavenumber ``sqrt(sum f_i^2)``. Works for 1D and N-D real
    signals. The DC bin is regularised by clipping frequencies below
    ``1/outer_scale`` and then overwriting the DC bin with the nearest
    nonzero-frequency bin, matching the existing ``create_kernel_spectral``
    behaviour.

    The Fourier-space exponent is ``-H`` because fractional integration of
    order H is convolution with the Riesz potential ``|r|^(H - d)``, whose
    Fourier transform is proportional to ``|f|^(-H)`` — the spatial
    dimension cancels out.

    Parameters
    ----------
    signal : ndarray
        Real-valued input, 1D or N-D.
    H : float
        Hurst exponent (order of the fractional integral).
    outer_scale : float, optional
        Low-frequency regularisation scale. Defaults to ``max(signal.shape)``.

    Returns
    -------
    ndarray
        Fractionally-integrated signal, same shape as input.
    """
    signal = B.asarray(signal)
    shape = signal.shape
    ndim = signal.ndim

    if outer_scale is None:
        outer_scale = max(shape)

    rfft_shape = shape[:-1] + (shape[-1] // 2 + 1,)

    # Build |f|^2 via broadcasted 1D axes to avoid meshgrid intermediates.
    freqs_squared = None
    for axis_idx in range(ndim):
        if axis_idx == ndim - 1:
            axis = B.rfftfreq(shape[-1], d=1.0)
        else:
            axis = B.fftfreq(shape[axis_idx], d=1.0)
        broadcast_shape = [1] * ndim
        broadcast_shape[axis_idx] = axis.size
        term = (axis.reshape(broadcast_shape)) ** 2
        if freqs_squared is None:
            # Promote to the half-spectrum shape via the + 0 trick on first add below
            freqs_squared = term
        else:
            freqs_squared = freqs_squared + term
        del axis, term
    # At this point freqs_squared has broadcasted shape equal to rfft_shape
    # (numpy/torch broadcasting materialises it on first binary op above).
    freqs = B.sqrt(freqs_squared)
    del freqs_squared

    min_freq = 1.0 / float(outer_scale)
    freqs_regularized = B.maximum(freqs, min_freq)
    del freqs

    kernel = freqs_regularized ** (-H)
    del freqs_regularized

    dc_index = (0,) * ndim
    if kernel.size == 1:
        kernel[dc_index] = 1.0
    else:
        neighbor_index = (1,) + (0,) * (ndim - 1)
        kernel[dc_index] = kernel[neighbor_index]

    # Ensure kernel broadcasts correctly against rfftn output.
    if kernel.shape != rfft_shape:
        # Broadcasting may have been virtual; materialise now.
        kernel = kernel + B.zeros(rfft_shape, dtype=kernel.dtype)

    spectrum = B.rfftn(signal) * kernel
    del kernel
    result = B.irfftn(spectrum, s=shape)
    del spectrum
    return result
