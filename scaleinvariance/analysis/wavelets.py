"""
Wavelet definitions for wavelet-fluctuation analysis.

A *wavelet* here is a discrete kernel ``g_r`` built at an integer scale (lag)
``r``; convolving a field with ``g_r`` and taking ``|.|`` gives the fluctuation
at scale ``r``. For a self-affine field with Hurst exponent ``H``,
``<|g_r * f|^q> ~ r^{xi(q)}`` with ``xi(q) = qH - K(q)``.

Normalization (the L1 / "fluctuation" convention)
--------------------------------------------------
Every kernel is built with the *same* L1 norm across scales,
``sum |g_r| = L1_NORM`` (= 2, matching the historical Haar and structure-
function kernels), and with the discrete mean removed so ``sum g_r = 0``
(one exact vanishing moment). Holding the L1 norm fixed is what makes the
fluctuation scale as ``r^H`` rather than ``r^{H+1/2}`` (the L2 / energy
convention used by the usual continuous wavelet transform). The absolute
constant only shifts the intercept of the scaling, never the slope, so it is a
convention, not a tuned parameter; different wavelets therefore share the same
slope but not the same amplitude.

Scale convention (lag -> sigma)
-------------------------------
The localized wavelets are anchored so that lag ``r`` equals the distance from
the central peak to the first trough; at the smallest lag ``r = 1`` that trough
sits on the immediate grid neighbour. For the Mexican hat the trough is at
``sqrt(3) sigma`` so ``sigma = r / sqrt(3)``; for the Morlet (omega0 = 6) the
first trough of the real part is at ``sigma * pi / omega0`` so
``sigma = omega0 r / pi``. Kernels are truncated at ``+/- 5 sigma`` for the
Mexican hat (its polynomial-times-Gaussian tail is slightly heavier) and
``+/- 4 sigma`` for the Morlet.

Because a localized wavelet's support is several sigma wide, its kernel is
wider than its nominal scale (``~5.8 r`` for the Mexican hat, ``~15 r`` for the
Morlet). A lag whose kernel would exceed the domain cannot be evaluated and is
returned as NaN by ``wavelet_fluctuation`` -- the largest reachable lag is
correspondingly smaller for the wider wavelets.

The same kernel width governs NaN contamination under
``nan_behavior='ignore'``: an output position is NaN whenever its full kernel
window contains any NaN, so a single NaN poisons a span as wide as the kernel
(``~r`` for Haar / structure function, ``~5.8 r`` / ``~15 r`` for the localized
wavelets). The structure-function wavelet would therefore be stricter under NaN
than the dedicated ``structure_function``: the mask counts the whole
width-``r+1`` window, including the zero taps between the two ``+/-1``
endpoints, whereas the dedicated increment only needs the two endpoints finite.
Because of this, ``wavelet_fluctuation`` auto-delegates to ``structure_function``
(with a warning) when the data contain NaNs and ``wavelet='structure_function'``.
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from .. import backend as B

L1_NORM = 2.0  # matches the historical Haar / structure-function kernels


@dataclass(frozen=True)
class Wavelet:
    """A wavelet usable for fluctuation analysis.

    Attributes
    ----------
    name : str
    build_kernel : callable
        ``build_kernel(lag) -> backend 1-D array`` at integer scale ``lag``.
    even_only : bool
        If True, only even lags are valid (the kernel needs an even width).
    is_complex : bool
        Whether the kernel is complex (the fluctuation is then the modulus of
        the complex convolution). Informational; the convolution backend
        detects complexity itself.
    valid_H_range : tuple of float
        Approximate range of Hurst exponents the wavelet can resolve, set by
        its number of vanishing moments. Documentation only -- not enforced.
    description : str
    """
    name: str
    build_kernel: Callable[[int], "object"]
    even_only: bool
    is_complex: bool
    valid_H_range: Tuple[float, float]
    description: str


def _finalize(shape, complex_kernel):
    """Mean-remove and L1-normalize a raw kernel shape (numpy 1-D array).

    Returns a backend array at the active precision.
    """
    shape = shape - shape.mean()                       # one exact vanishing moment
    shape = shape * (L1_NORM / np.abs(shape).sum())    # fixed L1 norm across scales
    np_dtype = B._active_complex_dtype_np() if complex_kernel else B._active_real_dtype_np()
    return B.asarray(shape.astype(np_dtype))


def _haar_kernel(lag):
    # Built with backend ops to stay BIT-IDENTICAL to the original
    # haar_fluctuation kernel (do not route through _finalize). The two halves
    # are means (+/- 1 each), so sum|g| = 2 and sum g = 0 already.
    kernel = B.ones(lag) / (lag / 2)
    kernel[:lag // 2] = kernel[:lag // 2] * -1
    return kernel


def _structure_function_kernel(lag):
    # "Poor man's wavelet": the first difference f(x+lag) - f(x). Two taps
    # (+1, -1) => sum|g| = 2, sum g = 0 exactly, for any lag. wavelet='haar'
    # aside, this reproduces structure_function (order 1) to numerical
    # precision; the dedicated structure_function() is the fast path.
    k = np.zeros(lag + 1, dtype=B._active_real_dtype_np())
    k[0] = 1.0
    k[-1] = -1.0
    return B.asarray(k)


def _mexican_hat_kernel(lag):
    # Ricker wavelet (2nd derivative of a Gaussian). Zero-crossings at +/-sigma,
    # troughs at +/-sqrt(3) sigma; lag == peak-to-trough => sigma = lag/sqrt(3).
    # Truncated at +/-5 sigma (its (1-u^2) prefactor gives a slightly heavier
    # tail than a pure Gaussian, so 5 sigma is used rather than 4).
    sigma = lag / np.sqrt(3.0)
    half = int(np.ceil(5.0 * sigma))
    u = np.arange(-half, half + 1, dtype=np.float64) / sigma
    shape = (1.0 - u ** 2) * np.exp(-0.5 * u ** 2)
    return _finalize(shape, complex_kernel=False)


def _morlet_kernel(lag, omega0=6.0):
    # Complex (analytic) Morlet with central frequency omega0 = 6. First trough
    # of the real part at sigma*pi/omega0; lag == peak-to-trough =>
    # sigma = omega0 lag / pi. The exp(-omega0^2/2) term enforces admissibility
    # (zero mean); ~1.5e-8 here.
    sigma = omega0 * lag / np.pi
    half = int(np.ceil(4.0 * sigma))
    u = np.arange(-half, half + 1, dtype=np.float64) / sigma
    shape = (np.exp(1j * omega0 * u) - np.exp(-0.5 * omega0 ** 2)) * np.exp(-0.5 * u ** 2)
    return _finalize(shape, complex_kernel=True)


WAVELETS = {
    'haar': Wavelet(
        name='haar', build_kernel=_haar_kernel, even_only=True, is_complex=False,
        valid_H_range=(-1.0, 1.0),
        description="Difference of the means of the two halves of a width-r window."),
    'structure_function': Wavelet(
        name='structure_function', build_kernel=_structure_function_kernel,
        even_only=False, is_complex=False, valid_H_range=(0.0, 1.0),
        description="First difference f(x+r)-f(x); the structure-function increment."),
    'mexican_hat': Wavelet(
        name='mexican_hat', build_kernel=_mexican_hat_kernel, even_only=False,
        is_complex=False, valid_H_range=(-1.0, 2.0),
        description="Ricker wavelet; two vanishing moments (resolves H up to ~2)."),
    'morlet': Wavelet(
        name='morlet', build_kernel=_morlet_kernel, even_only=False,
        is_complex=True, valid_H_range=(-1.0, 2.0),
        description="Complex Morlet (omega0=6); frequency-localized, phase-invariant modulus."),
}


def get_wavelet(name):
    """Return the Wavelet registered under ``name`` (raises on unknown name)."""
    if isinstance(name, Wavelet):
        return name
    if name not in WAVELETS:
        raise ValueError(
            f"Unknown wavelet '{name}'. Available: {sorted(WAVELETS)}")
    return WAVELETS[name]
