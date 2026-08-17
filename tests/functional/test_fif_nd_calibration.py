"""FIF_ND C1 calibration: the d-dimensional angular normalization.

The flux amplitude is set by the rate at which the kernel accumulates
alpha-power with scale. On the dx=2 lattice that rate carries a factor
Omega_d / 2^d (1 in 1-D, pi/2 in 2-D and 3-D), which the 1-D amplitude formula
C1**(1/alpha) does not contain. Reusing the 1-D formula in N-D — as every
release through 0.13.0 did — inflated the effective C1 by ~1.57x at every
alpha. Caught 2026-07-10 by cross-validation against Lovejoy's eps2D.m, whose
NDf constant is exactly this factor.
"""
import numpy as np
import pytest

import scaleinvariance as si
from scaleinvariance.simulation.FIF import (
    _angular_lograte_factor,
    extremal_levy,
    periodic_convolve_nd,
)
from scaleinvariance.simulation.kernels import create_kernel_LS2010


@pytest.fixture(autouse=True)
def float64_precision():
    """These are calibration measurements — run them in double, and put the
    global precision back so we do not leak state into the rest of the suite."""
    original = si.get_numerical_precision()
    si.set_numerical_precision('float64')
    yield
    si.set_numerical_precision(original)


def test_angular_factor_matches_omega_d_over_2_d():
    # Omega_d = 2 pi^(d/2) / Gamma(d/2): 2, 2pi, 4pi -> /2^d gives 1, pi/2, pi/2.
    assert _angular_lograte_factor(1) == pytest.approx(1.0)
    assert _angular_lograte_factor(2) == pytest.approx(np.pi / 2)
    assert _angular_lograte_factor(3) == pytest.approx(np.pi / 2)
    assert _angular_lograte_factor(4) == pytest.approx(np.pi ** 2 / 8)


def _trace_moment_C1(flux, n, q):
    """C1 from <eps_r^q> ~ lam^K(q) with K(q) = C1 (q^2 - q) at alpha=2."""
    flux = flux / flux.mean()
    lam, mom = [], []
    for r in (2, 4, 8, 16, 32, 64):
        coarse = flux.reshape(n // r, r, n // r, r).mean(axis=(1, 3))
        lam.append(n / r)
        mom.append((coarse ** q).mean())
    return np.polyfit(np.log(lam), np.log(mom), 1)[0] / (q ** 2 - q)


@pytest.mark.parametrize("q", [1.5, 2.0])
def test_2d_flux_recovers_requested_C1(q):
    """Trace moments on the 2-D flux read back the requested C1.

    Without the angular factor this lands near 0.14 for a requested 0.1.
    The window is wide on the low side because the estimator itself runs a
    few percent low on a finite domain (Lovejoy's own fields read 0.097).
    """
    n, alpha, C1 = 1024, 2.0, 0.1
    alpha_prime = 1.0 / (1.0 - 1.0 / alpha)
    kernel = create_kernel_LS2010(
        (n, n), -2.0 / alpha_prime, -2.0 / alpha, causal=False,
        outer_scale=None, final_power=1.0 / (alpha - 1.0))

    estimates = []
    for seed in (11, 12, 13):
        noise = extremal_levy(alpha, size=(n, n), seed=seed)
        integrated = si.backend.to_numpy(periodic_convolve_nd(noise, kernel))
        amplitude = (C1 / _angular_lograte_factor(2)) ** (1 / alpha)
        flux = np.exp(integrated.astype(np.float64) * amplitude)
        estimates.append(_trace_moment_C1(flux, n, q))

    C1_eff = float(np.mean(estimates))
    assert 0.075 < C1_eff < 0.115, f"q={q}: C1_eff={C1_eff} (requested {C1})"


def test_1d_flux_is_the_calibrated_reference():
    """1-D takes amplitude C1**(1/alpha) bare — that is what N-D calibrates to.

    Same estimator as the 2-D test, so the two are directly comparable: if 1-D
    reads back C1 with no factor and 2-D reads back C1 with pi/2 divided out,
    the angular factor is the whole of the difference.
    """
    n, alpha, C1, q = 2 ** 20, 2.0, 0.1, 2.0
    alpha_prime = 1.0 / (1.0 - 1.0 / alpha)
    kernel = create_kernel_LS2010(
        n, -1.0 / alpha_prime, -1.0 / alpha, causal=False,
        outer_scale=None, final_power=1.0 / (alpha - 1.0))

    estimates = []
    for seed in (21, 22, 23):
        noise = extremal_levy(alpha, size=n, seed=seed)
        integrated = si.backend.to_numpy(periodic_convolve_nd(noise, kernel))
        flux = np.exp(integrated.astype(np.float64) * C1 ** (1 / alpha))
        flux = flux / flux.mean()
        lam, mom = [], []
        for r in (4, 16, 64, 256, 1024):
            coarse = flux.reshape(n // r, r).mean(axis=1)
            lam.append(n / r)
            mom.append((coarse ** q).mean())
        estimates.append(np.polyfit(np.log(lam), np.log(mom), 1)[0] / (q ** 2 - q))

    C1_eff = float(np.mean(estimates))
    assert 0.075 < C1_eff < 0.115, f"1-D C1_eff={C1_eff} (requested {C1})"
