"""
Gross accuracy tests for simulation and analysis functions.

Uses small/fast simulations with wide tolerances. These verify that functions
return plausible values and haven't regressed badly — not that they are
numerically precise (see tests/numerical/ for that).
"""

import numpy as np
import pytest
import scaleinvariance

SIZE   = 2**13
N_SIMS = 10


def make_fif(H, C1, alpha, n=N_SIMS, kernel='LS2010'):
    return np.stack([
        scaleinvariance.FIF_1D(SIZE, alpha, C1, H=H,
                               kernel_construction_method_flux=kernel,
                               kernel_construction_method_observable=kernel)
        for _ in range(n)
    ])


def make_fbm_circulant(H, n=N_SIMS):
    return np.stack([scaleinvariance.fBm_1D_circulant(SIZE, H) for _ in range(n)])


# ============================================================
# Hurst estimation on fBm_1D_circulant, lags 100-1000
# ============================================================

class TestHurstFBmCirculant:
    """Structure function and Haar fluctuation Hurst on fBm, wide ±0.2 tolerance."""

    @pytest.mark.parametrize("H_true", [0.2, 0.4, 0.6, 0.8])
    def test_structure_function_hurst(self, H_true):
        data = make_fbm_circulant(H_true)
        H_est, _ = scaleinvariance.structure_function_hurst(
            data, min_sep=100, max_sep=1000, axis=1)
        assert abs(H_est - H_true) < 0.2, f"H_est={H_est:.3f}, H_true={H_true}"

    @pytest.mark.parametrize("H_true", [0.2, 0.4, 0.6, 0.8])
    def test_haar_fluctuation_hurst(self, H_true):
        data = make_fbm_circulant(H_true)
        H_est, _ = scaleinvariance.haar_fluctuation_hurst(
            data, min_sep=100, max_sep=1000, axis=1)
        assert abs(H_est - H_true) < 0.2, f"H_est={H_est:.3f}, H_true={H_true}"


# ============================================================
# Hurst estimation on FIF, both kernel methods
# ============================================================

class TestHurstFIF:

    @pytest.mark.parametrize("H_true", [0.3, 0.7])
    @pytest.mark.parametrize("kernel", ['LS2010', 'LS2010_spectral'])
    def test_structure_function_hurst(self, H_true, kernel):
        data = make_fif(H_true, C1=0.05, alpha=1.8, kernel=kernel)
        H_est, _ = scaleinvariance.structure_function_hurst(data, axis=1)
        assert abs(H_est - H_true) < 0.2, f"H_est={H_est:.3f}, H_true={H_true}, kernel={kernel}"

    @pytest.mark.parametrize("H_true", [0.3, 0.7])
    @pytest.mark.parametrize("kernel", ['LS2010', 'LS2010_spectral'])
    def test_haar_fluctuation_hurst(self, H_true, kernel):
        data = make_fif(H_true, C1=0.05, alpha=1.8, kernel=kernel)
        H_est, _ = scaleinvariance.haar_fluctuation_hurst(data, axis=1)
        assert abs(H_est - H_true) < 0.2, f"H_est={H_est:.3f}, H_true={H_true}, kernel={kernel}"

    @pytest.mark.parametrize("H_true", [0.3, 0.7])
    def test_spectral_hurst_fbm(self, H_true):
        data = make_fbm_circulant(H_true)
        H_est, _ = scaleinvariance.spectral_hurst(data, axis=1)
        assert abs(H_est - H_true) < 0.2, f"H_est={H_est:.3f}, H_true={H_true}"


# ============================================================
# Intermittency estimation
# ============================================================

class TestIntermittencyGrossAccuracy:

    @pytest.mark.parametrize("C1_true", [0.05, 0.15])
    @pytest.mark.parametrize("kernel", ['LS2010', 'LS2010_spectral'])
    def test_two_point_intermittency_exponent(self, C1_true, kernel):
        data = make_fif(H=0.5, C1=C1_true, alpha=1.8, kernel=kernel)
        C1_est, _ = scaleinvariance.two_point_intermittency_exponent(data, axis=1)
        assert 0.0 <= C1_est <= C1_true + 0.15, \
            f"C1_est={C1_est:.3f}, C1_true={C1_true}, kernel={kernel}"

    def test_two_point_intermittency_fbm_near_zero(self):
        # fBm has C1=0; estimate should be very small
        data = make_fbm_circulant(H=0.5)
        C1_est, _ = scaleinvariance.two_point_intermittency_exponent(data, axis=1)
        assert abs(C1_est) < 0.01, f"C1_est={C1_est:.4f} for fBm, expected ~0"

    @pytest.mark.parametrize("C1_true", [0.05, 0.15])
    def test_compute_K_q_function_C1(self, C1_true):
        data = make_fif(H=0.5, C1=C1_true, alpha=1.8)
        _, _, _, C1_fit, _ = scaleinvariance.compute_K_q_function(data, axis=1)
        assert 0.0 <= C1_fit <= C1_true + 0.2, \
            f"C1_fit={C1_fit:.3f}, C1_true={C1_true}"


# ============================================================
# K(q) near zero for fBm
# ============================================================

class TestFBmNonIntermittentKq:

    def test_K_near_zero_for_fbm(self):
        data = make_fbm_circulant(H=0.5)
        q_values, K_emp, *_ = scaleinvariance.compute_K_q_function(data, axis=1)
        assert np.all(np.abs(K_emp) < 0.15), \
            f"Max |K(q)|={np.max(np.abs(K_emp)):.3f}, expected ~0 for fBm"
