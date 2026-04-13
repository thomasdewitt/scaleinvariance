"""
Functional coverage tests for the scaleinvariance package.

Tests output shapes, mathematical identities, API contracts, and error
handling across the whole package. Not accuracy tests — see test_gross_accuracy.py.
"""

import numpy as np
import pytest
import scaleinvariance
from scaleinvariance.simulation.FIF import extremal_levy

SIZE  = 2**12
SMALL = 2**9


# ============================================================
# FIF_1D
# ============================================================

class TestFIF1D:

    @pytest.mark.parametrize("size", [512, 2**12])
    def test_output_shape(self, size):
        fif = scaleinvariance.FIF_1D(size, 1.8, 0.1, H=0.5)
        assert fif.shape == (size,)

    def test_no_nan_inf(self):
        fif = scaleinvariance.FIF_1D(SIZE, 1.8, 0.1, H=0.5)
        assert not np.any(np.isnan(fif))
        assert not np.any(np.isinf(fif))

    @pytest.mark.parametrize("H", [0.0, 0.3, 0.7])
    def test_positive_for_nonneg_H(self, H):
        fif = scaleinvariance.FIF_1D(SIZE, 1.8, 0.1, H=H)
        assert np.all(fif > 0)

    def test_mean_near_one(self):
        fif = scaleinvariance.FIF_1D(SIZE, 1.8, 0.1, H=0.5)
        assert abs(np.mean(fif) - 1.0) < 0.05

    def test_C1_zero_zero_mean(self):
        fif = scaleinvariance.FIF_1D(SIZE, 1.8, 0.0, H=0.5, causal=False)
        assert abs(np.mean(fif)) < 0.5

    def test_reproducibility_with_levy_noise(self):
        levy = extremal_levy(1.8, size=SIZE)
        fif1 = scaleinvariance.FIF_1D(SIZE, 1.8, 0.1, H=0.5, levy_noise=levy)
        fif2 = scaleinvariance.FIF_1D(SIZE, 1.8, 0.1, H=0.5, levy_noise=levy)
        np.testing.assert_array_equal(fif1, fif2)

    def test_periodic_false_shape(self):
        fif = scaleinvariance.FIF_1D(SIZE, 1.8, 0.1, H=0.5, periodic=False)
        assert fif.shape == (SIZE,)

    @pytest.mark.parametrize("kernel_obs", ['LS2010', 'spectral'])
    def test_kernel_methods_no_nan(self, kernel_obs):
        fif = scaleinvariance.FIF_1D(SIZE, 1.8, 0.1, H=0.5,
                                     kernel_construction_method_flux='LS2010',
                                     kernel_construction_method_observable=kernel_obs)
        assert not np.any(np.isnan(fif))

    @pytest.mark.parametrize("causal", [True, False])
    def test_causal_flag(self, causal):
        obs = 'LS2010' if causal else 'spectral'
        fif = scaleinvariance.FIF_1D(SIZE, 1.8, 0.1, H=0.5, causal=causal,
                                     kernel_construction_method_observable=obs)
        assert fif.shape == (SIZE,)
        assert not np.any(np.isnan(fif))


# ============================================================
# FIF_ND
# ============================================================

class TestFIFND:

    def test_2d_shape(self):
        fif = scaleinvariance.FIF_ND((64, 64), 1.8, 0.1, H=0.3)
        assert fif.shape == (64, 64)

    def test_2d_no_nan_inf(self):
        fif = scaleinvariance.FIF_ND((64, 64), 1.8, 0.1, H=0.3)
        assert not np.any(np.isnan(fif))
        assert not np.any(np.isinf(fif))

    def test_2d_positive(self):
        fif = scaleinvariance.FIF_ND((64, 64), 1.8, 0.1, H=0.3)
        assert np.all(fif > 0)

    def test_2d_mean_near_one(self):
        fif = scaleinvariance.FIF_ND((64, 64), 1.8, 0.1, H=0.3)
        assert abs(np.mean(fif) - 1.0) < 1e-4

    def test_periodic_tuple(self):
        fif = scaleinvariance.FIF_ND((64, 64), 1.8, 0.1, H=0.3, periodic=(True, False))
        assert fif.shape == (64, 64)
        assert not np.any(np.isnan(fif))

    def test_gsi_with_scale_metric(self):
        size = (64, 64)
        sim_size = (128, 128)
        metric = scaleinvariance.canonical_scale_metric(sim_size, ls=20.0, Hz=0.556)
        fif = scaleinvariance.FIF_ND(size, 1.8, 0.1, H=0.3, periodic=False,
                                     scale_metric=metric, scale_metric_dim=1 + 0.556,
                                     kernel_construction_method_observable='LS2010')
        assert fif.shape == size
        assert not np.any(np.isnan(fif))


# ============================================================
# fBm simulations
# ============================================================

class TestFBm1DCirculant:

    def test_shape(self):
        fbm = scaleinvariance.fBm_1D_circulant(SIZE, 0.5)
        assert fbm.shape == (SIZE,)

    def test_no_nan_inf(self):
        fbm = scaleinvariance.fBm_1D_circulant(SIZE, 0.5)
        assert not np.any(np.isnan(fbm))
        assert not np.any(np.isinf(fbm))

    def test_zero_mean(self):
        fbm = scaleinvariance.fBm_1D_circulant(SIZE, 0.5)
        assert abs(np.mean(fbm)) < 0.1

    def test_periodic_false_shape(self):
        fbm = scaleinvariance.fBm_1D_circulant(SIZE, 0.5, periodic=False)
        assert fbm.shape == (SIZE,)


class TestFBm1D:

    @pytest.mark.parametrize("causal", [True, False])
    def test_shape_and_no_nan(self, causal):
        fbm = scaleinvariance.fBm_1D(SIZE, 0.5, causal=causal)
        assert fbm.shape == (SIZE,)
        assert not np.any(np.isnan(fbm))

    def test_zero_mean(self):
        fbm = scaleinvariance.fBm_1D(SIZE, 0.5)
        assert abs(np.mean(fbm)) < 0.1

    @pytest.mark.parametrize("kernel", ['LS2010', 'spectral'])
    def test_kernel_methods_acausal(self, kernel):
        fbm = scaleinvariance.fBm_1D(SIZE, 0.5, causal=False, kernel_construction_method=kernel)
        assert not np.any(np.isnan(fbm))


class TestFBmNDCirculant:

    @pytest.mark.parametrize("size", [(128, 128), (32, 32, 32)])
    def test_shape_and_no_nan(self, size):
        fbm = scaleinvariance.fBm_ND_circulant(size, 0.5)
        assert fbm.shape == size
        assert not np.any(np.isnan(fbm))

    def test_zero_mean(self):
        fbm = scaleinvariance.fBm_ND_circulant((128, 128), 0.5)
        assert abs(np.mean(fbm)) < 0.1


# ============================================================
# canonical_scale_metric
# ============================================================

class TestCanonicalScaleMetric:

    def test_2d_shape(self):
        metric = scaleinvariance.canonical_scale_metric((64, 64), ls=10.0)
        assert metric.shape == (64, 64)

    def test_3d_shape(self):
        metric = scaleinvariance.canonical_scale_metric((32, 32, 32), ls=10.0)
        assert metric.shape == (32, 32, 32)

    def test_all_positive(self):
        metric = scaleinvariance.canonical_scale_metric((64, 64), ls=10.0)
        assert np.all(metric >= 0)

    def test_no_nan_inf(self):
        metric = scaleinvariance.canonical_scale_metric((64, 64), ls=10.0)
        assert not np.any(np.isnan(metric))
        assert not np.any(np.isinf(metric))

    def test_invalid_ls_raises(self):
        with pytest.raises(ValueError):
            scaleinvariance.canonical_scale_metric((64, 64), ls=0.0)

    def test_invalid_hz_raises(self):
        with pytest.raises(ValueError):
            scaleinvariance.canonical_scale_metric((64, 64), ls=10.0, Hz=0.0)


# ============================================================
# *_analysis functions
# ============================================================

@pytest.fixture(scope='module')
def fbm_1d():
    return scaleinvariance.fBm_1D_circulant(SIZE, 0.5)


class TestStructureFunctionAnalysis:

    def test_shapes_match(self, fbm_1d):
        lags, sf = scaleinvariance.structure_function(fbm_1d)
        assert lags.shape == sf.shape

    def test_lags_sorted_positive(self, fbm_1d):
        lags, _ = scaleinvariance.structure_function(fbm_1d)
        assert np.all(lags > 0)
        assert np.all(np.diff(lags) > 0)

    def test_sf_positive(self, fbm_1d):
        _, sf = scaleinvariance.structure_function(fbm_1d)
        assert np.all(sf > 0)

    def test_order_changes_values(self, fbm_1d):
        _, sf1 = scaleinvariance.structure_function(fbm_1d, order=1)
        _, sf2 = scaleinvariance.structure_function(fbm_1d, order=2)
        assert not np.allclose(sf1, sf2)

    def test_max_sep_respected(self, fbm_1d):
        max_sep = 100
        lags, _ = scaleinvariance.structure_function(fbm_1d, max_sep=max_sep)
        assert np.all(lags <= max_sep)


class TestHaarFluctuationAnalysis:

    def test_shapes_match(self, fbm_1d):
        lags, haar = scaleinvariance.haar_fluctuation(fbm_1d)
        assert lags.shape == haar.shape

    def test_lags_all_even(self, fbm_1d):
        lags, _ = scaleinvariance.haar_fluctuation(fbm_1d)
        assert np.all(lags % 2 == 0)

    def test_values_positive(self, fbm_1d):
        _, haar = scaleinvariance.haar_fluctuation(fbm_1d)
        assert np.all(haar > 0)

    def test_order_changes_values(self, fbm_1d):
        _, h1 = scaleinvariance.haar_fluctuation(fbm_1d, order=1)
        _, h2 = scaleinvariance.haar_fluctuation(fbm_1d, order=2)
        assert not np.allclose(h1, h2)

    def test_max_sep_respected(self, fbm_1d):
        max_sep = 100
        lags, _ = scaleinvariance.haar_fluctuation(fbm_1d, max_sep=max_sep)
        assert np.all(lags <= max_sep)

    def test_nan_behavior_raise(self):
        data = scaleinvariance.fBm_1D_circulant(SIZE, 0.5).copy()
        data[10] = np.nan
        with pytest.raises(ValueError):
            scaleinvariance.haar_fluctuation(data, nan_behavior='raise')

    def test_nan_behavior_ignore(self):
        data = scaleinvariance.fBm_1D_circulant(SIZE, 0.5).copy()
        data[10] = np.nan
        lags, haar = scaleinvariance.haar_fluctuation(data, nan_behavior='ignore')
        assert lags.shape == haar.shape


class TestSpectralAnalysis:

    def test_shapes_match(self, fbm_1d):
        freqs, psd = scaleinvariance.power_spectrum_binned(fbm_1d)
        assert freqs.shape == psd.shape

    def test_freqs_positive_sorted(self, fbm_1d):
        freqs, _ = scaleinvariance.power_spectrum_binned(fbm_1d)
        assert np.all(freqs > 0)
        assert np.all(np.diff(freqs) > 0)

    def test_psd_positive(self, fbm_1d):
        _, psd = scaleinvariance.power_spectrum_binned(fbm_1d)
        assert np.all(psd > 0)


# ============================================================
# *_hurst functions
# ============================================================

@pytest.fixture(scope='module')
def fbm_stack():
    return np.stack([scaleinvariance.fBm_1D_circulant(SIZE, 0.5) for _ in range(5)])


class TestHurstFunctions:

    @pytest.mark.parametrize("fn", [
        scaleinvariance.structure_function_hurst,
        scaleinvariance.haar_fluctuation_hurst,
        scaleinvariance.spectral_hurst,
    ])
    def test_return_fit_true_gives_five_tuple(self, fn, fbm_stack):
        result = fn(fbm_stack, return_fit=True, axis=1)
        assert len(result) == 5

    @pytest.mark.parametrize("fn", [
        scaleinvariance.structure_function_hurst,
        scaleinvariance.haar_fluctuation_hurst,
        scaleinvariance.spectral_hurst,
    ])
    def test_uncertainty_positive(self, fn, fbm_stack):
        _, uncertainty = fn(fbm_stack, axis=1)
        assert uncertainty > 0

    @pytest.mark.parametrize("fn", [
        scaleinvariance.structure_function_hurst,
        scaleinvariance.haar_fluctuation_hurst,
    ])
    def test_raises_on_tiny_array(self, fn):
        with pytest.raises(ValueError):
            fn(np.random.randn(8))

    @pytest.mark.parametrize("fn", [
        scaleinvariance.structure_function_hurst,
        scaleinvariance.haar_fluctuation_hurst,
    ])
    def test_axis_consistency(self, fn, fbm_stack):
        # Analyzing (5, SIZE) along axis=1 vs (SIZE, 5) along axis=0 should match
        H_ax1, _ = fn(fbm_stack, axis=1)
        H_ax0, _ = fn(fbm_stack.T, axis=0)
        assert abs(H_ax1 - H_ax0) < 1e-10

    @pytest.mark.parametrize("fn", [
        scaleinvariance.structure_function_hurst,
        scaleinvariance.haar_fluctuation_hurst,
        scaleinvariance.spectral_hurst,
    ])
    def test_works_on_1d_array(self, fn, fbm_1d):
        H, uncertainty = fn(fbm_1d)
        assert np.isfinite(H)
        assert uncertainty > 0


# ============================================================
# K_analytic
# ============================================================

class TestKFunction:

    def test_K_at_q1_is_zero(self):
        assert scaleinvariance.K_analytic(1.0, 0.1, 1.8) == 0.0

    @pytest.mark.parametrize("alpha", [1.5, 1.8, 2.0])
    def test_K_with_C1_zero_is_zero(self, alpha):
        q = np.array([0.5, 1.0, 1.5, 2.0])
        result = scaleinvariance.K_analytic(q, 0.0, alpha)
        np.testing.assert_array_equal(result, 0.0)

    def test_K_positive_for_q_gt_1(self):
        q = np.array([1.1, 1.5, 2.0, 2.5])
        K_vals = scaleinvariance.K_analytic(q, 0.1, 1.8)
        assert np.all(K_vals > 0)

    def test_K_negative_for_q_lt_1(self):
        q = np.array([0.1, 0.3, 0.5, 0.9])
        K_vals = scaleinvariance.K_analytic(q, 0.1, 1.8)
        assert np.all(K_vals < 0)

    def test_K_array_input(self):
        q = np.arange(0.1, 2.51, 0.1)
        result = scaleinvariance.K_analytic(q, 0.1, 1.8)
        assert result.shape == q.shape

    def test_K_matches_formula(self):
        q, C1, alpha = 2.0, 0.1, 1.8
        expected = (C1 / (alpha - 1)) * (q**alpha - q)
        assert abs(scaleinvariance.K_analytic(q, C1, alpha) - expected) < 1e-12


# ============================================================
# two_point_C1
# ============================================================

class TestTwoPointIntermittencyExponent:

    @pytest.fixture(scope='class')
    def fif_data(self):
        return np.stack([scaleinvariance.FIF_1D(SIZE, 1.8, 0.1, H=0.5) for _ in range(5)])

    @pytest.mark.parametrize("method", ['structure_function', 'haar_fluctuation'])
    def test_both_scaling_methods_run(self, fif_data, method):
        C1, uncertainty = scaleinvariance.two_point_C1(
            fif_data, scaling_method=method, axis=1)
        assert np.isfinite(C1)
        assert uncertainty > 0

    def test_invalid_scaling_method_raises(self, fif_data):
        with pytest.raises(ValueError):
            scaleinvariance.two_point_C1(fif_data, scaling_method='invalid', axis=1)

    def test_order_changes_result(self, fif_data):
        C1_2, _ = scaleinvariance.two_point_C1(fif_data, order=2, axis=1)
        C1_3, _ = scaleinvariance.two_point_C1(fif_data, order=3, axis=1)
        assert C1_2 != C1_3


# ============================================================
# Backend
# ============================================================

class TestBackend:

    def test_get_backend_returns_string(self):
        backend = scaleinvariance.get_backend()
        assert backend in ('numpy', 'torch')

    def test_set_backend_numpy(self):
        scaleinvariance.set_backend('numpy')
        assert scaleinvariance.get_backend() == 'numpy'

    def test_simulation_works_after_set_numpy(self):
        scaleinvariance.set_backend('numpy')
        fbm = scaleinvariance.fBm_1D_circulant(SMALL, 0.5)
        assert not np.any(np.isnan(fbm))

    def test_set_num_threads(self):
        scaleinvariance.set_num_threads(2)  # should not raise
