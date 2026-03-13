"""
Functional tests for compute_K_q_function.

Uses small/fast simulations. Tests correctness of return types, shapes,
mathematical identities (K(1)=0, sign of K), error handling, and gross
parameter recovery.
"""

import numpy as np
import pytest
import scaleinvariance

SIZE   = 2**10
N_SIMS = 4
H, C1, alpha = 0.5, 0.1, 1.8


@pytest.fixture(scope='module')
def fif_sims():
    return np.stack([
        scaleinvariance.FIF_1D(SIZE, alpha, C1, H=H) for _ in range(N_SIMS)
    ])


class TestReturnTypes:

    def test_shapes_consistent(self, fif_sims):
        q_values, K_emp, H_fit, C1_fit, alpha_fit = scaleinvariance.compute_K_q_function(
            fif_sims, axis=1)
        assert q_values.shape == K_emp.shape
        assert q_values.ndim == 1

    def test_scalar_outputs(self, fif_sims):
        _, _, H_fit, C1_fit, alpha_fit = scaleinvariance.compute_K_q_function(
            fif_sims, axis=1)
        for val in (H_fit, C1_fit, alpha_fit):
            assert np.isscalar(val) or val.ndim == 0
            assert np.isfinite(val)

    def test_default_q_values(self, fif_sims):
        q_values, *_ = scaleinvariance.compute_K_q_function(fif_sims, axis=1)
        np.testing.assert_allclose(q_values, np.arange(0.1, 2.51, 0.1))

    def test_custom_q_values(self, fif_sims):
        custom_q = np.array([0.5, 1.0, 1.5, 2.0])
        q_values, K_emp, *_ = scaleinvariance.compute_K_q_function(
            fif_sims, q_values=custom_q, axis=1)
        np.testing.assert_array_equal(q_values, custom_q)
        assert K_emp.shape == (4,)


class TestMathematicalIdentities:

    def test_K_at_q1_near_zero(self, fif_sims):
        # K(1) = 0 by definition regardless of parameters
        q_values, K_emp, *_ = scaleinvariance.compute_K_q_function(
            fif_sims, q_values=np.array([0.5, 1.0, 1.5]), axis=1)
        idx = np.where(q_values == 1.0)[0][0]
        assert abs(K_emp[idx]) < 0.1

    def test_K_positive_for_q_gt_1(self, fif_sims):
        q_values, K_emp, *_ = scaleinvariance.compute_K_q_function(fif_sims, axis=1)
        assert np.all(K_emp[q_values > 1.0] > -0.05)

    def test_K_negative_for_q_lt_1(self, fif_sims):
        q_values, K_emp, *_ = scaleinvariance.compute_K_q_function(fif_sims, axis=1)
        assert np.all(K_emp[q_values < 1.0] < 0.05)


class TestOptions:

    def test_scaling_method_structure_function(self, fif_sims):
        q, K, H_fit, C1_fit, alpha_fit = scaleinvariance.compute_K_q_function(
            fif_sims, scaling_method='structure_function', axis=1)
        assert np.all(np.isfinite(K))

    def test_scaling_method_haar_fluctuation(self, fif_sims):
        q, K, H_fit, C1_fit, alpha_fit = scaleinvariance.compute_K_q_function(
            fif_sims, scaling_method='haar_fluctuation', axis=1)
        assert np.all(np.isfinite(K))

    def test_hurst_fit_method_fixed(self, fif_sims):
        _, _, H_fit, C1_fit, alpha_fit = scaleinvariance.compute_K_q_function(
            fif_sims, hurst_fit_method='fixed', axis=1)
        assert np.isfinite(H_fit) and np.isfinite(C1_fit) and np.isfinite(alpha_fit)

    def test_hurst_fit_method_joint(self, fif_sims):
        _, _, H_fit, C1_fit, alpha_fit = scaleinvariance.compute_K_q_function(
            fif_sims, hurst_fit_method='joint', axis=1)
        assert np.isfinite(H_fit) and np.isfinite(C1_fit) and np.isfinite(alpha_fit)

    def test_nan_behavior_ignore_haar(self):
        data = np.stack([scaleinvariance.FIF_1D(SIZE, alpha, C1, H=H) for _ in range(N_SIMS)])
        data[0, :10] = np.nan
        scaleinvariance.compute_K_q_function(
            data, scaling_method='haar_fluctuation', nan_behavior='ignore', axis=1)


class TestErrorHandling:

    def test_invalid_scaling_method_raises(self, fif_sims):
        with pytest.raises(ValueError):
            scaleinvariance.compute_K_q_function(fif_sims, scaling_method='invalid', axis=1)

    def test_invalid_hurst_fit_method_raises(self, fif_sims):
        with pytest.raises(ValueError):
            scaleinvariance.compute_K_q_function(fif_sims, hurst_fit_method='invalid', axis=1)

    def test_too_few_q_values_fixed_raises(self, fif_sims):
        with pytest.raises(ValueError):
            scaleinvariance.compute_K_q_function(
                fif_sims, q_values=np.array([1.0, 1.5]), hurst_fit_method='fixed', axis=1)

    def test_too_few_q_values_joint_raises(self, fif_sims):
        with pytest.raises(ValueError):
            scaleinvariance.compute_K_q_function(
                fif_sims, q_values=np.array([0.5, 1.0, 1.5]), hurst_fit_method='joint', axis=1)


class TestGrossAccuracy:
    """Broad-tolerance checks — small sims, wide windows. These should always pass."""

    def test_fitted_C1_in_range(self, fif_sims):
        _, _, _, C1_fit, _ = scaleinvariance.compute_K_q_function(fif_sims, axis=1)
        assert 0.0 <= C1_fit <= 0.5, f"C1_fit={C1_fit:.3f} out of expected range"

    def test_fitted_alpha_in_range(self, fif_sims):
        _, _, _, _, alpha_fit = scaleinvariance.compute_K_q_function(fif_sims, axis=1)
        assert 0.5 <= alpha_fit <= 2.0, f"alpha_fit={alpha_fit:.3f} out of expected range"

    def test_fitted_H_in_range(self, fif_sims):
        _, _, H_fit, _, _ = scaleinvariance.compute_K_q_function(fif_sims, axis=1)
        assert 0.1 <= H_fit <= 0.9, f"H_fit={H_fit:.3f} out of expected range"
