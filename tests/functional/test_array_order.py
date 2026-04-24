"""
Tests for array-order support in structure_function and haar_fluctuation.

Before this change, `order` had to be a scalar. Both functions now accept
either a scalar (output shape (n_lags,), unchanged) or a 1-D array
(output shape (n_orders, n_lags)). The expensive per-lag work (|delta| or
|convolve|) is computed once and reused across orders.

Also covers the downstream refactor: K_empirical and two_point_C1 now make
a single vectorized call under the hood.
"""

import numpy as np
import pytest

from scaleinvariance import (
    fBm_1D_circulant,
    FIF_1D,
    structure_function,
    haar_fluctuation,
    K_empirical,
    two_point_C1,
)


@pytest.fixture
def fbm_1d():
    return fBm_1D_circulant(512, H=0.7, periodic=True)


class TestStructureFunctionArrayOrder:

    def test_scalar_order_backward_compat(self, fbm_1d):
        lags, sf = structure_function(fbm_1d, order=2)
        assert sf.shape == lags.shape
        assert sf.ndim == 1
        assert sf.dtype == np.float64

    def test_array_order_shape(self, fbm_1d):
        orders = np.array([1.0, 2.0, 3.0])
        lags, sf = structure_function(fbm_1d, order=orders)
        assert sf.shape == (len(orders), len(lags))
        assert sf.dtype == np.float64

    def test_array_order_row_matches_scalar_call(self, fbm_1d):
        orders = np.array([0.5, 1.0, 2.0, 3.0])
        lags, sf_arr = structure_function(fbm_1d, order=orders)
        for i, o in enumerate(orders):
            _, sf_scalar = structure_function(fbm_1d, order=float(o))
            np.testing.assert_allclose(sf_arr[i], sf_scalar, rtol=1e-12, atol=0.0)

    def test_list_of_orders_also_works(self, fbm_1d):
        lags, sf = structure_function(fbm_1d, order=[1.0, 2.0])
        assert sf.shape == (2, len(lags))

    def test_single_element_array_stays_2d(self, fbm_1d):
        """A 1-D array of length 1 is array-like, not scalar — shape should be (1, n_lags)."""
        lags, sf = structure_function(fbm_1d, order=np.array([2.0]))
        assert sf.shape == (1, len(lags))

    def test_nonpositive_in_array_raises(self, fbm_1d):
        with pytest.raises(ValueError, match="positive"):
            structure_function(fbm_1d, order=np.array([1.0, 0.0, 2.0]))
        with pytest.raises(ValueError, match="positive"):
            structure_function(fbm_1d, order=np.array([1.0, -0.5]))


class TestHaarFluctuationArrayOrder:

    def test_scalar_order_backward_compat(self, fbm_1d):
        lags, hf = haar_fluctuation(fbm_1d, order=2)
        assert hf.shape == lags.shape
        assert hf.ndim == 1
        assert hf.dtype == np.float64

    def test_array_order_shape(self, fbm_1d):
        orders = np.array([1.0, 2.0, 3.0])
        lags, hf = haar_fluctuation(fbm_1d, order=orders)
        assert hf.shape == (len(orders), len(lags))
        assert hf.dtype == np.float64

    def test_array_order_row_matches_scalar_call(self, fbm_1d):
        orders = np.array([0.5, 1.0, 2.0])
        lags, hf_arr = haar_fluctuation(fbm_1d, order=orders)
        for i, o in enumerate(orders):
            _, hf_scalar = haar_fluctuation(fbm_1d, order=float(o))
            # equal_nan because odd lags produce NaN at the same places in both
            np.testing.assert_allclose(hf_arr[i], hf_scalar, rtol=1e-12, atol=0.0,
                                       equal_nan=True)

    def test_list_of_orders_also_works(self, fbm_1d):
        lags, hf = haar_fluctuation(fbm_1d, order=[1.0, 2.0])
        assert hf.shape == (2, len(lags))

    def test_nonpositive_in_array_raises(self, fbm_1d):
        with pytest.raises(ValueError, match="positive"):
            haar_fluctuation(fbm_1d, order=np.array([1.0, 0.0]))


class TestIntermittencyRefactor:
    """K_empirical and two_point_C1 now issue a single vectorized call."""

    @pytest.fixture
    def fif_1d(self):
        np.random.seed(42)
        return FIF_1D(2048, alpha=1.8, C1=0.1, H=0.5, causal=False, periodic=True,
                      kernel_construction_method_observable='spectral')

    @pytest.mark.parametrize("scaling_method", ['structure_function', 'haar_fluctuation'])
    def test_K_empirical_runs_and_returns_expected_shapes(self, fif_1d, scaling_method):
        q_values = np.arange(0.5, 2.01, 0.25)
        q_out, K_vals, H_fit, C1_fit, alpha_fit = K_empirical(
            fif_1d, q_values=q_values, scaling_method=scaling_method
        )
        np.testing.assert_array_equal(q_out, q_values)
        assert K_vals.shape == q_values.shape
        assert np.all(np.isfinite(K_vals))
        assert 0.0 < C1_fit < 1.0
        assert 1.01 < alpha_fit <= 2.0

    @pytest.mark.parametrize("scaling_method", ['structure_function', 'haar_fluctuation'])
    def test_two_point_C1_runs(self, fif_1d, scaling_method):
        C1, sigma = two_point_C1(fif_1d, order=2, scaling_method=scaling_method)
        assert np.isfinite(C1)
        assert np.isfinite(sigma)
        assert sigma >= 0

    def test_two_point_C1_order_one_short_circuit(self, fif_1d):
        """K(1) = 0 by construction, so two_point_C1 returns (0, 0) at order=1."""
        C1, sigma = two_point_C1(fif_1d, order=1)
        assert C1 == 0.0
        assert sigma == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
