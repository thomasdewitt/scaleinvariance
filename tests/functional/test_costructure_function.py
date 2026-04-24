"""
Functional tests for costructure_function (two-field, two-exponent mixed structure function).
"""

import numpy as np
import pytest

from scaleinvariance import (
    fBm_1D_circulant,
    structure_function,
    costructure_function,
)


@pytest.fixture
def fbm_1d():
    return fBm_1D_circulant(512, H=0.7, periodic=True)


@pytest.fixture
def fbm_1d_other():
    return fBm_1D_circulant(512, H=0.3, periodic=True)


class TestSelfConsistency:
    """costructure_function(f, f, p, q) must equal structure_function(f, p+q).

    The analytic identity |d|^p * |d|^q = |d|^(p+q) holds exactly only for integer
    exponents under floating-point arithmetic; for general real exponents, the
    two computation paths (two pows + multiply vs single pow) disagree at the
    roundoff level. We therefore use a tolerance matched to the active precision
    (default float32).
    """

    @pytest.mark.parametrize("p,q", [(1.0, 1.0), (0.5, 1.5), (2.0, 0.5), (0.3, 0.7)])
    def test_same_field_matches_single_field(self, fbm_1d, p, q):
        lags_co, cosf = costructure_function(fbm_1d, fbm_1d, order1=p, order2=q)
        lags_sf, sf = structure_function(fbm_1d, order=p + q)

        np.testing.assert_array_equal(lags_co, lags_sf)
        # ~1e-7 is float32 machine epsilon; allow a generous margin for
        # accumulated roundoff in the two-pow path.
        np.testing.assert_allclose(cosf, sf, rtol=1e-5, atol=1e-7)

    def test_scalar_scalar_shape(self, fbm_1d):
        lags, cosf = costructure_function(fbm_1d, fbm_1d, order1=1.0, order2=1.0)
        assert cosf.shape == lags.shape
        assert cosf.ndim == 1


class TestArrayOrderShapes:
    """Output shape conventions for scalar/array order1, order2."""

    def test_order1_array_order2_scalar(self, fbm_1d, fbm_1d_other):
        orders = np.array([1.0, 2.0, 3.0])
        lags, cosf = costructure_function(fbm_1d, fbm_1d_other, order1=orders, order2=1.0)
        assert cosf.shape == (len(orders), len(lags))

    def test_order1_scalar_order2_array(self, fbm_1d, fbm_1d_other):
        orders = np.array([1.0, 2.0])
        lags, cosf = costructure_function(fbm_1d, fbm_1d_other, order1=1.0, order2=orders)
        assert cosf.shape == (len(orders), len(lags))

    def test_both_arrays(self, fbm_1d, fbm_1d_other):
        o1 = np.array([1.0, 2.0, 3.0])
        o2 = np.array([0.5, 1.5])
        lags, cosf = costructure_function(fbm_1d, fbm_1d_other, order1=o1, order2=o2)
        assert cosf.shape == (len(o1), len(o2), len(lags))

    def test_array_row_matches_scalar_call(self, fbm_1d, fbm_1d_other):
        """Each row of an array-order call must match an individual scalar-order call."""
        orders = np.array([1.0, 2.0, 3.0])
        lags, cosf_array = costructure_function(fbm_1d, fbm_1d_other,
                                                 order1=orders, order2=1.5)
        for i, o in enumerate(orders):
            _, cosf_scalar = costructure_function(fbm_1d, fbm_1d_other,
                                                   order1=float(o), order2=1.5)
            np.testing.assert_allclose(cosf_array[i], cosf_scalar, rtol=1e-12, atol=0.0)


class TestValidation:

    def test_shape_mismatch_raises(self):
        f1 = np.random.randn(100)
        f2 = np.random.randn(200)
        with pytest.raises(ValueError, match="same shape"):
            costructure_function(f1, f2)

    def test_ndim_mismatch_raises(self):
        f1 = np.random.randn(100)
        f2 = np.random.randn(10, 10)
        with pytest.raises(ValueError, match="same shape"):
            costructure_function(f1, f2)

    def test_nonpositive_order_raises(self, fbm_1d, fbm_1d_other):
        with pytest.raises(ValueError, match="positive"):
            costructure_function(fbm_1d, fbm_1d_other, order1=0.0, order2=1.0)
        with pytest.raises(ValueError, match="positive"):
            costructure_function(fbm_1d, fbm_1d_other, order1=1.0, order2=-0.5)
        with pytest.raises(ValueError, match="positive"):
            costructure_function(fbm_1d, fbm_1d_other, order1=np.array([1.0, -0.5]), order2=1.0)

    def test_bad_axis_raises(self, fbm_1d, fbm_1d_other):
        with pytest.raises(ValueError, match="out of bounds"):
            costructure_function(fbm_1d, fbm_1d_other, axis=5)


class TestNaNHandling:

    def test_nan_in_one_field_propagates_through_intersection(self):
        rng = np.random.default_rng(0)
        f1 = rng.standard_normal(512)
        f2 = rng.standard_normal(512)
        # Inject NaNs into f1 only; f2 stays finite.
        nan_idx = np.array([50, 100, 200, 300])
        f1[nan_idx] = np.nan

        lags, cosf = costructure_function(f1, f2, order1=1.0, order2=1.0,
                                          max_sep=16, lags='all')

        # Result must be finite at all computed lags (only a few NaNs, plenty of valid pairs).
        assert np.all(np.isfinite(cosf)), f"Unexpected NaNs in cosf: {cosf}"

        # Manually compute for lag=1 and compare.
        lag = 1
        d1 = np.abs(f1[lag:] - f1[:-lag])
        d2 = np.abs(f2[lag:] - f2[:-lag])
        expected = np.nanmean(d1 * d2)  # averages over positions where both finite
        # Find index of lag=1 in returned lags
        idx = int(np.where(lags == 1)[0][0])
        np.testing.assert_allclose(cosf[idx], expected, rtol=1e-10)


class TestIndependentFieldsScaling:
    """For two independent fBm fields with Hurst H1, H2, the (1,1) costructure
    factorizes: <|delta f1| * |delta f2|> = <|delta f1|> * <|delta f2|> ~ r^(H1+H2).
    """

    def test_scaling_slope_recovers_sum_of_hursts(self):
        H1, H2 = 0.7, 0.3
        # Use distinct seeds to keep the two fields independent.
        rng = np.random.default_rng(0)
        # fBm_1D_circulant uses np.random internally; manually re-seed.
        np.random.seed(1)
        f1 = fBm_1D_circulant(4096, H=H1, periodic=True)
        np.random.seed(2)
        f2 = fBm_1D_circulant(4096, H=H2, periodic=True)

        lags, cosf = costructure_function(f1, f2, order1=1.0, order2=1.0,
                                          max_sep=512, lags='powers of 1.2')

        # Fit slope on log-log in the scaling range.
        mask = (lags >= 4) & (lags <= 256)
        log_lags = np.log(lags[mask].astype(float))
        log_cosf = np.log(cosf[mask])
        slope, _ = np.polyfit(log_lags, log_cosf, 1)

        # Tolerance is loose because finite-sample scatter is substantial
        # on a single realization.
        assert abs(slope - (H1 + H2)) < 0.15, (
            f"Expected slope ~{H1 + H2}, got {slope:.3f}"
        )


class TestDefaults:

    def test_default_lags_is_powers_of_1_2(self, fbm_1d, fbm_1d_other):
        lags, _ = costructure_function(fbm_1d, fbm_1d_other)
        # Should be log-spaced (few lags, much less than N).
        assert 5 < len(lags) < 100
        assert lags[0] >= 1

    def test_returns_float64(self, fbm_1d, fbm_1d_other):
        _, cosf = costructure_function(fbm_1d, fbm_1d_other)
        assert cosf.dtype == np.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
