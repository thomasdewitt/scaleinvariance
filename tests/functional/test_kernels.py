"""
Tests for kernel construction methods (scaleinvariance.simulation.kernels).

Tests that naive and LS2010 kernels:
- Have correct shape and dtype
- Are symmetric (acausal) or one-sided (causal)
- Follow the expected power-law slope at intermediate lags
- Have finite, non-NaN values
- Respond correctly to outer_scale parameter
"""

import pytest
import numpy as np
from scaleinvariance.simulation.kernels import (
    create_kernel_naive, create_kernel_LS2010,
    create_kernel_spectral_odd,
)
from scaleinvariance.simulation.FIF import periodic_convolve
from scaleinvariance.simulation.fractional_integration import fractional_integral_spectral


class TestKernelNaive:

    @pytest.mark.parametrize("exponent", [-0.1, -0.3, -0.5, -0.7, -0.9])
    def test_shape_and_dtype(self, exponent):
        size = 256
        k = create_kernel_naive(size, exponent)
        assert k.shape == (size,)
        assert k.dtype == np.float64
        assert not np.any(np.isnan(k))
        assert not np.any(np.isinf(k))

    def test_symmetry_acausal(self):
        k = create_kernel_naive(256, -0.3, causal=False)
        # Kernel should be symmetric about center
        np.testing.assert_allclose(k[:128], k[128:][::-1], rtol=1e-10)

    def test_causal_zeros_first_half(self):
        k = create_kernel_naive(256, -0.3, causal=True)
        assert np.all(k[:128] == 0)
        assert np.any(k[128:] > 0)

    @pytest.mark.parametrize("exponent", [-0.2, -0.5, -0.8])
    def test_power_law_slope(self, exponent):
        """Check that kernel follows |x|^exponent at intermediate lags."""
        size = 4096
        k = np.array(create_kernel_naive(size, exponent, causal=False))
        mid = size // 2

        # Check at lags 50-500 (intermediate range)
        lags = np.array([50, 100, 200, 400])
        values = np.array([k[mid + lag] for lag in lags])
        expected = lags.astype(float) ** exponent

        # Ratios should be close to 1
        ratios = values / expected
        np.testing.assert_allclose(ratios, ratios[0], rtol=0.05)

    def test_outer_scale_suppresses_large_lags(self):
        size = 1024
        k_full = np.array(create_kernel_naive(size, -0.3, outer_scale=size))
        k_small = np.array(create_kernel_naive(size, -0.3, outer_scale=size // 4))
        mid = size // 2

        # With smaller outer scale, kernel should be more suppressed at large lags
        far_lag = size // 4
        assert abs(k_small[mid + far_lag]) < abs(k_full[mid + far_lag])

    def test_positive_values_acausal(self):
        """Acausal naive kernel should be strictly positive everywhere."""
        k = create_kernel_naive(512, -0.5, causal=False)
        assert np.all(k > 0)


class TestKernelLS2010:

    @pytest.mark.parametrize("exponent", [-0.1, -0.3, -0.5, -0.7, -0.9])
    def test_shape_and_dtype(self, exponent):
        size = 256
        norm_ratio_exp = -(exponent + 1.0)
        k = create_kernel_LS2010(size, exponent, norm_ratio_exp)
        assert k.shape == (size,)
        assert k.dtype == np.float64
        assert not np.any(np.isnan(k))

    def test_symmetry_acausal(self):
        k = np.array(create_kernel_LS2010(256, -0.3, -0.7, causal=False))
        np.testing.assert_allclose(k[:128], k[128:][::-1], rtol=1e-10)

    def test_causal_zeros_first_half(self):
        k = np.array(create_kernel_LS2010(256, -0.3, -0.7, causal=True))
        assert np.all(k[:128] == 0)

    @pytest.mark.parametrize("exponent", [-0.2, -0.5, -0.8])
    def test_power_law_slope(self, exponent):
        """LS2010 kernel should follow approximate power law at intermediate lags."""
        size = 4096
        norm_ratio_exp = -(exponent + 1.0)
        k = np.array(create_kernel_LS2010(size, exponent, norm_ratio_exp, causal=False))
        mid = size // 2

        lags = np.array([100, 200, 400])
        values = np.array([k[mid + lag] for lag in lags])

        # Estimate slope from log-log fit
        log_lags = np.log(lags.astype(float))
        log_vals = np.log(np.abs(values))
        slope = np.polyfit(log_lags, log_vals, 1)[0]

        # Slope should be within 20% of exponent (LS2010 has corrections)
        assert abs(slope - exponent) / abs(exponent) < 0.2, \
            f"Slope {slope:.3f} too far from exponent {exponent}"

    def test_nd_kernel(self):
        """Test 2D kernel construction."""
        size = (64, 64)
        k = create_kernel_LS2010(size, -1.7, -0.3)
        assert k.shape == size
        assert not np.any(np.isnan(k))
        # Should be symmetric under 180-degree rotation
        np.testing.assert_allclose(k, k[::-1, ::-1], rtol=1e-10)

    def test_outer_scale_suppresses_large_lags(self):
        size = 1024
        norm_ratio_exp = -0.7
        k_full = np.array(create_kernel_LS2010(size, -0.3, norm_ratio_exp, outer_scale=size))
        k_small = np.array(create_kernel_LS2010(size, -0.3, norm_ratio_exp, outer_scale=size // 4))
        mid = size // 2

        far_lag = size // 4
        assert abs(k_small[mid + far_lag]) < abs(k_full[mid + far_lag])


class TestKernelSpectralOdd:

    def test_shape(self):
        k = create_kernel_spectral_odd(256, -0.3)
        assert k.shape == (256,)
        assert not np.any(np.isnan(k))

    def test_rejects_causal(self):
        with pytest.raises(ValueError):
            create_kernel_spectral_odd(256, -0.3, causal=True)

    def test_rejects_nd(self):
        with pytest.raises(ValueError):
            create_kernel_spectral_odd((32, 32), -0.3)

    def test_antisymmetric_impulse_response(self):
        """IFFT of odd kernel should be antisymmetric: k(mid-j) = -k(mid+j)."""
        size = 512
        response = create_kernel_spectral_odd(size, -0.3)
        k = np.fft.fftshift(np.real(np.fft.ifft(response)))
        mid = size // 2
        right = k[mid + 1:]
        left_reversed = k[mid - 1:0:-1]
        np.testing.assert_allclose(left_reversed, -right, atol=1e-12)

    def test_zero_dc(self):
        """Odd kernel should have zero DC component."""
        response = create_kernel_spectral_odd(512, -0.5)
        assert abs(response[0]) < 1e-12

    def test_magnitude_is_power_law(self):
        """Odd kernel magnitude (excluding DC and Nyquist) should follow
        the expected power-law |f|^response_exponent."""
        size = 512
        exponent = -0.3
        response_exponent = -(1.0 + exponent)  # scaling_dimension=1
        resp = create_kernel_spectral_odd(size, exponent)
        freqs = np.fft.fftfreq(size, d=1.0)
        # Build expected magnitude from the regularized frequency grid.
        freqs_abs = np.abs(freqs)
        min_freq = 1.0 / float(size)
        freqs_reg = np.maximum(freqs_abs, min_freq)
        expected_mag = freqs_reg ** response_exponent
        expected_mag[0] = expected_mag[1]
        # Skip DC (0) and Nyquist (zeroed in odd kernel).
        mask = np.ones(size, dtype=bool)
        mask[0] = False
        mask[size // 2] = False
        np.testing.assert_allclose(np.abs(resp[mask]), expected_mag[mask], rtol=1e-10)

    def test_fif_1d_with_spectral_odd(self):
        """FIF_1D should run without error using spectral_odd observable kernel."""
        from scaleinvariance import FIF_1D
        result = FIF_1D(256, alpha=1.8, C1=0.1, H=0.3, causal=False,
                        kernel_construction_method_observable='spectral_odd',
                        kernel_construction_method_flux='LS2010')
        assert result.shape == (256,)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestKernelConsistency:
    """Cross-method consistency checks."""

    @pytest.mark.parametrize("exponent", [-0.3, -0.7])
    def test_naive_ls2010_same_slope(self, exponent):
        """Both methods should give similar power-law slopes at intermediate lags."""
        size = 4096
        norm_ratio_exp = -(exponent + 1.0)
        k_naive = np.array(create_kernel_naive(size, exponent))
        k_ls2010 = np.array(create_kernel_LS2010(size, exponent, norm_ratio_exp))
        mid = size // 2

        lags = np.array([100, 200, 400, 800])

        for k, name in [(k_naive, 'naive'), (k_ls2010, 'LS2010')]:
            values = np.array([k[mid + lag] for lag in lags])
            log_lags = np.log(lags.astype(float))
            log_vals = np.log(np.abs(values))
            slope = np.polyfit(log_lags, log_vals, 1)[0]
            assert abs(slope - exponent) / abs(exponent) < 0.25, \
                f"{name} slope {slope:.3f} too far from {exponent}"

    def test_peak_at_center(self):
        """Both kernels should peak near the center."""
        size = 512
        for method in ['naive', 'LS2010']:
            if method == 'naive':
                k = np.array(create_kernel_naive(size, -0.5))
            else:
                k = np.array(create_kernel_LS2010(size, -0.5, -0.5))
            peak_idx = np.argmax(k)
            # Peak should be within 1 of center
            assert abs(peak_idx - size // 2) <= 1, \
                f"{method} peak at {peak_idx}, expected near {size // 2}"


class TestFractionalIntegralSpectral:
    """Tests for fractional_integral_spectral."""

    def test_identity_at_H_zero_1d(self):
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(512)
        out = fractional_integral_spectral(signal, H=0.0)
        np.testing.assert_allclose(out, signal, rtol=1e-12, atol=1e-12)

    def test_identity_at_H_zero_2d(self):
        rng = np.random.default_rng(0)
        signal = rng.standard_normal((64, 64))
        out = fractional_integral_spectral(signal, H=0.0)
        np.testing.assert_allclose(out, signal, rtol=1e-12, atol=1e-12)

    def test_identity_at_H_zero_3d(self):
        rng = np.random.default_rng(0)
        signal = rng.standard_normal((16, 16, 16))
        out = fractional_integral_spectral(signal, H=0.0)
        np.testing.assert_allclose(out, signal, rtol=1e-12, atol=1e-12)

    def test_real_output(self):
        """Output should be real-valued for a real input."""
        rng = np.random.default_rng(7)
        signal = rng.standard_normal(256)
        out = fractional_integral_spectral(signal, H=0.3)
        assert np.isrealobj(out)
        assert not np.any(np.isnan(out))
