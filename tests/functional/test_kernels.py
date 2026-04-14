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
from scaleinvariance.simulation.fractional_integration import (
    fractional_integral_spectral,
    broken_fractional_integral_spectral,
    _isotropic_rfft_frequencies,
)
from scaleinvariance import get_numerical_precision, set_numerical_precision

def _expected_dtype():
    return np.float32 if get_numerical_precision() == 'float32' else np.float64


class TestKernelNaive:

    @pytest.mark.parametrize("exponent", [-0.1, -0.3, -0.5, -0.7, -0.9])
    def test_shape_and_dtype(self, exponent):
        size = 256
        k = create_kernel_naive(size, exponent)
        assert k.shape == (size,)
        assert k.dtype == _expected_dtype()
        assert not np.any(np.isnan(np.array(k)))
        assert not np.any(np.isinf(np.array(k)))

    def test_symmetry_acausal(self):
        k = np.array(create_kernel_naive(256, -0.3, causal=False))
        # Kernel should be symmetric about center
        np.testing.assert_allclose(k[:128], k[128:][::-1], rtol=1e-4)

    def test_causal_zeros_first_half(self):
        k = np.array(create_kernel_naive(256, -0.3, causal=True))
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
        assert k.dtype == _expected_dtype()
        assert not np.any(np.isnan(np.array(k)))

    def test_symmetry_acausal(self):
        k = np.array(create_kernel_LS2010(256, -0.3, -0.7, causal=False))
        np.testing.assert_allclose(k[:128], k[128:][::-1], rtol=1e-4)

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
        assert not np.any(np.isnan(np.array(k)))
        # Should be symmetric under 180-degree rotation
        np.testing.assert_allclose(k, k[::-1, ::-1], rtol=1e-4)

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
        k = np.array(create_kernel_spectral_odd(256, -0.3))
        assert k.shape == (256,)
        assert not np.any(np.isnan(np.array(k)))

    def test_rejects_causal(self):
        with pytest.raises(ValueError):
            create_kernel_spectral_odd(256, -0.3, causal=True)

    def test_rejects_nd(self):
        with pytest.raises(ValueError):
            create_kernel_spectral_odd((32, 32), -0.3)

    def test_antisymmetric_impulse_response(self):
        """IFFT of odd kernel should be antisymmetric: k(mid-j) = -k(mid+j)."""
        size = 512
        response = np.array(create_kernel_spectral_odd(size, -0.3), dtype=np.complex128)
        k = np.fft.fftshift(np.real(np.fft.ifft(response)))
        mid = size // 2
        right = k[mid + 1:]
        left_reversed = k[mid - 1:0:-1]
        np.testing.assert_allclose(left_reversed, -right, atol=1e-6)

    def test_zero_dc(self):
        """Odd kernel should have zero DC component."""
        response = np.array(create_kernel_spectral_odd(512, -0.5))
        assert abs(response[0]) < 1e-6

    def test_magnitude_is_power_law(self):
        """Odd kernel magnitude (excluding DC and Nyquist) should follow
        the expected power-law |f|^response_exponent."""
        size = 512
        exponent = -0.3
        response_exponent = -(1.0 + exponent)  # scaling_dimension=1
        resp = np.array(create_kernel_spectral_odd(size, exponent))
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
        np.testing.assert_allclose(np.abs(resp[mask]), expected_mag[mask], rtol=1e-4)

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

    def test_raises_at_H_zero(self):
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(512)
        with pytest.raises(ValueError):
            fractional_integral_spectral(signal, H=0.0)

    def test_real_output(self):
        """Output should be real-valued for a real input."""
        rng = np.random.default_rng(7)
        signal = rng.standard_normal(256)
        out = fractional_integral_spectral(signal, H=0.3)
        out = np.array(out)
        assert np.isrealobj(out)
        assert not np.any(np.isnan(out))

    def test_preserves_positivity_of_flux_like_input(self):
        """A unit-mean positive input (like the FIF flux) stays positive after
        integration because the DC bin is filled from the nearest non-DC bin,
        giving the kernel no spectral notch at DC."""
        rng = np.random.default_rng(1)
        size = 1024
        flux = np.exp(0.2 * rng.standard_normal(size))  # strictly positive
        flux = flux / np.mean(flux)  # unit mean
        out = np.array(fractional_integral_spectral(flux, H=0.4))
        assert np.all(out > 0)

    def test_raises_on_overflow_float32(self):
        """With float32 (default) and default outer_scale, a huge H should trip
        the overflow guard."""
        assert get_numerical_precision() == 'float32'
        rng = np.random.default_rng(4)
        signal = rng.standard_normal(1024)
        with pytest.raises(OverflowError):
            # 1024^50 ≫ float32 max
            fractional_integral_spectral(signal, H=50.0)

    def test_large_H_works_in_float64(self):
        """Same H that overflowed float32 should run cleanly in float64."""
        try:
            set_numerical_precision('float64')
            rng = np.random.default_rng(5)
            signal = rng.standard_normal(1024)
            out = np.array(fractional_integral_spectral(signal, H=20.0))
            assert np.all(np.isfinite(out))
        finally:
            set_numerical_precision('float32')

    def test_negative_H_no_overflow_check(self):
        """Negative H never overflows (kernel <= 1) regardless of size."""
        rng = np.random.default_rng(6)
        signal = rng.standard_normal(512)
        out = np.array(fractional_integral_spectral(signal, H=-50.0))
        assert np.all(np.isfinite(out))


class TestBrokenFractionalIntegralSpectral:
    """Tests for broken_fractional_integral_spectral."""

    def test_raises_when_both_H_zero(self):
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(512)
        with pytest.raises(ValueError):
            broken_fractional_integral_spectral(
                signal, H_small_scale=0.0, H_large_scale=0.0,
                transition_wavelength=32,
            )

    def test_runs_when_one_H_zero(self):
        rng = np.random.default_rng(1)
        signal = rng.standard_normal(512)
        out = np.array(broken_fractional_integral_spectral(
            signal, H_small_scale=0.3, H_large_scale=0.0,
            transition_wavelength=32,
        ))
        assert np.all(np.isfinite(out))

    def test_equals_single_exponent_when_H_equal_1d(self):
        rng = np.random.default_rng(2)
        signal = rng.standard_normal(512)
        H = 0.4
        single = np.array(fractional_integral_spectral(signal, H=H))
        broken = np.array(broken_fractional_integral_spectral(
            signal, H_small_scale=H, H_large_scale=H,
            transition_wavelength=32,
        ))
        np.testing.assert_allclose(broken, single, rtol=1e-4, atol=1e-4)

    def test_equals_single_exponent_when_H_equal_2d(self):
        rng = np.random.default_rng(3)
        signal = rng.standard_normal((64, 64))
        H = 0.3
        single = np.array(fractional_integral_spectral(signal, H=H))
        broken = np.array(broken_fractional_integral_spectral(
            signal, H_small_scale=H, H_large_scale=H,
            transition_wavelength=8,
        ))
        np.testing.assert_allclose(broken, single, rtol=1e-4, atol=1e-4)

    def test_real_output(self):
        rng = np.random.default_rng(4)
        signal = rng.standard_normal(256)
        out = np.array(broken_fractional_integral_spectral(
            signal, H_small_scale=0.2, H_large_scale=0.7,
            transition_wavelength=16,
        ))
        assert np.isrealobj(out)
        assert np.all(np.isfinite(out))

    def test_kernel_continuity_at_transition(self):
        """Reconstruct the kernel and check the two branches agree at k_t."""
        size = 4096
        H_small, H_large = 0.25, 0.7
        transition_wavelength = 32.0
        outer_scale = size

        freqs = np.array(_isotropic_rfft_frequencies((size,)))
        min_freq = 1.0 / float(outer_scale)
        freqs_reg = np.maximum(freqs, min_freq)
        k_t = 1.0 / transition_wavelength

        small_prefactor = k_t ** (H_small - H_large)
        large_branch = freqs_reg ** (-H_large)
        small_branch = small_prefactor * freqs_reg ** (-H_small)
        kernel = np.where(freqs_reg <= k_t, large_branch, small_branch)

        # Find bins straddling k_t and confirm the two branch values agree there.
        # The kernel at the highest bin <= k_t (from large branch) and the
        # lowest bin > k_t (from small branch) should be within a small factor
        # of each other given grid discreteness.
        below_idx = np.searchsorted(freqs_reg, k_t, side='right') - 1
        above_idx = below_idx + 1
        # Continuity: evaluate both branches at k_t exactly
        at_kt_large = k_t ** (-H_large)
        at_kt_small = small_prefactor * k_t ** (-H_small)
        np.testing.assert_allclose(at_kt_large, at_kt_small, rtol=1e-12)
        # Sanity: the actual kernel values at straddling bins should each be
        # within a factor of 2 of at_kt_large (grid resolution effect).
        assert 0.5 * at_kt_large < kernel[below_idx] < 2.0 * at_kt_large
        assert 0.5 * at_kt_large < kernel[above_idx] < 2.0 * at_kt_large

    def test_slopes_in_each_regime(self):
        """Applied to white noise, the output PSD slopes should approximately
        match -2*H on each side of k_t. Loose tolerance; smoke-level only."""
        size = 8192
        H_small, H_large = 0.3, 0.7
        transition_wavelength = 64.0
        rng = np.random.default_rng(7)
        white = rng.standard_normal(size)
        out = np.array(broken_fractional_integral_spectral(
            white, H_small_scale=H_small, H_large_scale=H_large,
            transition_wavelength=transition_wavelength,
        )).astype(np.float64)
        # rfft power spectrum
        freqs_rfft = np.fft.rfftfreq(size, d=1.0)
        power = np.abs(np.fft.rfft(out)) ** 2
        k_t = 1.0 / transition_wavelength
        # Pick well-separated bin ranges inside each regime.
        large_mask = (freqs_rfft > 4 / size) & (freqs_rfft < k_t / 2)
        small_mask = (freqs_rfft > 2 * k_t) & (freqs_rfft < 0.25)
        slope_large = np.polyfit(
            np.log(freqs_rfft[large_mask]), np.log(power[large_mask]), 1
        )[0]
        slope_small = np.polyfit(
            np.log(freqs_rfft[small_mask]), np.log(power[small_mask]), 1
        )[0]
        # Expect slopes near -2*H; allow ±0.4 tolerance for finite-N variance
        assert abs(slope_large - (-2 * H_large)) < 0.4, slope_large
        assert abs(slope_small - (-2 * H_small)) < 0.4, slope_small

    def test_raises_on_overflow_float32(self):
        assert get_numerical_precision() == 'float32'
        rng = np.random.default_rng(8)
        signal = rng.standard_normal(1024)
        with pytest.raises(OverflowError):
            broken_fractional_integral_spectral(
                signal, H_small_scale=0.3, H_large_scale=50.0,
                transition_wavelength=32,
            )

    def test_raises_on_prefactor_overflow(self):
        """When transition_wavelength >> outer_scale with a big H gap, the
        continuity prefactor k_t^(H_small-H_large) dominates and should
        trigger the guard even though outer_scale^max(H) is modest."""
        assert get_numerical_precision() == 'float32'
        rng = np.random.default_rng(9)
        signal = rng.standard_normal(1024)
        with pytest.raises(OverflowError):
            broken_fractional_integral_spectral(
                signal,
                H_small_scale=0.0,   # actually the prefactor depends on the gap
                H_large_scale=20.0,
                transition_wavelength=1e9,  # k_t tiny → big prefactor exponent
                outer_scale=1024,
            )

    def test_raises_on_nonpositive_transition_wavelength(self):
        rng = np.random.default_rng(10)
        signal = rng.standard_normal(256)
        with pytest.raises(ValueError, match="transition_wavelength"):
            broken_fractional_integral_spectral(
                signal, H_small_scale=0.3, H_large_scale=0.5,
                transition_wavelength=0,
            )
        with pytest.raises(ValueError, match="transition_wavelength"):
            broken_fractional_integral_spectral(
                signal, H_small_scale=0.3, H_large_scale=0.5,
                transition_wavelength=-10,
            )

    def test_raises_on_nonpositive_outer_scale(self):
        rng = np.random.default_rng(11)
        signal = rng.standard_normal(256)
        with pytest.raises(ValueError, match="outer_scale"):
            broken_fractional_integral_spectral(
                signal, H_small_scale=0.3, H_large_scale=0.5,
                transition_wavelength=32, outer_scale=0,
            )
        with pytest.raises(ValueError, match="outer_scale"):
            fractional_integral_spectral(signal, H=0.3, outer_scale=-5)
