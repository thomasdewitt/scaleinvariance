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


class TestKernelLS2010PerAxisCausal:
    """Tests for the per-axis causal extension on create_kernel_LS2010."""

    def test_per_axis_causal_zeroes_correct_half(self):
        """causal=(True, False) zeros the axis-0 negative half only."""
        k = np.array(create_kernel_LS2010((64, 64), -1.7, -0.3, causal=(True, False)))
        assert np.all(k[:32, :] == 0)
        assert np.any(k[32:, :] != 0)

    def test_all_axes_causal_keeps_one_orthant(self):
        k = np.array(create_kernel_LS2010((32, 32), -1.7, -0.3, causal=True))
        assert np.all(k[:16, :] == 0)
        assert np.all(k[:, :16] == 0)
        assert np.any(k[16:, 16:] != 0)

    def test_causal_axis1_only(self):
        """causal=(False, True) zeros the axis-1 negative half only."""
        k = np.array(create_kernel_LS2010((64, 64), -1.7, -0.3, causal=(False, True)))
        assert np.all(k[:, :32] == 0)
        assert np.any(k[:, 32:] != 0)

    def test_raises_on_tuple_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            create_kernel_LS2010((64, 64), -1.7, -0.3, causal=(True, True, True))


class TestFractionalIntegralSpectralOddAxes:
    """Tests for odd_axes on fractional_integral_spectral."""

    def test_1d_real_output_zero_mean(self):
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(1024)
        out = np.array(fractional_integral_spectral(signal, H=0.4, odd_axes=True))
        assert np.isrealobj(out)
        assert abs(out.mean()) < 1e-5
        assert np.all(np.isfinite(out))

    def test_2d_both_odd_real_output(self):
        """k=2 makes the kernel purely real (i^2 = -1); output is real and zero-mean."""
        rng = np.random.default_rng(1)
        signal = rng.standard_normal((64, 64))
        out = np.array(fractional_integral_spectral(signal, H=0.3, odd_axes=True))
        assert np.isrealobj(out)
        assert abs(out.mean()) < 1e-5

    def test_3d_all_odd_real_output(self):
        """k=3 → purely imaginary kernel + Hermitian → still real-valued output."""
        rng = np.random.default_rng(2)
        signal = rng.standard_normal((16, 16, 16))
        out = np.array(fractional_integral_spectral(signal, H=0.2, odd_axes=True))
        assert np.isrealobj(out)
        assert abs(out.mean()) < 1e-5

    def test_matches_legacy_create_kernel_spectral_odd_1d(self):
        """The new spectral path with odd_axes=True must match the legacy
        create_kernel_spectral_odd convolution in 1D up to floating-point
        precision."""
        from scaleinvariance.simulation.FIF import periodic_convolve
        H = 0.4
        size = 1024
        rng = np.random.default_rng(3)
        signal = rng.standard_normal(size)
        legacy_kernel = create_kernel_spectral_odd(
            size, -1.0 + H, causal=False, outer_scale=size
        )
        legacy_out = np.array(periodic_convolve(signal, legacy_kernel, kernel_is_fourier=True))
        new_out = np.array(fractional_integral_spectral(signal, H=H, odd_axes=True))
        # Both produce mathematically identical outputs (same Fourier response,
        # same DC/Nyquist handling).
        np.testing.assert_allclose(legacy_out, new_out, rtol=1e-4, atol=1e-4)

    def test_odd_path_preserves_magnitude_spectrum(self):
        """|FFT(K_odd · X)|^2 / |FFT(X)|^2 should match |f|^(-2H) (radial PSD)
        away from DC/Nyquist — odd kernels only change phases, not magnitudes."""
        H = 0.3
        size = 1024
        rng = np.random.default_rng(4)
        signal = rng.standard_normal(size)
        even_out = np.array(fractional_integral_spectral(signal, H=H))
        odd_out = np.array(fractional_integral_spectral(signal, H=H, odd_axes=True))
        # Compare ratio of power spectra at intermediate frequencies.
        from numpy.fft import rfft
        psd_even = np.abs(rfft(even_out)) ** 2
        psd_odd = np.abs(rfft(odd_out)) ** 2
        # Use mid-range bins (avoid DC and Nyquist, where the odd kernel
        # explicitly zeros).
        mid = slice(10, size // 4)
        np.testing.assert_allclose(psd_odd[mid], psd_even[mid], rtol=1e-3)


class TestFIF_ND_OddAndCausal:
    """End-to-end tests for the new N-D parameters."""

    def test_2d_odd_output_is_zero_mean(self):
        from scaleinvariance import FIF_ND
        out = FIF_ND((64, 64), alpha=1.8, C1=0.1, H=0.3, periodic=True,
                     observable_kernel_odd_axes=True)
        assert abs(out.mean()) < 1e-5
        assert abs(out.std() - 1.0) < 1e-5

    def test_2d_odd_with_ls2010_raises(self):
        """odd_axes is restricted to the spectral observable path."""
        from scaleinvariance import FIF_ND
        with pytest.raises(ValueError, match="spectral"):
            FIF_ND((64, 64), alpha=1.8, C1=0.1, H=0.3, periodic=True,
                   kernel_construction_method_observable='LS2010',
                   observable_kernel_odd_axes=True)

    def test_2d_causal_runs_and_is_finite(self):
        from scaleinvariance import FIF_ND
        out = FIF_ND((64, 64), alpha=1.8, C1=0.1, H=0.3, periodic=True,
                     kernel_construction_method_observable='LS2010',
                     causal=True)
        assert np.all(np.isfinite(out))

    def test_2d_per_axis_causal(self):
        from scaleinvariance import FIF_ND
        out = FIF_ND((64, 64), alpha=1.8, C1=0.1, H=0.3, periodic=True,
                     kernel_construction_method_observable='LS2010',
                     causal=(True, False))
        assert np.all(np.isfinite(out))

    def test_2d_causal_axis0_odd_axis1_requires_spectral_raises(self):
        """Mixed causal + odd on different axes still hits the LS2010-only
        path for causal, which is incompatible with odd_axes."""
        from scaleinvariance import FIF_ND
        # LS2010 + odd raises; the user-facing message is the LS2010+odd one
        # because LS2010 is required for causal.
        with pytest.raises(ValueError, match="spectral"):
            FIF_ND((64, 64), alpha=1.8, C1=0.1, H=0.3, periodic=True,
                   kernel_construction_method_observable='LS2010',
                   causal=(True, False),
                   observable_kernel_odd_axes=(False, True))
        # The same combo with spectral observable raises because causal
        # requires LS2010.
        with pytest.raises(ValueError, match="Spectral observable"):
            FIF_ND((64, 64), alpha=1.8, C1=0.1, H=0.3, periodic=True,
                   kernel_construction_method_observable='spectral',
                   causal=(True, False),
                   observable_kernel_odd_axes=(False, True))

    def test_2d_causal_with_spectral_observable_raises(self):
        from scaleinvariance import FIF_ND
        with pytest.raises(ValueError, match="Spectral observable"):
            FIF_ND((64, 64), alpha=1.8, C1=0.1, H=0.3, periodic=True,
                   causal=True)  # default observable is 'spectral'

    def test_2d_same_axis_causal_and_odd_raises(self):
        from scaleinvariance import FIF_ND
        with pytest.raises(ValueError, match="cannot be both causal and odd"):
            FIF_ND((64, 64), alpha=1.8, C1=0.1, H=0.3, periodic=True,
                   kernel_construction_method_observable='LS2010',
                   causal=True, observable_kernel_odd_axes=True)

    def test_C1_zero_with_causal_raises(self):
        from scaleinvariance import FIF_ND
        with pytest.raises(ValueError, match="fBm cannot be causal"):
            FIF_ND((64, 64), alpha=1.8, C1=0, H=0.3, periodic=True, causal=True)

    def test_C1_zero_with_odd_raises(self):
        from scaleinvariance import FIF_ND
        with pytest.raises(ValueError, match="C1=0"):
            FIF_ND((64, 64), alpha=1.8, C1=0, H=0.3, periodic=True,
                   observable_kernel_odd_axes=True)

    def test_H_zero_with_odd_raises(self):
        from scaleinvariance import FIF_ND
        with pytest.raises(ValueError, match="H=0"):
            FIF_ND((64, 64), alpha=1.8, C1=0.1, H=0.0, periodic=True,
                   observable_kernel_odd_axes=True)


class TestOddKernelPSDEquivalence:
    """The odd-axes Fourier multiplier ``i^k · Π sign(f_j)`` has unit
    modulus, so the magnitude spectrum of the output is unchanged.

    Two test layers:
      1. At the ``fractional_integral_spectral`` level: applying the
         operator with vs without ``odd_axes`` to the **same** input must
         produce outputs whose PSDs match bin-by-bin (within float noise),
         except at DC / Nyquist along odd axes (which are zeroed in the odd
         case).
      2. At the ``FIF_1D``/``FIF_ND`` level: the per-output normalization
         differs (unit-mean vs zero-mean/unit-std), so PSDs match only up to
         a single global scalar. The test verifies the *ratio* is
         approximately constant across inertial-range bins.
    """

    # ----- 1. Operator level: bin-by-bin match -----

    @staticmethod
    def _inertial_mask_1d_full(n):
        """1D inertial-range mask on the full FFT grid (signed frequencies)."""
        freqs = np.fft.fftfreq(n, d=1.0)
        return (np.abs(freqs) > 4.0 / n) & (np.abs(freqs) < 1.0 / 8.0)

    def test_operator_1d_psd_match_bin_by_bin(self):
        """fractional_integral_spectral with vs without odd_axes — same input
        → identical bin-by-bin PSDs (inertial range)."""
        from scaleinvariance.simulation.fractional_integration import fractional_integral_spectral
        size = 2048
        H = 0.3
        rng = np.random.default_rng(11)
        flux = np.exp(0.5 * rng.standard_normal(size))
        flux /= flux.mean()
        even = np.asarray(fractional_integral_spectral(flux, H=H))
        odd = np.asarray(fractional_integral_spectral(flux, H=H, odd_axes=True))
        psd_even = np.abs(np.fft.fft(even)) ** 2
        psd_odd = np.abs(np.fft.fft(odd)) ** 2
        mask = self._inertial_mask_1d_full(size)
        np.testing.assert_allclose(psd_odd[mask], psd_even[mask], rtol=1e-4)

    def test_operator_2d_psd_match_bin_by_bin_both_odd(self):
        """2D, both axes odd."""
        from scaleinvariance.simulation.fractional_integration import fractional_integral_spectral
        size = (256, 256)
        H = 0.3
        rng = np.random.default_rng(12)
        flux = np.exp(0.4 * rng.standard_normal(size))
        flux /= flux.mean()
        even = np.asarray(fractional_integral_spectral(flux, H=H))
        odd = np.asarray(fractional_integral_spectral(flux, H=H, odd_axes=True))
        psd_even = np.abs(np.fft.fftn(even)) ** 2
        psd_odd = np.abs(np.fft.fftn(odd)) ** 2
        fx, fy = np.meshgrid(np.fft.fftfreq(size[0]), np.fft.fftfreq(size[1]),
                             indexing='ij')
        fmag = np.sqrt(fx**2 + fy**2)
        # Exclude DC line on each odd axis (and the Nyquist for even-N axes).
        mask = (fmag > 4.0 / max(size)) & (fmag < 1.0 / 8.0)
        mask &= (fx != 0) & (fy != 0)
        if size[0] % 2 == 0:
            mask &= (np.abs(fx - 0.5) > 1e-12) & (np.abs(fx + 0.5) > 1e-12)
        if size[1] % 2 == 0:
            mask &= (np.abs(fy - 0.5) > 1e-12) & (np.abs(fy + 0.5) > 1e-12)
        np.testing.assert_allclose(psd_odd[mask], psd_even[mask], rtol=1e-4)

    def test_operator_2d_psd_match_bin_by_bin_single_axis(self):
        """2D, only axis 0 odd."""
        from scaleinvariance.simulation.fractional_integration import fractional_integral_spectral
        size = (256, 256)
        H = 0.4
        rng = np.random.default_rng(13)
        flux = np.exp(0.4 * rng.standard_normal(size))
        flux /= flux.mean()
        even = np.asarray(fractional_integral_spectral(flux, H=H))
        odd = np.asarray(fractional_integral_spectral(flux, H=H, odd_axes=(True, False)))
        psd_even = np.abs(np.fft.fftn(even)) ** 2
        psd_odd = np.abs(np.fft.fftn(odd)) ** 2
        fx, fy = np.meshgrid(np.fft.fftfreq(size[0]), np.fft.fftfreq(size[1]),
                             indexing='ij')
        fmag = np.sqrt(fx**2 + fy**2)
        mask = (fmag > 4.0 / max(size)) & (fmag < 1.0 / 8.0)
        # Only axis 0 is odd → exclude DC and Nyquist along that axis only.
        mask &= (fx != 0)
        if size[0] % 2 == 0:
            mask &= (np.abs(fx - 0.5) > 1e-12) & (np.abs(fx + 0.5) > 1e-12)
        np.testing.assert_allclose(psd_odd[mask], psd_even[mask], rtol=1e-4)

    # ----- 2. FIF level: PSDs match up to a constant scalar -----

    @staticmethod
    def _constant_ratio_test(psd_even, psd_odd, mask, cv_tol=0.02):
        """Assert that psd_odd[mask] / psd_even[mask] is approximately
        constant across bins (low coefficient of variation)."""
        ratio = psd_odd[mask] / psd_even[mask]
        ratio_cv = ratio.std() / ratio.mean()
        assert ratio_cv < cv_tol, (
            f"Ratio is not approximately constant: mean={ratio.mean():.4f}, "
            f"std={ratio.std():.4f}, cv={ratio_cv:.4f} (tol={cv_tol})"
        )

    def test_fif_1d_psd_ratio_is_constant(self):
        """FIF_1D output PSDs (even vs odd, same noise) differ only by an
        overall scale (mean-normalization vs std-normalization). Bin-wise
        ratio across the inertial range should be near-constant."""
        from scaleinvariance import FIF_1D
        from scaleinvariance.simulation.FIF import extremal_levy
        size = 2048
        alpha, C1, H = 1.8, 0.1, 0.3
        noise = extremal_levy(alpha=alpha, size=size, seed=11)
        even = np.asarray(FIF_1D(size, alpha, C1, H, levy_noise=noise))
        odd = np.asarray(FIF_1D(size, alpha, C1, H, levy_noise=noise,
                                 observable_kernel_odd_axes=True))
        psd_even = np.abs(np.fft.fft(even)) ** 2
        psd_odd = np.abs(np.fft.fft(odd)) ** 2
        mask = self._inertial_mask_1d_full(size)
        self._constant_ratio_test(psd_even, psd_odd, mask, cv_tol=0.02)

    def test_fif_nd_psd_ratio_is_constant_both_odd(self):
        from scaleinvariance import FIF_ND
        from scaleinvariance.simulation.FIF import _extremal_levy_core
        from scaleinvariance import backend as B
        size = (256, 256)
        alpha, C1, H = 1.8, 0.1, 0.3
        np.random.seed(7)
        noise_np = B.to_numpy(_extremal_levy_core(alpha, size=size[0] * size[1])).reshape(size)
        even = np.asarray(FIF_ND(size, alpha, C1, H, periodic=True, levy_noise=noise_np))
        odd = np.asarray(FIF_ND(size, alpha, C1, H, periodic=True, levy_noise=noise_np,
                                observable_kernel_odd_axes=True))
        psd_even = np.abs(np.fft.fftn(even)) ** 2
        psd_odd = np.abs(np.fft.fftn(odd)) ** 2
        fx, fy = np.meshgrid(np.fft.fftfreq(size[0]), np.fft.fftfreq(size[1]),
                             indexing='ij')
        fmag = np.sqrt(fx**2 + fy**2)
        mask = (fmag > 4.0 / max(size)) & (fmag < 1.0 / 8.0)
        mask &= (fx != 0) & (fy != 0)
        if size[0] % 2 == 0:
            mask &= (np.abs(fx - 0.5) > 1e-12) & (np.abs(fx + 0.5) > 1e-12)
        if size[1] % 2 == 0:
            mask &= (np.abs(fy - 0.5) > 1e-12) & (np.abs(fy + 0.5) > 1e-12)
        self._constant_ratio_test(psd_even, psd_odd, mask, cv_tol=0.02)

    def test_fif_nd_psd_ratio_is_constant_single_axis(self):
        from scaleinvariance import FIF_ND
        from scaleinvariance.simulation.FIF import _extremal_levy_core
        from scaleinvariance import backend as B
        size = (256, 256)
        alpha, C1, H = 1.8, 0.1, 0.4
        np.random.seed(13)
        noise_np = B.to_numpy(_extremal_levy_core(alpha, size=size[0] * size[1])).reshape(size)
        even = np.asarray(FIF_ND(size, alpha, C1, H, periodic=True, levy_noise=noise_np))
        odd = np.asarray(FIF_ND(size, alpha, C1, H, periodic=True, levy_noise=noise_np,
                                observable_kernel_odd_axes=(True, False)))
        psd_even = np.abs(np.fft.fftn(even)) ** 2
        psd_odd = np.abs(np.fft.fftn(odd)) ** 2
        fx, fy = np.meshgrid(np.fft.fftfreq(size[0]), np.fft.fftfreq(size[1]),
                             indexing='ij')
        fmag = np.sqrt(fx**2 + fy**2)
        mask = (fmag > 4.0 / max(size)) & (fmag < 1.0 / 8.0)
        mask &= (fx != 0)
        if size[0] % 2 == 0:
            mask &= (np.abs(fx - 0.5) > 1e-12) & (np.abs(fx + 0.5) > 1e-12)
        self._constant_ratio_test(psd_even, psd_odd, mask, cv_tol=0.02)


class TestFIF_1D_OddAndCausalValidation:
    """Validation tests for new FIF_1D parameters."""

    def test_spectral_odd_method_emits_deprecation(self):
        import warnings
        from scaleinvariance import FIF_1D
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            FIF_1D(128, alpha=1.8, C1=0.1, H=0.3,
                   kernel_construction_method_observable='spectral_odd')
        assert any(issubclass(wi.category, DeprecationWarning) for wi in w)

    def test_observable_kernel_odd_axes_zero_mean_output(self):
        from scaleinvariance import FIF_1D
        out = FIF_1D(1024, alpha=1.8, C1=0.1, H=0.3,
                     observable_kernel_odd_axes=True)
        assert abs(out.mean()) < 1e-5
        assert abs(out.std() - 1.0) < 1e-5

    def test_causal_and_odd_same_axis_raises(self):
        from scaleinvariance import FIF_1D
        with pytest.raises(ValueError, match="cannot be both causal and odd"):
            FIF_1D(128, alpha=1.8, C1=0.1, H=0.3,
                   causal=True, observable_kernel_odd_axes=True)

    def test_naive_observable_with_odd_raises(self):
        from scaleinvariance import FIF_1D
        with pytest.raises(ValueError, match="naive"):
            FIF_1D(128, alpha=1.8, C1=0.1, H=0.3,
                   kernel_construction_method_observable='naive',
                   observable_kernel_odd_axes=True)

    def test_H_zero_with_odd_raises(self):
        from scaleinvariance import FIF_1D
        with pytest.raises(ValueError, match="H=0"):
            FIF_1D(128, alpha=1.8, C1=0.1, H=0.0,
                   observable_kernel_odd_axes=True)


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
