"""
Basic functionality tests for scaleinvariance package.

Tests all simulation methods and analysis functions with small arrays
to ensure they execute without errors and produce valid output.
"""

import pytest
import numpy as np
from scaleinvariance import (
    fBm_1D_circulant, fBm_ND_circulant, fBm_1D,
    FIF_1D, FIF_ND,
    structure_function_hurst, haar_fluctuation_hurst, spectral_hurst,
    structure_function_analysis, haar_fluctuation_analysis, spectral_analysis,
    set_backend, get_backend
)


class TestfBmSimulations:
    """Test fBm simulation methods."""

    @pytest.mark.parametrize("H", [-0.3, 0.1, 0.3, 0.5, 0.7, 0.9])
    def test_fBm_1D_circulant(self, H):
        """Test 1D circulant fBm with various H values."""
        size = 128
        result = fBm_1D_circulant(size, H, periodic=True)

        assert result.shape == (size,), f"Expected shape ({size},), got {result.shape}"
        assert not np.any(np.isnan(result)), "Output contains NaN values"
        assert not np.any(np.isinf(result)), "Output contains infinite values"
        assert result.dtype == np.float64, f"Expected dtype float64, got {result.dtype}"

    @pytest.mark.parametrize("H", [0.1, 0.3, 0.5, 0.7, 0.9])
    @pytest.mark.parametrize("periodic", [True, False])
    def test_fBm_1D_circulant_periodic(self, H, periodic):
        """Test 1D circulant fBm with periodic/non-periodic modes."""
        size = 128
        result = fBm_1D_circulant(size, H, periodic=periodic)

        assert result.shape == (size,), f"Expected shape ({size},), got {result.shape}"
        assert not np.any(np.isnan(result)), "Output contains NaN values"

    @pytest.mark.parametrize("H", [0.1, 0.3, 0.5, 0.7, 0.9])
    @pytest.mark.parametrize("ndim", [2, 3])
    def test_fBm_ND_circulant(self, H, ndim):
        """Test N-D circulant fBm with various H values and dimensions."""
        if ndim == 2:
            size = (128, 128)
        else:  # 3D
            size = (32, 32, 32)  # Smaller for 3D to keep test fast

        result = fBm_ND_circulant(size, H, periodic=True)

        assert result.shape == size, f"Expected shape {size}, got {result.shape}"
        assert not np.any(np.isnan(result)), "Output contains NaN values"
        assert not np.any(np.isinf(result)), "Output contains infinite values"
        assert result.dtype == np.float64, f"Expected dtype float64, got {result.dtype}"

    @pytest.mark.parametrize("H", [-0.3, 0.0, 0.3, 0.7, 1.0, 1.3])
    @pytest.mark.parametrize("causal", [True, False])
    def test_fBm_1D(self, H, causal):
        """Test 1D fBm (fractional integration) with various H and causality."""
        size = 128
        result = fBm_1D(size, H, causal=causal, periodic=True)

        assert result.shape == (size,), f"Expected shape ({size},), got {result.shape}"
        assert not np.any(np.isnan(result)), "Output contains NaN values"
        assert not np.any(np.isinf(result)), "Output contains infinite values"
        assert result.dtype == np.float64, f"Expected dtype float64, got {result.dtype}"


class TestFIFSimulations:
    """Test FIF (multifractal) simulation methods."""

    @pytest.mark.parametrize("alpha,C1,H", [
        (1.5, 0.05, 0.3),
        (1.8, 0.1, 0.5),
        (1.9, 0.15, 0.7),
        (1.5, 0.2, 0.0),   # H=0 case
        (1.8, 0.1, -0.5),  # Negative H
    ])
    @pytest.mark.parametrize("causal", [True, False])
    def test_FIF_1D_basic(self, alpha, C1, H, causal):
        """Test 1D FIF with various parameter combinations."""
        size = 128
        result = FIF_1D(size, alpha, C1, H, causal=causal, periodic=True)

        assert result.shape == (size,), f"Expected shape ({size},), got {result.shape}"
        assert not np.any(np.isnan(result)), "Output contains NaN values"
        assert not np.any(np.isinf(result)), "Output contains infinite values"
        assert result.dtype == np.float64, f"Expected dtype float64, got {result.dtype}"

    @pytest.mark.parametrize("periodic", [True, False])
    def test_FIF_1D_periodic(self, periodic):
        """Test 1D FIF with periodic/non-periodic modes."""
        size = 128
        result = FIF_1D(size, alpha=1.8, C1=0.1, H=0.3, causal=True, periodic=periodic)

        assert result.shape == (size,), f"Expected shape ({size},), got {result.shape}"
        assert not np.any(np.isnan(result)), "Output contains NaN values"

    def test_FIF_1D_C1_zero_routes_to_fBm(self):
        """Test that C1=0 routes to fBm (requires causal=False)."""
        size = 128
        result = FIF_1D(size, alpha=1.8, C1=0, H=0.3, causal=False, periodic=True)

        assert result.shape == (size,), f"Expected shape ({size},), got {result.shape}"
        assert not np.any(np.isnan(result)), "Output contains NaN values"

    @pytest.mark.parametrize("alpha,C1,H", [
        (1.5, 0.05, 0.3),
        (1.8, 0.1, 0.5),
        (1.9, 0.15, 0.7),
        (1.5, 0.2, 0.0),   # H=0 case
    ])
    @pytest.mark.parametrize("ndim", [2, 3])
    def test_FIF_ND_basic(self, alpha, C1, H, ndim):
        """Test N-D FIF with various parameter combinations."""
        if ndim == 2:
            size = (128, 128)
        else:  # 3D
            size = (32, 32, 32)  # Smaller for 3D to keep test fast

        result = FIF_ND(size, alpha, C1, H, periodic=False)

        assert result.shape == size, f"Expected shape {size}, got {result.shape}"
        assert not np.any(np.isnan(result)), "Output contains NaN values"
        assert not np.any(np.isinf(result)), "Output contains infinite values"
        assert result.dtype == np.float64, f"Expected dtype float64, got {result.dtype}"

    def test_FIF_ND_C1_zero_routes_to_fBm(self):
        """Test that C1=0 routes to fBm for N-D case."""
        size = (128, 128)
        result = FIF_ND(size, alpha=1.8, C1=0, H=0.3, periodic=False)

        assert result.shape == size, f"Expected shape {size}, got {result.shape}"
        assert not np.any(np.isnan(result)), "Output contains NaN values"

    @pytest.mark.parametrize("periodic", [
        True,
        False,
        (True, False),   # Per-axis periodicity
        (False, True),
    ])
    def test_FIF_ND_periodic(self, periodic):
        """Test N-D FIF with various periodicity modes."""
        size = (128, 128)
        result = FIF_ND(size, alpha=1.8, C1=0.1, H=0.3, periodic=periodic)

        assert result.shape == size, f"Expected shape {size}, got {result.shape}"
        assert not np.any(np.isnan(result)), "Output contains NaN values"


class TestAnalysisFunctions:
    """Test analysis methods on simulated data."""

    @pytest.fixture
    def test_data_1d(self):
        """Generate 1D test data for analysis."""
        np.random.seed(42)
        return fBm_1D_circulant(512, H=0.7, periodic=True)

    @pytest.fixture
    def test_data_2d(self):
        """Generate 2D test data for analysis."""
        np.random.seed(42)
        return fBm_ND_circulant((256, 256), H=0.6, periodic=True)

    def test_structure_function_analysis(self, test_data_1d):
        """Test structure function analysis."""
        lags, sf_values = structure_function_analysis(test_data_1d, order=2, axis=0)

        assert len(lags) > 0, "No lags returned"
        assert len(sf_values) == len(lags), "Lags and values length mismatch"
        assert not np.all(np.isnan(sf_values)), "All SF values are NaN"
        assert np.issubdtype(lags.dtype, np.integer), f"Expected lags dtype integer, got {lags.dtype}"
        assert sf_values.dtype == np.float64, f"Expected values dtype float64, got {sf_values.dtype}"

    def test_haar_fluctuation_analysis(self, test_data_1d):
        """Test Haar fluctuation analysis."""
        lags, haar_values = haar_fluctuation_analysis(test_data_1d, order=1, axis=0)

        assert len(lags) > 0, "No lags returned"
        assert len(haar_values) == len(lags), "Lags and values length mismatch"
        assert not np.all(np.isnan(haar_values)), "All Haar values are NaN"
        assert np.issubdtype(lags.dtype, np.integer), f"Expected lags dtype integer, got {lags.dtype}"
        assert haar_values.dtype == np.float64, f"Expected values dtype float64, got {haar_values.dtype}"

    def test_spectral_analysis(self, test_data_1d):
        """Test spectral analysis."""
        freqs, psd = spectral_analysis(test_data_1d, nbins=30, axis=0)

        assert len(freqs) > 0, "No frequencies returned"
        assert len(psd) == len(freqs), "Frequencies and PSD length mismatch"
        assert not np.all(np.isnan(psd)), "All PSD values are NaN"
        assert freqs.dtype == np.float64, f"Expected freqs dtype float64, got {freqs.dtype}"
        assert psd.dtype == np.float64, f"Expected psd dtype float64, got {psd.dtype}"

    def test_structure_function_hurst(self, test_data_1d):
        """Test Hurst estimation via structure function."""
        H_est, H_err = structure_function_hurst(test_data_1d, axis=0)

        assert not np.isnan(H_est), "Estimated H is NaN"
        assert not np.isnan(H_err), "H uncertainty is NaN"
        assert H_err > 0, "H uncertainty should be positive"
        assert 0 < H_est < 1.5, f"H estimate {H_est} outside reasonable range"
        assert isinstance(H_est, (float, np.floating)), f"H_est should be float, got {type(H_est)}"

    def test_haar_fluctuation_hurst(self, test_data_1d):
        """Test Hurst estimation via Haar fluctuation."""
        H_est, H_err = haar_fluctuation_hurst(test_data_1d, axis=0)

        assert not np.isnan(H_est), "Estimated H is NaN"
        assert not np.isnan(H_err), "H uncertainty is NaN"
        assert H_err > 0, "H uncertainty should be positive"
        assert 0 < H_est < 1.5, f"H estimate {H_est} outside reasonable range"
        assert isinstance(H_est, (float, np.floating)), f"H_est should be float, got {type(H_est)}"

    def test_spectral_hurst(self, test_data_1d):
        """Test Hurst estimation via spectral method."""
        H_est, H_err = spectral_hurst(test_data_1d, axis=0)

        assert not np.isnan(H_est), "Estimated H is NaN"
        assert not np.isnan(H_err), "H uncertainty is NaN"
        assert H_err > 0, "H uncertainty should be positive"
        assert 0 < H_est < 1.5, f"H estimate {H_est} outside reasonable range"
        assert isinstance(H_est, (float, np.floating)), f"H_est should be float, got {type(H_est)}"

    def test_2d_analysis(self, test_data_2d):
        """Test analysis functions work on 2D data."""
        # Test along first axis
        lags_sf, sf_values = structure_function_analysis(test_data_2d, order=2, axis=0)
        assert len(lags_sf) > 0 and not np.all(np.isnan(sf_values))

        lags_haar, haar_values = haar_fluctuation_analysis(test_data_2d, order=1, axis=0)
        assert len(lags_haar) > 0 and not np.all(np.isnan(haar_values))

        freqs, psd = spectral_analysis(test_data_2d, nbins=30, axis=0)
        assert len(freqs) > 0 and not np.all(np.isnan(psd))

        # Test Hurst estimation on 2D data
        H_sf, _ = structure_function_hurst(test_data_2d, axis=0)
        assert not np.isnan(H_sf) and 0 < H_sf < 1.5


class TestBackendConsistency:
    """Test that backend operations maintain float64 dtype."""

    @pytest.fixture(params=['numpy', 'torch'], autouse=True)
    def setup_backend(self, request):
        """Setup and teardown for each backend test."""
        original_backend = get_backend()
        backend = request.param

        # Skip torch tests if torch is not available
        if backend == 'torch':
            try:
                import torch
            except ImportError:
                pytest.skip("PyTorch not installed")

        set_backend(backend)
        yield backend
        set_backend(original_backend)

    def test_fif_1d_dtype_consistency(self):
        """Ensure FIF_1D always returns float64."""
        result = FIF_1D(128, alpha=1.8, C1=0.1, H=0.3)
        assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"

    def test_fif_nd_dtype_consistency(self):
        """Ensure FIF_ND always returns float64."""
        result = FIF_ND((64, 64), alpha=1.8, C1=0.1, H=0.3)
        assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"

    def test_fbm_1d_dtype_consistency(self):
        """Ensure fBm_1D always returns float64."""
        result = fBm_1D(128, H=0.7)
        assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"

    def test_fbm_nd_dtype_consistency(self):
        """Ensure fBm_ND always returns float64."""
        result = fBm_ND_circulant((64, 64), H=0.7)
        assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"


if __name__ == "__main__":
    # Allow running directly with: python test_basic_functionality.py
    pytest.main([__file__, "-v"])
