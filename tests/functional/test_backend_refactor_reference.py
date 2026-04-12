"""
Test that backend-refactored code produces results matching pre-refactor reference.

Reference data was generated ONCE by generate_backend_refactor_reference.py
using BOTH numpy and torch backends BEFORE the refactor. DO NOT REGENERATE.

Three test classes:
  TestSimulationNumpy  — numpy backend sims vs numpy reference (bit-identical)
  TestSimulationTorch  — torch backend sims vs torch reference (bit-identical on CPU)
  TestAnalysis         — analysis on fixed input, any backend (tolerance varies)

To test a specific backend:
  uv run pytest tests/functional/test_backend_refactor_reference.py -v
  uv run pytest ... -k "Numpy"
  uv run pytest ... -k "Torch"
"""

import numpy as np
import os
import pytest
import warnings

import scaleinvariance as si

try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), 'backend_refactor_reference.npz')
SEED = 42

# Simulation keys (without prefix) and their parameters
SIM_CASES = {
    'fbm_1d_circulant':          lambda: si.fBm_1D_circulant(1024, H=0.5, periodic=True),
    'fbm_1d_circulant_aperiodic': lambda: si.fBm_1D_circulant(1024, H=0.5, periodic=False),
    'fbm_2d_circulant':          lambda: si.fBm_ND_circulant((64, 64), H=0.4, periodic=True),
    'fbm_1d_conv':               lambda: si.fBm_1D(1024, H=0.5, periodic=True),
    'fif_1d':                    lambda: si.FIF_1D(1024, alpha=1.8, C1=0.1, H=0.3, periodic=True),
    'fif_1d_aperiodic':          lambda: si.FIF_1D(1024, alpha=1.8, C1=0.1, H=0.3, periodic=False),
    'fif_1d_alpha2':             lambda: si.FIF_1D(1024, alpha=2.0, C1=0.05, H=0.5, periodic=True),
    'fif_2d':                    lambda: si.FIF_ND((64, 64), alpha=1.8, C1=0.1, H=0.3),
}


@pytest.fixture(scope='module')
def ref():
    if not os.path.exists(REFERENCE_PATH):
        pytest.skip(
            "Reference data not found. Run generate_backend_refactor_reference.py "
            "BEFORE starting the refactor."
        )
    return dict(np.load(REFERENCE_PATH, allow_pickle=False))


def _assert_close(actual, expected, name, rtol, atol):
    """Assert arrays are close, with informative error message."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    # Handle NaN: NaN in same positions is OK
    nan_match = np.isnan(actual) == np.isnan(expected)
    assert np.all(nan_match), (
        f"{name}: NaN positions differ. "
        f"actual NaN count={np.sum(np.isnan(actual))}, "
        f"expected NaN count={np.sum(np.isnan(expected))}"
    )

    finite = np.isfinite(actual) & np.isfinite(expected)
    if not np.any(finite):
        return

    np.testing.assert_allclose(
        actual[finite], expected[finite],
        rtol=rtol, atol=atol,
        err_msg=f"{name}: values differ beyond tolerance (rtol={rtol}, atol={atol})"
    )


# =============================================================================
# Simulation: numpy backend vs numpy reference (bit-identical)
# =============================================================================

class TestSimulationNumpy:

    @pytest.fixture(autouse=True)
    def _setup_backend(self):
        prev = si.get_backend()
        si.set_backend('numpy')
        si.set_numerical_precision('float64')
        yield
        si.set_backend(prev)

    @pytest.mark.parametrize('case_name', SIM_CASES.keys())
    def test_sim(self, ref, case_name):
        key = f'np_sim_{case_name}'
        if key not in ref:
            pytest.skip(f"Reference key {key} not found")
        np.random.seed(SEED)
        result = SIM_CASES[case_name]()
        _assert_close(result, ref[key], key, rtol=0.0, atol=0.0)


# =============================================================================
# Simulation: torch CPU backend vs torch reference (bit-identical)
# =============================================================================

class TestSimulationTorchCPU:

    @pytest.fixture(autouse=True)
    def _setup_backend(self):
        if not _torch_available:
            pytest.skip("torch not available")
        prev = si.get_backend()
        si.set_backend('torch')
        si.set_numerical_precision('float64')
        yield
        si.set_backend(prev)

    @pytest.mark.parametrize('case_name', SIM_CASES.keys())
    def test_sim(self, ref, case_name):
        key = f'torch_sim_{case_name}'
        if key not in ref:
            pytest.skip(f"Reference key {key} not found")
        torch.manual_seed(SEED)
        result = SIM_CASES[case_name]()
        _assert_close(result, ref[key], key, rtol=0.0, atol=0.0)


# =============================================================================
# Simulation: torch GPU backend vs torch reference (near-identical)
# =============================================================================

class TestSimulationTorchGPU:
    """Compare GPU simulation against torch CPU reference.
    Small tolerance for GPU nondeterminism in reductions/FFT."""

    @pytest.fixture(autouse=True)
    def _setup_backend(self):
        if not _torch_available or not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        prev_backend = si.get_backend()
        si.set_backend('torch')
        si.set_numerical_precision('float64')
        # set_device will be available after Phase 1c of the refactor
        if not hasattr(si, 'set_device'):
            pytest.skip("set_device not yet implemented")
        si.set_device('cuda')
        yield
        si.set_device('cpu')
        si.set_backend(prev_backend)

    @pytest.mark.parametrize('case_name', SIM_CASES.keys())
    def test_sim(self, ref, case_name):
        key = f'torch_sim_{case_name}'
        if key not in ref:
            pytest.skip(f"Reference key {key} not found")
        torch.manual_seed(SEED)
        result = SIM_CASES[case_name]()
        # GPU float64 FFT may differ slightly from CPU
        _assert_close(result, ref[key], f'gpu_{key}', rtol=1e-10, atol=1e-12)


# =============================================================================
# Analysis: deterministic on fixed input, tested across backends
# =============================================================================

class TestAnalysisNumpyBackend:
    """Analysis with numpy backend should be bit-identical to reference."""

    @pytest.fixture(autouse=True)
    def _setup(self, ref):
        prev = si.get_backend()
        si.set_backend('numpy')
        si.set_numerical_precision('float64')
        self.signal_1d = ref['np_sim_fbm_1d_circulant']
        self.signal_2d = ref['np_sim_fbm_2d_circulant']
        self.fif_signal = ref['np_sim_fif_1d']
        self.rtol = 0.0
        self.atol = 0.0
        yield
        si.set_backend(prev)

    def test_structure_function_order1(self, ref):
        lags, values = si.structure_function_analysis(self.signal_1d, order=1, axis=0)
        _assert_close(lags, ref['analysis_sf_lags'], 'sf_lags', self.rtol, self.atol)
        _assert_close(values, ref['analysis_sf_values'], 'sf_values', self.rtol, self.atol)

    def test_structure_function_order2(self, ref):
        lags, values = si.structure_function_analysis(self.signal_1d, order=2, axis=0)
        _assert_close(lags, ref['analysis_sf_lags_o2'], 'sf_lags_o2', self.rtol, self.atol)
        _assert_close(values, ref['analysis_sf_values_o2'], 'sf_values_o2', self.rtol, self.atol)

    def test_structure_function_hurst(self, ref):
        H, unc = si.structure_function_hurst(self.signal_1d, axis=0)
        _assert_close(H, ref['analysis_sf_H'], 'sf_H', self.rtol, self.atol)
        _assert_close(unc, ref['analysis_sf_unc'], 'sf_unc', self.rtol, self.atol)

    def test_haar_fluctuation_order1(self, ref):
        lags, values = si.haar_fluctuation_analysis(self.signal_1d, order=1, axis=0)
        _assert_close(lags, ref['analysis_haar_lags'], 'haar_lags', self.rtol, self.atol)
        _assert_close(values, ref['analysis_haar_values'], 'haar_values', self.rtol, self.atol)

    def test_haar_fluctuation_order2(self, ref):
        lags, values = si.haar_fluctuation_analysis(self.signal_1d, order=2, axis=0)
        _assert_close(lags, ref['analysis_haar_lags_o2'], 'haar_lags_o2', self.rtol, self.atol)
        _assert_close(values, ref['analysis_haar_values_o2'], 'haar_values_o2', self.rtol, self.atol)

    def test_haar_fluctuation_hurst(self, ref):
        H, unc = si.haar_fluctuation_hurst(self.signal_1d, axis=0)
        _assert_close(H, ref['analysis_haar_H'], 'haar_H', self.rtol, self.atol)
        _assert_close(unc, ref['analysis_haar_unc'], 'haar_unc', self.rtol, self.atol)

    def test_spectral_analysis(self, ref):
        freq, psd = si.spectral_analysis(self.signal_1d, axis=0)
        _assert_close(freq, ref['analysis_spec_freq'], 'spec_freq', self.rtol, self.atol)
        _assert_close(psd, ref['analysis_spec_psd'], 'spec_psd', self.rtol, self.atol)

    def test_spectral_hurst(self, ref):
        H, unc = si.spectral_hurst(self.signal_1d, axis=0)
        _assert_close(H, ref['analysis_spec_H'], 'spec_H', self.rtol, self.atol)
        _assert_close(unc, ref['analysis_spec_unc'], 'spec_unc', self.rtol, self.atol)

    def test_compute_K_q_function(self, ref):
        q_vals = np.arange(0.1, 2.51, 0.1)
        kq_q, kq_K, kq_H, kq_C1, kq_alpha = si.compute_K_q_function(
            self.fif_signal, q_values=q_vals, scaling_method='structure_function',
            hurst_fit_method='fixed'
        )
        _assert_close(kq_q, ref['analysis_kq_q'], 'kq_q', self.rtol, self.atol)
        _assert_close(kq_K, ref['analysis_kq_K'], 'kq_K', self.rtol, self.atol)
        _assert_close(kq_H, ref['analysis_kq_H'], 'kq_H', self.rtol, self.atol)
        _assert_close(kq_C1, ref['analysis_kq_C1'], 'kq_C1', self.rtol, self.atol)
        _assert_close(kq_alpha, ref['analysis_kq_alpha'], 'kq_alpha', self.rtol, self.atol)

    def test_two_point_intermittency(self, ref):
        c1, unc = si.two_point_intermittency_exponent(self.fif_signal)
        _assert_close(c1, ref['analysis_c1_val'], 'c1_val', self.rtol, self.atol)
        _assert_close(unc, ref['analysis_c1_unc'], 'c1_unc', self.rtol, self.atol)

    def test_structure_function_2d(self, ref):
        lags, values = si.structure_function_analysis(self.signal_2d, order=1, axis=0)
        _assert_close(lags, ref['analysis_sf2d_lags'], 'sf2d_lags', self.rtol, self.atol)
        _assert_close(values, ref['analysis_sf2d_values'], 'sf2d_values', self.rtol, self.atol)

    def test_haar_fluctuation_2d(self, ref):
        lags, values = si.haar_fluctuation_analysis(self.signal_2d, order=1, axis=0)
        _assert_close(lags, ref['analysis_haar2d_lags'], 'haar2d_lags', self.rtol, self.atol)
        _assert_close(values, ref['analysis_haar2d_values'], 'haar2d_values', self.rtol, self.atol)

    def test_spectral_2d(self, ref):
        freq, psd = si.spectral_analysis(self.signal_2d, axis=0)
        _assert_close(freq, ref['analysis_spec2d_freq'], 'spec2d_freq', self.rtol, self.atol)
        _assert_close(psd, ref['analysis_spec2d_psd'], 'spec2d_psd', self.rtol, self.atol)

    def test_haar_nan_path(self, ref):
        test_nan = self.signal_1d.copy()
        test_nan[100:110] = np.nan
        lags, values = si.haar_fluctuation_analysis(test_nan, order=1, axis=0, nan_behavior='ignore')
        _assert_close(lags, ref['analysis_haar_nan_lags'], 'haar_nan_lags', self.rtol, self.atol)
        _assert_close(values, ref['analysis_haar_nan_values'], 'haar_nan_values', self.rtol, self.atol)


class TestAnalysisTorchBackend:
    """Analysis with torch backend on same input data.
    Torch float64 FFT/reductions may differ at ~1e-14 level."""

    @pytest.fixture(autouse=True)
    def _setup(self, ref):
        if not _torch_available:
            pytest.skip("torch not available")
        prev = si.get_backend()
        si.set_backend('torch')
        si.set_numerical_precision('float64')
        self.signal_1d = ref['np_sim_fbm_1d_circulant']
        self.signal_2d = ref['np_sim_fbm_2d_circulant']
        self.fif_signal = ref['np_sim_fif_1d']
        self.rtol = 1e-12
        self.atol = 1e-14
        yield
        si.set_backend(prev)

    def test_structure_function_order1(self, ref):
        lags, values = si.structure_function_analysis(self.signal_1d, order=1, axis=0)
        _assert_close(lags, ref['analysis_sf_lags'], 'sf_lags', self.rtol, self.atol)
        _assert_close(values, ref['analysis_sf_values'], 'sf_values', self.rtol, self.atol)

    def test_structure_function_order2(self, ref):
        lags, values = si.structure_function_analysis(self.signal_1d, order=2, axis=0)
        _assert_close(lags, ref['analysis_sf_lags_o2'], 'sf_lags_o2', self.rtol, self.atol)
        _assert_close(values, ref['analysis_sf_values_o2'], 'sf_values_o2', self.rtol, self.atol)

    def test_structure_function_hurst(self, ref):
        H, unc = si.structure_function_hurst(self.signal_1d, axis=0)
        _assert_close(H, ref['analysis_sf_H'], 'sf_H', self.rtol, self.atol)
        _assert_close(unc, ref['analysis_sf_unc'], 'sf_unc', self.rtol, self.atol)

    def test_haar_fluctuation_order1(self, ref):
        lags, values = si.haar_fluctuation_analysis(self.signal_1d, order=1, axis=0)
        _assert_close(lags, ref['analysis_haar_lags'], 'haar_lags', self.rtol, self.atol)
        _assert_close(values, ref['analysis_haar_values'], 'haar_values', self.rtol, self.atol)

    def test_haar_fluctuation_order2(self, ref):
        lags, values = si.haar_fluctuation_analysis(self.signal_1d, order=2, axis=0)
        _assert_close(lags, ref['analysis_haar_lags_o2'], 'haar_lags_o2', self.rtol, self.atol)
        _assert_close(values, ref['analysis_haar_values_o2'], 'haar_values_o2', self.rtol, self.atol)

    def test_haar_fluctuation_hurst(self, ref):
        H, unc = si.haar_fluctuation_hurst(self.signal_1d, axis=0)
        _assert_close(H, ref['analysis_haar_H'], 'haar_H', self.rtol, self.atol)
        _assert_close(unc, ref['analysis_haar_unc'], 'haar_unc', self.rtol, self.atol)

    def test_spectral_analysis(self, ref):
        freq, psd = si.spectral_analysis(self.signal_1d, axis=0)
        _assert_close(freq, ref['analysis_spec_freq'], 'spec_freq', self.rtol, self.atol)
        _assert_close(psd, ref['analysis_spec_psd'], 'spec_psd', self.rtol, self.atol)

    def test_spectral_hurst(self, ref):
        H, unc = si.spectral_hurst(self.signal_1d, axis=0)
        _assert_close(H, ref['analysis_spec_H'], 'spec_H', self.rtol, self.atol)
        _assert_close(unc, ref['analysis_spec_unc'], 'spec_unc', self.rtol, self.atol)

    def test_compute_K_q_function(self, ref):
        q_vals = np.arange(0.1, 2.51, 0.1)
        kq_q, kq_K, kq_H, kq_C1, kq_alpha = si.compute_K_q_function(
            self.fif_signal, q_values=q_vals, scaling_method='structure_function',
            hurst_fit_method='fixed'
        )
        _assert_close(kq_q, ref['analysis_kq_q'], 'kq_q', self.rtol, self.atol)
        _assert_close(kq_K, ref['analysis_kq_K'], 'kq_K', self.rtol, self.atol)
        _assert_close(kq_H, ref['analysis_kq_H'], 'kq_H', self.rtol, self.atol)
        _assert_close(kq_C1, ref['analysis_kq_C1'], 'kq_C1', self.rtol, self.atol)
        _assert_close(kq_alpha, ref['analysis_kq_alpha'], 'kq_alpha', self.rtol, self.atol)

    def test_two_point_intermittency(self, ref):
        c1, unc = si.two_point_intermittency_exponent(self.fif_signal)
        _assert_close(c1, ref['analysis_c1_val'], 'c1_val', self.rtol, self.atol)
        _assert_close(unc, ref['analysis_c1_unc'], 'c1_unc', self.rtol, self.atol)

    def test_structure_function_2d(self, ref):
        lags, values = si.structure_function_analysis(self.signal_2d, order=1, axis=0)
        _assert_close(lags, ref['analysis_sf2d_lags'], 'sf2d_lags', self.rtol, self.atol)
        _assert_close(values, ref['analysis_sf2d_values'], 'sf2d_values', self.rtol, self.atol)

    def test_haar_fluctuation_2d(self, ref):
        lags, values = si.haar_fluctuation_analysis(self.signal_2d, order=1, axis=0)
        _assert_close(lags, ref['analysis_haar2d_lags'], 'haar2d_lags', self.rtol, self.atol)
        _assert_close(values, ref['analysis_haar2d_values'], 'haar2d_values', self.rtol, self.atol)

    def test_spectral_2d(self, ref):
        freq, psd = si.spectral_analysis(self.signal_2d, axis=0)
        _assert_close(freq, ref['analysis_spec2d_freq'], 'spec2d_freq', self.rtol, self.atol)
        _assert_close(psd, ref['analysis_spec2d_psd'], 'spec2d_psd', self.rtol, self.atol)

    def test_haar_nan_path(self, ref):
        test_nan = self.signal_1d.copy()
        test_nan[100:110] = np.nan
        lags, values = si.haar_fluctuation_analysis(test_nan, order=1, axis=0, nan_behavior='ignore')
        _assert_close(lags, ref['analysis_haar_nan_lags'], 'haar_nan_lags', self.rtol, self.atol)
        _assert_close(values, ref['analysis_haar_nan_values'], 'haar_nan_values', self.rtol, self.atol)


class TestAnalysisTorchGPU:
    """Analysis with torch GPU backend on same input data.
    Wider tolerance for GPU nondeterminism."""

    @pytest.fixture(autouse=True)
    def _setup(self, ref):
        if not _torch_available or not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if not hasattr(si, 'set_device'):
            pytest.skip("set_device not yet implemented")
        prev = si.get_backend()
        si.set_backend('torch')
        si.set_numerical_precision('float64')
        si.set_device('cuda')
        self.signal_1d = ref['np_sim_fbm_1d_circulant']
        self.signal_2d = ref['np_sim_fbm_2d_circulant']
        self.fif_signal = ref['np_sim_fif_1d']
        self.rtol = 1e-10
        self.atol = 1e-12
        yield
        si.set_device('cpu')
        si.set_backend(prev)

    def test_structure_function_order1(self, ref):
        lags, values = si.structure_function_analysis(self.signal_1d, order=1, axis=0)
        _assert_close(lags, ref['analysis_sf_lags'], 'sf_lags', self.rtol, self.atol)
        _assert_close(values, ref['analysis_sf_values'], 'sf_values', self.rtol, self.atol)

    def test_haar_fluctuation_order1(self, ref):
        lags, values = si.haar_fluctuation_analysis(self.signal_1d, order=1, axis=0)
        _assert_close(lags, ref['analysis_haar_lags'], 'haar_lags', self.rtol, self.atol)
        _assert_close(values, ref['analysis_haar_values'], 'haar_values', self.rtol, self.atol)

    def test_spectral_analysis(self, ref):
        freq, psd = si.spectral_analysis(self.signal_1d, axis=0)
        _assert_close(freq, ref['analysis_spec_freq'], 'spec_freq', self.rtol, self.atol)
        _assert_close(psd, ref['analysis_spec_psd'], 'spec_psd', self.rtol, self.atol)

    def test_spectral_hurst(self, ref):
        H, unc = si.spectral_hurst(self.signal_1d, axis=0)
        _assert_close(H, ref['analysis_spec_H'], 'spec_H', self.rtol, self.atol)
        _assert_close(unc, ref['analysis_spec_unc'], 'spec_unc', self.rtol, self.atol)
