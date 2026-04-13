"""
Tests for the set_numerical_precision / get_numerical_precision API and the
overflow guards added in v0.10.0.

Parametrized across (backend, precision) so every simulation method is
exercised at numpy-float32, numpy-float64, torch-float32, and torch-float64.
"""
import warnings

import numpy as np
import pytest

import scaleinvariance
from scaleinvariance import (
    FIF_1D, FIF_ND,
    fBm_1D, fBm_1D_circulant, fBm_ND_circulant,
    get_backend, set_backend,
    get_numerical_precision, set_numerical_precision,
    to_numpy,
)
from scaleinvariance.simulation.FIF import extremal_levy


BACKEND_PRECISION = [
    ('numpy', 'float64'),
    ('numpy', 'float32'),
    ('torch', 'float64'),
    ('torch', 'float32'),
]


def _real_np_dtype(precision):
    return np.float32 if precision == 'float32' else np.float64


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------

class TestPrecisionAPI:

    def test_default_is_float64(self):
        assert get_numerical_precision() == 'float64'

    def test_round_trip(self):
        original = get_numerical_precision()
        try:
            set_numerical_precision('float32')
            assert get_numerical_precision() == 'float32'
            set_numerical_precision('float64')
            assert get_numerical_precision() == 'float64'
        finally:
            set_numerical_precision(original)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="float32|float64"):
            set_numerical_precision('float16')

    def test_module_level_export(self):
        # The two names must be available directly on the package.
        assert scaleinvariance.set_numerical_precision is set_numerical_precision
        assert scaleinvariance.get_numerical_precision is get_numerical_precision


# ---------------------------------------------------------------------------
# Output dtype tracking
# ---------------------------------------------------------------------------

@pytest.fixture(params=BACKEND_PRECISION, autouse=False,
                ids=lambda p: f"{p[0]}-{p[1]}")
def precision_env(request):
    """Save and restore (backend, precision) for each test."""
    backend, precision = request.param

    if backend == 'torch':
        pytest.importorskip('torch')

    original_backend = get_backend()
    original_precision = get_numerical_precision()
    try:
        set_backend(backend)
        set_numerical_precision(precision)
        yield backend, precision
    finally:
        set_numerical_precision(original_precision)
        set_backend(original_backend)


class TestOutputDtype:

    def test_fBm_1D_circulant_dtype(self, precision_env):
        backend, precision = precision_env
        out = fBm_1D_circulant(128, H=0.5, periodic=True)
        assert out.dtype == _real_np_dtype(precision), \
            f"{backend}+{precision}: got {out.dtype}"

    def test_fBm_ND_circulant_dtype(self, precision_env):
        backend, precision = precision_env
        out = fBm_ND_circulant((64, 64), H=0.5, periodic=True)
        assert out.dtype == _real_np_dtype(precision)

    def test_fBm_1D_dtype(self, precision_env):
        backend, precision = precision_env
        out = fBm_1D(128, H=0.7, causal=False, periodic=True)
        assert out.dtype == _real_np_dtype(precision)

    def test_FIF_1D_dtype(self, precision_env):
        backend, precision = precision_env
        out = FIF_1D(128, alpha=1.8, C1=0.1, H=0.3)
        assert out.dtype == _real_np_dtype(precision)

    def test_FIF_ND_dtype(self, precision_env):
        backend, precision = precision_env
        out = FIF_ND((64, 64), alpha=1.8, C1=0.1, H=0.3, periodic=True)
        assert out.dtype == _real_np_dtype(precision)

    def test_extremal_levy_dtype(self, precision_env):
        backend, precision = precision_env
        out = extremal_levy(alpha=1.8, size=256)
        assert out.dtype == _real_np_dtype(precision)


# ---------------------------------------------------------------------------
# Float32 numerical equivalence with float64 (loose tolerance)
# ---------------------------------------------------------------------------

class TestFloat32MatchesFloat64:
    """At a seeded RNG state, float32 output should be close to float64 cast
    down. We use loose tolerances because the random draws themselves diverge
    at float32 precision once any transcendentals (``cos``/``sin``/``exp``)
    are applied to a float32 state.
    """

    def _run_both(self, fn):
        original_backend = get_backend()
        original_precision = get_numerical_precision()
        try:
            set_backend('numpy')
            set_numerical_precision('float64')
            np.random.seed(7)
            out64 = fn()
            set_numerical_precision('float32')
            np.random.seed(7)
            out32 = fn()
            return out64, out32
        finally:
            set_numerical_precision(original_precision)
            set_backend(original_backend)

    def test_fBm_1D_circulant_close(self):
        out64, out32 = self._run_both(lambda: fBm_1D_circulant(1024, H=0.4))
        assert out32.dtype == np.float32 and out64.dtype == np.float64
        np.testing.assert_allclose(out32.astype(np.float64), out64,
                                   rtol=1e-3, atol=1e-3)

    def test_fBm_ND_circulant_close(self):
        out64, out32 = self._run_both(lambda: fBm_ND_circulant((64, 64), H=0.4))
        np.testing.assert_allclose(out32.astype(np.float64), out64,
                                   rtol=1e-3, atol=1e-3)

    def test_FIF_1D_close(self):
        out64, out32 = self._run_both(
            lambda: FIF_1D(1024, alpha=1.8, C1=0.05, H=0.3)
        )
        # FIF is multiplicative cascade; float32 can drift more than fBm.
        # Compare statistics, not pointwise values.
        assert abs(float(np.mean(out32)) - float(np.mean(out64))) < 0.05
        assert abs(float(np.std(out32)) - float(np.std(out64))) < 0.1


# ---------------------------------------------------------------------------
# Overflow guard: exp clip warns and returns finite output
# ---------------------------------------------------------------------------

class TestExpClipGuard:

    def test_clip_helper_float32_warns_and_is_finite(self):
        """Directly hit the guard with an array guaranteed to exceed the
        float32 exp range."""
        from scaleinvariance.simulation.FIF import _clip_and_exp_flux
        original_precision = get_numerical_precision()
        try:
            set_numerical_precision('float32')
            scaled = np.array([0.0, 50.0, 100.0, 200.0, -500.0],
                              dtype=np.float32)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                flux = _clip_and_exp_flux(scaled, "test_case")
            flux = to_numpy(flux)
            assert np.all(np.isfinite(flux))
            assert any(
                issubclass(w.category, RuntimeWarning)
                and 'saturated' in str(w.message)
                for w in caught
            ), "expected RuntimeWarning about flux saturation"
        finally:
            set_numerical_precision(original_precision)

    def test_clip_helper_float64_does_not_engage_for_moderate(self):
        """At float64 with moderate inputs, the guard must not engage."""
        from scaleinvariance.simulation.FIF import _clip_and_exp_flux
        original_precision = get_numerical_precision()
        try:
            set_numerical_precision('float64')
            scaled = np.array([0.0, 5.0, -10.0, 20.0], dtype=np.float64)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                flux = _clip_and_exp_flux(scaled, "test_case")
            flux = to_numpy(flux)
            assert np.all(np.isfinite(flux))
            # exp of moderate inputs never saturates at float64.
            assert not any(
                issubclass(w.category, RuntimeWarning)
                and 'saturated' in str(w.message)
                for w in caught
            )
            np.testing.assert_allclose(flux, np.exp(scaled), rtol=1e-15)
        finally:
            set_numerical_precision(original_precision)

    def test_FIF_1D_moderate_C1_does_not_warn(self):
        """At default float64 and moderate parameters, the guard must not
        engage (otherwise it would change default-case outputs)."""
        original_precision = get_numerical_precision()
        try:
            set_numerical_precision('float64')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                out = FIF_1D(1024, alpha=1.8, C1=0.1, H=0.3)
            assert np.all(np.isfinite(out))
            assert not any(
                issubclass(w.category, RuntimeWarning)
                and 'saturated' in str(w.message)
                for w in caught
            )
        finally:
            set_numerical_precision(original_precision)


# ---------------------------------------------------------------------------
# scale_metric validation in FIF_ND
# ---------------------------------------------------------------------------

class TestScaleMetricValidation:

    def test_nonpositive_scale_metric_raises(self):
        size = (32, 32)
        sim_size = (64, 64)
        bad = np.ones(sim_size)
        bad[0, 0] = 0.0  # zero triggers distance**negative → inf
        with pytest.raises(ValueError, match="finite and strictly positive"):
            FIF_ND(size, alpha=1.8, C1=0.1, H=0.3, periodic=False,
                   scale_metric=bad, scale_metric_dim=2.0,
                   kernel_construction_method_observable='LS2010')

    def test_nan_scale_metric_raises(self):
        size = (32, 32)
        sim_size = (64, 64)
        bad = np.ones(sim_size)
        bad[5, 5] = np.nan
        with pytest.raises(ValueError, match="finite and strictly positive"):
            FIF_ND(size, alpha=1.8, C1=0.1, H=0.3, periodic=False,
                   scale_metric=bad, scale_metric_dim=2.0,
                   kernel_construction_method_observable='LS2010')
