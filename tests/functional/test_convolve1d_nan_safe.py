"""Tests for `backend.convolve1d(nan_safe=True)`.

Compares the FFT-based NaN-safe path (used on the torch backend) to the
scipy direct reference (still used on the numpy backend) across a range
of shapes, axes, precisions, and NaN densities.
"""

import numpy as np
import pytest
from scipy.signal import convolve as scipy_convolve

from scaleinvariance import backend as B


@pytest.fixture(autouse=True)
def _restore_backend_state():
    """Snapshot and restore global backend state — this file toggles
    backend/device/precision per case and must not leak into other tests."""
    saved = (B.get_backend(), B.get_device(), B.get_numerical_precision())
    yield
    B.set_backend(saved[0])
    B.set_device(saved[1])
    B.set_numerical_precision(saved[2])


def _scipy_ref(sig, kernel, axis):
    kshape = [1] * sig.ndim
    kshape[axis % sig.ndim] = len(kernel)
    return scipy_convolve(
        sig, np.asarray(kernel).reshape(kshape), mode="valid", method="direct"
    )


def _make_signal(shape, dtype, nan_frac, rng):
    sig = rng.standard_normal(shape).astype(dtype)
    if nan_frac > 0:
        sig[rng.random(shape) < nan_frac] = np.nan
    return sig


_CASES = [
    ((12,), 0, 4),
    ((16, 64), -1, 6),
    ((8, 8, 128), 2, 16),
    ((4, 16, 8, 64), 3, 8),
]

_BACKENDS = [("numpy", "cpu")]
try:
    import torch  # noqa: F401

    _BACKENDS.append(("torch", "cpu"))
    if torch.cuda.is_available():
        _BACKENDS.append(("torch", "cuda"))
except ImportError:
    pass


@pytest.mark.parametrize("backend,device", _BACKENDS)
@pytest.mark.parametrize("shape,axis,n_ker", _CASES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nan_frac", [0.0, 0.05])
def test_convolve1d_nan_safe_matches_scipy(
    backend, device, shape, axis, n_ker, dtype, nan_frac
):
    """FFT-NaN-safe output matches scipy direct within FFT tolerance."""
    B.set_backend(backend)
    if backend == "torch":
        B.set_device(device)
    B.set_numerical_precision("float32" if dtype is np.float32 else "float64")

    rng = np.random.default_rng(0)
    sig = _make_signal(shape, dtype, nan_frac, rng)
    kernel = rng.standard_normal(n_ker).astype(dtype)

    out = B.to_numpy(B.convolve1d(sig, kernel, axis=axis, nan_safe=True))
    expected = _scipy_ref(sig, kernel, axis)

    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))

    finite = ~np.isnan(out)
    rtol = 1e-4 if dtype is np.float32 else 1e-10
    np.testing.assert_allclose(
        out[finite], expected[finite], rtol=rtol, atol=rtol
    )


def test_convolve1d_nan_safe_agrees_with_regular_fft_when_no_nans():
    """Without NaNs, nan_safe=True must match the plain FFT path."""
    B.set_backend("numpy")
    B.set_numerical_precision("float64")
    rng = np.random.default_rng(1)
    sig = rng.standard_normal((8, 64))
    kernel = rng.standard_normal(8)
    a = B.to_numpy(B.convolve1d(sig, kernel, axis=-1, nan_safe=False))
    b = B.to_numpy(B.convolve1d(sig, kernel, axis=-1, nan_safe=True))
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("backend,device", _BACKENDS)
@pytest.mark.parametrize("nan_safe", [False, True])
def test_convolve1d_empty_kernel_raises(backend, device, nan_safe):
    """Empty kernel must raise on both backends, both nan_safe settings.

    Pre-regression: scipy's direct path raised. The torch FFT-nan-safe
    path used to silently return garbage; the dispatcher now guards it.
    """
    B.set_backend(backend)
    if backend == "torch":
        B.set_device(device)
    sig = np.arange(5, dtype=np.float32)
    with pytest.raises(ValueError):
        B.convolve1d(sig, np.array([], dtype=np.float32), nan_safe=nan_safe)


@pytest.mark.parametrize("backend,device", _BACKENDS)
def test_convolve1d_nan_safe_count_exact_at_large_axis(backend, device):
    """Validity count must stay bit-exact past the float32 2**24 threshold.

    Reproduces the case where an int-typed cumsum disagrees with the
    float32 prefix-sum approach: a single NaN near the end of a >2**24
    axis. With the int64 fix, torch must mark the affected windows NaN
    exactly as scipy direct would.
    """
    pytest.importorskip("torch")
    import torch as _torch

    B.set_backend(backend)
    if backend == "torch":
        B.set_device(device)
    B.set_numerical_precision("float32")

    n = (1 << 24) + 32
    n_ker = 4
    miss = n - 5
    sig = np.zeros(n, dtype=np.float32)
    sig[miss] = np.nan
    kernel = np.ones(n_ker, dtype=np.float32)

    out = B.to_numpy(B.convolve1d(sig, kernel, axis=0, nan_safe=True))
    assert out.shape == (n - n_ker + 1,)
    nan_positions = np.where(np.isnan(out))[0]
    expected = np.arange(max(0, miss - n_ker + 1), miss + 1)
    np.testing.assert_array_equal(nan_positions, expected)
