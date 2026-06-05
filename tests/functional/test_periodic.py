"""Tests for periodic (toroidal) structure function and Haar fluctuation.

Covers:
  - periodic SF / Haar match independent numpy references (all backends);
  - the toroidal reflection identity S(r) = S(L - r);
  - costructure self-consistency under periodic=True;
  - NaN-safe periodic Haar vs a masked reference, and the raise path;
  - the max_sep > L//2 ValueError for SF and Haar;
  - numpy <-> torch parity;
  - Hurst recovery on a truly-periodic fBm (via Haar, for convergence).
"""

import numpy as np
import pytest

import scaleinvariance as si
from scaleinvariance import backend as B


@pytest.fixture(autouse=True)
def _restore_backend_state():
    """This file toggles backend/precision per case; restore afterwards."""
    saved = (B.get_backend(), B.get_device(), B.get_numerical_precision())
    yield
    B.set_backend(saved[0])
    B.set_device(saved[1])
    B.set_numerical_precision(saved[2])


_BACKENDS = ["numpy"]
try:
    import torch  # noqa: F401

    _BACKENDS.append("torch")
except ImportError:
    pass


# --------------------------------------------------------------------------
# Independent numpy references
# --------------------------------------------------------------------------

def _ref_sf_periodic(x, lag, q=1.0):
    return float(np.mean(np.abs(np.roll(x, -lag) - x) ** q))


def _haar_kernel(lag):
    k = np.ones(lag) / (lag / 2)
    k[: lag // 2] *= -1
    return k


def _ref_haar_periodic(x, lag):
    L = len(x)
    kp = np.zeros(L)
    kp[:lag] = _haar_kernel(lag)
    conv = np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(kp), n=L)
    return float(np.mean(np.abs(conv)))


def _ref_haar_periodic_nan(x, lag):
    L = len(x)
    kp = np.zeros(L)
    kp[:lag] = _haar_kernel(lag)
    clean = np.where(np.isnan(x), 0.0, x)
    conv = np.fft.irfft(np.fft.rfft(clean) * np.fft.rfft(kp), n=L)
    valid = (~np.isnan(x)).astype(np.int64)
    # trailing length-`lag` circular window count for each output position
    cnt = np.array([valid[(np.arange(p - lag + 1, p + 1)) % L].sum() for p in range(L)])
    conv = np.where(cnt == lag, conv, np.nan)
    if np.all(np.isnan(conv)):  # matches the package's all-NaN guard
        return np.nan
    return float(np.nanmean(np.abs(conv)))


# --------------------------------------------------------------------------
# Correctness vs reference
# --------------------------------------------------------------------------

@pytest.mark.parametrize("backend", _BACKENDS)
def test_structure_function_periodic_matches_reference(backend):
    B.set_backend(backend)
    B.set_numerical_precision("float64")
    x = np.asarray(B.to_numpy(si.fBm_1D_circulant(2048, H=0.7, periodic=True)))

    for q in (1.0, 2.0, 0.5):
        lags, sf = si.structure_function(x, order=q, periodic=True)
        ref = np.array([_ref_sf_periodic(x, int(l), q) for l in lags])
        assert np.allclose(B.to_numpy(sf), ref, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_haar_fluctuation_periodic_matches_reference(backend):
    B.set_backend(backend)
    B.set_numerical_precision("float64")
    x = np.asarray(B.to_numpy(si.fBm_1D_circulant(2048, H=0.6, periodic=True)))

    lags, hf = si.haar_fluctuation(x, order=1, periodic=True)
    ref = np.array([_ref_haar_periodic(x, int(l)) for l in lags])
    assert np.allclose(B.to_numpy(hf), ref, atol=1e-9, rtol=1e-9)


def test_reflection_identity_S_r_equals_S_Lminus_r():
    """S(r) = S(L - r) exactly on a periodic domain (per-realization)."""
    B.set_backend("numpy")
    B.set_numerical_precision("float64")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(64)
    L = len(x)
    for r in (1, 5, 13, 27):
        assert _ref_sf_periodic(x, r) == pytest.approx(_ref_sf_periodic(x, L - r), abs=1e-12)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_costructure_self_consistency_periodic(backend):
    """costructure_function(f, f, p, q, periodic) == structure_function(f, p+q, periodic)."""
    B.set_backend(backend)
    B.set_numerical_precision("float64")
    x = si.fBm_1D_circulant(1024, H=0.5, periodic=True)

    lc, cv = si.costructure_function(x, x, order1=1.0, order2=1.5, periodic=True)
    ls, sv = si.structure_function(x, order=2.5, periodic=True)
    assert np.array_equal(lc, ls)
    assert np.allclose(B.to_numpy(cv), B.to_numpy(sv), atol=1e-9, rtol=1e-9)


# --------------------------------------------------------------------------
# Lag ceiling and raises
# --------------------------------------------------------------------------

def test_sf_default_max_sep_capped_at_half():
    """SF (and costructure) obey S(r)=S(L-r), so lags are capped at L//2."""
    B.set_backend("numpy")
    L = 2048
    x = si.fBm_1D_circulant(L, H=0.5, periodic=True)
    lags_sf, _ = si.structure_function(x, periodic=True)
    assert lags_sf.max() <= L // 2


def test_haar_default_max_sep_not_capped_at_half():
    """Haar has no reflection symmetry, so lags extend past L//2 up to L-1."""
    B.set_backend("numpy")
    L = 2048
    x = si.fBm_1D_circulant(L, H=0.5, periodic=True)
    lags_hf, _ = si.haar_fluctuation(x, periodic=True)
    assert lags_hf.max() > L // 2


def test_haar_no_reflection_identity():
    """Regression guard: F(r) != F(L-r) for Haar (unlike the structure fn).

    L=8, x_j = cos(pi*j/2): periodic Haar gives F(2)=1, F(6)=1/3, while the
    structure function gives S(2)=S(6).
    """
    L = 8
    x = np.cos(np.pi * np.arange(L) / 2)
    assert _ref_haar_periodic(x, 2) == pytest.approx(1.0)
    assert _ref_haar_periodic(x, 6) == pytest.approx(1.0 / 3.0)
    # The structure function, by contrast, IS reflection-symmetric here.
    assert _ref_sf_periodic(x, 2) == pytest.approx(_ref_sf_periodic(x, 6))


@pytest.mark.parametrize("backend", _BACKENDS)
def test_sf_max_sep_above_half_raises(backend):
    B.set_backend(backend)
    L = 1024
    x = si.fBm_1D_circulant(L, H=0.5, periodic=True)
    with pytest.raises(ValueError):
        si.structure_function(x, periodic=True, max_sep=L // 2 + 2)
    with pytest.raises(ValueError):
        si.costructure_function(x, x, periodic=True, max_sep=L // 2 + 2)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_haar_max_sep_above_half_accepted(backend):
    """Haar accepts max_sep > L//2 and stays correct at those large lags."""
    B.set_backend(backend)
    B.set_numerical_precision("float64")
    L = 1024
    x = np.asarray(B.to_numpy(si.fBm_1D_circulant(L, H=0.6, periodic=True)))
    lags, hf = si.haar_fluctuation(x, periodic=True, max_sep=L - 1, lags="all")
    assert lags.max() > L // 2
    ref = np.array([_ref_haar_periodic(x, int(l)) for l in lags])
    assert np.allclose(B.to_numpy(hf), ref, atol=1e-9, rtol=1e-9)


# --------------------------------------------------------------------------
# NaN handling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("backend", _BACKENDS)
def test_periodic_haar_nan_ignore_matches_reference(backend):
    B.set_backend(backend)
    B.set_numerical_precision("float64")
    x = np.asarray(B.to_numpy(si.fBm_1D_circulant(1024, H=0.6, periodic=True))).copy()
    x[10] = np.nan
    x[500] = np.nan
    x[1000] = np.nan  # near the seam: tests wraparound window masking

    lags, hf = si.haar_fluctuation(x, periodic=True, nan_behavior="ignore")
    ref = np.array([_ref_haar_periodic_nan(x, int(l)) for l in lags])
    got = B.to_numpy(hf)
    finite = np.isfinite(ref)
    assert np.allclose(got[finite], ref[finite], atol=1e-9, rtol=1e-9)
    assert np.array_equal(np.isfinite(got), finite)


def test_periodic_haar_nan_raise():
    B.set_backend("numpy")
    x = np.asarray(B.to_numpy(si.fBm_1D_circulant(512, H=0.5, periodic=True))).copy()
    x[3] = np.nan
    with pytest.raises(ValueError):
        si.haar_fluctuation(x, periodic=True)  # nan_behavior defaults to 'raise'


def test_periodic_sf_handles_nan_via_nanmean():
    B.set_backend("numpy")
    B.set_numerical_precision("float64")
    x = np.asarray(B.to_numpy(si.fBm_1D_circulant(512, H=0.5, periodic=True))).copy()
    x[7] = np.nan
    lags, sf = si.structure_function(x, periodic=True)
    assert np.all(np.isfinite(B.to_numpy(sf)))


# --------------------------------------------------------------------------
# Backend parity
# --------------------------------------------------------------------------

@pytest.mark.skipif("torch" not in _BACKENDS, reason="torch not installed")
def test_numpy_torch_parity_periodic():
    B.set_numerical_precision("float64")
    x = np.asarray(B.to_numpy(si.fBm_1D_circulant(1024, H=0.6, periodic=True))).copy()
    xn = x.copy()
    xn[20] = np.nan
    xn[600] = np.nan

    B.set_backend("numpy")
    _, sf_np = si.structure_function(x, order=[1.0, 2.0], periodic=True)
    _, hf_np = si.haar_fluctuation(xn, periodic=True, nan_behavior="ignore")

    B.set_backend("torch")
    _, sf_t = si.structure_function(x, order=[1.0, 2.0], periodic=True)
    _, hf_t = si.haar_fluctuation(xn, periodic=True, nan_behavior="ignore")

    assert np.allclose(B.to_numpy(sf_np), B.to_numpy(sf_t), atol=1e-10, rtol=1e-10)
    hn, ht = B.to_numpy(hf_np), B.to_numpy(hf_t)
    finite = np.isfinite(hn)
    assert np.array_equal(finite, np.isfinite(ht))
    assert np.allclose(hn[finite], ht[finite], atol=1e-10, rtol=1e-10)


# --------------------------------------------------------------------------
# Hurst recovery on a truly-periodic field (Haar, for convergence)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("backend", _BACKENDS)
def test_periodic_haar_hurst_recovery(backend):
    B.set_backend(backend)
    B.set_numerical_precision("float64")
    H_true = 0.7
    data = np.stack(
        [np.asarray(B.to_numpy(si.fBm_1D_circulant(4096, H=H_true, periodic=True)))
         for _ in range(8)],
        axis=0,
    )
    H_per, _ = si.haar_fluctuation_hurst(data, axis=1, periodic=True)
    H_ape, _ = si.haar_fluctuation_hurst(data, axis=1, periodic=False)
    # Recovers the true exponent...
    assert abs(H_per - H_true) < 0.1
    # ...and periodic tracks aperiodic closely on a genuinely periodic field.
    assert abs(H_per - H_ape) < 0.05
