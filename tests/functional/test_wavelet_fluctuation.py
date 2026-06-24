"""
Correctness tests for the general wavelet-fluctuation machinery.

These generate their own data at runtime (no stored regression blobs): the
structure_function wavelet must reproduce structure_function, every wavelet must
recover the Hurst exponent of fBm, the complex Morlet must return a real
modulus, and the K_empirical / two_point_C1 wavelet plumbing (and its
raise-rule) must behave.

(The one-time check that the refactor left haar_fluctuation BIT-IDENTICAL was
run against a pre-refactor baseline at commit time and then removed -- there is
deliberately no regression data in git.)
"""

import numpy as np
import pytest

import scaleinvariance as si


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    si.set_backend('numpy')
    si.set_numerical_precision('float32')


def test_structure_function_wavelet_matches_structure_function():
    """The 'structure_function' wavelet reproduces structure_function (order 1)."""
    si.set_backend('numpy')
    si.set_numerical_precision('float64')
    x = si.fBm_1D_circulant(4096, H=0.4)
    lw, vw = si.wavelet_fluctuation(x, wavelet='structure_function', order=1)
    ls, vs = si.structure_function(x, order=1)
    common = np.intersect1d(lw, ls)
    vw_c = vw[np.isin(lw, common)]
    vs_c = vs[np.isin(ls, common)]
    assert np.allclose(vw_c, vs_c, rtol=1e-10, atol=0)


def test_haar_wrapper_matches_wavelet_fluctuation():
    """haar_fluctuation is exactly wavelet_fluctuation(wavelet='haar')."""
    si.set_backend('numpy')
    si.set_numerical_precision('float64')
    x = si.fBm_1D_circulant(4096, H=0.4)
    l1, v1 = si.haar_fluctuation(x, order=[1.0, 2.0])
    l2, v2 = si.wavelet_fluctuation(x, wavelet='haar', order=[1.0, 2.0])
    np.testing.assert_array_equal(l1, l2)
    np.testing.assert_array_equal(v1, v2)


@pytest.mark.parametrize('wavelet', ['haar', 'structure_function', 'mexican_hat', 'morlet'])
def test_hurst_recovery_fbm(wavelet):
    """Every wavelet recovers fBm H within tolerance (averaged over realizations)."""
    si.set_backend('numpy')
    si.set_numerical_precision('float64')
    H_true = 0.4
    data = np.stack([si.fBm_1D_circulant(8192, H=H_true) for _ in range(40)], axis=0)
    H, unc = si.wavelet_fluctuation_hurst(data, wavelet=wavelet, axis=1,
                                          min_sep=8, max_sep=512)
    assert abs(H - H_true) < 0.05, f'{wavelet}: H={H:.3f}'


def test_mexican_hat_resolves_high_H():
    """Mexican hat (2 vanishing moments) recovers H=0.8."""
    si.set_backend('numpy')
    si.set_numerical_precision('float64')
    data = np.stack([si.fBm_1D_circulant(8192, H=0.8) for _ in range(40)], axis=0)
    H_mex, _ = si.wavelet_fluctuation_hurst(data, wavelet='mexican_hat', axis=1,
                                            min_sep=8, max_sep=512)
    assert abs(H_mex - 0.8) < 0.05, f'mexican_hat H={H_mex:.3f}'


def test_morlet_returns_real_finite_fluctuations():
    """The complex Morlet yields a real, positive, finite fluctuation (modulus)."""
    si.set_backend('numpy')
    si.set_numerical_precision('float64')
    x = si.fBm_1D_circulant(4096, H=0.5)
    lags, vals = si.wavelet_fluctuation(x, wavelet='morlet', order=1)
    finite = np.isfinite(vals)
    assert finite.sum() > 5
    assert np.all(np.asarray(vals)[finite] > 0)
    assert not np.iscomplexobj(vals)


def test_oversized_kernel_lags_are_nan():
    """Lags whose (wide) kernel exceeds the domain are returned as NaN, not an error."""
    si.set_backend('numpy')
    si.set_numerical_precision('float64')
    x = si.fBm_1D_circulant(2048, H=0.5)
    lags, vals = si.wavelet_fluctuation(x, wavelet='morlet', order=1, lags='all')
    assert np.isfinite(vals[0])    # small lags computable
    assert np.isnan(vals[-1])      # large lags: morlet support ~15x lag


def test_unknown_wavelet_raises():
    si.set_backend('numpy')
    x = si.fBm_1D_circulant(1024, H=0.5)
    with pytest.raises(ValueError, match='Unknown wavelet'):
        si.wavelet_fluctuation(x, wavelet='not_a_wavelet')


def test_array_order_matches_scalar():
    """Array-order output equals the per-order scalar calls (mexican_hat)."""
    si.set_backend('numpy')
    si.set_numerical_precision('float64')
    x = si.fBm_1D_circulant(4096, H=0.4)
    _, v_arr = si.wavelet_fluctuation(x, wavelet='mexican_hat', order=[1.0, 2.0])
    _, v1 = si.wavelet_fluctuation(x, wavelet='mexican_hat', order=1.0)
    _, v2 = si.wavelet_fluctuation(x, wavelet='mexican_hat', order=2.0)
    np.testing.assert_array_equal(v_arr[0], v1)
    np.testing.assert_array_equal(v_arr[1], v2)


def test_K_empirical_wavelet_path():
    si.set_backend('numpy')
    si.set_numerical_precision('float64')
    fif = np.stack([si.FIF_1D(8192, alpha=1.8, C1=0.1, H=0.3) for _ in range(20)], axis=0)
    q, K, H, C1, alpha = si.K_empirical(
        fif, axis=1, scaling_method='wavelet_fluctuation', wavelet='mexican_hat',
        min_sep=30, max_sep=1000)
    assert 0.2 < H < 0.45
    assert 0.0 < C1 < 0.25


def test_two_point_C1_wavelet_path():
    si.set_backend('numpy')
    si.set_numerical_precision('float64')
    fif = np.stack([si.FIF_1D(8192, alpha=2.0, C1=0.1, H=0.3) for _ in range(20)], axis=0)
    C1, _ = si.two_point_C1(fif, axis=1, scaling_method='wavelet_fluctuation',
                            wavelet='haar', min_sep=30, max_sep=1000)
    assert 0.0 < C1 < 0.25


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_convolve1d_complex_kernel_dispatch(backend):
    """convolve1d picks rfft vs fft from the kernel: real and complex-but-real
    kernels give a real result via rfft (the latter must not crash rfft);
    a genuinely complex kernel gives the complex convolution."""
    import scaleinvariance.backend as B
    from scipy.signal import convolve as sconv
    if backend == 'torch':
        try:
            B.set_backend('torch')
        except Exception:
            pytest.skip('torch unavailable')
    else:
        B.set_backend('numpy')
    B.set_numerical_precision('float64')
    rng = np.random.default_rng(3)
    sig = rng.standard_normal(128)
    k_real = np.array([1.0, 0.0, -1.0])
    k_cplx_real = k_real.astype(np.complex128)        # complex dtype, zero imag
    k_cplx = np.array([1 + 2j, 0 + 0j, -1 + 1j])

    out_real = B.to_numpy(B.convolve1d(sig, k_real, axis=0))
    out_cr = B.to_numpy(B.convolve1d(sig, k_cplx_real, axis=0))
    out_cplx = B.to_numpy(B.convolve1d(sig, k_cplx, axis=0))

    assert not np.iscomplexobj(out_real)
    assert not np.iscomplexobj(out_cr)              # the bug Codex caught
    np.testing.assert_allclose(out_cr, out_real, rtol=1e-12)
    assert np.iscomplexobj(out_cplx)
    np.testing.assert_allclose(out_cplx, sconv(sig, k_cplx, mode='valid'), rtol=1e-10)
    B.set_backend('numpy')


@pytest.mark.parametrize('func', ['K_empirical', 'two_point_C1'])
def test_wavelet_with_wrong_scaling_method_raises(func):
    si.set_backend('numpy')
    x = si.fBm_1D_circulant(2048, H=0.5)
    fn = getattr(si, func)
    with pytest.raises(ValueError, match='wavelet'):
        fn(x, scaling_method='structure_function', wavelet='morlet')
