"""
Tests for the chunked / out-of-core FFT engine (backend.set_fft_device).

The chunked engine decomposes N-D FFTs into per-axis batched 1D FFTs
streamed through the FFT device. Here the FFT device is 'cpu' with a tiny
chunk size, which exercises exactly the same code paths as GPU offload
(slab iteration, contiguity handling, write-back) minus the PCIe transfer.

Chunked results are mathematically identical but not bit-identical to the
direct transforms (per-axis ordering differs), so comparisons use small
tolerances.
"""

import numpy as np
import pytest

import scaleinvariance as si
from scaleinvariance import backend as B

try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False

pytestmark = pytest.mark.skipif(not _torch_available, reason="torch not available")


@pytest.fixture
def torch_chunked():
    """Torch backend with chunked FFTs forced (tiny chunks, cpu FFT device)."""
    prev_backend = si.get_backend()
    prev_precision = si.get_numerical_precision()
    si.set_backend('torch')
    si.set_numerical_precision('float64')
    si.set_fft_device('cpu', chunk_bytes=256)  # tiny: many chunks even for small arrays
    yield
    si.set_fft_device(None)
    si.set_numerical_precision(prev_precision)
    si.set_backend(prev_backend)


def _direct_and_chunked(fn, *args, **kwargs):
    """Evaluate a backend FFT with offload disabled and enabled."""
    si.set_fft_device(None)
    direct = B.to_numpy(fn(*args, **kwargs))
    si.set_fft_device('cpu', chunk_bytes=256)
    chunked = B.to_numpy(fn(*args, **kwargs))
    return direct, chunked


SHAPES = [(8, 6), (5, 8), (7, 9), (4, 6, 10), (3, 5, 7)]


class TestChunkedMatchesDirect:

    @pytest.mark.parametrize('shape', SHAPES)
    def test_rfftn(self, torch_chunked, shape):
        rng = np.random.default_rng(1)
        x = B.asarray(rng.standard_normal(shape))
        direct, chunked = _direct_and_chunked(B.rfftn, x)
        np.testing.assert_allclose(chunked, direct, rtol=1e-12, atol=1e-13)

    @pytest.mark.parametrize('shape', SHAPES)
    def test_irfftn_round_trip(self, torch_chunked, shape):
        rng = np.random.default_rng(2)
        x_np = rng.standard_normal(shape)
        x = B.asarray(x_np)
        spec = B.rfftn(x)
        out = B.to_numpy(B.irfftn(spec, s=shape))
        np.testing.assert_allclose(out, x_np, rtol=1e-12, atol=1e-13)

    @pytest.mark.parametrize('shape', SHAPES)
    def test_irfftn(self, torch_chunked, shape):
        rng = np.random.default_rng(3)
        spec_np = (rng.standard_normal(shape[:-1] + (shape[-1] // 2 + 1,))
                   + 1j * rng.standard_normal(shape[:-1] + (shape[-1] // 2 + 1,)))
        spec = B.asarray(spec_np)
        direct, chunked = _direct_and_chunked(B.irfftn, spec, s=shape)
        np.testing.assert_allclose(chunked, direct, rtol=1e-12, atol=1e-13)

    @pytest.mark.parametrize('shape', SHAPES)
    def test_fftn_ifftn(self, torch_chunked, shape):
        rng = np.random.default_rng(4)
        x_np = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        x = B.asarray(x_np)
        direct, chunked = _direct_and_chunked(B.fftn, x)
        np.testing.assert_allclose(chunked, direct, rtol=1e-12, atol=1e-13)
        direct, chunked = _direct_and_chunked(B.ifftn, x)
        np.testing.assert_allclose(chunked, direct, rtol=1e-12, atol=1e-13)

    def test_float32(self, torch_chunked):
        si.set_numerical_precision('float32')
        rng = np.random.default_rng(5)
        x = B.asarray(rng.standard_normal((16, 12)).astype(np.float32))
        direct, chunked = _direct_and_chunked(B.rfftn, x)
        np.testing.assert_allclose(chunked, direct, rtol=1e-5, atol=1e-6)

    def test_1d_transforms_unaffected(self, torch_chunked):
        """1D transforms never chunk; they must keep working with a device set."""
        rng = np.random.default_rng(6)
        x_np = rng.standard_normal(64)
        x = B.asarray(x_np)
        out = B.to_numpy(B.irfft(B.rfft(x), n=64))
        np.testing.assert_allclose(out, x_np, rtol=1e-12, atol=1e-13)


class TestChunkedSemantics:

    def test_irfftn_does_not_mutate_input_by_default(self, torch_chunked):
        rng = np.random.default_rng(7)
        spec_np = (rng.standard_normal((6, 5)) + 1j * rng.standard_normal((6, 5)))
        spec = B.asarray(spec_np)
        before = B.to_numpy(spec).copy()
        B.irfftn(spec, s=(6, 8))
        np.testing.assert_array_equal(B.to_numpy(spec), before)

    def test_irfftn_cropped_raises_with_offload(self, torch_chunked):
        spec = B.asarray(np.ones((8, 5), dtype=np.complex128))
        with pytest.raises(ValueError, match="Chunked irfftn"):
            B.irfftn(spec, s=(4, 8))  # cropped: spectrum rows != s[0]

    def test_invalid_chunk_bytes(self, torch_chunked):
        with pytest.raises(ValueError):
            si.set_fft_device('cpu', chunk_bytes=0)

    def test_get_fft_device(self, torch_chunked):
        assert si.get_fft_device() == 'cpu'
        si.set_fft_device(None)
        assert si.get_fft_device() is None

    def test_numpy_backend_unaffected(self):
        prev_backend = si.get_backend()
        try:
            si.set_backend('numpy')
            si.set_fft_device('cpu', chunk_bytes=256)
            rng = np.random.default_rng(8)
            x = rng.standard_normal((8, 6))
            out = B.irfftn(B.rfftn(x), s=(8, 6))
            np.testing.assert_allclose(out, x, rtol=1e-12, atol=1e-13)
        finally:
            si.set_fft_device(None)
            si.set_backend(prev_backend)


class TestSimulationsWithOffload:
    """Full simulations with chunked FFTs must match the direct results."""

    def test_fif_nd_matches_direct(self, torch_chunked):
        torch.manual_seed(123)
        si.set_fft_device(None)
        direct = si.FIF_ND((32, 32), alpha=1.8, C1=0.1, H=0.3, periodic=False)
        torch.manual_seed(123)
        si.set_fft_device('cpu', chunk_bytes=256)
        chunked = si.FIF_ND((32, 32), alpha=1.8, C1=0.1, H=0.3, periodic=False)
        np.testing.assert_allclose(chunked, direct, rtol=1e-10, atol=1e-12)

    def test_fif_nd_odd_axes_matches_direct(self, torch_chunked):
        torch.manual_seed(321)
        si.set_fft_device(None)
        direct = si.FIF_ND((16, 16), alpha=1.8, C1=0.1, H=0.3, periodic=True,
                           observable_kernel_odd_axes=(True, False))
        torch.manual_seed(321)
        si.set_fft_device('cpu', chunk_bytes=256)
        chunked = si.FIF_ND((16, 16), alpha=1.8, C1=0.1, H=0.3, periodic=True,
                            observable_kernel_odd_axes=(True, False))
        np.testing.assert_allclose(chunked, direct, rtol=1e-10, atol=1e-12)

    def test_fif_nd_ls2010_observable_matches_direct(self, torch_chunked):
        torch.manual_seed(99)
        si.set_fft_device(None)
        direct = si.FIF_ND((16, 16), alpha=1.7, C1=0.05, H=0.2, periodic=True,
                           kernel_construction_method_observable='LS2010')
        torch.manual_seed(99)
        si.set_fft_device('cpu', chunk_bytes=256)
        chunked = si.FIF_ND((16, 16), alpha=1.7, C1=0.05, H=0.2, periodic=True,
                            kernel_construction_method_observable='LS2010')
        np.testing.assert_allclose(chunked, direct, rtol=1e-10, atol=1e-12)

    def test_fbm_nd_matches_direct(self, torch_chunked):
        torch.manual_seed(7)
        si.set_fft_device(None)
        direct = si.fBm_ND_circulant((16, 32), H=0.4, periodic=True)
        torch.manual_seed(7)
        si.set_fft_device('cpu', chunk_bytes=256)
        chunked = si.fBm_ND_circulant((16, 32), H=0.4, periodic=True)
        np.testing.assert_allclose(chunked, direct, rtol=1e-10, atol=1e-12)

    def test_fif_3d_matches_direct(self, torch_chunked):
        torch.manual_seed(11)
        si.set_fft_device(None)
        direct = si.FIF_ND((8, 8, 8), alpha=1.8, C1=0.1, H=0.3, periodic=True)
        torch.manual_seed(11)
        si.set_fft_device('cpu', chunk_bytes=256)
        chunked = si.FIF_ND((8, 8, 8), alpha=1.8, C1=0.1, H=0.3, periodic=True)
        np.testing.assert_allclose(chunked, direct, rtol=1e-10, atol=1e-12)
