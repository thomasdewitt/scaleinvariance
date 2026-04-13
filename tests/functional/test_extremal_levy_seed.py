"""Tests for extremal_levy seed reproducibility and cross-alpha correlation."""

import numpy as np
import pytest

from scaleinvariance import (
    get_backend, set_backend,
    get_numerical_precision, set_numerical_precision,
    get_device, set_device,
)
from scaleinvariance.simulation.FIF import extremal_levy


def _available_configs():
    """Generate (backend, precision, device) tuples, skipping unavailable."""
    configs = []
    for backend in ('numpy', 'torch'):
        for precision in ('float32', 'float64'):
            if backend == 'numpy':
                configs.append((backend, precision, 'cpu'))
            else:
                configs.append((backend, precision, 'cpu'))
                configs.append((backend, precision, 'cuda'))
    return configs


ALL_CONFIGS = _available_configs()


@pytest.fixture(params=ALL_CONFIGS,
                ids=lambda c: f"{c[0]}-{c[1]}-{c[2]}")
def env(request):
    backend, precision, device = request.param
    if backend == 'torch':
        torch = pytest.importorskip('torch')
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    elif device != 'cpu':
        pytest.skip("numpy only supports cpu")

    orig_backend = get_backend()
    orig_precision = get_numerical_precision()
    orig_device = get_device()
    try:
        set_backend(backend)
        set_numerical_precision(precision)
        set_device(device)
        yield backend, precision, device
    finally:
        set_device(orig_device)
        set_numerical_precision(orig_precision)
        set_backend(orig_backend)


SIZE = 10000


class TestExtremalLevySeed:

    def test_same_seed_reproduces(self, env):
        a = extremal_levy(1.8, size=SIZE, seed=42)
        b = extremal_levy(1.8, size=SIZE, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_differs(self, env):
        a = extremal_levy(1.8, size=SIZE, seed=42)
        b = extremal_levy(1.8, size=SIZE, seed=99)
        assert not np.allclose(a, b)

    def test_cross_alpha_correlation(self, env):
        a = extremal_levy(1.7, size=SIZE, seed=42)
        b = extremal_levy(1.71, size=SIZE, seed=42)
        corr = np.corrcoef(a, b)[0, 1]
        assert corr > 0.8
