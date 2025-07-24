import pytest
import numpy as np
from scaleinvariance.analysis import structure_function_hurst, haar_fluctuation_hurst, spectral_hurst


def test_structure_function_hurst():
    data = np.random.randn(100)
    with pytest.raises(NotImplementedError):
        structure_function_hurst(data)


def test_haar_fluctuation_hurst():
    data = np.random.randn(100)
    with pytest.raises(NotImplementedError):
        haar_fluctuation_hurst(data)


def test_spectral_hurst():
    data = np.random.randn(100)
    with pytest.raises(NotImplementedError):
        spectral_hurst(data)