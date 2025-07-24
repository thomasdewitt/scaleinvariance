import pytest
from scaleinvariance.simulation import fractionally_integrated_flux, fractional_brownian_motion


def test_fractionally_integrated_flux():
    with pytest.raises(NotImplementedError):
        fractionally_integrated_flux(1000, alpha=0.5)


def test_fractional_brownian_motion():
    with pytest.raises(NotImplementedError):
        fractional_brownian_motion(1000, hurst=0.7)