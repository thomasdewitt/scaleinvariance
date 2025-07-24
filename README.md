# ScaleInvariance

⚠️ **This package is under active development. Current functionality is limited to Hurst exponent estimation and fractional Brownian motion simulation.**

Simulation and analysis tools for scale-invariant processes and multifractal fields.

## Current Features

### Simulation
- **1D fractional Brownian motion**: `acausal_fBm_1D()` - Spectral synthesis method
- **2D fractional Brownian motion**: `acausal_fBm_2D()` - Isotropic 2D fields with proper frequency normalization

### Hurst Exponent Estimation
- **Haar fluctuation method**: `haar_fluctuation_hurst()`
- **Structure function method**: `structure_function_hurst()`  
- **Spectral method**: `spectral_hurst()`

All methods support multi-dimensional arrays and return uncertainty estimates.

## Installation

```bash
pip install scaleinvariance
```

## Basic Usage

```python
from scaleinvariance import acausal_fBm_1D, acausal_fBm_2D, haar_fluctuation_hurst

# Generate 1D fractional Brownian motion
fBm_1d = acausal_fBm_1D(1024, H=0.7)

# Generate 2D fractional Brownian motion  
fBm_2d = acausal_fBm_2D((512, 1024), H=0.7)

# Estimate Hurst exponent
H_est, H_err = haar_fluctuation_hurst(fBm_1d)
print(f"Estimated H = {H_est:.3f} ± {H_err:.3f}")
```

## Testing

```bash
# Test 1D fBm generation and Hurst estimation
python tests/test_acausal_fBm_hurst_estimation.py 0.7

# Test 2D fBm with isotropy validation
python tests/test_2d_fbm.py 0.7
```

## Planned Features

- Fractionally integrated flux (FIF) simulation
- Advanced multifractal analysis tools
- Additional Hurst estimation methods
- Comprehensive documentation

## Requirements

- Python ≥ 3.8
- NumPy, SciPy, PyTorch, Matplotlib