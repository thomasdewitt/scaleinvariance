# scaleinvariance

Simulation and analysis tools for scale-invariant processes and multifractal fields.

## Current Features

### Hurst Exponent Estimation

- **Haar fluctuation method**: `haar_fluctuation_hurst()`
- **Structure function method**: `structure_function_hurst()`
- **Spectral method**: `spectral_hurst()`

All methods support multi-dimensional arrays, averaging over dimensions that are orthogonal to the specified dimension along which spectra are calculated (specified by `axis`. Plotting data and fit line may be returned with `return_fit=True`.

### Simulation

`pytorch` is leveraged for parallel and efficient simulation.

- **1D fractional Brownian motion**: `acausal_fBm_1D()` - Spectral synthesis method
- **2D fractional Brownian motion**: `acausal_fBm_2D()` - Isotropic 2D fields with proper frequency normalization
- **1D fractionally integrated flux (FIF)**: `FIF_1D()` - Multifractal cascade simulation; causal/acausal
- **2D fractionally integrated flux (FIF)**: `FIF_2D()` - Isotropic 2D multifractals

## Installation

```bash
pip install scaleinvariance
```

## Basic Usage

```python
from scaleinvariance import acausal_fBm_1D, acausal_fBm_2D, FIF_1D, haar_fluctuation_hurst

# Generate 1D fractional Brownian motion
fBm_1d = acausal_fBm_1D(1024, H=0.7)

# Generate 2D fractional Brownian motion  
fBm_2d = acausal_fBm_2D((512, 1024), H=0.7)

# Generate multifractal FIF timeseries
fif = FIF_1D(2**16, H=0.3, C1=0.1)

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

## Examples

See the `examples/` directory for comprehensive demonstrations:

- **`fif_comparison_demo.py`**: Compare Hurst estimation methods on multifractal FIF simulations with different intermittency parameters
- **`multi_dataset_haar_analysis.py`**: Real-world data analysis using Haar fluctuation method

Run examples:

```bash
python examples/fif_comparison_demo.py
```

Data source for LGMR: https://www.ncei.noaa.gov/access/paleo-search/study/33112

## Planned Features

- Advanced multifractal analysis tools
- Comprehensive documentation

## Requirements

- Python ≥ 3.8
- NumPy, SciPy, PyTorch, Matplotlib