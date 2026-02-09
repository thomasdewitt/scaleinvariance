---
name: scaleinvariance
description: Use when working with scaleinvariance package for multifractal field simulation and analysis - Hurst exponent estimation, fractional Brownian motion (fBm), fractionally integrated flux (multiplicative cascades) (FIF), intermittency analysis, and Generalized Scale Invariance (GSI). Also use for scaling analysis of time series, turbulence data, atmospheric fields, or any task involving self-similar, cascade, or multifractal processes or fields.
---

# scaleinvariance Package Reference

Simulation and analysis tools for **multifractal fields and time series**.

**Documentation**: See CLAUDE.md in repository root.

## Critical Usage Note

**When analyzing multiple arrays (e.g., multiple time series or spatial snapshots), concatenate them along a NEW axis and pass ALL AT ONCE.** Analysis functions are NOT linear operations - do NOT loop and accumulate results.

```python
import numpy as np
import scaleinvariance

# Say we have 10 independent time series, each of length 2048
time_series_list = [generate_some_data() for _ in range(10)]

# CORRECT - stack along new axis (axis 0) and compute along original axis (axis 1)
stacked = np.stack(time_series_list, axis=0)  # Shape: (10, 2048)
H, err = scaleinvariance.structure_function_hurst(stacked, axis=1)  # Analyze along axis 1

# WRONG - do NOT loop and combine
# results = []
# for ts in time_series_list:
#     H, err = scaleinvariance.structure_function_hurst(ts)  # NO!
#     results.append(H)
# H_avg = np.mean(results)  # INCORRECT - these are not linear operations!
```

This applies to all analysis functions: `structure_function_hurst`, `haar_fluctuation_hurst`, `spectral_hurst`, `two_point_intermittency_exponent`, and their underlying `*_analysis` functions.

## When to Use Which Function

### Hurst Exponent Estimation

| Task                                        | Function                   | Notes                                                                                                                       |
| ------------------------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Recommended** cases when 0<H<1            | `structure_function_hurst` | First-order structure function scaling, robust                                                                              |
| Wavelet-based alternative, works for -1<H<1 | `haar_fluctuation_hurst`   | Uses Haar wavelet convolution, first order                                                                                  |
| Spectral method, works for all H            | `spectral_hurst`           | Uses power spectrum β = 2H + 1, uses second-order statistics so will not be equivalent to the above for multifractal fields |

All three methods estimate the same H parameter but may give slightly different results. Structure function is recommended for most cases.

### Underlying Scaling Analysis

**Usually, the *_hurst methods above are preferred to the functions in this section, as the spectrum/scaling function may be obtained from them using a kwarg described below**

| Task                      | Function                      | Notes                               |
| ------------------------- | ----------------------------- | ----------------------------------- |
| Structure function values | `structure_function_analysis` | Returns lags and S_n(r) values      |
| Haar fluctuation values   | `haar_fluctuation_analysis`   | Returns lags and fluctuation values |
| Power spectral density    | `spectral_analysis`           | Returns binned frequencies and PSD  |

### Intermittency Analysis

| Task                  | Function                           | Notes                                                                 |
| --------------------- | ---------------------------------- | --------------------------------------------------------------------- |
| Estimate C1 parameter | `two_point_intermittency_exponent` | Uses two-point scaling to estimate intermittency                      |
| Compute K(q) function | `K`                                | Returns Lovejoy and Schertzer's K(q) = (C1/(alpha-1)) * (q^alpha - q) |

### Simulation

| Task                            | Function            | Notes                                          |
| ------------------------------- | ------------------- | ---------------------------------------------- |
| 1D multifractal (FIF)           | `FIF_1D`            | Fractionally integrated flux with H, C1, alpha |
| N-D multifractal (FIF)          | `FIF_ND`            | 2D, 3D, etc. with GSI support                  |
| 1D fBm (spectral)               | `fBm_1D_circulant`  | Fast spectral synthesis, periodic              |
| N-D fBm (spectral)              | `fBm_ND_circulant`  | 2D, 3D, 4D, etc. spectral synthesis, isotropic |
| 1D fBm (fractional integration) | `fBm_1D`            | Extended H range (-0.5, 1.5)                   |

### Generalized Scale Invariance (GSI)

| Task                     | Function                 | Notes                                                                                |
| ------------------------ | ------------------------ | ------------------------------------------------------------------------------------ |
| Anisotropic scale metric | `canonical_scale_metric` | Creates stratified/anisotropic distance metric, for use in FIF and fBm methods above |

### Backend Control

| Task        | Function                                         | Notes                                               |
| ----------- | ------------------------------------------------ | --------------------------------------------------- |
| Set backend | `set_backend('numpy')` or `set_backend('torch')` | Torch is often much faster but is a huge dependency |
| Get backend | `get_backend()`                                  | Returns current backend name                        |
| Set threads | `set_num_threads(n)`                             | Default: 90% CPU count                              |

## Function Signatures

### Hurst Exponent Estimation

```python
scaleinvariance.structure_function_hurst(
    data,               # N-dimensional array
    min_sep=None,       # Min lag for fit (default: 1 or 4 depending on size)
    max_sep=None,       # Max lag for fit (default: size/8 or size-1 if small)
    axis=0,             # Axis along which to compute
    return_fit=False    # If True, also return lags, values, fit_line
) -> (H, uncertainty) | (H, uncertainty, lags, sf_values, fit_line)
```

```python
scaleinvariance.haar_fluctuation_hurst(
    data,               # N-dimensional array
    min_sep=None,       # Min lag for fit
    max_sep=None,       # Max lag for fit
    axis=0,             # Axis along which to compute
    return_fit=False    # If True, also return lags, values, fit_line
) -> (H, uncertainty) | (H, uncertainty, lags, haar_values, fit_line)
```

```python
scaleinvariance.spectral_hurst(
    data,                    # N-dimensional array
    max_wavelength=None,     # Max wavelength for fit (default: size/8 or size)
    min_wavelength=None,     # Min wavelength for fit (default: 2 or 8)
    nbins=50,                # Number of log bins for PSD
    axis=0,                  # Axis along which to compute FFT
    return_fit=False         # If True, also return freqs, psd, fit_line
) -> (H, uncertainty) | (H, uncertainty, freqs, psd_values, fit_line)
```

### Scaling Analysis Functions

```python
scaleinvariance.structure_function_analysis(
    data,                    # N-dimensional array
    order=1,                 # Order n of structure function S_n(r)
    max_sep=None,            # Maximum lag to compute
    axis=0,                  # Axis along which to compute
    lags='powers of 1.2'     # 'all', 'powers of X', or array of lags
) -> (lags, sf_values)
```

```python
scaleinvariance.haar_fluctuation_analysis(
    data,                    # N-dimensional array
    order=1,                 # Order of fluctuation
    max_sep=None,            # Maximum lag to compute
    axis=0,                  # Axis along which to compute
    lags='powers of 1.2',    # 'all', 'powers of X', or array of lags
    nan_behavior='raise'     # 'raise' or 'ignore'
) -> (lags, haar_values)
```

```python
scaleinvariance.spectral_analysis(
    data,                    # N-dimensional array
    max_wavelength=None,     # Maximum wavelength to include
    min_wavelength=None,     # Minimum wavelength to include
    nbins=50,                # Number of log bins
    axis=0                   # Axis along which to compute FFT
) -> (binned_freq, binned_psd)
```

### Intermittency Analysis

```python
scaleinvariance.two_point_intermittency_exponent(
    data,                           # N-dimensional array
    order=2,                        # Second order for exponent (default: 2)
    assumed_alpha=2.0,              # Levy stability parameter (default: Gaussian)
    scaling_method='structure_function',  # or 'haar_fluctuation'
    min_sep=None,                   # Min separation for scaling
    max_sep=None,                   # Max separation for scaling
    axis=0                          # Axis along which to compute
) -> (C1, uncertainty)   # NOTE: uncertainty from error propagation is approximate
```

```python
scaleinvariance.K(
    q,       # Order of moment (float or array)
    C1,      # Codimension of mean (intermittency parameter)
    alpha    # Levy stability parameter
) -> K_value   # K(q) = (C1/(alpha-1)) * (q^alpha - q)
```

### FIF Simulation

```python
scaleinvariance.FIF_1D(
    size,                      # Length (must be even, power of 2 recommended)
    alpha,                     # Levy stability parameter in (0, 2), != 1
    C1,                        # Codimension of mean (intermittency), must be >= 0
                               # C1 = 0 routes to fBm (requires causal=False)
    H,                         # Hurst exponent in (-1, 1)
    levy_noise=None,           # Pre-generated noise for reproducibility
    causal=True,               # Use causal kernels (must be False for C1=0)
    outer_scale=None,          # Large-scale cutoff (default: size)
    outer_scale_width_factor=2.0,  # Transition width control
    kernel_construction_method='LS2010',  # or 'naive', don't change this unless explicitely asked
    periodic=True              # Double size internally to eliminate artifacts
) -> numpy.ndarray   # Normalized by mean (or zero mean for H < 0)
```

```python
scaleinvariance.FIF_ND(
    size,                      # Tuple of dimensions, e.g., (512, 512) or (256, 256, 128)
    alpha,                     # Levy stability parameter in (0, 2), != 1
    C1,                        # Codimension of mean (intermittency), must be >= 0
                               # C1 = 0 routes to fBm_ND_circulant()
    H,                         # Hurst exponent in (0, 1)
    levy_noise=None,           # Pre-generated noise for reproducibility
    outer_scale=None,          # Large-scale cutoff (default: max(size))
    outer_scale_width_factor=2.0,  # Transition width control
    kernel_construction_method='LS2010',  # Only LS2010 for N-D
    periodic=False,            # Bool or tuple of bool for per-axis periodicity
    scale_metric=None,         # Custom GSI distance metric
    scale_metric_dim=None      # Scaling dimension (default: spatial dimension)
) -> numpy.ndarray   # Normalized by mean
```

### fBm Simulation

Circulant methods are preferred. fBm uses a convolution with a power law kernel similar to FIF, but this is not as accurate. However, fBm_1D can be used to limit the outer scale or control the noise, or make causal simulations, for example.

```python
scaleinvariance.fBm_1D_circulant(
    size,      # Length (must be power of 2)
    H          # Hurst parameter (typical: 0-1, but negative supported)
) -> numpy.ndarray   # Zero mean, unit std, periodic
```

```python
scaleinvariance.fBm_ND_circulant(
    size,      # Tuple of dimensions, e.g., (512, 512) for 2D, (256, 256, 128) for 3D (must be powers of 2)
    H,         # Hurst parameter (typical: 0-1, but negative supported)
    periodic=True  # Bool or tuple of bool for per-axis periodicity control
) -> numpy.ndarray   # Zero mean, unit std
```

```python
scaleinvariance.fBm_1D(
    size,                      # Length (must be even, power of 2 recommended)
    H,                         # Hurst exponent in (-0.5, 1.5)
    causal=True,               # Use causal kernels
    outer_scale=None,          # Large-scale cutoff
    outer_scale_width_factor=2.0,
    periodic=True,             # Eliminate periodicity artifacts
    gaussian_noise=None,       # Pre-generated noise
    kernel_construction_method='naive'  # or 'LS2010'
) -> numpy.ndarray   # Zero mean, unit std
```

### GSI Scale Metric

```python
scaleinvariance.canonical_scale_metric(
    size,              # (nx, nz) for 2D or (nx, ny, nz) for 3D
    ls,                # Spheroscale (characteristic horizontal scale), must be > 0
    Hz=5/9             # Anisotropy exponent (default: atmospheric stratification)
) -> numpy.ndarray   # Scale metric with dx=2 spacing (matches LS2010 kernels)
```

**IMPORTANT for GSI with FIF_ND**: When using `canonical_scale_metric`, you must also specify `scale_metric_dim`:

- For 2D: `scale_metric_dim = 1 + Hz`
- For 3D: `scale_metric_dim = 2 + Hz`
- For 4D: `scale_metric_dim = 3 + Hz`
- ...

```python
# Example: 2D GSI simulation
size = (256, 256)
sim_size = (512, 512)  # Doubled for periodic=False
Hz = 0.556
ls = 100.0
metric = scaleinvariance.canonical_scale_metric(sim_size, ls=ls, Hz=Hz)
fif = scaleinvariance.FIF_ND(size, alpha=1.8, C1=0.1, H=0.3, periodic=False,
                              scale_metric=metric, scale_metric_dim=1 + Hz)
```

## Parameter Interpretation

### Hurst Exponent (H)

- H = 0.5: Random walk / uncorrelated increments
- H > 0.5: Persistent / positively correlated increments
- H < 0.5: Anti-persistent / negatively correlated increments
- FIF_1D supports H in (-1, 1); negative H gives differenced processes

### Intermittency (C1)

- C1 = 0: No intermittency (monofractal) - FIF functions route internally to fBm
  - FIF_1D with C1=0 requires causal=False (fBm cannot be causal)
  - FIF_ND with C1=0 works normally (no causal parameter)
- C1 > 0: Multifractal intermittency
- Typical atmospheric values: C1 ~ 0.05-0.2

### Levy Stability (alpha)

- alpha = 2: Gaussian (log-normal multifractal)
- alpha < 2: Heavy-tailed (more extreme events)
- alpha = 1 is singular and not supported

### Anisotropy (Hz)

- Hz = 1: Isotropic (standard Euclidean)
- Hz < 1: Stratified/layered structures (vertical scales slower)
- Hz = 5/9 ~ 0.556: Typical atmospheric stratification

## Per-Axis Periodicity (FIF_ND)

Control periodicity separately for each axis:

```python
# Periodic in x, non-periodic in z (for e.g., horizontal periodicity, vertical boundaries)
fif = scaleinvariance.FIF_ND((512, 256), alpha=1.7, C1=0.1, H=0.3,
                              periodic=(True, False))
```

## Backend Selection

PyTorch backend provides speedups, often quite significant, for simulations:

```python
import scaleinvariance

# Check current backend
print(scaleinvariance.get_backend())  # 'numpy' or 'torch'

# Force numpy backend
scaleinvariance.set_backend('numpy')

# Use torch if available (auto-detected by default)
scaleinvariance.set_backend('torch')

# Control threading
scaleinvariance.set_num_threads(8)
```

All functions return NumPy arrays regardless of backend.

## Complete Examples

### Estimate Hurst exponent with plotting

```python
import numpy as np
import matplotlib.pyplot as plt
import scaleinvariance

# Generate test data
fif = scaleinvariance.FIF_1D(4096, alpha=1.8, C1=0.1, H=0.4)

# Estimate H with fit details
H, err, lags, sf, fit = scaleinvariance.structure_function_hurst(fif, return_fit=True)
print(f"H = {H:.3f} +/- {err:.3f}")

# Plot
plt.figure(figsize=(6, 4))
plt.loglog(lags, sf, 'o', label='Data')
plt.loglog(lags, fit, '-', label=f'Fit: H={H:.2f}')
plt.xlabel('Lag'); plt.ylabel('S_1(r)')
plt.legend(); plt.show()
```

### Analyze multiple realizations correctly

```python
import numpy as np
import scaleinvariance

# Generate multiple realizations
n_realizations = 20
size = 2048
H_true = 0.35

# Stack into 2D array: (n_realizations, size)
data = np.stack([
    scaleinvariance.FIF_1D(size, alpha=1.8, C1=0.1, H=H_true)
    for _ in range(n_realizations)
], axis=0)

# Analyze ALL at once, along axis 1
H_est, H_err = scaleinvariance.structure_function_hurst(data, axis=1)
print(f"Estimated H = {H_est:.3f} +/- {H_err:.3f} (true: {H_true})")
```

### 2D FIF with GSI anisotropy

```python
import numpy as np
import matplotlib.pyplot as plt
import scaleinvariance

# Parameters
size = (256, 256)
Hz = 0.556  # Atmospheric stratification
ls = 50.0   # Spheroscale

# Create doubled domain for non-periodic simulation
sim_size = (512, 512)
metric = scaleinvariance.canonical_scale_metric(sim_size, ls=ls, Hz=Hz)

# Generate GSI multifractal
fif = scaleinvariance.FIF_ND(size, alpha=1.7, C1=0.1, H=0.3, periodic=False,
                              scale_metric=metric, scale_metric_dim=1 + Hz)

plt.figure(figsize=(6, 5))
plt.imshow(np.log(fif), cmap='viridis', aspect='auto')
plt.colorbar(label='log(FIF)')
plt.xlabel('x'); plt.ylabel('z (vertical)')
plt.title('Anisotropic GSI multifractal')
plt.show()
```
