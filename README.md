# scaleinvariance

![FIF 2D Example](FIF_2D_H0.3_C10.200_alpha2.0_magma_log_seed2.png)

Simulation and analysis tools for scale-invariant processes and multifractal fields.

[Documentation](https://scaleinvariance.readthedocs.io/en/latest/index.html)

View example FIF simulation output in the [Multifractal Explorer](https://thomasddewitt.com/thought-cloud/multifractal-explorer/)

## Current Features

### Analysis

#### Hurst exponent estimation

- **Haar fluctuation method**: `haar_fluctuation_hurst()`
- **Structure function method**: `structure_function_hurst()`
- **Spectral method**: `spectral_hurst()`

All methods support multi-dimensional arrays, averaging over dimensions that are orthogonal to the specified dimension along which spectra are calculated (specified by `axis`. Data and fit line for plotting may be returned with `return_fit=True`.

#### Structure functions

- **Structure function**: `structure_function()` - Order-`q` structure function (`q` may be scalar or array-valued)
- **Co-structure function**: `costructure_function()` - Mixed-moment structure function for two fields

### Simulation

- **1D fractionally integrated flux (FIF)**: `FIF_1D()` - Multifractal cascade simulation; causal/acausal
- **N-D fractionally integrated flux (FIF)**: `FIF_ND()` - Isotropic N-D multifractals for arbitrary dimensions, with optional per-axis causality (Example shown above)
- **1D fractional Brownian motion**: `fBm_1D_circulant()` - Fast spectral synthesis
- **N-D fractional Brownian motion**: `fBm_ND_circulant()` - Isotropic N-D (2D, 3D, 4D, etc.) fBm fields
- **1D fBm (fractional integration)**: `fBm_1D()` - Extended Hurst range (-0.5, 1.5) with causal/acausal kernels
- **Generalized Scale Invariance (GSI)**: `FIF_ND()` accepts a custom `scale_metric` together with the GSI elliptical dimension `elliptical_dim` (`D_el`) for anisotropic / stratified multifractals. `canonical_scale_metric()` builds the canonical stratified metric (last axis anisotropic with stratification exponent `Hz`).

For FIF simulation methods (`FIF_1D`, `FIF_ND`), the currently supported range is `0.5 <= alpha <= 2` with `alpha != 1`.
The kernel construction method `'LS2010'` (Lovejoy & Schertzer 2010) is used for flux kernels. For observable kernels, both `'LS2010'` and `'spectral'` are supported. GSI with a custom `scale_metric` requires `kernel_construction_method_observable='LS2010'`.

## Installation

```bash
pip install scaleinvariance
```

#### Agent Skill (Highly Recommended for Agents)

An agent skill is included in this repository. For Claude Code:

```bash
mkdir -p ~/.claude/skills/scaleinvariance
cp agent-skills/scaleinvariance/SKILL.md ~/.claude/skills/scaleinvariance/
```

Codex:

```bash
mkdir -p ~/.codex/skills/scaleinvariance
cp agent-skills/scaleinvariance/SKILL.md ~/.codex/skills/scaleinvariance/
```


## Performance

By default, scaleinvariance uses NumPy for all computations. However, if PyTorch is installed, the package automatically detects it and uses PyTorch for simulation functions, providing potentially **huge** performance gains (depending on your machine):

```bash
# NumPy-only installation (minimal dependencies)
pip install scaleinvariance

# With PyTorch for faster simulations
pip install scaleinvariance[torch]
```

Control backend explicitly:

```python
import scaleinvariance

# Check current backend
print(scaleinvariance.get_backend())  # 'torch' if available, else 'numpy'

# Force specific backend
scaleinvariance.set_backend('numpy')  # Always use NumPy
scaleinvariance.set_backend('torch')  # Use torch (raises error if not installed)

# Configure threading (defaults to 90% CPU count)
scaleinvariance.set_num_threads(8)
```

## Basic Usage

```python
from scaleinvariance import (
    fBm_1D_circulant, fBm_ND_circulant, FIF_1D, FIF_ND,
    canonical_scale_metric, haar_fluctuation_hurst,
)

# Generate 1D fractional Brownian motion
fBm_1d = fBm_1D_circulant(1024, H=0.7)

# Generate 2D fractional Brownian motion
fBm_2d = fBm_ND_circulant((512, 512), H=0.7)

# Generate 3D fractional Brownian motion
fBm_3d = fBm_ND_circulant((256, 256, 128), H=0.7)

# Generate multifractal FIF timeseries
fif = FIF_1D(2**16, alpha=1.8, C1=0.1, H=0.3)

# Override FIF kernels explicitly if needed
fif_custom = FIF_1D(
    2**16,
    alpha=1.8,
    C1=0.1,
    H=0.3,
    causal=False,
    kernel_construction_method_flux='LS2010',
    kernel_construction_method_observable='spectral',
)

# Anisotropic 2D GSI multifractal (canonical stratified metric)
size = (256, 256)
sim_size = (512, 512)        # non-periodic -> doubled domain for the metric
Hz = 0.556                    # canonical atmospheric stratification
metric = canonical_scale_metric(sim_size, ls=100.0, Hz=Hz)
fif_gsi = FIF_ND(
    size, alpha=1.8, C1=0.1, H=0.3, periodic=False,
    scale_metric=metric, elliptical_dim=1 + Hz,
    kernel_construction_method_observable='LS2010',
)

# Estimate Hurst exponent
H_est, H_err = haar_fluctuation_hurst(fBm_1d)
print(f"Estimated H = {H_est:.3f} ± {H_err:.3f}")
```

## Kernel Selection

For FIF simulations:

- `kernel_construction_method_flux` controls the cascade kernel.
- `kernel_construction_method_observable` controls the final observable kernel.
- Flux kernels support `'LS2010'` (and deprecated `'naive'` for 1D only).
- Observable kernels support `'spectral'` (default) and `'LS2010'`. `FIF_1D` also supports `'spectral_odd'` and deprecated `'naive'`.
- The old `kernel_construction_method=` argument is no longer accepted by `FIF_1D` or `FIF_ND`.
- `naive` FIF kernels remain available only for comparison and debugging and emit a warning because they are not remotely accurate.

For fBm:

- `fBm_ND_circulant()` is recommended. It uses a spectral synthesis method and produces essentially perfectly scale invariant fields.
- `fBm_1D()` is included for comparsion or for causal simulations.

## Generalized Scale Invariance (GSI)

`FIF_ND()` supports anisotropic / stratified multifractals via two
parameters:

- `scale_metric`: an N-D array of generalized distances on the
  simulation grid. Must match the simulation domain shape after
  accounting for periodicity (domain is doubled along non-periodic
  axes). Values must be finite and strictly positive everywhere.
- `elliptical_dim`: the GSI elliptical dimension `D_el`. For canonical
  stratified anisotropy with exponent `Hz`, `D_el = d - 1 + Hz` in `d`
  dimensions (i.e. `1 + Hz` in 2D, `2 + Hz` in 3D, …). Defaults to the
  spatial dimension (isotropic).

The helper `canonical_scale_metric(size, ls, Hz)` builds the standard
stratified metric — the last axis is anisotropic with exponent `Hz`,
all other axes are isotropic — on the `dx=2` grid expected by the
LS2010 kernel construction.

GSI with a custom `scale_metric` currently requires
`kernel_construction_method_observable='LS2010'`; it is not supported
on the default `'spectral'` observable path, and is therefore
incompatible with `observable_kernel_odd_axes`.

> Note: the older keyword `scale_metric_dim` has been renamed to
> `elliptical_dim`. The old name still works but emits a
> `DeprecationWarning` and will be removed in a future release.

## Examples

Work in progress. See [Multifractal Explorer](https://thomasddewitt.com/thought-cloud/multifractal-explorer/) for visuals.

Run examples:

```bash
python examples/fif_comparison_demo.py
```

Data source for LGMR: https://www.ncei.noaa.gov/access/paleo-search/study/33112

## Testing

Tests are organised into three directories under `tests/`:

| Directory           | Purpose                                                                                                                     | How to run                           |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `tests/functional/` | Sanity checks and gross-accuracy tests. Should always pass.                                                                 | `pytest`                             |
| `tests/numerical/`  | Theory-validation scripts run directly as Python programs. Some produce failures due to finite-size/discretization effects. | `python tests/numerical/<script>.py` |
| `tests/visual/`     | Scripts that generate plots for manual inspection.                                                                          | `python tests/visual/<script>.py`    |


FIF validation scripts, which test scaling over multiple ranges of scale, live in `tests/numerical/`. They are designed to be run as standalone Python programs, not via `pytest`, and they generate many large FIF realizations to reach statistical convergence. These scripts are also known to produce some failures, especially near grid scales, because finite-size effects are not fully mitigated by the LS2010 corrections or their spectral alternatives.

Run the functional suite:

```bash
pytest
```

Run a specific visual test:

```bash
python tests/visual/test_1D_fif_structure_function.py 0.3 0.1 1.8
```

## Requirements

- Python ≥ 3.10
- NumPy, SciPy
- PyTorch (optional, for simulation speedup)
