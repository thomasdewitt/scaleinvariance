# Examples

This directory contains demonstration scripts showing the capabilities of the scaleinvariance package.

## fif_comparison_demo.py

This script demonstrates estimation of the Hurst exponent estimation in intermittent multifractal fields. It generates Fractionally Integrated Flux (FIF) simulations with different levels of intermittency (measured by $C_1$) and compares available estimation methods.

### What the plot shows:

- **Top 4 panels**: Time series of FIF simulations with different parameter combinations but the same random seed. This shows the effect of intermittency: values that are unusually large in the nonintermittent case truly "blow up" when $C_1$ is increased.

- **Middle panels**: Haar fluctuation analysis (averaged over 10 realizations)

- **Bottom panels**: Power spectral analysis (averaged over 10 realizations)

### Key takeaways:

1. **First-order (Haar fluctuations/structure function) vs second-order (spectral) methods:**
   
   First-order statistics more accurately estimate $H$ as they are less affected by intermittency (see output below). Second-order statistics such as the spectrum used here underestimate $H$ unless intermittency corrections are included. In `spectral_hurst`, they are not. This is particularly true for $H=0$ because the multifractal simulations are more accurate in this case. The reason is that there is a single fractional integral for $H=0$ in the FIF model, as opposed to two for $H\ne 0$, and these introduce numerical error.

2. **Structure functions are only valid for $0<H<1$**:
   
   When $H$ is negative, the structure function saturates and returns $H=0$ (see below).

Output:

```
============================================================
Summary of Hurst estimates (10 simulation ensemble):
============================================================
H=0.0, C1=0.001:
  True H: 0.00
  Haar:   -0.101 ± 0.010
  Structure: 0.075 ± 0.001
  Spectral: -0.029 ± 0.007

H=0.0, C1=0.1:
  True H: 0.00
  Haar:   -0.068 ± 0.007
  Structure: 0.048 ± 0.002
  Spectral: -0.269 ± 0.011

H=0.3, C1=0.001:
  True H: 0.30
  Haar:   0.318 ± 0.003
  Structure: 0.343 ± 0.002
  Spectral: 0.334 ± 0.005

H=0.3, C1=0.1:
  True H: 0.30
  Haar:   0.342 ± 0.006
  Structure: 0.354 ± 0.003
  Spectral: 0.209 ± 0.014

H=-0.3, C1=0.001:
  True H: -0.30
  Haar:   -0.434 ± 0.002
  Structure: 0.004 ± 0.000
  Spectral: -0.385 ± 0.008

H=-0.3, C1=0.1:
  True H: -0.30
  Haar:   -0.301 ± 0.002
  Structure: 0.033 ± 0.002
  Spectral: -0.529 ± 0.014
```

### Usage:

```bash
python fif_comparison_demo.py
```

The script saves the comparison plot as `fif_comparison_demo.png` in the same directory.



## multi_dataset_haar_analysis.py

Work in progress to demonstrate Haar fluctuation analysis using real world data.