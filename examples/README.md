# Examples

This directory contains demonstration scripts showing the capabilities of the scaleinvariance package.

## fif_comparison_demo.py

This script demonstrates estimation of the Hurst exponent estimation in intermittent multifractal fields. It generates Fractionally Integrated Flux (FIF) simulations with different levels of intermittency (measured by $C_1$) and compares available estimation methods.

### What the plot shows:

- **Top 4 panels**: Time series of FIF simulations with different parameter combinations but the same random seed. Values that are unusually large in the nonintermittent case truly "blow up" when $C_1$ is increased.

- **Middle panels**: Haar fluctuation analysis (averaged over 10 realizations)

- **Bottom panels**: Power spectral analysis (averaged over 10 realizations)

### Key takeaways:

1. **First-order (Haar fluctuations/structure function) vs second-order (spectral) methods:**
   
   First-order statistics more accurately estimate $H$ as they are less affected by intermittency. Second-order statistics, such as the spectrum used here, underestimate $H$ unless intermittency corrections are included. In `spectral_hurst`, they are not.

2. **Structure functions are only valid for $0<H<1$**:
   
   When $H$ is negative, the structure function saturates and returns $H=0.$

Output:

```
============================================================
Summary of Hurst estimates:
============================================================
H=-0.3, C1=0.01:
  True H: -0.30
  Haar:   -0.413 ± 0.008
  Spectral: -0.495 ± 0.032
  Structure: 0.014 ± 0.001

H=-0.3, C1=0.1:
  True H: -0.30
  Haar:   -0.297 ± 0.008
  Spectral: -0.571 ± 0.023
  Structure: 0.039 ± 0.002

H=0.3, C1=0.01:
  True H: 0.30
  Haar:   0.326 ± 0.002
  Spectral: 0.223 ± 0.030
  Structure: 0.348 ± 0.002

H=0.3, C1=0.1:
  True H: 0.30
  Haar:   0.337 ± 0.004
  Spectral: 0.089 ± 0.025
  Structure: 0.345 ± 0.004
```

### Usage:

```bash
python fif_comparison_demo.py
```

The script saves the comparison plot as `fif_comparison_demo.png` in the same directory.