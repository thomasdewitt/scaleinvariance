#!/usr/bin/env python3
"""
K(q) Empirical vs Theoretical Test

Generates FIF_1D simulations, estimates K(q) empirically using both
fit_hurst=False and fit_hurst=True, and compares to theoretical K(q).

Usage: python test_Kq_estimation.py [H] [C1] [alpha]
Examples:
  python test_Kq_estimation.py 0.3 0.1 1.8
  python test_Kq_estimation.py 0.5 0.15 1.6
  python test_Kq_estimation.py 0.7 0.05 2.0
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import scaleinvariance

# ---- Configuration ----
KERNEL_METHOD_FLUX = 'LS2010'
KERNEL_METHOD_OBS  = 'spectral'
FIT_MIN        = 10
FIT_MAX        = 10000
SIM_SIZE       = 2**16
N_SIMS         = 20
SCALING_METHOD = 'haar_fluctuation'   # 'structure_function' or 'haar_fluctuation'

# ---- CLI ----
H     = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
C1    = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 1.8

print("=" * 55)
print("K(q) ESTIMATION TEST")
print("=" * 55)
print(f"\nParameters: H={H}, C1={C1}, alpha={alpha}")
print(f"Config: {N_SIMS} sims, size={SIM_SIZE}, fit=[{FIT_MIN},{FIT_MAX}], flux={KERNEL_METHOD_FLUX}, obs={KERNEL_METHOD_OBS}, scaling={SCALING_METHOD}")
print(f'Using backend "{scaleinvariance.get_backend()}"')

# ---- Generate simulations ----
print(f"\nGenerating {N_SIMS} FIF simulations...")
all_sims = []
for i in range(N_SIMS):
    if i % 20 == 0:
        print(f"  Simulation {i+1}/{N_SIMS}...")
    all_sims.append(scaleinvariance.FIF_1D(
        SIM_SIZE, alpha, C1, H=H,
        kernel_construction_method_flux=KERNEL_METHOD_FLUX,
        kernel_construction_method_observable=KERNEL_METHOD_OBS,
    ))
sims = np.vstack(all_sims)  # shape: (N_SIMS, SIM_SIZE)

# ---- Estimate K(q): fit_hurst=False (H from order-1 scaling, fit C1 and alpha) ----
print(f"\nEstimating K(q) with hurst_fit_method='fixed'...")
q_values, K_empirical, H_fit1, C1_fit1, alpha_fit1 = scaleinvariance.K_empirical(
    sims, scaling_method=SCALING_METHOD, hurst_fit_method='fixed', min_sep=FIT_MIN, max_sep=FIT_MAX, axis=1)
print(f"  H={H_fit1:.3f}  C1={C1_fit1:.3f}  alpha={alpha_fit1:.3f}  (input: H={H}, C1={C1}, alpha={alpha})")

# ---- Estimate K(q): joint fit of H, C1, alpha ----
print(f"Estimating K(q) with hurst_fit_method='joint'...")
_, _, H_fit2, C1_fit2, alpha_fit2 = scaleinvariance.K_empirical(
    sims, scaling_method=SCALING_METHOD, hurst_fit_method='joint', min_sep=FIT_MIN, max_sep=FIT_MAX, axis=1)
print(f"  H={H_fit2:.3f}  C1={C1_fit2:.3f}  alpha={alpha_fit2:.3f}  (input: H={H}, C1={C1}, alpha={alpha})")

# ---- Plot ----
K_theoretical = scaleinvariance.K_analytic(q_values, C1, alpha)
K_fitted1     = scaleinvariance.K_analytic(q_values, C1_fit1, alpha_fit1)
K_fitted2     = scaleinvariance.K_analytic(q_values, C1_fit2, alpha_fit2)

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(q_values, K_theoretical, '-',  color='steelblue',   linewidth=2,
        label=f'Theoretical  H={H}, C₁={C1}, α={alpha}')
ax.plot(q_values, K_empirical,   'o',  color='gray',         markersize=5,
        label='Empirical K(q)')
ax.plot(q_values, K_fitted1,     '--', color='darkorange',   linewidth=2,
        label=f'Fit (H fixed)  H={H_fit1:.3f}, C₁={C1_fit1:.3f}, α={alpha_fit1:.3f}')
ax.plot(q_values, K_fitted2,     '--', color='mediumseagreen', linewidth=2,
        label=f'Fit (H free)   H={H_fit2:.3f}, C₁={C1_fit2:.3f}, α={alpha_fit2:.3f}')
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')

ax.set_xlabel('q')
ax.set_ylabel('K(q)')
ax.set_title(
    f'K(q) estimation:  H={H}, C₁={C1}, α={alpha}\n'
    f'{N_SIMS} sims × {SIM_SIZE:,} pts, fit=[{FIT_MIN},{FIT_MAX}], kernel={KERNEL_METHOD}'
)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nTest complete!")
