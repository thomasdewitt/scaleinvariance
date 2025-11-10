#!/usr/bin/env python3
"""
Compare kernel construction methods visually.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from scaleinvariance.simulation.FIF import create_kernel_LS2010, create_kernel_naive

# Parameters
size = 2**12
alpha = 1.8
C1 = 0.15
H = 0.3
outer_scale = size / 100

# Calculate kernel parameters
alpha_prime = 1.0 / (1.0 - 1.0/alpha)
flux_exponent = -1.0 / alpha_prime
flux_norm_ratio_exp = -1.0 / alpha
flux_final_power = 1.0 / (alpha - 1.0)

H_exponent = -1.0 + H
H_norm_ratio_exp = -H

# Create kernels
print("Creating LS2010 flux kernel...")
kernel1_ls = create_kernel_LS2010(size, flux_exponent, flux_norm_ratio_exp,
                                  causal=False, outer_scale=outer_scale,
                                  outer_scale_width_factor=2.0,
                                  final_power=flux_final_power)

print("Creating naive flux kernel...")
kernel1_naive = create_kernel_naive(size, flux_exponent, causal=False,
                                   outer_scale=outer_scale,
                                   outer_scale_width_factor=2.0)

print("Creating LS2010 H kernel...")
kernel2_ls = create_kernel_LS2010(size, H_exponent, H_norm_ratio_exp,
                                  causal=False, outer_scale=outer_scale,
                                  outer_scale_width_factor=2.0,
                                  final_power=None)

print("Creating naive H kernel...")
kernel2_naive = create_kernel_naive(size, H_exponent, causal=False,
                                   outer_scale=outer_scale,
                                   outer_scale_width_factor=2.0)

# Convert to numpy
kernel1_ls = kernel1_ls.cpu().numpy()
kernel1_naive = kernel1_naive.cpu().numpy()
kernel2_ls = kernel2_ls.cpu().numpy()
kernel2_naive = kernel2_naive.cpu().numpy()

# Create distance array for plotting
distance_ls = np.arange(-(size - 1), size, 2)
distance_naive = np.arange(-size//2, size//2)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Flux kernel comparison
ax = axes[0, 0]
ax.loglog(np.abs(distance_ls), kernel1_ls, 'b-', linewidth=2, label='LS2010', alpha=0.7)
ax.loglog(np.abs(distance_naive), kernel1_naive, 'r--', linewidth=1.5, label='Naive', alpha=0.7)
ax.axvline(outer_scale, color='k', linestyle=':', linewidth=1.5, alpha=0.7, label='Outer scale')
ax.set_xlabel('Distance')
ax.set_ylabel('Kernel value')
ax.set_title('Flux Kernel Comparison')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

# Flux kernel ratio
ax = axes[0, 1]
# Interpolate naive to LS2010 grid for comparison
from scipy.interpolate import interp1d
interp_naive = interp1d(np.abs(distance_naive), kernel1_naive, bounds_error=False, fill_value=0)
kernel1_naive_interp = interp_naive(np.abs(distance_ls))
ratio = kernel1_ls / (kernel1_naive_interp + 1e-100)
ax.semilogx(np.abs(distance_ls), ratio, 'g-', linewidth=1.5)
ax.axvline(outer_scale, color='k', linestyle=':', linewidth=1.5, alpha=0.7)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('Distance')
ax.set_ylabel('LS2010 / Naive')
ax.set_title('Flux Kernel Ratio')
ax.grid(True, alpha=0.3)

# H kernel comparison
ax = axes[1, 0]
ax.loglog(np.abs(distance_ls), np.abs(kernel2_ls), 'b-', linewidth=2, label='LS2010', alpha=0.7)
ax.loglog(np.abs(distance_naive), np.abs(kernel2_naive), 'r--', linewidth=1.5, label='Naive', alpha=0.7)
ax.axvline(outer_scale, color='k', linestyle=':', linewidth=1.5, alpha=0.7, label='Outer scale')
ax.set_xlabel('Distance')
ax.set_ylabel('|Kernel value|')
ax.set_title('H Kernel Comparison')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

# H kernel ratio
ax = axes[1, 1]
interp_naive_h = interp1d(np.abs(distance_naive), kernel2_naive, bounds_error=False, fill_value=0)
kernel2_naive_interp = interp_naive_h(np.abs(distance_ls))
ratio_h = kernel2_ls / (kernel2_naive_interp + 1e-100)
ax.semilogx(np.abs(distance_ls), ratio_h, 'g-', linewidth=1.5)
ax.axvline(outer_scale, color='k', linestyle=':', linewidth=1.5, alpha=0.7)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('Distance')
ax.set_ylabel('LS2010 / Naive')
ax.set_title('H Kernel Ratio')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nDone!")
