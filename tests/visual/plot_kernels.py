#!/usr/bin/env python3
"""
Kernel Comparison Plot

Plots 1D kernels from all construction methods (naive, LS2010, spectral) for visual comparison.

Panel 1: Full kernel (centered) — linear x, log y
Panel 2: Positive half only — loglog

Usage: python plot_kernels.py [exponent] [--causal]
Examples:
  python plot_kernels.py -0.3
  python plot_kernels.py -0.7 --causal
  python plot_kernels.py        # defaults to exponent=-0.3, acausal
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scaleinvariance.simulation.kernels import (
    create_kernel_naive, create_kernel_LS2010, create_kernel_spectral
)

def main():
    # Parse arguments
    exponent = -0.3
    causal = False

    args = [a for a in sys.argv[1:] if a != '--causal']
    if '--causal' in sys.argv:
        causal = True
    if args:
        try:
            exponent = float(args[0])
        except ValueError:
            print("Error: exponent must be a number")
            sys.exit(1)

    size = 1024
    outer_scale = size

    # LS2010 needs norm_ratio_exponent; use -H convention (exponent = -1+H → H = exponent+1)
    H_equiv = exponent + 1.0
    norm_ratio_exp = -H_equiv

    print(f"Kernel comparison: size={size}, exponent={exponent}, causal={causal}")

    # Build kernels
    k_naive = np.array(create_kernel_naive(size, exponent, causal=causal, outer_scale=outer_scale))
    k_ls2010 = np.array(create_kernel_LS2010(size, exponent, norm_ratio_exp, causal=causal, outer_scale=outer_scale))
    k_spectral = np.array(create_kernel_spectral(size, exponent, causal=causal, outer_scale=outer_scale))

    # x-axis: centered indices
    x_full = np.arange(size) - size // 2

    # Colors
    c_naive = '#2196F3'
    c_ls2010 = '#9C27B0'
    c_spectral = '#FF9800'

    causal_str = "causal" if causal else "acausal"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel 1: Full kernel, linear x, log y
    ax = axes[0]
    ax.semilogy(x_full, np.abs(k_naive), color=c_naive, lw=1.5, label='naive', alpha=0.85)
    ax.semilogy(x_full, np.abs(k_ls2010), color=c_ls2010, lw=1.5, label='LS2010', alpha=0.85)
    ax.semilogy(x_full, np.abs(k_spectral), color=c_spectral, lw=1.5, label='spectral', alpha=0.85)
    ax.set_xlabel('Index (centered)')
    ax.set_ylabel('|K(x)|')
    ax.set_title(f'Full kernel ({causal_str})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, which='both')

    # Panel 2: Positive half, loglog
    ax = axes[1]
    mid = size // 2
    x_pos = np.arange(1, mid)  # skip center (x=0)

    # For naive, center is at half-integer, so positive half starts at mid
    k_naive_pos = np.abs(k_naive[mid:mid + len(x_pos)])
    k_ls2010_pos = np.abs(k_ls2010[mid:mid + len(x_pos)])
    k_spectral_pos = np.abs(k_spectral[mid:mid + len(x_pos)])

    ax.loglog(x_pos, k_naive_pos, color=c_naive, lw=1.5, label='naive', alpha=0.85)
    ax.loglog(x_pos, k_ls2010_pos, color=c_ls2010, lw=1.5, label='LS2010', alpha=0.85)
    ax.loglog(x_pos, k_spectral_pos, color=c_spectral, lw=1.5, label='spectral', alpha=0.85)

    # Reference power law
    ref_x = x_pos[5:]
    ref_y = k_naive_pos[len(x_pos) // 4] * (ref_x / ref_x[len(ref_x) // 4 - 5]) ** exponent
    ax.loglog(ref_x, ref_y, 'k--', lw=1, alpha=0.5, label=f'x^{exponent:.2f}')

    ax.set_xlabel('Lag x')
    ax.set_ylabel('|K(x)|')
    ax.set_title(f'Positive half (exp={exponent})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, which='both')

    fig.suptitle(f'Kernel methods comparison  |  exponent = {exponent}, {causal_str}', fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
