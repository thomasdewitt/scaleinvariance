#!/usr/bin/env python3
"""
Kernel Comparison Plot

Plots 1D kernels from all construction methods (naive, LS2010, spectral, spectral_odd)
for visual comparison.

Panel 1: Full kernel (centered) — linear x, log y
Panel 2: Positive half only — loglog

Usage: python plot_kernels.py [exponent] [--causal] [--save-dir DIR]
Examples:
  python plot_kernels.py -0.3
  python plot_kernels.py -0.7 --causal
  python plot_kernels.py --save-dir /tmp/plots
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scaleinvariance.simulation.kernels import (
    create_kernel_naive, create_kernel_LS2010, create_kernel_spectral,
    create_kernel_spectral_odd,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Kernel Comparison Plot')
    parser.add_argument('exponent', nargs='?', type=float, default=-0.3, help='Power-law exponent (default: -0.3)')
    parser.add_argument('--causal', action='store_true', help='Use causal kernels')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save plots (default: show)')
    return parser.parse_args()


def main():
    args = parse_args()
    exponent = args.exponent
    causal = args.causal
    save_dir = args.save_dir

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    size = 1024
    outer_scale = size

    # LS2010 needs norm_ratio_exponent; use -H convention (exponent = -1+H → H = exponent+1)
    H_equiv = exponent + 1.0
    norm_ratio_exp = -H_equiv

    print(f"Kernel comparison: size={size}, exponent={exponent}, causal={causal}")

    # Build kernels
    k_naive = np.array(create_kernel_naive(size, exponent, causal=causal, outer_scale=outer_scale))
    k_ls2010 = np.array(create_kernel_LS2010(size, exponent, norm_ratio_exp, causal=causal, outer_scale=outer_scale))
    k_spectral = np.array(create_kernel_spectral(size, exponent, causal=False, outer_scale=outer_scale))

    # Odd kernel (non-causal only)
    k_odd = np.array(create_kernel_spectral_odd(size, exponent, causal=False, outer_scale=outer_scale))

    # x-axis: centered indices
    x_full = np.arange(size) - size // 2

    # Colors
    c_naive = '#2196F3'
    c_ls2010 = '#9C27B0'
    c_spectral = '#FF9800'
    c_odd = '#d62728'

    causal_str = "causal" if causal else "acausal"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel 1: Full kernel, linear x, log y
    ax = axes[0]
    ax.semilogy(x_full, np.abs(k_naive), color=c_naive, lw=1.5, label='naive', alpha=0.85)
    ax.semilogy(x_full, np.abs(k_ls2010), color=c_ls2010, lw=1.5, label='LS2010', alpha=0.85)
    ax.semilogy(x_full, np.abs(k_spectral), color=c_spectral, lw=1.5, label='spectral', alpha=0.85)
    ax.semilogy(x_full, np.abs(k_odd), color=c_odd, lw=1.5, label='spectral_odd', alpha=0.85)
    ax.set_xlabel('Index (centered)')
    ax.set_ylabel('|K(x)|')
    ax.set_title(f'Full kernel ({causal_str})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, which='both')

    # Panel 2: Positive half, loglog
    ax = axes[1]
    mid = size // 2
    x_pos = np.arange(1, mid)  # skip center (x=0)

    # For naive, center is at half-integer, so positive half starts at mid
    k_naive_pos = np.abs(k_naive[mid:mid + len(x_pos)])
    k_ls2010_pos = np.abs(k_ls2010[mid:mid + len(x_pos)])
    k_spectral_pos = np.abs(k_spectral[mid:mid + len(x_pos)])
    k_odd_pos = np.abs(k_odd[mid:mid + len(x_pos)])

    ax.loglog(x_pos, k_naive_pos, color=c_naive, lw=1.5, label='naive', alpha=0.85)
    ax.loglog(x_pos, k_ls2010_pos, color=c_ls2010, lw=1.5, label='LS2010', alpha=0.85)
    ax.loglog(x_pos, k_spectral_pos, color=c_spectral, lw=1.5, label='spectral', alpha=0.85)
    ax.loglog(x_pos, k_odd_pos, color=c_odd, lw=1.5, label='spectral_odd', alpha=0.85)

    # Reference power law
    ref_x = x_pos[5:]
    ref_y = k_naive_pos[len(x_pos) // 4] * (ref_x / ref_x[len(ref_x) // 4 - 5]) ** exponent
    ax.loglog(ref_x, ref_y, 'k--', lw=1, alpha=0.5, label=f'x^{exponent:.2f}')

    ax.set_xlabel('Lag x')
    ax.set_ylabel('|K(x)|')
    ax.set_title(f'Positive half (exp={exponent})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, which='both')

    fig.suptitle(f'Kernel methods comparison  |  exponent = {exponent}, {causal_str}', fontsize=11, y=1.01)
    plt.tight_layout()

    if save_dir is not None:
        filename = f'kernels_exp{exponent}_{causal_str}.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
