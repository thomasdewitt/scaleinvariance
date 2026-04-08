#!/usr/bin/env python3
"""
FIF Spectral Analysis Test Script

Tests FIF_1D simulation from scaleinvariance package and compares spectra
against theoretical expectations including multifractal scaling corrections.

Usage: python test_1D_fif_spectra.py [H] [C1] [alpha] [causal] [--save-dir DIR]
Examples:
  python test_1D_fif_spectra.py 0.7
  python test_1D_fif_spectra.py 0.7 0.08 1.8
  python test_1D_fif_spectra.py 0.5 0.1 1.6 True
  python test_1D_fif_spectra.py 0.7 0.15 1.8 False --save-dir /tmp/plots
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scaleinvariance


def parse_args():
    parser = argparse.ArgumentParser(description='FIF Spectral Analysis Test')
    parser.add_argument('H', nargs='?', type=float, default=0.0, help='Hurst exponent (default: 0.0)')
    parser.add_argument('C1', nargs='?', type=float, default=0.15, help='C1 parameter (default: 0.15)')
    parser.add_argument('alpha', nargs='?', type=float, default=1.8, help='Alpha parameter (default: 1.8)')
    parser.add_argument('causal', nargs='?', type=str, default='False', help='Causal (default: False)')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save plots (default: show)')
    return parser.parse_args()


FLUX_METHODS = [
    ('LS2010',           'b',       'LS2010'),
    ('LS2010_highorder', '#9467bd', 'LS2010 highorder'),
    ('naive',            '#ff7f0e', 'naive'),
]


def main():
    args = parse_args()
    H = args.H
    C1 = args.C1
    alpha = args.alpha
    causal = args.causal.lower() in ['true', '1', 'yes', 't']
    save_dir = args.save_dir

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    print("=" * 50)
    print("FIF SPECTRAL ANALYSIS TEST")
    print("=" * 50)

    # Parameters
    size = 2**18
    n_sims = 300
    outer_scale = size / 100

    # Observable kernel: spectral unless causal
    if causal:
        obs_method = 'LS2010'
    else:
        obs_method = 'spectral'

    print(f"\nParameters:")
    print(f"  Size: {size:,}")
    print(f"  H: {H}")
    print(f"  Alpha: {alpha}")
    print(f"  C1: {C1}")
    print(f"  Causal: {causal}")
    print(f"  Observable kernel: {obs_method}")
    print(f"  Number of simulations: {n_sims}")
    print("  Noise usage: one explicit Levy-noise realization per simulation, reused across kernels")

    print(f"\nGenerating {n_sims} paired FIF simulations...")

    # Collect data for each flux method
    all_data = {method: [] for method, _, _ in FLUX_METHODS}

    for i in range(n_sims):
        if i % 50 == 0:
            print(f"  Realization {i+1}/{n_sims}...")

        levy_noise = scaleinvariance.simulation.FIF.extremal_levy(alpha, size=size)

        for method, _, _ in FLUX_METHODS:
            field = scaleinvariance.FIF_1D(
                size, alpha, C1, H=H, causal=causal,
                levy_noise=levy_noise,
                kernel_construction_method_flux=method,
                kernel_construction_method_observable=obs_method,
                outer_scale=outer_scale,
                periodic=(obs_method == 'spectral'),
            )
            all_data[method].append(field)

    # Compute spectra
    results = {}
    for method, color, label in FLUX_METHODS:
        stacked = np.vstack(all_data[method])
        H_est, H_err, freqs, spectrum, fit_line = scaleinvariance.spectral_hurst(
            stacked, return_fit=True, axis=1, min_wavelength=2.1)
        beta = 2 * H_est + 1
        results[method] = dict(
            H_est=H_est, H_err=H_err, freqs=freqs,
            spectrum=spectrum, fit_line=fit_line, beta=beta,
            color=color, label=label,
        )

    # Theoretical expectations
    theoretical_K2 = C1 * (2**alpha - 2) / (alpha - 1)
    theoretical_beta_fif = 2 * H + 1 - theoretical_K2

    plt.figure(figsize=(10, 8))

    # --- Top: raw spectra ---
    plt.subplot(2, 1, 1)

    for method, _, _ in FLUX_METHODS:
        r = results[method]
        plt.loglog(r['freqs'], r['spectrum'], color=r['color'], linewidth=2,
                   label=f"FIF {r['label']} (H={r['H_est']:.2f}, \u03b2={r['beta']:.2f})")

    # Theoretical reference lines
    ref_r = results['LS2010']
    ref_freqs_mask = (ref_r['freqs'] > ref_r['freqs'][len(ref_r['freqs'])//8]) & \
                     (ref_r['freqs'] < ref_r['freqs'][7*len(ref_r['freqs'])//8])
    ref_freqs = ref_r['freqs'][ref_freqs_mask]
    ref_power = ref_r['spectrum'][len(ref_r['freqs'])//4]
    ref_freq = ref_r['freqs'][len(ref_r['freqs'])//4]

    theoretical_line_fif = ref_power * (ref_freqs / ref_freq) ** (-theoretical_beta_fif)
    plt.loglog(ref_freqs, theoretical_line_fif, 'k--', linewidth=2,
               label=f'FIF theory (\u03b2={theoretical_beta_fif:.2f})')

    theoretical_beta_fbm = 2 * H + 1
    theoretical_line_fbm = ref_power * (ref_freqs / ref_freq) ** (-theoretical_beta_fbm)
    plt.loglog(ref_freqs, theoretical_line_fbm, 'k:', linewidth=0.5,
               label=f'Equivalent nonintermittent (\u03b2={theoretical_beta_fbm:.2f})')

    plt.xlabel('Frequency')
    plt.ylabel('Power Spectral Density')
    causal_str = "causal" if causal else "acausal"
    plt.title(f'FIF Raw Spectral Analysis (H={H}, \u03b1={alpha}, C\u2081={C1}, {causal_str})')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3, which='both')
    plt.axvline(1 / outer_scale, color='k', linestyle=':', linewidth=1.5, alpha=0.7)

    # --- Bottom: normalized spectra ---
    plt.subplot(2, 1, 2)

    for method, _, _ in FLUX_METHODS:
        r = results[method]
        mid_idx = len(r['freqs']) // 2
        mid_freq = r['freqs'][mid_idx]
        mid_power = r['spectrum'][mid_idx]
        normalized = (r['spectrum'] * r['freqs'] ** theoretical_beta_fif) / \
                     (mid_power * mid_freq ** theoretical_beta_fif)
        plt.loglog(r['freqs'], normalized, color=r['color'], linewidth=2,
                   label=f"FIF {r['label']} normalized")

    # fBm reference
    ref_freqs_full = ref_r['freqs']
    mid_idx = len(ref_freqs_full) // 2
    mid_freq = ref_freqs_full[mid_idx]
    mid_power = ref_r['spectrum'][mid_idx]
    fbm_line = ref_power * (ref_freqs_full / ref_freq) ** (-theoretical_beta_fbm)
    normalized_fbm = (fbm_line * ref_freqs_full ** theoretical_beta_fif) / \
                     (mid_power * mid_freq ** theoretical_beta_fif)
    plt.loglog(ref_freqs_full, normalized_fbm, 'k:', linewidth=0.5,
               label='fBm normalized by FIF theory')

    # Flat reference
    ref_norm = (ref_r['spectrum'] * ref_r['freqs'] ** theoretical_beta_fif) / \
               (mid_power * mid_freq ** theoretical_beta_fif)
    median_level = np.median(ref_norm[len(ref_freqs_full)//4:3*len(ref_freqs_full)//4])
    plt.loglog(ref_freqs_full, np.full_like(ref_freqs_full, median_level), 'k--', linewidth=1,
               label='Flat reference')

    plt.xlabel('Frequency')
    plt.ylabel('Power \u00d7 f^\u03b2 (normalized)')
    plt.title('Multifractal-Normalized Spectra (flat = perfect agreement with theory)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3, which='both')
    plt.axvline(1 / outer_scale, color='k', linestyle=':', linewidth=1.5, alpha=0.7)
    plt.ylim(1e-2, 1e2)

    plt.tight_layout()

    if save_dir is not None:
        filename = f'spectra_H{H}_C1{C1}_alpha{alpha}_{causal_str}.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {filepath}")
    else:
        plt.show()

    print(f"\nTest complete!")
    return results


if __name__ == "__main__":
    print(f'Using backend "{scaleinvariance.get_backend()}"')
    results = main()
