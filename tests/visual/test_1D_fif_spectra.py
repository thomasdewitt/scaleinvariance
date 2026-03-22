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
    n_sims = 300    # Number of simulations for averaging
    outer_scale = size/100

    print(f"\nParameters:")
    print(f"  Size: {size:,}")
    print(f"  H: {H}")
    print(f"  Alpha: {alpha}")
    print(f"  C1: {C1}")
    print(f"  Causal: {causal}")
    print(f"  Number of simulations: {n_sims}")
    print("  Noise usage: one explicit Levy-noise realization per simulation, reused across kernels")

    print(f"\nGenerating {n_sims} paired FIF simulations...")
    all_fif_data = []
    all_fif_spectral_data = []
    all_fif_uncorrected_data = []
    all_fif_spectral_odd_data = []

    for i in range(n_sims):
        if i % 50 == 0:
            print(f"  Realization {i+1}/{n_sims}...")

        levy_noise = scaleinvariance.simulation.FIF.extremal_levy(alpha, size=size)

        fif_field = scaleinvariance.FIF_1D(
            size,
            alpha,
            C1,
            H=H,
            causal=causal,
            levy_noise=levy_noise,
            kernel_construction_method_flux='LS2010',
            kernel_construction_method_observable='LS2010',
            outer_scale=outer_scale,
        )
        all_fif_data.append(fif_field)

        fif_field_spectral = scaleinvariance.FIF_1D(
            size,
            alpha,
            C1,
            H=H,
            causal=causal,
            levy_noise=levy_noise,
            kernel_construction_method_observable='LS2010_spectral',
            kernel_construction_method_flux='LS2010_spectral',
            outer_scale=outer_scale,
            periodic=True,
        )
        all_fif_spectral_data.append(fif_field_spectral)

        fif_field_uncorrected = scaleinvariance.FIF_1D(
            size,
            alpha,
            C1,
            H=H,
            causal=causal,
            levy_noise=levy_noise,
            kernel_construction_method_flux='naive',
            kernel_construction_method_observable='naive',
            outer_scale=outer_scale,
        )
        all_fif_uncorrected_data.append(fif_field_uncorrected)

        # Spectral odd observable kernel (non-causal, periodic)
        if not causal and H != 0:
            fif_field_spectral_odd = scaleinvariance.FIF_1D(
                size,
                alpha,
                C1,
                H=H,
                causal=False,
                levy_noise=levy_noise,
                kernel_construction_method_flux='LS2010_spectral',
                kernel_construction_method_observable='spectral_odd',
                outer_scale=outer_scale,
                periodic=True,
            )
            all_fif_spectral_odd_data.append(fif_field_spectral_odd)

    # Concatenate all simulations and analyze with spectral_hurst
    all_fif_data = np.vstack(all_fif_data)
    H_fif, H_err_fif, freqs, mean_fif_spectrum, fit_line = scaleinvariance.spectral_hurst(
        all_fif_data, return_fit=True, axis=1, min_wavelength=2.1)
    beta_fif = 2 * H_fif + 1

    all_fif_spectral_data = np.vstack(all_fif_spectral_data)
    H_fif_spectral, H_err_fif_spectral, freqs_spectral, mean_fif_spectral_spectrum, fit_line_spectral = scaleinvariance.spectral_hurst(
        all_fif_spectral_data, return_fit=True, axis=1, min_wavelength=2.1)
    beta_fif_spectral = 2 * H_fif_spectral + 1

    all_fif_uncorrected_data = np.vstack(all_fif_uncorrected_data)
    H_fif_uncorrected, H_err_fif_uncorrected, freqs_uncorrected, mean_fif_uncorrected_spectrum, fit_line_uncorrected = scaleinvariance.spectral_hurst(
        all_fif_uncorrected_data, return_fit=True, axis=1, min_wavelength=2.1)
    beta_fif_uncorrected = 2 * H_fif_uncorrected + 1

    has_odd = len(all_fif_spectral_odd_data) > 0
    if has_odd:
        all_fif_spectral_odd_data = np.vstack(all_fif_spectral_odd_data)
        H_fif_odd, H_err_fif_odd, freqs_odd, mean_fif_odd_spectrum, fit_line_odd = scaleinvariance.spectral_hurst(
            all_fif_spectral_odd_data, return_fit=True, axis=1, min_wavelength=2.1)
        beta_fif_odd = 2 * H_fif_odd + 1

    # Theoretical expectations
    theoretical_K2 = C1 * (2**alpha - 2) / (alpha - 1)  # K(q=2)
    theoretical_beta_fif = 2*H+1 - theoretical_K2

    plt.figure(figsize=(10, 8))

    # Create two subplots: raw spectra and normalized
    plt.subplot(2, 1, 1)

    # Plot raw averaged spectra
    plt.loglog(freqs, mean_fif_spectrum, 'b-', linewidth=2,
              label=f'FIF LS2010 (H={H_fif:.2f}, \u03b2={beta_fif:.2f})')
    plt.loglog(freqs_spectral, mean_fif_spectral_spectrum, '#2ca02c', linewidth=2,
              label=f'FIF spectral (H={H_fif_spectral:.2f}, \u03b2={beta_fif_spectral:.2f})')
    plt.loglog(freqs_uncorrected, mean_fif_uncorrected_spectrum, '#ff7f0e', linewidth=2,
              label=f'FIF naive (H={H_fif_uncorrected:.2f}, \u03b2={beta_fif_uncorrected:.2f})')
    if has_odd:
        plt.loglog(freqs_odd, mean_fif_odd_spectrum, '#d62728', linewidth=2,
                  label=f'FIF spectral_odd (H={H_fif_odd:.2f}, \u03b2={beta_fif_odd:.2f})')

    # Add theoretical reference lines
    ref_freq_range = (freqs > freqs[len(freqs)//8]) & (freqs < freqs[7*len(freqs)//8])
    ref_freqs = freqs[ref_freq_range]

    # FIF theoretical line
    ref_power_fif = mean_fif_spectrum[len(freqs)//4]
    ref_freq_fif = freqs[len(freqs)//4]
    theoretical_line_fif = ref_power_fif * (ref_freqs/ref_freq_fif)**(-theoretical_beta_fif)
    plt.loglog(ref_freqs, theoretical_line_fif, 'k--', linewidth=2,
              label=f'FIF theory (\u03b2={theoretical_beta_fif:.2f})')

    # fBm (non-intermittent) theoretical line
    theoretical_beta_fbm = 2 * H + 1
    theoretical_line_fbm = ref_power_fif * (ref_freqs/ref_freq_fif)**(-theoretical_beta_fbm)
    plt.loglog(ref_freqs, theoretical_line_fbm, 'k:', linewidth=0.5,
              label=f'Equivalent nonintermittent (\u03b2={theoretical_beta_fbm:.2f})')

    plt.xlabel('Frequency')
    plt.ylabel('Power Spectral Density')
    causal_str = "causal" if causal else "acausal"
    plt.title(f'FIF Raw Spectral Analysis (H={H}, \u03b1={alpha}, C\u2081={C1}, {causal_str})')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3, which='both')

    # Add outer scale vertical line (frequency = 1/outer_scale)
    plt.axvline(1/outer_scale, color='k', linestyle=':', linewidth=1.5, alpha=0.7)

    # Second subplot: Normalized by multifractal theory
    plt.subplot(2, 1, 2)

    # Normalize all spectra by middle frequency power to show relative deviations
    mid_idx = len(freqs) // 2
    mid_freq = freqs[mid_idx]
    mid_power_corrected = mean_fif_spectrum[mid_idx]
    mid_power_spectral = mean_fif_spectral_spectrum[mid_idx]
    mid_power_uncorrected = mean_fif_uncorrected_spectrum[mid_idx]

    # Normalize FIF spectra by their theoretical slope AND middle frequency power
    normalized_fif_spectrum = (mean_fif_spectrum * (freqs**(theoretical_beta_fif))) / (mid_power_corrected * (mid_freq**(theoretical_beta_fif)))
    normalized_fif_spectral_spectrum = (mean_fif_spectral_spectrum * (freqs_spectral**(theoretical_beta_fif))) / (mid_power_spectral * (mid_freq**(theoretical_beta_fif)))
    normalized_fif_uncorrected_spectrum = (mean_fif_uncorrected_spectrum * (freqs_uncorrected**(theoretical_beta_fif))) / (mid_power_uncorrected * (mid_freq**(theoretical_beta_fif)))

    # Also normalize a theoretical fBm spectrum for comparison
    theoretical_line_fbm_full = ref_power_fif * (freqs/ref_freq_fif)**(-theoretical_beta_fbm)
    normalized_fbm_theoretical = (theoretical_line_fbm_full * (freqs**(theoretical_beta_fif))) / (mid_power_corrected * (mid_freq**(theoretical_beta_fif)))

    plt.loglog(freqs, normalized_fif_spectrum, 'b-', linewidth=2,
              label='FIF LS2010 normalized')
    plt.loglog(freqs_spectral, normalized_fif_spectral_spectrum, '#2ca02c', linewidth=2,
              label='FIF spectral normalized')
    plt.loglog(freqs_uncorrected, normalized_fif_uncorrected_spectrum, '#ff7f0e', linewidth=2,
              label='FIF naive normalized')
    if has_odd:
        mid_power_odd = mean_fif_odd_spectrum[mid_idx]
        normalized_fif_odd_spectrum = (mean_fif_odd_spectrum * (freqs_odd**(theoretical_beta_fif))) / (mid_power_odd * (mid_freq**(theoretical_beta_fif)))
        plt.loglog(freqs_odd, normalized_fif_odd_spectrum, '#d62728', linewidth=2,
                  label='FIF spectral_odd normalized')
    plt.loglog(freqs, normalized_fbm_theoretical, 'k:', linewidth=0.5,
              label='fBm normalized by FIF theory')

    # Add flat reference line
    median_level = np.median(normalized_fif_spectrum[len(freqs)//4:3*len(freqs)//4])
    plt.loglog(freqs, np.full_like(freqs, median_level), 'k--', linewidth=1,
              label='Flat reference')

    plt.xlabel('Frequency')
    plt.ylabel('Power \u00d7 f^\u03b2 (normalized)')
    plt.title('Multifractal-Normalized Spectra (flat = perfect agreement with theory)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3, which='both')

    # Add outer scale vertical line (frequency = 1/outer_scale)
    plt.axvline(1/outer_scale, color='k', linestyle=':', linewidth=1.5, alpha=0.7)

    # Set y-limits to span 4 orders of magnitude, centered around 1.0 (normalized level)
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
    return freqs, mean_fif_spectrum

if __name__ == "__main__":
    # scaleinvariance.set_backend('numpy')
    print(f'Using backend "{scaleinvariance.get_backend()}"')
    freqs, fif_spectrum = main()
