#!/usr/bin/env python3
"""
FIF Structure Function Analysis Test Script

Tests FIF_1D simulation from scaleinvariance package and compares structure functions
against theoretical expectations including multifractal scaling corrections.

Usage: python test_1D_fif_structure_function.py [H] [C1] [alpha] [causal] [--save-dir DIR]
Examples:
  python test_1D_fif_structure_function.py 0.7
  python test_1D_fif_structure_function.py 0.7 0.08 1.8
  python test_1D_fif_structure_function.py 0.5 0.1 1.6 True
  python test_1D_fif_structure_function.py 0.7 0.15 1.8 False --save-dir /tmp/plots
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scaleinvariance


def parse_args():
    parser = argparse.ArgumentParser(description='FIF Structure Function Analysis Test')
    parser.add_argument('H', nargs='?', type=float, default=0.3, help='Hurst exponent (default: 0.3)')
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
    print("FIF STRUCTURE FUNCTION ANALYSIS TEST")
    print("=" * 50)

    # Parameters
    size = 2**20
    n_sims = 10    # Number of simulations for averaging
    outer_scale = size

    print(f"\nParameters:")
    print(f"  Size: {size:,}")
    print(f"  H: {H}")
    print(f"  Alpha: {alpha}")
    print(f"  C1: {C1}")
    print(f"  Causal: {causal}")
    print(f"  Number of simulations: {n_sims}")

    # Generate corrected FIF simulations
    print(f"\nGenerating {n_sims} corrected FIF simulations...")
    all_fif_data = []

    for i in range(n_sims):
        if i%2 == 0: print(f"  Simulation {i+1}/{n_sims}...",)

        # Generate corrected FIF simulation
        fif_field = scaleinvariance.FIF_1D(size, alpha, C1, H=H, causal=causal,
                                         kernel_construction_method_flux='LS2010',
                                         kernel_construction_method_observable='LS2010',
                                         outer_scale=outer_scale)
        all_fif_data.append(fif_field)

    # Concatenate all simulations and analyze with structure_function_analysis
    all_fif_data = np.vstack(all_fif_data)
    lags, mean_fif_sf = scaleinvariance.structure_function_analysis(
        all_fif_data, order=2, axis=1)

    # Generate uncorrected FIF simulations for comparison
    print(f"\nGenerating {n_sims} uncorrected FIF simulations...")
    all_fif_uncorrected_data = []

    for i in range(n_sims):
        if i%2 == 0: print(f"  Uncorrected simulation {i+1}/{n_sims}...",)

        # Generate uncorrected FIF simulation
        fif_field_uncorrected = scaleinvariance.FIF_1D(size, alpha, C1, H=H, causal=causal,
                                                      kernel_construction_method_flux='naive',
                                                      kernel_construction_method_observable='naive',
                                                      outer_scale=outer_scale)
        all_fif_uncorrected_data.append(fif_field_uncorrected)

    # Concatenate uncorrected simulations and analyze
    all_fif_uncorrected_data = np.vstack(all_fif_uncorrected_data)
    lags_uncorrected, mean_fif_uncorrected_sf = scaleinvariance.structure_function_analysis(
        all_fif_uncorrected_data, order=2, axis=1)

    # Generate spectral_odd FIF simulations (non-causal only, H != 0 only)
    has_odd = not causal and H != 0
    if has_odd:
        print(f"\nGenerating {n_sims} spectral_odd FIF simulations...")
        all_fif_odd_data = []
        for i in range(n_sims):
            if i%2 == 0: print(f"  Spectral_odd simulation {i+1}/{n_sims}...",)
            fif_field_odd = scaleinvariance.FIF_1D(size, alpha, C1, H=H, causal=False,
                                                   kernel_construction_method_flux='LS2010_spectral',
                                                   kernel_construction_method_observable='spectral_odd',
                                                   outer_scale=outer_scale, periodic=True)
            all_fif_odd_data.append(fif_field_odd)
        all_fif_odd_data = np.vstack(all_fif_odd_data)
        lags_odd, mean_fif_odd_sf = scaleinvariance.structure_function_analysis(
            all_fif_odd_data, order=2, axis=1)

    # Generate spectral (even) FIF simulations for comparison (non-causal only, H != 0 only)
    has_spectral = not causal and H != 0
    if has_spectral:
        print(f"\nGenerating {n_sims} spectral (even) FIF simulations...")
        all_fif_spectral_data = []
        for i in range(n_sims):
            if i%2 == 0: print(f"  Spectral simulation {i+1}/{n_sims}...",)
            fif_field_spectral = scaleinvariance.FIF_1D(size, alpha, C1, H=H, causal=False,
                                                        kernel_construction_method_flux='LS2010_spectral',
                                                        kernel_construction_method_observable='LS2010_spectral',
                                                        outer_scale=outer_scale, periodic=True)
            all_fif_spectral_data.append(fif_field_spectral)
        all_fif_spectral_data = np.vstack(all_fif_spectral_data)
        lags_spectral, mean_fif_spectral_sf = scaleinvariance.structure_function_analysis(
            all_fif_spectral_data, order=2, axis=1)


    # Theoretical expectations
    theoretical_K2 = C1 * (2**alpha - 2) / (alpha - 1)  # K(q=2)
    theoretical_zeta_fif = 2*H - theoretical_K2


    plt.figure(figsize=(10, 8))

    # Create two subplots: raw structure functions and normalized
    plt.subplot(2, 1, 1)

    # Plot raw averaged structure functions
    plt.loglog(lags, mean_fif_sf, 'b-', linewidth=2,
              label='FIF LS2010')
    plt.loglog(lags_uncorrected, mean_fif_uncorrected_sf, '#ff7f0e', linewidth=2,
              label='FIF naive')
    if has_spectral:
        plt.loglog(lags_spectral, mean_fif_spectral_sf, '#2ca02c', linewidth=2,
                  label='FIF spectral')
    if has_odd:
        plt.loglog(lags_odd, mean_fif_odd_sf, '#d62728', linewidth=2,
                  label='FIF spectral_odd')

    # Add theoretical reference lines
    ref_lag_range = (lags < lags[7*len(lags)//8])
    ref_lags = lags[ref_lag_range]

    # FIF theoretical line
    ref_sf_fif = mean_fif_sf[len(lags)//4]
    ref_lag_fif = lags[len(lags)//4]
    theoretical_line_fif = ref_sf_fif * (ref_lags/ref_lag_fif)**(theoretical_zeta_fif)
    plt.loglog(ref_lags, theoretical_line_fif, 'k--', linewidth=2,
              label=f'FIF theory (\u03b6\u2082={theoretical_zeta_fif:.2f})')

    # fBm (non-intermittent) theoretical line
    theoretical_zeta_fbm = 2 * H
    theoretical_line_fbm = ref_sf_fif * (ref_lags/ref_lag_fif)**(theoretical_zeta_fbm)
    plt.loglog(ref_lags, theoretical_line_fbm, 'k:', linewidth=0.5,
              label=f'Equivalent nonintermittent SF (\u03b6\u2082={theoretical_zeta_fbm:.2f})')

    plt.xlabel('Lag')
    plt.ylabel('Structure Function S\u2082(r)')
    causal_str = "causal" if causal else "acausal"
    plt.title(f'FIF Raw Structure Function Analysis (H={H}, \u03b1={alpha}, C\u2081={C1}, {causal_str})')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3, which='both')

    # Add outer scale vertical line
    plt.axvline(outer_scale, color='k', linestyle=':', linewidth=1.5, alpha=0.7)

    # Second subplot: Normalized by multifractal theory
    plt.subplot(2, 1, 2)

    # Normalize all structure functions by middle lag power to show relative deviations
    mid_idx = len(lags) // 2
    mid_lag = lags[mid_idx]
    mid_sf_corrected = mean_fif_sf[mid_idx]
    mid_sf_uncorrected = mean_fif_uncorrected_sf[mid_idx]

    # Normalize FIF structure functions by their theoretical slope AND middle lag power
    normalized_fif_sf = (mean_fif_sf / (lags**(theoretical_zeta_fif))) / (mid_sf_corrected / (mid_lag**(theoretical_zeta_fif)))
    normalized_fif_uncorrected_sf = (mean_fif_uncorrected_sf / (lags_uncorrected**(theoretical_zeta_fif))) / (mid_sf_uncorrected / (mid_lag**(theoretical_zeta_fif)))

    # Also normalize a theoretical fBm structure function for comparison
    theoretical_line_fbm_full = ref_sf_fif * (lags/ref_lag_fif)**(theoretical_zeta_fbm)
    normalized_fbm_theoretical = (theoretical_line_fbm_full / (lags**(theoretical_zeta_fif))) / (mid_sf_corrected / (mid_lag**(theoretical_zeta_fif)))

    plt.loglog(lags, normalized_fif_sf, 'b-', linewidth=2,
              label='FIF LS2010 normalized')
    plt.loglog(lags_uncorrected, normalized_fif_uncorrected_sf, '#ff7f0e', linewidth=2,
              label='FIF naive normalized')

    if has_spectral:
        mid_sf_spectral = mean_fif_spectral_sf[mid_idx]
        normalized_fif_spectral_sf = (mean_fif_spectral_sf / (lags_spectral**(theoretical_zeta_fif))) / (mid_sf_spectral / (mid_lag**(theoretical_zeta_fif)))
        plt.loglog(lags_spectral, normalized_fif_spectral_sf, '#2ca02c', linewidth=2,
                  label='FIF spectral normalized')

    if has_odd:
        mid_sf_odd = mean_fif_odd_sf[mid_idx]
        normalized_fif_odd_sf = (mean_fif_odd_sf / (lags_odd**(theoretical_zeta_fif))) / (mid_sf_odd / (mid_lag**(theoretical_zeta_fif)))
        plt.loglog(lags_odd, normalized_fif_odd_sf, '#d62728', linewidth=2,
                  label='FIF spectral_odd normalized')

    plt.loglog(lags, normalized_fbm_theoretical, 'k:', linewidth=0.5,
              label='fBm normalized by FIF theory')

    # Add flat reference line
    median_level = np.median(normalized_fif_sf[len(lags)//4:3*len(lags)//4])
    plt.loglog(lags, np.full_like(lags, median_level), 'k--', linewidth=1,
              label='Flat reference')


    plt.xlabel('Lag')
    plt.ylabel('S\u2082(r) / r^\u03b6\u2082 (normalized)')
    plt.title('Multifractal-Normalized Structure Functions (flat = perfect agreement with theory)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3, which='both')

    # Add outer scale vertical line
    plt.axvline(outer_scale, color='k', linestyle=':', linewidth=1.5, alpha=0.7)

    # Set y-limits to span 4 orders of magnitude, centered around 1.0 (normalized level)
    plt.ylim(1e-2, 1e2)

    plt.tight_layout()

    if save_dir is not None:
        filename = f'structure_function_H{H}_C1{C1}_alpha{alpha}_{causal_str}.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {filepath}")
    else:
        plt.show()

    print(f"\nTest complete!")
    return lags, mean_fif_sf

if __name__ == "__main__":
    lags, fif_sf = main()
