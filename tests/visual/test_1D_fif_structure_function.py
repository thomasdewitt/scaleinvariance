#!/usr/bin/env python3
"""
FIF Structure Function Analysis Test Script

Tests FIF_1D simulation from scaleinvariance package and compares structure functions
against theoretical expectations including multifractal scaling corrections.

Usage: python test_1D_fif_structure_function.py [H] [C1] [alpha] [causal]
Examples:
  python test_1D_fif_structure_function.py 0.7
  python test_1D_fif_structure_function.py 0.7 0.08 1.8
  python test_1D_fif_structure_function.py 0.5 0.1 1.6 True
  python test_1D_fif_structure_function.py 0.7 0.15 1.8 False
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import scaleinvariance

def main():
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            H = float(sys.argv[1])
        except ValueError:
            print("Error: H must be a number")
            sys.exit(1)
    else:
        H = 0.3  # Default H value

    if len(sys.argv) > 2:
        try:
            C1 = float(sys.argv[2])
        except ValueError:
            print("Error: C1 must be a number")
            sys.exit(1)
    else:
        C1 = 0.15  # Default C1 value

    if len(sys.argv) > 3:
        try:
            alpha = float(sys.argv[3])
        except ValueError:
            print("Error: alpha must be a number")
            sys.exit(1)
    else:
        alpha = 1.8  # Default alpha value

    if len(sys.argv) > 4:
        try:
            causal = sys.argv[4].lower() in ['true', '1', 'yes', 't']
        except ValueError:
            print("Error: causal must be True/False")
            sys.exit(1)
    else:
        causal = False  # Default causal value

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
                                         kernel_construction_method='LS2010', outer_scale=outer_scale)
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
                                                      kernel_construction_method='naive', outer_scale=outer_scale)
        all_fif_uncorrected_data.append(fif_field_uncorrected)

    # Concatenate uncorrected simulations and analyze
    all_fif_uncorrected_data = np.vstack(all_fif_uncorrected_data)
    lags_uncorrected, mean_fif_uncorrected_sf = scaleinvariance.structure_function_analysis(
        all_fif_uncorrected_data, order=2, axis=1)


    # Theoretical expectations
    theoretical_K2 = C1 * (2**alpha - 2) / (alpha - 1)  # K(q=2)
    theoretical_zeta_fif = 2*H - theoretical_K2


    plt.figure(figsize=(10, 8))

    # Create two subplots: raw structure functions and normalized
    plt.subplot(2, 1, 1)

    # Plot raw averaged structure functions
    plt.loglog(lags, mean_fif_sf, 'b-', linewidth=2,
              label='FIF corrected')
    plt.loglog(lags_uncorrected, mean_fif_uncorrected_sf, 'orange', linewidth=2,
              label='FIF uncorrected')

    # Add theoretical reference lines
    ref_lag_range = (lags < lags[7*len(lags)//8])
    ref_lags = lags[ref_lag_range]

    # FIF theoretical line
    ref_sf_fif = mean_fif_sf[len(lags)//4]
    ref_lag_fif = lags[len(lags)//4]
    theoretical_line_fif = ref_sf_fif * (ref_lags/ref_lag_fif)**(theoretical_zeta_fif)
    plt.loglog(ref_lags, theoretical_line_fif, 'g--', linewidth=2,
              label=f'FIF theory (ζ₂={theoretical_zeta_fif:.2f})')

    # fBm (non-intermittent) theoretical line
    theoretical_zeta_fbm = 2 * H
    theoretical_line_fbm = ref_sf_fif * (ref_lags/ref_lag_fif)**(theoretical_zeta_fbm)
    plt.loglog(ref_lags, theoretical_line_fbm, 'r--', linewidth=0.5,
              label=f'Equivalent nonintermittent SF (ζ₂={theoretical_zeta_fbm:.2f})')

    plt.xlabel('Lag')
    plt.ylabel('Structure Function S₂(r)')
    causal_str = "causal" if causal else "acausal"
    plt.title(f'FIF Raw Structure Function Analysis (H={H}, α={alpha}, C₁={C1}, {causal_str})')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')

    # Add outer scale vertical line
    plt.axvline(outer_scale, color='k', linestyle=':', linewidth=1.5, alpha=0.7, label='Outer scale')

    ylims_first = plt.ylim()

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
              label='FIF corrected normalized')
    plt.loglog(lags_uncorrected, normalized_fif_uncorrected_sf, 'orange', linewidth=2,
              label='FIF uncorrected normalized')
    plt.loglog(lags, normalized_fbm_theoretical, 'r--', linewidth=0.5,
              label='fBm normalized by FIF theory')

    # Add flat reference line
    median_level = np.median(normalized_fif_sf[len(lags)//4:3*len(lags)//4])
    plt.loglog(lags, np.full_like(lags, median_level), 'g--', linewidth=1,
              label='Flat reference')


    plt.xlabel('Lag')
    plt.ylabel('S₂(r) / r^ζ₂ (normalized)')
    plt.title('Multifractal-Normalized Structure Functions (flat = perfect agreement with theory)')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')

    # Add outer scale vertical line
    plt.axvline(outer_scale, color='k', linestyle=':', linewidth=1.5, alpha=0.7, label='Outer scale')

    # Set y-limits to span 4 orders of magnitude, centered around 1.0 (normalized level)
    plt.ylim(1e-2, 1e2)

    plt.tight_layout()
    plt.show()

    print(f"\nTest complete!")
    return lags, mean_fif_sf

if __name__ == "__main__":
    lags, fif_sf = main()
