#!/usr/bin/env python3
"""
fBm Structure Function Analysis Test Script

Tests fBm_1D simulation from scaleinvariance package with three methods:
- fBm_1D with naive kernel
- fBm_1D with LS2010 kernel (finite-size corrections)
- fBm_1D_circulant (spectral method)

Compares structure functions against theoretical expectations for fractional Brownian motion.

Usage: python test_1D_fbm_integration_structure_function.py [H] [causal]
Examples:
  python test_1D_fbm_integration_structure_function.py 0.7
  python test_1D_fbm_integration_structure_function.py 0.3 True
  python test_1D_fbm_integration_structure_function.py -0.3 False
  python test_1D_fbm_integration_structure_function.py 1.2 True
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
        H = 0.7  # Default H value

    if len(sys.argv) > 2:
        try:
            causal = sys.argv[2].lower() in ['true', '1', 'yes', 't']
        except ValueError:
            print("Error: causal must be True/False")
            sys.exit(1)
    else:
        causal = False  # Default causal value

    print("=" * 50)
    print("fBm STRUCTURE FUNCTION ANALYSIS TEST")
    print("Testing: naive, LS2010, and circulant methods")
    print("=" * 50)

    # Parameters
    size = 2**20
    n_sims = 10    # Number of simulations for averaging
    outer_scale = size

    print(f"\nParameters:")
    print(f"  Size: {size:,}")
    print(f"  H: {H}")
    print(f"  Causal: {causal}")
    print(f"  Number of simulations: {n_sims}")

    # Generate fBm simulations using integration method (naive kernel)
    print(f"\nGenerating {n_sims} fBm simulations using integration method (naive kernel)...")
    all_fbm_naive_data = []

    for i in range(n_sims):
        if i%2 == 0:
            print(f"  Naive simulation {i+1}/{n_sims}...")

        # Generate fBm using integration method with naive kernel
        fbm_field = scaleinvariance.fBm_1D(size, H, causal=causal, outer_scale=outer_scale,
                                          kernel_construction_method='naive')
        all_fbm_naive_data.append(fbm_field)

    # Concatenate all simulations and analyze with structure_function_analysis
    all_fbm_naive_data = np.vstack(all_fbm_naive_data)
    lags, mean_fbm_naive_sf = scaleinvariance.structure_function_analysis(
        all_fbm_naive_data, order=2, axis=1)

    # Generate fBm simulations using integration method (LS2010 kernel)
    print(f"\nGenerating {n_sims} fBm simulations using integration method (LS2010 kernel)...")
    all_fbm_ls2010_data = []

    for i in range(n_sims):
        if i%2 == 0:
            print(f"  LS2010 simulation {i+1}/{n_sims}...")

        # Generate fBm using integration method with LS2010 kernel
        fbm_field = scaleinvariance.fBm_1D(size, H, causal=causal, outer_scale=outer_scale,
                                          kernel_construction_method='LS2010')
        all_fbm_ls2010_data.append(fbm_field)

    # Concatenate all simulations and analyze with structure_function_analysis
    all_fbm_ls2010_data = np.vstack(all_fbm_ls2010_data)
    lags_ls2010, mean_fbm_ls2010_sf = scaleinvariance.structure_function_analysis(
        all_fbm_ls2010_data, order=2, axis=1)

    # Generate reference fBm using circulant method for comparison (only if H in valid range)
    if 0 < H < 1:
        print(f"\nGenerating {n_sims} reference fBm simulations using circulant method...")
        all_fbm_spectral_data = []

        for i in range(n_sims):
            if i%2 == 0:
                print(f"  Circulant simulation {i+1}/{n_sims}...")

            # Generate fBm using circulant method
            fbm_spectral = scaleinvariance.fBm_1D_circulant(size, H)
            all_fbm_spectral_data.append(fbm_spectral)

        # Concatenate spectral simulations and analyze
        all_fbm_spectral_data = np.vstack(all_fbm_spectral_data)
        lags_spectral, mean_fbm_spectral_sf = scaleinvariance.structure_function_analysis(
            all_fbm_spectral_data, order=2, axis=1)
    else:
        all_fbm_spectral_data = None

    # Theoretical expectations for fBm: ζ₂ = 2H
    theoretical_zeta = 2 * H

    plt.figure(figsize=(10, 8))

    # Create two subplots: raw structure functions and normalized
    plt.subplot(2, 1, 1)

    # Plot raw averaged structure functions
    plt.loglog(lags, mean_fbm_naive_sf, 'b-', linewidth=2,
              label='fBm_1D (naive)')
    plt.loglog(lags_ls2010, mean_fbm_ls2010_sf, 'purple', linewidth=2,
              label='fBm_1D (LS2010)')

    if all_fbm_spectral_data is not None:
        plt.loglog(lags_spectral, mean_fbm_spectral_sf, 'orange', linewidth=2,
                  label='fBm_1D_circulant')

    # Add theoretical reference line
    ref_lag_range = (lags < lags[7*len(lags)//8])
    ref_lags = lags[ref_lag_range]

    # fBm theoretical line
    ref_sf = mean_fbm_naive_sf[len(lags)//4]
    ref_lag = lags[len(lags)//4]
    theoretical_line = ref_sf * (ref_lags/ref_lag)**(theoretical_zeta)
    plt.loglog(ref_lags, theoretical_line, 'g--', linewidth=2,
              label=f'fBm theory (ζ₂={theoretical_zeta:.2f})')

    plt.xlabel('Lag')
    plt.ylabel('Structure Function S₂(r)')
    causal_str = "causal" if causal else "acausal"
    plt.title(f'fBm Raw Structure Function Analysis (H={H}, {causal_str})')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')

    # Add outer scale vertical line
    plt.axvline(outer_scale, color='k', linestyle=':', linewidth=1.5, alpha=0.7, label='Outer scale')

    # Second subplot: Normalized by theory
    plt.subplot(2, 1, 2)

    # Normalize all structure functions by middle lag power to show relative deviations
    mid_idx = len(lags) // 2
    mid_lag = lags[mid_idx]
    mid_sf_naive = mean_fbm_naive_sf[mid_idx]
    mid_sf_ls2010 = mean_fbm_ls2010_sf[mid_idx]

    # Normalize fBm structure functions by their theoretical slope AND middle lag power
    normalized_fbm_naive_sf = (mean_fbm_naive_sf / (lags**(theoretical_zeta))) / (mid_sf_naive / (mid_lag**(theoretical_zeta)))
    normalized_fbm_ls2010_sf = (mean_fbm_ls2010_sf / (lags_ls2010**(theoretical_zeta))) / (mid_sf_ls2010 / (mid_lag**(theoretical_zeta)))

    plt.loglog(lags, normalized_fbm_naive_sf, 'b-', linewidth=2,
              label='fBm_1D (naive) normalized')
    plt.loglog(lags_ls2010, normalized_fbm_ls2010_sf, 'purple', linewidth=2,
              label='fBm_1D (LS2010) normalized')

    if all_fbm_spectral_data is not None:
        mid_sf_spectral = mean_fbm_spectral_sf[mid_idx]
        normalized_fbm_spectral_sf = (mean_fbm_spectral_sf / (lags_spectral**(theoretical_zeta))) / (mid_sf_spectral / (mid_lag**(theoretical_zeta)))
        plt.loglog(lags_spectral, normalized_fbm_spectral_sf, 'orange', linewidth=2,
                  label='fBm_1D_circulant normalized')

    # Add flat reference line
    median_level = np.median(normalized_fbm_naive_sf[len(lags)//4:3*len(lags)//4])
    plt.loglog(lags, np.full_like(lags, median_level), 'g--', linewidth=1,
              label='Flat reference')

    plt.xlabel('Lag')
    plt.ylabel('S₂(r) / r^ζ₂ (normalized)')
    plt.title('Normalized Structure Functions (flat = perfect agreement with theory)')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')

    # Add outer scale vertical line
    plt.axvline(outer_scale, color='k', linestyle=':', linewidth=1.5, alpha=0.7, label='Outer scale')

    # Set y-limits to span 4 orders of magnitude, centered around 1.0 (normalized level)
    plt.ylim(1e-2, 1e2)

    plt.tight_layout()
    plt.show()

    print(f"\nTest complete!")
    return lags, mean_fbm_naive_sf, mean_fbm_ls2010_sf

if __name__ == "__main__":
    lags, fbm_naive_sf, fbm_ls2010_sf = main()
