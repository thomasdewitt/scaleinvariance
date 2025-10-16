#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from scaleinvariance import acausal_fBm_2D, haar_fluctuation_hurst, structure_function_hurst, spectral_hurst

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_2d_fbm.py H")
        print("Example: python test_2d_fbm.py 0.7")
        sys.exit(1)
    
    H = float(sys.argv[1])
    size = 2048

    print(f"Generating 2D fBm with H={H}, size={int(size/2)}x{size}")
    fbm_2d_rect = acausal_fBm_2D((int(size/2), size), H)
    print(f"Shape: {fbm_2d_rect.shape}")
    print(f"Mean: {np.mean(fbm_2d_rect):.6f}")
    print(f"Std: {np.std(fbm_2d_rect):.6f}")

    # Hurst estimation along both axes for isotropy test
    print(f"\nHurst estimation along both axes (H_true={H}):")
    
    # Get Hurst estimates with analysis data for both axes
    H_haar_1, H_haar_err_1, lags_haar_1, fluct_haar_1, fit_haar_1 = haar_fluctuation_hurst(fbm_2d_rect, axis=1, return_fit=True)
    H_struct_1, H_struct_err_1, lags_struct_1, fluct_struct_1, fit_struct_1 = structure_function_hurst(fbm_2d_rect, axis=1, return_fit=True)
    
    H_haar_0, H_haar_err_0, lags_haar_0, fluct_haar_0, fit_haar_0 = haar_fluctuation_hurst(fbm_2d_rect, axis=0, return_fit=True)
    H_struct_0, H_struct_err_0, lags_struct_0, fluct_struct_0, fit_struct_0 = structure_function_hurst(fbm_2d_rect, axis=0, return_fit=True)
    
    # Spectral analysis along both axes
    H_spec_1, H_spec_err_1, k_vals_1, power_vals_1, fit_spec_1 = spectral_hurst(fbm_2d_rect, axis=1, return_fit=True)
    H_spec_0, H_spec_err_0, k_vals_0, power_vals_0, fit_spec_0 = spectral_hurst(fbm_2d_rect, axis=0, return_fit=True)
    
    print(f"Haar method (axis 1): H = {H_haar_1:.3f} ± {H_haar_err_1:.3f}")
    print(f"Haar method (axis 0): H = {H_haar_0:.3f} ± {H_haar_err_0:.3f}")
    print(f"Structure function (axis 1): H = {H_struct_1:.3f} ± {H_struct_err_1:.3f}")
    print(f"Structure function (axis 0): H = {H_struct_0:.3f} ± {H_struct_err_0:.3f}")
    print(f"Spectral method (axis 1): H = {H_spec_1:.3f} ± {H_spec_err_1:.3f}")
    print(f"Spectral method (axis 0): H = {H_spec_0:.3f} ± {H_spec_err_0:.3f}")
    print(f"Isotropy check: |H_axis1 - H_axis0| = {abs(H_haar_1 - H_haar_0):.3f} (Haar), {abs(H_struct_1 - H_struct_0):.3f} (Struct), {abs(H_spec_1 - H_spec_0):.3f} (Spectral)")

    # Visualization
    fig = plt.figure(figsize=(15, 10))
    
    # Show the 2D field
    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.imshow(fbm_2d_rect, cmap='RdBu_r', aspect='equal')
    ax1.set_title(f'2D fBm (H={H}, {int(size/2)}x{size})')

    # Plot Haar fluctuation analysis (both axes)
    ax2 = plt.subplot(2, 2, 2)
    ax2.loglog(lags_haar_1, fluct_haar_1, 'b.-', alpha=0.7, label='Haar axis 1')
    ax2.loglog(lags_haar_1, fit_haar_1, 'b--', 
               label=f'Fit axis 1: H={H_haar_1:.3f}±{H_haar_err_1:.3f}')
    ax2.loglog(lags_haar_0, fluct_haar_0, 'c.--', alpha=0.7, label='Haar axis 0')
    ax2.loglog(lags_haar_0, fit_haar_0, 'c:', 
               label=f'Fit axis 0: H={H_haar_0:.3f}±{H_haar_err_0:.3f}')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Haar Fluctuation')
    ax2.set_title('Haar Fluctuation Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot structure function analysis (both axes)
    ax3 = plt.subplot(2, 2, 3)
    ax3.loglog(lags_struct_1, fluct_struct_1, 'r.-', alpha=0.7, label='Struct func axis 1')
    ax3.loglog(lags_struct_1, fit_struct_1, 'r--',
               label=f'Fit axis 1: H={H_struct_1:.3f}±{H_struct_err_1:.3f}')
    ax3.loglog(lags_struct_0, fluct_struct_0, 'm.--', alpha=0.7, label='Struct func axis 0')
    ax3.loglog(lags_struct_0, fit_struct_0, 'm:',
               label=f'Fit axis 0: H={H_struct_0:.3f}±{H_struct_err_0:.3f}')
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('Structure Function')
    ax3.set_title('Structure Function Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot spectral analysis (both axes)
    ax4 = plt.subplot(2, 2, 4)
    ax4.loglog(k_vals_1, power_vals_1, 'g.-', alpha=0.7, label='Power spec axis 1')
    ax4.loglog(k_vals_1, fit_spec_1, 'g--',
               label=f'Fit axis 1: H={H_spec_1:.3f}±{H_spec_err_1:.3f}')
    ax4.loglog(k_vals_0, power_vals_0, 'y.--', alpha=0.7, label='Power spec axis 0')
    ax4.loglog(k_vals_0, fit_spec_0, 'y:',
               label=f'Fit axis 0: H={H_spec_0:.3f}±{H_spec_err_0:.3f}')
    ax4.set_xlabel('Wavenumber |k|')
    ax4.set_ylabel('Power')
    ax4.set_title('Power Spectrum Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()