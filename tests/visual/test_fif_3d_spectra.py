#!/usr/bin/env python3
"""
FIF 3D Spectral Analysis Test Script

Tests FIF_ND (3D) simulation from scaleinvariance package and compares spectra
against theoretical expectations including multifractal scaling corrections.
Also visualizes the 3D field.

Usage: python test_fif_3d_spectra.py [H] [C1] [alpha] [size] [nsims]
Examples:
  python test_fif_3d_spectra.py 0.7
  python test_fif_3d_spectra.py 0.7 0.08 1.8
  python test_fif_3d_spectra.py 0.5 0.1 1.6 512
  python test_fif_3d_spectra.py 0.7 0.15 1.8 256
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import scaleinvariance

def K(q, alpha, C1):
    """Scaling function from multifractal theory"""
    if alpha == 1: return C1 * q * np.log(q)
    return C1 * (q**alpha - q) / (alpha - 1)

def calculate_3d_spectrum_and_visualize(alpha, C1, H, size=1024, n_samples=10):
    """
    Calculate and plot the power spectrum, Haar fluctuation, and structure function
    analysis of FIF_ND (3D) simulations, compare with theoretical scaling, and visualize the 3D field.
    
    Parameters:
        alpha (float): Multifractal index for the Lévy noise.
        C1 (float): Codimension of the mean.
        H (float): Hurst exponent.
        size (int): Size of each simulation (square domain).
        n_samples (int): Number of simulations to average over for analysis.
    
    Returns:
        dict: Results from all three analysis methods
    """
    print(f"Generating {n_samples} FIF_ND (3D) simulations of size {(size//2, size//2, size*2)}...")
    
    # Generate all FIF fields
    all_fields = []
    for i in range(n_samples):
        data_2d = scaleinvariance.FIF_ND(size=(size//2, size//2, size*2), alpha=alpha, C1=C1, H=H)
        all_fields.append(data_2d)
        if (i+1) % max(1, n_samples//5) == 0:
            print(f"  Generated {i+1}/{n_samples}")
    
    all_fields = np.array(all_fields)  # Shape: (n_samples, size, size)
    field_for_display = all_fields[0][0,:,:].copy()
    
    # Perform all three analyses using the efficient axis parameter
    print("Running spectral analysis...")
    H_spectral, H_spectral_err, freqs, psd, fit_line_spectral = scaleinvariance.spectral_hurst(
        all_fields, min_wavelength=1, max_wavelength=None, axis=3, return_fit=True  # Analyze along last axis (rows)
    )
    
    print("Running Haar fluctuation analysis...")
    H_haar, H_haar_err, lags_haar, fluct, fit_line_haar = scaleinvariance.haar_fluctuation_hurst(
        all_fields, min_sep=1, max_sep=None, axis=3, return_fit=True  # Analyze along last axis (rows)
    )
    
    print("Running structure function analysis...")
    H_sf, H_sf_err, lags_sf, sf_vals, fit_line_sf = scaleinvariance.structure_function_hurst(
        all_fields, min_sep=1, max_sep=None, axis=3, return_fit=True  # Analyze along last axis (rows)
    )
    
    # Calculate theoretical H value
    theoretical_H = H + K(1, alpha, C1)
    
    
    # Calculate theoretical spectrum for comparison
    beta = 2*H - K(2, alpha, C1) + 1
    theoretical_spectrum = freqs ** (-beta)
    theoretical_fBm_spec = freqs ** (-2*H-1)
    
    # Scale to match simulation at reference point
    ref_idx = min(10, len(psd)-1)
    if np.isfinite(theoretical_spectrum[ref_idx]) and theoretical_spectrum[ref_idx] > 0:
        scale_factor_multifractal = psd[ref_idx] / theoretical_spectrum[ref_idx]
        theoretical_spectrum *= scale_factor_multifractal
        
        scale_factor_fbm = psd[ref_idx] / theoretical_fBm_spec[ref_idx]
        theoretical_fBm_spec *= scale_factor_fbm
    
    # Create figure with 2x2 subplot layout
    fig = plt.figure(figsize=(12, 10))
    
    # Plot 1: 2D Field visualization
    ax1 = plt.subplot(2, 2, 1)
    # Use log scaling for better visualization of multifractal structure
    field_log = np.log10(np.maximum(field_for_display, np.min(field_for_display[field_for_display > 0])))
    im = ax1.pcolormesh(field_log, cmap='plasma', shading='auto')
    ax1.set_title(f'Slice of 3D FIF Field (log₁₀)\nα={alpha:.2f}, C₁={C1:.3f}, H={H:.2f}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    
    # Plot 2: Power Spectrum
    ax2 = plt.subplot(2, 2, 2)
    ax2.loglog(freqs, psd, 'b-', alpha=0.8, label=f'Spectral: H={H_spectral:.3f}±{H_spectral_err:.3f}')
    
    # Create theoretical line with slope -beta anchored at center point
    center_idx = len(freqs) // 2
    center_freq, center_psd = freqs[center_idx], psd[center_idx]
    beta_theoretical = 2*H - K(2, alpha, C1) + 1
    theoretical_line = center_psd * (freqs / center_freq) ** (-beta_theoretical)
    ax2.loglog(freqs, theoretical_line, 'g--', alpha=0.8, label=f'Theory: β={beta_theoretical:.2f}')
    
    ax2.set_title('Power Spectrum')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Power')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()
    
    # Plot 3: Haar Fluctuation
    ax3 = plt.subplot(2, 2, 3)
    ax3.loglog(lags_haar, fluct, 'r-', alpha=0.8, label=f'Haar: H={H_haar:.3f}±{H_haar_err:.3f}')
    
    # Create theoretical line with slope H anchored at center point
    center_idx_haar = len(lags_haar) // 2
    center_lag, center_fluct = lags_haar[center_idx_haar], fluct[center_idx_haar]
    theoretical_haar_line = center_fluct * (lags_haar / center_lag) ** H
    ax3.loglog(lags_haar, theoretical_haar_line, 'g--', alpha=0.8, label=f'Theory: H={H:.2f}')
    
    ax3.set_title('Haar Fluctuation')
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('Fluctuation')
    ax3.grid(True, which="both", ls="-", alpha=0.2)
    ax3.legend()
    
    # Plot 4: Structure Function
    ax4 = plt.subplot(2, 2, 4)
    ax4.loglog(lags_sf, sf_vals, 'g-', alpha=0.8, label=f'SF: H={H_sf:.3f}±{H_sf_err:.3f}')
    
    # Create theoretical line with slope H anchored at center point
    center_idx_sf = len(lags_sf) // 2
    center_lag_sf, center_sf = lags_sf[center_idx_sf], sf_vals[center_idx_sf]
    theoretical_sf_line = center_sf * (lags_sf / center_lag_sf) ** H
    ax4.loglog(lags_sf, theoretical_sf_line, 'g--', alpha=0.8, label=f'Theory: H={H:.2f}')
    
    ax4.set_title('Structure Function')
    ax4.set_xlabel('Lag')
    ax4.set_ylabel('SF')
    ax4.grid(True, which="both", ls="-", alpha=0.2)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    
    return {
        'theoretical_H': theoretical_H,
        'spectral': {'H': H_spectral, 'H_err': H_spectral_err, 'freqs': freqs, 'psd': psd},
        'haar': {'H': H_haar, 'H_err': H_haar_err, 'lags': lags_haar, 'fluct': fluct},
        'structure_function': {'H': H_sf, 'H_err': H_sf_err, 'lags': lags_sf, 'sf_vals': sf_vals},
        'field': field_for_display
    }

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
        C1 = 0.1  # Default C1 value
    
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
            size = int(sys.argv[4])
            if size <= 0 or size % 2 != 0:
                raise ValueError("Size must be a positive even number")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        size = 128  # Default size

    if len(sys.argv) > 5:
        try:
            nsims = int(sys.argv[5])
            if nsims <= 0:
                raise ValueError("N sims must be positive")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        nsims = 10  # Default
    
    # Validate parameters
    if not (0 < alpha <= 2):
        print("Warning: alpha should be in (0, 2] for physical validity")
    if not (-1 < H < 1):
        print("Error: H must be in (-1, 1)")
        sys.exit(1)
    # if C1 <= 0:
    #     print("Error: C1 must be positive")
    #     sys.exit(1)
    
    print("="*60)
    print("FIF_3D Spectral Analysis and Visualization")
    print("="*60)
    print(f"Parameters: α={alpha:.3f}, C₁={C1:.3f}, H={H:.3f}")
    print(f"Domain size: {size//2}×{size*2}")
    print("="*60)
    
    # Run the analysis
    try:
        results = calculate_3d_spectrum_and_visualize(
            alpha, C1, H, size=size, n_samples=nsims
        )
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()