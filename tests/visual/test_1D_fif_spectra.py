#!/usr/bin/env python3
"""
FIF Spectral Analysis Test Script

Tests FIF_1D simulation from scaleinvariance package and compares spectra
against theoretical expectations including multifractal scaling corrections.

Usage: python test_fif_spectra.py [H] [C1] [alpha] [causal]
Examples: 
  python test_fif_spectra.py 0.7
  python test_fif_spectra.py 0.7 0.08 1.8
  python test_fif_spectra.py 0.5 0.1 1.6 True
  python test_fif_spectra.py 0.7 0.15 1.8 False
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
        H = 0.0  # Default H value
    
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
    print("FIF SPECTRAL ANALYSIS TEST")
    print("=" * 50)
    
    # Parameters
    size = 2**20
    n_sims = 100    # Number of simulations for averaging
    outer_scale = size/100
    
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
    
    # Concatenate all simulations and analyze with spectral_hurst
    all_fif_data = np.vstack(all_fif_data)
    H_fif, H_err_fif, freqs, mean_fif_spectrum, fit_line = scaleinvariance.spectral_hurst(
        all_fif_data, return_fit=True, axis=1, min_wavelength=2.1)
    beta_fif = 2 * H_fif + 1
    
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
    H_fif_uncorrected, H_err_fif_uncorrected, freqs_uncorrected, mean_fif_uncorrected_spectrum, fit_line_uncorrected = scaleinvariance.spectral_hurst(
        all_fif_uncorrected_data, return_fit=True, axis=1, min_wavelength=2.1)
    beta_fif_uncorrected = 2 * H_fif_uncorrected + 1
    
    
    # Theoretical expectations
    theoretical_K2 = C1 * (2**alpha - 2) / (alpha - 1)  # K(q=2)
    theoretical_beta_fif = 2*H+1 - theoretical_K2
    
    
    plt.figure(figsize=(10, 8))
    
    # Create two subplots: raw spectra and normalized
    plt.subplot(2, 1, 1)
    
    # Plot raw averaged spectra
    plt.loglog(freqs, mean_fif_spectrum, 'b-', linewidth=2, 
              label=f'FIF corrected (H={H_fif:.2f}, β={beta_fif:.2f})')
    plt.loglog(freqs_uncorrected, mean_fif_uncorrected_spectrum, 'orange', linewidth=2, 
              label=f'FIF uncorrected (H={H_fif_uncorrected:.2f}, β={beta_fif_uncorrected:.2f})')
    
    # Add theoretical reference lines
    ref_freq_range = (freqs > freqs[len(freqs)//8]) & (freqs < freqs[7*len(freqs)//8])
    ref_freqs = freqs[ref_freq_range]
    
    # FIF theoretical line
    ref_power_fif = mean_fif_spectrum[len(freqs)//4]
    ref_freq_fif = freqs[len(freqs)//4]
    theoretical_line_fif = ref_power_fif * (ref_freqs/ref_freq_fif)**(-theoretical_beta_fif)
    plt.loglog(ref_freqs, theoretical_line_fif, 'g--', linewidth=2,
              label=f'FIF theory (β={theoretical_beta_fif:.2f})')
    
    # fBm (non-intermittent) theoretical line
    theoretical_beta_fbm = 2 * H + 1
    theoretical_line_fbm = ref_power_fif * (ref_freqs/ref_freq_fif)**(-theoretical_beta_fbm)
    plt.loglog(ref_freqs, theoretical_line_fbm, 'r--', linewidth=0.5,
              label=f'Equivalent nonintermittent spectrum (β={theoretical_beta_fbm:.2f})')
    
    plt.xlabel('Frequency')
    plt.ylabel('Power Spectral Density')
    causal_str = "causal" if causal else "acausal"
    plt.title(f'FIF Raw Spectral Analysis (H={H}, α={alpha}, C₁={C1}, {causal_str})')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')

    # Add outer scale vertical line (frequency = 1/outer_scale)
    plt.axvline(1/outer_scale, color='k', linestyle=':', linewidth=1.5, alpha=0.7, label='Outer scale')

    ylims_first = plt.ylim()
    
    # Second subplot: Normalized by multifractal theory
    plt.subplot(2, 1, 2)
    
    # Normalize all spectra by middle frequency power to show relative deviations
    mid_idx = len(freqs) // 2
    mid_freq = freqs[mid_idx]
    mid_power_corrected = mean_fif_spectrum[mid_idx]
    mid_power_uncorrected = mean_fif_uncorrected_spectrum[mid_idx]
    
    # Normalize FIF spectra by their theoretical slope AND middle frequency power
    normalized_fif_spectrum = (mean_fif_spectrum * (freqs**(theoretical_beta_fif))) / (mid_power_corrected * (mid_freq**(theoretical_beta_fif)))
    normalized_fif_uncorrected_spectrum = (mean_fif_uncorrected_spectrum * (freqs_uncorrected**(theoretical_beta_fif))) / (mid_power_uncorrected * (mid_freq**(theoretical_beta_fif)))
    
    # Also normalize a theoretical fBm spectrum for comparison  
    theoretical_line_fbm_full = ref_power_fif * (freqs/ref_freq_fif)**(-theoretical_beta_fbm)
    normalized_fbm_theoretical = (theoretical_line_fbm_full * (freqs**(theoretical_beta_fif))) / (mid_power_corrected * (mid_freq**(theoretical_beta_fif)))
    
    plt.loglog(freqs, normalized_fif_spectrum, 'b-', linewidth=2,
              label='FIF corrected normalized')
    plt.loglog(freqs_uncorrected, normalized_fif_uncorrected_spectrum, 'orange', linewidth=2,
              label='FIF uncorrected normalized')
    plt.loglog(freqs, normalized_fbm_theoretical, 'r--', linewidth=0.5,
              label='fBm normalized by FIF theory')
    
    # Add flat reference line
    median_level = np.median(normalized_fif_spectrum[len(freqs)//4:3*len(freqs)//4])
    plt.loglog(freqs, np.full_like(freqs, median_level), 'g--', linewidth=1,
              label='Flat reference')
    
    
    plt.xlabel('Frequency')
    plt.ylabel('Power × f^β (normalized)')
    plt.title('Multifractal-Normalized Spectra (flat = perfect agreement with theory)')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')

    # Add outer scale vertical line (frequency = 1/outer_scale)
    plt.axvline(1/outer_scale, color='k', linestyle=':', linewidth=1.5, alpha=0.7, label='Outer scale')

    # Set y-limits to span 4 orders of magnitude, centered around 1.0 (normalized level)
    plt.ylim(1e-2, 1e2)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTest complete!")
    return freqs, mean_fif_spectrum

if __name__ == "__main__":
    # scaleinvariance.set_backend('numpy')
    print(f'Using backend "{scaleinvariance.get_backend()}"')
    freqs, fif_spectrum = main()