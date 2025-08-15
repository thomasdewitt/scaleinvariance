#!/usr/bin/env python3
"""
FIF 2D Spectral Analysis Test Script

Tests FIF_2D simulation from scaleinvariance package and compares spectra
against theoretical expectations including multifractal scaling corrections.
Also visualizes the 2D field.

Usage: python test_fif_2d_spectra.py [H] [C1] [alpha] [size] [nsims]
Examples: 
  python test_fif_2d_spectra.py 0.7
  python test_fif_2d_spectra.py 0.7 0.08 1.8
  python test_fif_2d_spectra.py 0.5 0.1 1.6 512
  python test_fif_2d_spectra.py 0.7 0.15 1.8 256
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import scaleinvariance

def K(q, alpha, C1):
    """Scaling function from multifractal theory"""
    return C1 * (q**alpha - q) / (alpha - 1)

def calculate_2d_spectrum_and_visualize(alpha, C1, H, size=1024, n_samples=10):
    """
    Calculate and plot the power spectrum of FIF_2D simulations along one dimension,
    compare with theoretical spectrum, and visualize the 2D field.
    
    Parameters:
        alpha (float): Multifractal index for the Lévy noise.
        C1 (float): Codimension of the mean.
        H (float): Hurst exponent.
        size (int): Size of each simulation (square domain).
        n_samples (int): Number of simulations to average over for spectrum.
    
    Returns:
        tuple: (frequencies, average_power_spectrum, theoretical_spectrum)
    """
    print(f"Generating {n_samples} FIF_2D simulations of size {size}x{size}...")
    
    all_spectra = []
    field_for_display = None
    
    for i in range(n_samples):
        # Generate 2D FIF field
        data_2d = scaleinvariance.FIF_2D(size=size, alpha=alpha, C1=C1, H=H)
        
        # Save first field for visualization
        if i == 0:
            field_for_display = data_2d.copy()
        
        # Take 1D slices along rows (axis=1) and average their spectra
        slice_spectra = []
        # Take every 8th row to reduce computation but maintain statistics
        for row_idx in range(0, size, max(1, size//128)):
            row_data = data_2d[row_idx, :]
            fft_data = np.fft.fft(row_data)
            power_spectrum = np.abs(fft_data[:size//2+1])**2
            slice_spectra.append(power_spectrum[1:])  # Remove DC component
        
        # Average spectra from all rows for this realization
        avg_slice_spectrum = np.mean(slice_spectra, axis=0)
        all_spectra.append(avg_slice_spectrum)
        
        if (i+1) % max(1, n_samples//5) == 0:
            print(f"  Completed {i+1}/{n_samples}")
    
    # Average over all realizations
    avg_spectrum = np.mean(all_spectra, axis=0)
    frequencies = np.fft.rfftfreq(size)[1:]
    
    # Bin the spectrum for cleaner visualization
    binned_freq = []
    binned_psd = []
    nbins = 30
    bins = np.logspace(np.log10(frequencies.min()), np.log10(frequencies.max()), nbins + 1)
    
    for i in range(nbins):
        in_bin = (frequencies >= bins[i]) & (frequencies < bins[i+1])
        if np.any(in_bin):
            binned_freq.append(np.mean(frequencies[in_bin]))
            binned_psd.append(np.mean(avg_spectrum[in_bin]))
    
    binned_freq = np.array(binned_freq)
    binned_psd = np.array(binned_psd)
    
    # Calculate theoretical spectrum
    beta = 2*H - K(2, alpha, C1) + 1
    theoretical_spectrum = binned_freq ** (-beta)
    theoretical_fBm_spec = binned_freq ** (-2*H-1)
    
    # Scale to match simulation at reference point
    ref_idx = min(10, len(binned_psd)-1)
    if np.isfinite(theoretical_spectrum[ref_idx]) and theoretical_spectrum[ref_idx] > 0:
        scale_factor_multifractal = binned_psd[ref_idx] / theoretical_spectrum[ref_idx]
        theoretical_spectrum *= scale_factor_multifractal
        
        scale_factor_fbm = binned_psd[ref_idx] / theoretical_fBm_spec[ref_idx]
        theoretical_fBm_spec *= scale_factor_fbm
    
    # Create figure with three subplots
    fig = plt.figure(figsize=(16, 5))
    
    # Plot 1: 2D Field visualization with log scaling and fun colors
    ax1 = plt.subplot(131)
    # Use log scaling for better visualization of multifractal structure
    field_log = np.log10(np.maximum(field_for_display, np.min(field_for_display[field_for_display > 0])))
    im = ax1.pcolormesh(field_log, cmap='plasma', shading='auto')
    ax1.set_title(f'2D FIF Field (log₁₀)\nα={alpha:.2f}, C₁={C1:.3f}, H={H:.2f}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    
    # Plot 2: Power Spectrum
    ax2 = plt.subplot(132)
    ax2.loglog(binned_freq, binned_psd, 'b-', alpha=0.8, label='Average Simulation Spectrum')
    ax2.loglog(binned_freq, theoretical_spectrum, 'g--', alpha=0.8, label='Theoretical Multifractal Spectrum')
    ax2.loglog(binned_freq, theoretical_fBm_spec, 'r--', alpha=0.8, label='Theoretical fBm Spectrum')
    
    ax2.set_title('1D Power Spectrum (row-averaged)')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Power')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()
    
    # Add resolution marker
    nyquist_freq = 0.5
    ax2.axvline(x=nyquist_freq, color='black', linestyle='-', linewidth=2, alpha=0.8, label='Nyquist')
    
    # Plot 3: Spectrum Ratio (Error analysis)
    ax3 = plt.subplot(133)
    ratio = binned_psd / theoretical_spectrum
    ax3.loglog(binned_freq, ratio, 'b-', alpha=0.8, label='Simulation/Theory Ratio')
    ax3.loglog(binned_freq, np.ones_like(binned_freq), 'g--', alpha=0.8, label='Perfect Agreement')
    ax3.set_ylim(1e-2, 1e2)
    ax3.set_title('Spectrum Agreement')
    ax3.set_xlabel('Frequency')
    ax3.set_ylabel('Spectrum Ratio')
    ax3.grid(True, which="both", ls="-", alpha=0.2)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\nField Statistics:")
    print(f"  Min: {np.min(field_for_display):.3f}")
    print(f"  Max: {np.max(field_for_display):.3f}")
    print(f"  Mean: {np.mean(field_for_display):.3f}")
    print(f"  Std: {np.std(field_for_display):.3f}")
    
    return frequencies, avg_spectrum, theoretical_spectrum

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
        size = 1024  # Default size

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
    if C1 <= 0:
        print("Error: C1 must be positive")
        sys.exit(1)
    
    print("="*60)
    print("FIF_2D Spectral Analysis and Visualization")
    print("="*60)
    print(f"Parameters: α={alpha:.3f}, C₁={C1:.3f}, H={H:.3f}")
    print(f"Domain size: {size}×{size}")
    print("="*60)
    
    # Run the analysis
    try:
        freq, sim_spec, theo_spec = calculate_2d_spectrum_and_visualize(
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