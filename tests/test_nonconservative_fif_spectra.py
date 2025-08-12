#!/usr/bin/env python3
"""
Test script for FIF_1D spectral properties
Compares simulated power spectra with theoretical expectations
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('/Users/thomas/Box Sync/scaleinvariance')
import scaleinvariance

def K(q, alpha, C1):
    """Scaling function from multifractal theory"""
    return C1 * (q**alpha - q) / (alpha - 1)

def calculate_spectra(alpha, C1, H, n_samples=10, size=2**16, causal=True, show_error=True):
    """
    Calculate and plot the power spectrum of FIF simulations and compare with theoretical spectrum.
    
    Parameters:
        alpha (float): Multifractal index for the Lévy noise.
        C1 (float): Codimension of the mean.
        H (float): Hurst exponent.
        n_samples (int): Number of simulations to average over.
        size (int): Size of each simulation (should be a power of 2 for efficient FFT).
        causal (bool): Whether to use causal kernels.
        show_error (bool): Whether to show error plot or raw spectra.
    
    Returns:
        tuple: (frequencies, average_power_spectrum, theoretical_spectrum)
    """
    device = torch.device("cpu")
    downsample_factor = 4
    all_spectra = torch.zeros((n_samples, size//2+1), device=device)
    
    print(f"Generating {n_samples} FIF simulations...")
    for i in range(n_samples):
        data = scaleinvariance.FIF_1D(size=size, alpha=alpha, C1=C1, H=H, causal=causal)
        data = torch.as_tensor(data, device=device)
        
        fft_data = torch.fft.fft(data)
        power_spectrum = torch.abs(fft_data[:size//2+1])**2
        # Clip extreme values to prevent overflow in averaging  
        power_spectrum = torch.clamp(power_spectrum, max=1e30)
        all_spectra[i] = power_spectrum
        
        if (i+1) % 5 == 0:
            print(f"  Completed {i+1}/{n_samples}")
    
    avg_spectrum = torch.mean(all_spectra, axis=0)[1:]  # Remove DC component
    frequencies = torch.fft.rfftfreq(size, device=device)[1:]

    # Bin the spectrum for cleaner visualization
    binned_freq = []
    binned_psd = []
    nbins = 30
    bins = torch.logspace(torch.log10(frequencies.min()), torch.log10(frequencies.max()), nbins + 1, device=device)
    
    for i in range(nbins):
        in_bin = (frequencies >= bins[i]) & (frequencies < bins[i+1])
        if torch.any(in_bin):
            binned_freq.append(torch.mean(frequencies[in_bin]))
            binned_psd.append(torch.mean(avg_spectrum[in_bin]))
    
    binned_freq = torch.tensor(binned_freq, device=device)
    binned_psd = torch.tensor(binned_psd, device=device)
    
    # Calculate theoretical spectrum
    beta = 2*H - K(2, alpha, C1) + 1
    theoretical_spectrum = binned_freq ** (-beta)
    theoretical_fBm_spec = binned_freq ** (-2*H-1)
    
    # Check for infinite values in theoretical spectrum
    if torch.any(torch.isinf(theoretical_spectrum)):
        # Clip to avoid infinities
        theoretical_spectrum = torch.clamp(theoretical_spectrum, max=1e50)
    
    # Scale to match simulation at reference point (10th bin)
    ref_idx = min(10, len(binned_psd)-1)  # Use 10th bin as reference
    if torch.isfinite(theoretical_spectrum[ref_idx]) and theoretical_spectrum[ref_idx] > 0:
        scale_factor_multifractal = binned_psd[ref_idx] / theoretical_spectrum[ref_idx]
        theoretical_spectrum *= scale_factor_multifractal
        
        # Scale fBm spectrum to same reference point
        scale_factor_fbm = binned_psd[ref_idx] / theoretical_fBm_spec[ref_idx]
        theoretical_fBm_spec *= scale_factor_fbm
    
    # Calculate Haar fluctuations
    # Concatenate all FIF realizations along axis 0 for batch processing
    all_fif_data = []
    for i in range(n_samples):
        data = scaleinvariance.FIF_1D(size=size, alpha=alpha, C1=C1, H=H, causal=causal)
        all_fif_data.append(data[::downsample_factor])  # Downsample for efficiency
    
    fif_array = np.stack(all_fif_data, axis=0)  # Shape: (n_samples, size//16)
    
    # Calculate Haar fluctuations along axis 1 (spatial dimension)
    lags, haar_flucs = scaleinvariance.haar_fluctuation_analysis(fif_array, axis=1)
    lags = lags * downsample_factor  # Adjust lags to account for downsampling
    
    # Plot results - two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Check for extreme values that might cause plotting issues
    if torch.max(binned_psd) > 1e20:
        print(f"Warning: Extremely large spectrum values detected (max={torch.max(binned_psd):.2e})")
        print("This may indicate numerical instability or extreme multifractal parameters")
    
    # Plot 1: Power Spectrum
    if show_error:
        ax1.loglog(binned_freq.cpu().numpy(), (binned_psd/theoretical_spectrum).cpu().numpy(), 
                  'b-', alpha=0.8, label='Simulation/Theory Ratio')
        ax1.loglog(binned_freq.cpu().numpy(), np.ones_like(binned_freq.cpu().numpy()), 
                  'r--', alpha=0.8, label='Perfect Agreement')
        ax1.set_ylim(1e-3, 2e1)
        ax1.set_ylabel('Spectrum Ratio')
    else:
        ax1.loglog(binned_freq.cpu().numpy(), binned_psd.cpu().numpy(), 
                  'b-', alpha=0.8, label='Average Simulation Spectrum')
        ax1.loglog(binned_freq.cpu().numpy(), theoretical_spectrum.cpu().numpy(), 
                  'g--', alpha=0.8, label='Theoretical Multifractal Spectrum')
        ax1.loglog(binned_freq.cpu().numpy(), theoretical_fBm_spec.cpu().numpy(), 
                  'r--', alpha=0.8, label='Theoretical fBm Spectrum')
        ax1.set_ylabel('Power')
    
    causality_str = "Causal" if causal else "Non-causal"
    ax1.set_title(f'Power Spectrum ({causality_str})')
    ax1.set_xlabel('Frequency')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()
    
    # Add resolution marker for spectrum (Nyquist frequency of original simulation)
    nyquist_freq = 0.5  # Nyquist frequency in original simulation units
    ax1.axvline(x=nyquist_freq, color='black', linestyle='-', linewidth=2, alpha=0.8, label='Nyquist')
    
    # Plot 2: Haar Fluctuations
    ax2.loglog(lags, haar_flucs, 'b-', alpha=0.8, label='Haar Fluctuations')
    
    # Theoretical Haar scaling: F(τ) ~ τ^H
    ref_idx_haar = len(lags) // 2  # Middle of the lag range
    scale_factor_haar = haar_flucs[ref_idx_haar] / (lags[ref_idx_haar] ** H)
    theoretical_haar = scale_factor_haar * (lags ** H)
    ax2.loglog(lags, theoretical_haar, 'r--', alpha=0.8, label=f'Theory: τ^{H:.2f}')
    
    # Add resolution marker for Haar (grid scale of original simulation)
    grid_scale = 1.0  # Grid scale in original simulation units
    ax2.axvline(x=grid_scale, color='black', linestyle='-', linewidth=2, alpha=0.8, label='Grid scale')
    
    ax2.set_title(f'Haar Fluctuations')
    ax2.set_xlabel('Lag τ')
    ax2.set_ylabel('Fluctuation F(τ)')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()
    
    # Overall title
    fig.suptitle(f'FIF Analysis: α={alpha:.2f}, C1={C1:.3f}, H={H:.2f}')
    plt.tight_layout()
    plt.show()
    
    return frequencies.cpu().numpy(), avg_spectrum.cpu().numpy(), theoretical_spectrum.cpu().numpy()

def run_spectral_tests():
    """Run all spectral tests"""
    
    print("="*60)
    print("FIF_1D Spectral Validation Tests")
    print("="*60)
    
    # Test 1: Intermittent case - Causal
    print("\n1. INTERMITTENT CASE - CAUSAL")
    alpha1 = np.random.uniform(1.0, 2.0)
    C1_1 = np.random.uniform(0.07, 0.3)
    H1 = np.random.uniform(-1.0, 1.0)
    n_samples = 30
    size = 2**18
    print(f"Parameters: α={alpha1:.3f}, C1={C1_1:.3f}, H={H1:.3f}")
    
    calculate_spectra(alpha1, C1_1, H1, n_samples=n_samples, size=size, causal=True, show_error=False)
    
    # Test 2: Intermittent case - Non-causal
    print("\n2. INTERMITTENT CASE - NON-CAUSAL")
    alpha2 = np.random.uniform(1.0, 2.0)
    C1_2 = np.random.uniform(0.1, 1.0)
    H2 = np.random.uniform(-1.0, 1.0)
    print(f"Parameters: α={alpha2:.3f}, C1={C1_2:.3f}, H={H2:.3f}")
    
    calculate_spectra(alpha2, C1_2, H2, n_samples=n_samples, size=size, causal=False, show_error=False)
    
    # Test 3: Non-intermittent case - Causal
    print("\n3. NON-INTERMITTENT CASE - CAUSAL")
    alpha3 = np.random.uniform(1.0, 2.0)
    C1_3 = 0.001
    H3 = np.random.uniform(-1.0, 1.0)
    print(f"Parameters: α={alpha3:.3f}, C1={C1_3:.3f}, H={H3:.3f}")
    
    calculate_spectra(alpha3, C1_3, H3, n_samples=n_samples, size=size, causal=True, show_error=False)
    
    # Test 4: Non-intermittent case - Non-causal
    print("\n4. NON-INTERMITTENT CASE - NON-CAUSAL")
    alpha4 = np.random.uniform(1.0, 2.0)
    C1_4 = 0.001
    H4 = np.random.uniform(-1.0, 1.0)
    print(f"Parameters: α={alpha4:.3f}, C1={C1_4:.3f}, H={H4:.3f}")
    
    calculate_spectra(alpha4, C1_4, H4, n_samples=n_samples, size=size, causal=False, show_error=False)
    
    print("\n" + "="*60)
    print("All spectral tests completed!")
    print("="*60)

if __name__ == "__main__":
    run_spectral_tests()