#!/usr/bin/env python3
"""
FIF Comparison Demo

Shows how different multifractal parameters affect the same underlying noise.
Creates 4 simulations using identical Lévy noise but different H and C1 values,
then compares their time series, Haar fluctuations, and spectral Hurst estimates.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import scaleinvariance

def main():
    # Parameters
    size = 2**22
    alpha = 1.8
    np.random.seed(42)  # For reproducible results
    
    # Generate shared Lévy noise
    print("Generating shared Lévy noise...")
    shared_noise = scaleinvariance.simulation.FIF.extremal_levy(alpha, size)
    
    # Create 4 simulations with same noise, different parameters
    params = [
        (-0.3, 0.01, 'H=-0.3, C1=0.01'),
        (-0.3, 0.1, 'H=-0.3, C1=0.1'), 
        (0.3, 0.01, 'H=0.3, C1=0.01'),
        (0.3, 0.1, 'H=0.3, C1=0.1')
    ]
    
    simulations = []
    print("Generating FIF simulations with shared noise...")
    for H, C1, label in params:
        fif = scaleinvariance.FIF_1D(size, alpha, C1, H, levy_noise=shared_noise, causal=False)
        simulations.append((fif, H, C1, label))
        print(f"  Generated {label}")
    
    # Create figure with subplots - 4 time series plots + 4 analysis plots
    fig = plt.figure(figsize=(12, 16))
    
    # Top 2x2: Individual FIF time series plots (H=-0.3 left, H=0.3 right)
    ax1 = plt.subplot(4, 2, 1)  # H=-0.3, C1=0.01
    ax2 = plt.subplot(4, 2, 2)  # H=0.3, C1=0.01
    ax3 = plt.subplot(4, 2, 3)  # H=-0.3, C1=0.1
    ax4 = plt.subplot(4, 2, 4)  # H=0.3, C1=0.1
    
    # Plot each simulation individually
    x = np.arange(len(simulations[0][0]))
    
    # H=-0.3, C1=0.01
    fif, H, C1, _ = simulations[0]
    ax1.plot(x[::64], fif[::64], 'k-', alpha=0.8, linewidth=0.5)
    ax1.set_title(f'H={H}, C1={C1}')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # H=0.3, C1=0.01
    fif, H, C1, _ = simulations[2]
    ax2.plot(x[::64], fif[::64], 'k-', alpha=0.8, linewidth=0.5)
    ax2.set_title(f'H={H}, C1={C1}')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # H=-0.3, C1=0.1
    fif, H, C1, _ = simulations[1]
    ax3.plot(x[::64], fif[::64], 'k-', alpha=0.8, linewidth=0.5)
    ax3.set_title(f'H={H}, C1={C1}')
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # H=0.3, C1=0.1
    fif, H, C1, _ = simulations[3]
    ax4.plot(x[::64], fif[::64], 'k-', alpha=0.8, linewidth=0.5)
    ax4.set_title(f'H={H}, C1={C1}')
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    # Analysis plots - row 2: Haar fluctuations
    ax5 = plt.subplot(4, 2, 5)  # Haar H=-0.3
    ax6 = plt.subplot(4, 2, 6)  # Haar H=0.3
    
    # Analysis plots - row 3: Spectral
    ax7 = plt.subplot(4, 2, 7)  # Spectral H=-0.3  
    ax8 = plt.subplot(4, 2, 8)  # Spectral H=0.3
    
    # Generate 10 simulations for each parameter combination for analysis
    print("Generating 10 simulations for each parameter combination...")
    analysis_sims = {}
    for H, C1, label in params:
        fif_array = []
        for _ in range(10):
            fif_temp = scaleinvariance.FIF_1D(size, alpha, C1, H, causal=False)
            fif_array.append(fif_temp[::256])
        analysis_sims[(H, C1)] = np.stack(fif_array, axis=0)
        print(f"  Generated 10 sims for {label}")
    
    # Haar analysis for H=-0.3 cases (averaged over 10 sims)
    print("Computing Haar fluctuations (10 sim average)...")
    for i, (_, H, C1, label) in enumerate([simulations[0], simulations[1]]):
        fif_batch = analysis_sims[(H, C1)]
        H_est, H_err, lags, flucs, fit = scaleinvariance.haar_fluctuation_hurst(fif_batch, axis=1, return_fit=True, max_sep=100)
        
        color = 'b-' if i == 0 else 'r-'
        ax5.loglog(lags * 256, flucs, color, alpha=0.8, label=label)
        ax5.loglog(lags * 256, fit, '--', color=color[0], alpha=0.6, 
                  label=f'H={H_est:.3f}±{H_err:.3f}')
    
    ax5.set_title('Haar Fluctuations: H = -0.3')
    ax5.set_xlabel('Lag τ')
    ax5.set_ylabel('Fluctuation F(τ)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Haar analysis for H=0.3 cases (averaged over 10 sims)
    for i, (_, H, C1, label) in enumerate([simulations[2], simulations[3]]):
        fif_batch = analysis_sims[(H, C1)]
        H_est, H_err, lags, flucs, fit = scaleinvariance.haar_fluctuation_hurst(fif_batch, axis=1, return_fit=True, max_sep=100)
        
        color = 'b-' if i == 0 else 'r-'
        ax6.loglog(lags * 256, flucs, color, alpha=0.8, label=label)
        ax6.loglog(lags * 256, fit, '--', color=color[0], alpha=0.6,
                  label=f'H={H_est:.3f}±{H_err:.3f}')
    
    ax6.set_title('Haar Fluctuations: H = 0.3')
    ax6.set_xlabel('Lag τ')
    ax6.set_ylabel('Fluctuation F(τ)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Spectral analysis for H=-0.3 cases (averaged over 10 sims)
    print("Computing spectral Hurst estimates (10 sim average)...")
    for i, (_, H, C1, label) in enumerate([simulations[0], simulations[1]]):
        fif_batch = analysis_sims[(H, C1)]
        H_est, H_err, freqs, psd, fit = scaleinvariance.spectral_hurst(fif_batch, axis=1, return_fit=True)
        
        color = 'b-' if i == 0 else 'r-'
        ax7.loglog(freqs / 256, psd, color, alpha=0.8, label=label)
        ax7.loglog(freqs / 256, fit, '--', color=color[0], alpha=0.6,
                  label=f'H={H_est:.3f}±{H_err:.3f}')
    
    ax7.set_title('Power Spectra: H = -0.3')
    ax7.set_xlabel('Frequency')
    ax7.set_ylabel('Power')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Spectral analysis for H=0.3 cases (averaged over 10 sims)
    for i, (_, H, C1, label) in enumerate([simulations[2], simulations[3]]):
        fif_batch = analysis_sims[(H, C1)]
        H_est, H_err, freqs, psd, fit = scaleinvariance.spectral_hurst(fif_batch, axis=1, return_fit=True)
        
        color = 'b-' if i == 0 else 'r-'
        ax8.loglog(freqs / 256, psd, color, alpha=0.8, label=label)
        ax8.loglog(freqs / 256, fit, '--', color=color[0], alpha=0.6,
                  label=f'H={H_est:.3f}±{H_err:.3f}')
    
    ax8.set_title('Power Spectra: H = 0.3')
    ax8.set_xlabel('Frequency')
    ax8.set_ylabel('Power')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    
    # Overall title and layout
    fig.suptitle(f'FIF Comparison: α={alpha}, Shared Lévy Noise', fontsize=14)
    plt.tight_layout()
    plt.savefig('fif_comparison_demo.png', dpi=300, bbox_inches='tight')
    print("Saved plot as 'fif_comparison_demo.png'")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary of Hurst estimates:")
    print("="*60)
    for fif, H_true, C1, label in simulations:
        H_haar, H_haar_err = scaleinvariance.haar_fluctuation_hurst(fif[::256], max_sep=100)
        H_spec, H_spec_err = scaleinvariance.spectral_hurst(fif[::256])
        H_struct, H_struct_err = scaleinvariance.structure_function_hurst(fif[::256], max_sep=100)
        print(f"{label}:")
        print(f"  True H: {H_true:.2f}")
        print(f"  Haar:   {H_haar:.3f} ± {H_haar_err:.3f}")
        print(f"  Spectral: {H_spec:.3f} ± {H_spec_err:.3f}")
        print(f"  Structure: {H_struct:.3f} ± {H_struct_err:.3f}")
        print()

if __name__ == "__main__":
    main()