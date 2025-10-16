#!/usr/bin/env python3
import matplotlib.pyplot as plt
from scaleinvariance import FIF_1D
import torch

size = 2048
alpha = 2.0
C1 = 0.01
spike_loc = 1024
plot = True

for H in [0, 0.3]:
    for method in ['naive','LS2010']:
        noise = torch.zeros(size)
        noise[spike_loc] = 10

        # Generate causal and acausal
        fif_causal = FIF_1D(size, alpha, C1, H, levy_noise=noise, causal=True, periodic=True, kernel_construction_method=method)
        fif_acausal = FIF_1D(size, alpha, C1, H, levy_noise=noise, causal=False, periodic=True, kernel_construction_method=method)

        # If the system is causal, it does not know the spike is about to happen
        # We can look for symmetry about spike location
        # It will not be the same for all kernel construction methods, however, so this will not be exact
        if abs(fif_causal[spike_loc-10]-fif_causal[spike_loc+10]) < 10 * abs(fif_acausal[spike_loc-10]-fif_acausal[spike_loc+10]):
            print(f'Symmetry test failed for H={H} for kernel method {method}: inspect plot?')
        if (max(fif_causal[spike_loc-1:spike_loc+3]) != max(fif_causal)) or (max(fif_acausal[spike_loc-1:spike_loc+3]) != max(fif_acausal)):
            # print(fif_causal[spike_loc-1:spike_loc+3], max(fif_causal))
            # print(fif_acausal[spike_loc-1:spike_loc+3], max(fif_acausal))
            print(f'Spike location test failed for H={H} for kernel method {method}; inspect plot?')
        else:
            print(f'Causality tests passed for H={H} for kernel method {method}')
        

        if plot:
            # Plot
            fig, axes = plt.subplots(3, 1, figsize=(10, 6))
            axes[0].plot(fif_causal, 'k-', linewidth=0.5)
            axes[0].set_title('Causal')
            # axes[0].axhline(1, color='r', linestyle='--', alpha=0.3)
            axes[1].plot(fif_acausal, 'k-', linewidth=0.5)
            axes[1].set_title('Acausal')
            axes[2].plot(noise, 'k-', linewidth=0.5)
            axes[2].set_title('Noise')
            # axes[1].axhline(1, color='r', linestyle='--', alpha=0.3)
            plt.tight_layout()
            # plt.savefig('/Users/thomas/Box Sync/scaleinvariance/test_causal.png', dpi=150)
            plt.show()
            plt.clf()