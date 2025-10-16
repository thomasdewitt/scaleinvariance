#!/usr/bin/env python3
"""
Verify whether missing ifftshift in periodic_convolve is actually a bug.
Tests both symmetric and asymmetric (causal) kernels.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cpu")
dtype = torch.float64

def periodic_convolve_original(signal, kernel):
    """Original version without ifftshift"""
    fft_signal = torch.fft.fft(signal)
    fft_kernel = torch.fft.fft(kernel)
    convolved = torch.fft.ifft(fft_signal * fft_kernel)
    return convolved.real

def periodic_convolve_fixed(signal, kernel):
    """Fixed version with ifftshift"""
    kernel_shifted = torch.fft.ifftshift(kernel)
    fft_signal = torch.fft.fft(signal)
    fft_kernel = torch.fft.fft(kernel_shifted)
    convolved = torch.fft.ifft(fft_signal * fft_kernel)
    return convolved.real

def test_symmetric_kernel():
    """Test with a symmetric kernel (like non-causal FIF)"""
    size = 128
    signal = torch.randn(size, dtype=dtype, device=device)

    # Create symmetric kernel centered at size//2
    t = torch.abs(torch.arange(-size//2, size//2, dtype=dtype, device=device))
    t[t==0] = 1
    kernel = t ** (-0.5)  # Power-law kernel

    result_original = periodic_convolve_original(signal, kernel)
    result_fixed = periodic_convolve_fixed(signal, kernel)

    diff = torch.abs(result_original - result_fixed).max().item()
    print(f"Symmetric kernel - Max difference: {diff:.6e}")
    return diff

def test_causal_kernel():
    """Test with a causal kernel (like causal FIF)"""
    size = 128
    signal = torch.randn(size, dtype=dtype, device=device)

    # Create causal kernel centered at size//2
    t = torch.abs(torch.arange(-size//2, size//2, dtype=dtype, device=device))
    t[t==0] = 1
    kernel = t ** (-0.5)
    kernel[:size//2] = 0  # Make it causal

    result_original = periodic_convolve_original(signal, kernel)
    result_fixed = periodic_convolve_fixed(signal, kernel)

    diff = torch.abs(result_original - result_fixed).max().item()
    print(f"Causal kernel - Max difference: {diff:.6e}")

    # Plot to visualize
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Plot kernels
    axes[0, 0].plot(kernel.cpu().numpy(), 'k-')
    axes[0, 0].set_title('Original Kernel (centered at size//2)')
    axes[0, 0].axvline(size//2, color='r', linestyle='--', alpha=0.5)

    kernel_shifted = torch.fft.ifftshift(kernel)
    axes[0, 1].plot(kernel_shifted.cpu().numpy(), 'k-')
    axes[0, 1].set_title('After ifftshift (zero at index 0)')
    axes[0, 1].axvline(0, color='r', linestyle='--', alpha=0.5)

    # Plot results
    axes[1, 0].plot(result_original.cpu().numpy(), label='Original', alpha=0.7)
    axes[1, 0].plot(result_fixed.cpu().numpy(), label='Fixed', alpha=0.7)
    axes[1, 0].set_title('Convolution Results')
    axes[1, 0].legend()

    # Plot difference
    axes[1, 1].plot((result_original - result_fixed).cpu().numpy(), 'r-')
    axes[1, 1].set_title(f'Difference (max={diff:.2e})')

    plt.tight_layout()
    plt.savefig('/Users/thomas/Box Sync/scaleinvariance/fftshift_bug_test.png', dpi=150)
    print("Saved visualization to fftshift_bug_test.png")

    return diff

if __name__ == '__main__':
    print("Testing periodic_convolve with and without ifftshift\n")

    diff_sym = test_symmetric_kernel()
    diff_causal = test_causal_kernel()

    print(f"\nConclusion:")
    if diff_sym > 1e-10 or diff_causal > 1e-10:
        print("✗ BUG CONFIRMED: ifftshift is needed for correct convolution")
    else:
        print("✓ No difference found - ifftshift may not be necessary")
