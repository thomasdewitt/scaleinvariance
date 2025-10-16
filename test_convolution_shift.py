#!/usr/bin/env python3
"""
Test whether we should shift the signal, kernel, both, or neither in FFT convolution.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cpu")
dtype = torch.float64

def test_impulse_shift():
    """
    Test with impulse response kernel that should shift signal by k positions.
    If convolution is correct, output should be shifted version of input.
    """
    size = 16
    shift_amount = 3

    # Create a simple signal (a spike)
    signal = torch.zeros(size, dtype=dtype, device=device)
    signal[5] = 1.0

    # Create impulse kernel centered at size//2 + shift_amount
    # This represents a delay by shift_amount
    kernel = torch.zeros(size, dtype=dtype, device=device)
    kernel[size//2 + shift_amount] = 1.0

    print("Testing impulse shift convolution")
    print(f"Input spike at index: 5")
    print(f"Kernel impulse at index: {size//2 + shift_amount} (lag = {shift_amount})")

    # Method 1: Shift only kernel
    kernel_shifted = torch.fft.ifftshift(kernel)
    result1 = torch.fft.ifft(torch.fft.fft(signal) * torch.fft.fft(kernel_shifted)).real

    # Method 2: Shift both
    signal_shifted = torch.fft.ifftshift(signal)
    kernel_shifted = torch.fft.ifftshift(kernel)
    result2 = torch.fft.ifft(torch.fft.fft(signal_shifted) * torch.fft.fft(kernel_shifted)).real
    result2 = torch.fft.fftshift(result2)  # Shift back

    # Method 3: Shift neither
    result3 = torch.fft.ifft(torch.fft.fft(signal) * torch.fft.fft(kernel)).real

    # Expected: spike should appear at (5 + shift_amount) % size
    expected_index = (5 + shift_amount) % size

    print(f"\nExpected output spike at index: {expected_index}")
    print(f"\nMethod 1 (shift kernel only):")
    print(f"  Max at index: {torch.argmax(result1).item()}")
    print(f"  Max value: {torch.max(result1).item():.3f}")

    print(f"\nMethod 2 (shift both):")
    print(f"  Max at index: {torch.argmax(result2).item()}")
    print(f"  Max value: {torch.max(result2).item():.3f}")

    print(f"\nMethod 3 (shift neither):")
    print(f"  Max at index: {torch.argmax(result3).item()}")
    print(f"  Max value: {torch.max(result3).item():.3f}")

    # Check which method gives correct result
    correct_method = None
    if torch.argmax(result1).item() == expected_index:
        correct_method = "Method 1 (shift kernel only)"
    if torch.argmax(result2).item() == expected_index:
        if correct_method:
            correct_method += " AND Method 2"
        else:
            correct_method = "Method 2 (shift both)"
    if torch.argmax(result3).item() == expected_index:
        if correct_method:
            correct_method += " AND Method 3"
        else:
            correct_method = "Method 3 (shift neither)"

    print(f"\n✓ Correct method: {correct_method}")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    axes[0, 0].stem(signal.cpu().numpy())
    axes[0, 0].set_title('Signal')
    axes[0, 0].axvline(5, color='r', linestyle='--', alpha=0.3)

    axes[0, 1].stem(kernel.cpu().numpy())
    axes[0, 1].set_title(f'Kernel (impulse at {size//2 + shift_amount})')
    axes[0, 1].axvline(size//2, color='g', linestyle='--', alpha=0.3, label='center')
    axes[0, 1].legend()

    axes[0, 2].stem(kernel_shifted.cpu().numpy())
    axes[0, 2].set_title('Kernel after ifftshift')

    axes[1, 0].stem(result1.cpu().numpy())
    axes[1, 0].set_title('Method 1: shift kernel only')
    axes[1, 0].axvline(expected_index, color='r', linestyle='--', alpha=0.3)

    axes[1, 1].stem(result2.cpu().numpy())
    axes[1, 1].set_title('Method 2: shift both')
    axes[1, 1].axvline(expected_index, color='r', linestyle='--', alpha=0.3)

    axes[1, 2].stem(result3.cpu().numpy())
    axes[1, 2].set_title('Method 3: shift neither')
    axes[1, 2].axvline(expected_index, color='r', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/thomas/Box Sync/scaleinvariance/convolution_shift_test.png', dpi=150)
    print("\nSaved visualization to convolution_shift_test.png")

if __name__ == '__main__':
    test_impulse_shift()
