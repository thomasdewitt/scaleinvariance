#!/usr/bin/env python3
"""
Compare old vs new Hanning window implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

device = torch.device("cpu")
dtype = torch.float64

def apply_outer_scale_old(kernel, distance, outer_scale, width_factor=2.0):
    """Old method with explicit cutoffs"""
    if not isinstance(distance, torch.Tensor):
        distance = torch.as_tensor(distance, device=device, dtype=dtype)

    transition_width = outer_scale * width_factor
    lower_edge = outer_scale - transition_width / 2.0
    upper_edge = outer_scale + transition_width / 2.0

    window = torch.ones_like(distance)
    in_transition = (distance >= lower_edge) & (distance <= upper_edge)

    normalized_dist = (distance[in_transition] - lower_edge) / transition_width
    window[in_transition] = 0.5 * (1.0 + torch.cos(torch.pi * normalized_dist))
    window[distance > upper_edge] = 0.0

    return kernel * window

def apply_outer_scale_new(kernel, distance, outer_scale, width_factor=2.0):
    """New method with clamp"""
    if not isinstance(distance, torch.Tensor):
        distance = torch.as_tensor(distance, device=device, dtype=dtype)

    transition_width = outer_scale * width_factor
    lower_edge = outer_scale - transition_width / 2.0
    upper_edge = outer_scale + transition_width / 2.0

    normalized_dist = torch.clamp((distance - lower_edge) / transition_width, 0.0, 1.0)
    window = 0.5 * (1.0 + torch.cos(torch.pi * normalized_dist))

    return kernel * window

# Parameters
size = 2**12
outer_scale = 1000.0
width_factor = 2.0

# Create simple test kernel (power law)
distance = torch.abs(torch.arange(-size//2, size//2, dtype=dtype, device=device) + 0.5)
kernel = distance ** (-0.5)

# Apply both methods
kernel_old = apply_outer_scale_old(kernel.clone(), distance, outer_scale, width_factor)
kernel_new = apply_outer_scale_new(kernel.clone(), distance, outer_scale, width_factor)

# Convert to numpy
distance_np = distance.cpu().numpy()
kernel_np = kernel.cpu().numpy()
kernel_old_np = kernel_old.cpu().numpy()
kernel_new_np = kernel_new.cpu().numpy()

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Full comparison
ax = axes[0]
ax.loglog(distance_np, kernel_np, 'k-', linewidth=1, label='Original kernel', alpha=0.3)
ax.loglog(distance_np, kernel_old_np, 'b-', linewidth=2, label='Old (explicit cutoffs)', alpha=0.7)
ax.loglog(distance_np, kernel_new_np, 'r--', linewidth=2, label='New (clamp)', alpha=0.7)
ax.axvline(outer_scale, color='g', linestyle=':', linewidth=1.5, alpha=0.7, label='Outer scale')
ax.axvline(outer_scale - outer_scale * width_factor / 2, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Transition edges')
ax.axvline(outer_scale + outer_scale * width_factor / 2, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('Distance')
ax.set_ylabel('Kernel value')
ax.set_title('Hanning Window Methods Comparison')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

# Difference (zoomed near outer scale)
ax = axes[1]
diff = kernel_old_np - kernel_new_np
ax.semilogx(distance_np, diff, 'purple', linewidth=2)
ax.axvline(outer_scale, color='g', linestyle=':', linewidth=1.5, alpha=0.7, label='Outer scale')
ax.axvline(outer_scale - outer_scale * width_factor / 2, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.axvline(outer_scale + outer_scale * width_factor / 2, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('Distance')
ax.set_ylabel('Old - New')
ax.set_title('Difference Between Methods')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nOuter scale: {outer_scale}")
print(f"Width factor: {width_factor}")
print(f"Transition width: {outer_scale * width_factor}")
print(f"Lower edge: {outer_scale - outer_scale * width_factor / 2}")
print(f"Upper edge: {outer_scale + outer_scale * width_factor / 2}")
print(f"\nMax absolute difference: {np.max(np.abs(diff)):.2e}")
print("Done!")
