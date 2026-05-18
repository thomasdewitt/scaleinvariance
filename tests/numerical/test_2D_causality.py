#!/usr/bin/env python3
"""
N-D causality smoke test for FIF_ND.

Drives a 2D FIF with a single-pixel spike and inspects asymmetry about the
spike location for various ``causal`` settings. Mirrors the structure of
``test_1D_causality.py``.

A causal axis means the spike location should mark the start (not the
middle) of the response along that axis. With both axes causal, the
response is confined to the bottom-right quadrant relative to the spike.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from scaleinvariance import FIF_ND


size = (256, 256)
alpha = 2.0
C1 = 0.01
H = 0.3
spike_loc = (128, 128)
plot = True


for causal in [True, (True, False), (False, True), False]:
    label = f"causal={causal!r}"

    noise = torch.zeros(size)
    noise[spike_loc] = 10.0

    fif = FIF_ND(
        size, alpha, C1, H,
        levy_noise=noise,
        causal=causal,
        periodic=True,
        kernel_construction_method_flux='LS2010',
        kernel_construction_method_observable='LS2010',
    )

    # For each "causal" axis we expect the response to be one-sided: most of
    # the response lies on the positive-coordinate side of the spike (rows
    # for axis 0, columns for axis 1).
    causal_axes = []
    if causal is True:
        causal_axes = [0, 1]
    elif isinstance(causal, tuple):
        causal_axes = [i for i, c in enumerate(causal) if c]

    for ax in causal_axes:
        # Compare integrated response on the "positive" half vs "negative" half
        # of the axis, measured relative to the spike location.
        if ax == 0:
            past = float(np.sum(np.abs(fif[:spike_loc[0], :])))
            future = float(np.sum(np.abs(fif[spike_loc[0]+1:, :])))
        else:
            past = float(np.sum(np.abs(fif[:, :spike_loc[1]])))
            future = float(np.sum(np.abs(fif[:, spike_loc[1]+1:])))
        ratio = past / future if future > 0 else float('inf')
        status = "✓" if ratio < 0.5 else "✗"
        print(f"  {label} axis={ax}: past/future = {ratio:.3f}  {status}")

    if not causal_axes:
        print(f"  {label}: (acausal — symmetry check skipped)")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(np.log(fif + 1e-12), cmap='magma', aspect='auto')
        axes[0].set_title(f'{label} — log(FIF)')
        axes[0].axhline(spike_loc[0], color='cyan', lw=0.5)
        axes[0].axvline(spike_loc[1], color='cyan', lw=0.5)
        axes[1].imshow(noise.numpy(), cmap='gray', aspect='auto')
        axes[1].set_title('Noise (spike)')
        plt.tight_layout()
        plt.show()
        plt.clf()
