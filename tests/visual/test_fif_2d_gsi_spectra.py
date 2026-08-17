#!/usr/bin/env python3
"""
FIF 2D GSI visual test.

Canonical anisotropic scale metric: the last axis (z) scales with exponent Hz
relative to the first (x), so a separation dz enters the metric as dz^(1/Hz)
and exponents along z are those along x divided by Hz.

Usage: python test_fif_2d_gsi_spectra.py [size] [Hz] [ls] [H] [nsims]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import scaleinvariance as si

size = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
Hz = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
ls = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
H = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
nsims = int(sys.argv[5]) if len(sys.argv) > 5 else 5

si.set_device('cuda')

alpha, C1 = 2, 0.01
lag_lo, lag_hi = 30, 100
q_values = np.arange(0.2, 2.51, 0.2)

print(f"{size}x{size} periodic, {nsims} sims, alpha={alpha}, C1={C1}, H={H}")
print(f"GSI: Hz={Hz}, ls={ls:g}, D_el={1+Hz:.3f}; expected H_x={H:.3f}, H_z={H/Hz:.3f}")

metric = si.canonical_scale_metric((size, size), ls=ls, Hz=Hz)
fields = np.array([
    si.FIF_ND((size, size), alpha=alpha, C1=C1, H=H, periodic=True,
              scale_metric=metric, elliptical_dim=1 + Hz,
              kernel_construction_method_observable='LS2010')
    for _ in range(nsims)
])
print("fields generated")

freqs_x, psd_x = si.power_spectrum_binned(fields, axis=1)
freqs_z, psd_z = si.power_spectrum_binned(fields, axis=2)

qs_x, Kq_x, H_x, C1_x, alpha_x = si.K_empirical(
    fields, q_values=q_values, scaling_method='wavelet_fluctuation',
    wavelet='mexican_hat', min_sep=lag_lo, max_sep=lag_hi, axis=1, periodic=True)
qs_z, Kq_z, H_z, C1_z, alpha_z = si.K_empirical(
    fields, q_values=q_values, scaling_method='wavelet_fluctuation',
    wavelet='mexican_hat', min_sep=lag_lo, max_sep=lag_hi, axis=2, periodic=True)

lags_x, fluct_x = si.wavelet_fluctuation(fields, wavelet='mexican_hat', order=1,
                                         axis=1, periodic=True)
lags_z, fluct_z = si.wavelet_fluctuation(fields, wavelet='mexican_hat', order=1,
                                         axis=2, periodic=True)

K_theory = C1 * (q_values**alpha - q_values) / (alpha - 1)
xi2 = 2 * H - C1 * (2**alpha - 2) / (alpha - 1)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

ax = axes[0, 0]
field = fields[0]
ax.pcolormesh(np.log10(np.maximum(field, field[field > 0].min())),
              cmap='plasma', shading='auto')
ax.set_title(f'field (log$_{{10}}$)  Hz={Hz:g}, ls={ls:g}, H={H:g}')
ax.set_xlabel('z (anisotropic)')
ax.set_ylabel('x')
ax.set_aspect('equal')

ax = axes[0, 1]
ax.loglog(freqs_x, psd_x, color='tab:blue', label='x')
ax.loglog(freqs_z, psd_z, color='tab:red', label='z')
mid = len(freqs_x) // 2
ax.loglog(freqs_x, psd_x[mid] * (freqs_x / freqs_x[mid]) ** -(1 + xi2),
          '--', color='tab:blue', alpha=0.6, label=f'β={1+xi2:.2f}')
ax.loglog(freqs_z, psd_z[mid] * (freqs_z / freqs_z[mid]) ** -(1 + xi2 / Hz),
          '--', color='tab:red', alpha=0.6, label=f'β={1+xi2/Hz:.2f}')
ax.axvline(1 / ls, color='k', ls=':', alpha=0.5)
ax.set_title('power spectrum')
ax.set_xlabel('frequency')
ax.set_ylabel('power')
ax.legend(fontsize=8)

ax = axes[1, 0]
ax.plot(qs_x, Kq_x, 'o-', color='tab:blue', ms=3,
        label=f'x: C1={C1_x:.3f}, α={alpha_x:.2f}')
ax.plot(qs_z, Kq_z, 'o-', color='tab:red', ms=3,
        label=f'z: C1={C1_z:.3f}, α={alpha_z:.2f}')
ax.plot(q_values, K_theory, 'k--', alpha=0.6, label=f'C1={C1:g}, α={alpha:g}')
ax.set_title(f'K(q), mexican hat, lags {lag_lo}–{lag_hi}')
ax.set_xlabel('q')
ax.set_ylabel('K(q)')
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.loglog(lags_x, fluct_x, color='tab:blue', label=f'x: H={H_x:.3f} (expect {H:.3f})')
ax.loglog(lags_z, fluct_z, color='tab:red', label=f'z: H={H_z:.3f} (expect {H/Hz:.3f})')
ax.axvspan(lag_lo, lag_hi, color='k', alpha=0.12)
ax.set_title('mexican hat fluctuation, q=1')
ax.set_xlabel('lag')
ax.set_ylabel('fluctuation')
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
