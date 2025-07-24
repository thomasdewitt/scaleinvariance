import sys
import numpy as np
import matplotlib.pyplot as plt
from scaleinvariance import acausal_fBm, structure_function_hurst, haar_fluctuation_hurst, spectral_hurst

if len(sys.argv) != 2:
    print("Usage: python test_acausal_fBm_hurst_estimation.py <H>")
    print("Example: python test_acausal_fBm_hurst_estimation.py 0.7")
    sys.exit(1)

H = float(sys.argv[1])
print(f"Testing fBm with H = {H}")

# Generate fBm - create 2D array with 2 time series
size = 2**20
fBm1 = acausal_fBm(size, H)
fBm2 = acausal_fBm(size, H)
fBm = np.stack([fBm1, fBm2], axis=0)
print(f"Generated 2D fBm array with shape: {fBm.shape}")

# Calculate Hurst estimates
try:
    h_sf, h_sf_err, lags_sf, sf_values, fit_sf = structure_function_hurst(fBm, axis=1, return_fit=True)
    print(f"Structure function: H = {h_sf:.3f} ± {h_sf_err:.3f}")
except NotImplementedError:
    h_sf = None
    print("Structure function: Not implemented")

try:
    h_haar, h_haar_err, lags_haar, haar_values, fit_haar = haar_fluctuation_hurst(fBm, axis=1, return_fit=True)
    print(f"Haar fluctuation: H = {h_haar:.3f} ± {h_haar_err:.3f}")
except NotImplementedError:
    h_haar = None
    print("Haar fluctuation: Not implemented")

try:
    h_spec, h_spec_err, freqs_spec, psd_values, fit_spec = spectral_hurst(fBm, axis=1, return_fit=True)
    print(f"Spectral method: H = {h_spec:.3f} ± {h_spec_err:.3f}")
except NotImplementedError:
    h_spec = None
    print("Spectral method: Not implemented")

# Plotting
plt.style.use('seaborn-v0_8')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Timeseries - plot only first series
ax1.plot(fBm[0, ::2**9], color='#2E8B57', linewidth=0.6)
ax1.set_title(f'fBm Timeseries (H = {H})', fontsize=12, fontweight='bold')
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')
ax1.grid(True, alpha=0.3)

# Structure function plot
if h_sf is not None:
    ax2.loglog(lags_sf, sf_values, 'o', color='#1f77b4', markersize=4, alpha=0.7, label='Data')
    ax2.loglog(lags_sf, fit_sf, '-', color='red', linewidth=2, label=f'Fit: H = {h_sf:.3f} ± {h_sf_err:.3f}')
    ax2.legend()
ax2.set_title('Structure Function Analysis', fontsize=12, fontweight='bold')
ax2.set_xlabel('Lag')
ax2.set_ylabel('Structure Function')
ax2.grid(True, alpha=0.3)

# Haar fluctuation plot
if h_haar is not None:
    ax3.loglog(lags_haar, haar_values, 's', color='#ff7f0e', markersize=4, alpha=0.7, label='Data')
    ax3.loglog(lags_haar, fit_haar, '-', color='red', linewidth=2, label=f'Fit: H = {h_haar:.3f} ± {h_haar_err:.3f}')
    ax3.legend()
ax3.set_title('Haar Fluctuation Analysis', fontsize=12, fontweight='bold')
ax3.set_xlabel('Scale')
ax3.set_ylabel('Fluctuation')
ax3.grid(True, alpha=0.3)

# Spectral plot
if h_spec is not None:
    ax4.loglog(freqs_spec, psd_values, '^', color='#d62728', markersize=4, alpha=0.7, label='Data')
    ax4.loglog(freqs_spec, fit_spec, '-', color='red', linewidth=2, label=f'Fit: H = {h_spec:.3f} ± {h_spec_err:.3f}')
    ax4.legend()
ax4.set_title('Spectral Analysis', fontsize=12, fontweight='bold')
ax4.set_xlabel('Frequency')
ax4.set_ylabel('Power')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()