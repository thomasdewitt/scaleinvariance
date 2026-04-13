"""
Generate reference data for the backend refactor.

RUN THIS ONCE BEFORE THE REFACTOR BEGINS.
DO NOT RE-RUN AFTER THE REFACTOR HAS STARTED — the whole point is to compare
new code against old code's output.

Produces: tests/functional/backend_refactor_reference.npz

Contains three groups:
  np_sim_*   — simulation outputs from numpy backend (seeded)
  torch_sim_* — simulation outputs from torch backend (seeded)
  analysis_*  — analysis outputs computed on numpy sim data (deterministic)
"""

import numpy as np
import sys
import os

import scaleinvariance as si
import torch

SEED = 42


def _seed_numpy():
    np.random.seed(SEED)


def _seed_torch():
    torch.manual_seed(SEED)


def _run_simulations(prefix):
    """Run the simulation suite, return dict with prefixed keys."""
    results = {}
    backend = si.get_backend()
    seed_fn = _seed_torch if backend == 'torch' else _seed_numpy

    # fBm_1D_circulant
    seed_fn()
    results[f'{prefix}_fbm_1d_circulant'] = si.fBm_1D_circulant(1024, H=0.5, periodic=True)

    seed_fn()
    results[f'{prefix}_fbm_1d_circulant_aperiodic'] = si.fBm_1D_circulant(1024, H=0.5, periodic=False)

    # fBm_ND_circulant (2D)
    seed_fn()
    results[f'{prefix}_fbm_2d_circulant'] = si.fBm_ND_circulant((64, 64), H=0.4, periodic=True)

    # fBm_1D (convolution-based)
    seed_fn()
    results[f'{prefix}_fbm_1d_conv'] = si.fBm_1D(1024, H=0.5, periodic=True)

    # FIF_1D
    seed_fn()
    results[f'{prefix}_fif_1d'] = si.FIF_1D(1024, alpha=1.8, C1=0.1, H=0.3, periodic=True)

    seed_fn()
    results[f'{prefix}_fif_1d_aperiodic'] = si.FIF_1D(1024, alpha=1.8, C1=0.1, H=0.3, periodic=False)

    # FIF_1D different alpha
    seed_fn()
    results[f'{prefix}_fif_1d_alpha2'] = si.FIF_1D(1024, alpha=2.0, C1=0.05, H=0.5, periodic=True)

    # FIF_ND (2D)
    seed_fn()
    results[f'{prefix}_fif_2d'] = si.FIF_ND((64, 64), alpha=1.8, C1=0.1, H=0.3)

    return results


def _run_analysis(test_signal_1d, test_signal_2d, fif_signal):
    """Run the analysis suite on provided signals. Deterministic (no seeding needed)."""
    results = {}

    # --- Structure function ---
    lags, vals = si.structure_function(test_signal_1d, order=1, axis=0)
    results['analysis_sf_lags'] = lags
    results['analysis_sf_values'] = vals

    lags, vals = si.structure_function(test_signal_1d, order=2, axis=0)
    results['analysis_sf_lags_o2'] = lags
    results['analysis_sf_values_o2'] = vals

    H, unc = si.structure_function_hurst(test_signal_1d, axis=0)
    results['analysis_sf_H'] = np.float64(H)
    results['analysis_sf_unc'] = np.float64(unc)

    # --- Haar fluctuation ---
    lags, vals = si.haar_fluctuation(test_signal_1d, order=1, axis=0)
    results['analysis_haar_lags'] = lags
    results['analysis_haar_values'] = vals

    lags, vals = si.haar_fluctuation(test_signal_1d, order=2, axis=0)
    results['analysis_haar_lags_o2'] = lags
    results['analysis_haar_values_o2'] = vals

    H, unc = si.haar_fluctuation_hurst(test_signal_1d, axis=0)
    results['analysis_haar_H'] = np.float64(H)
    results['analysis_haar_unc'] = np.float64(unc)

    # --- Spectral ---
    freq, psd = si.power_spectrum_binned(test_signal_1d, axis=0)
    results['analysis_spec_freq'] = freq
    results['analysis_spec_psd'] = psd

    H, unc = si.spectral_hurst(test_signal_1d, axis=0)
    results['analysis_spec_H'] = np.float64(H)
    results['analysis_spec_unc'] = np.float64(unc)

    # --- K(q) function (on FIF signal — it's multifractal) ---
    q_vals = np.arange(0.1, 2.51, 0.1)
    kq_q, kq_K, kq_H, kq_C1, kq_alpha = si.K_empirical(
        fif_signal, q_values=q_vals, scaling_method='structure_function',
        hurst_fit_method='fixed'
    )
    results['analysis_kq_q'] = kq_q
    results['analysis_kq_K'] = kq_K
    results['analysis_kq_H'] = np.float64(kq_H)
    results['analysis_kq_C1'] = np.float64(kq_C1)
    results['analysis_kq_alpha'] = np.float64(kq_alpha)

    # --- Two-point intermittency ---
    c1, unc = si.two_point_C1(fif_signal)
    results['analysis_c1_val'] = np.float64(c1)
    results['analysis_c1_unc'] = np.float64(unc)

    # --- 2D analysis ---
    lags, vals = si.structure_function(test_signal_2d, order=1, axis=0)
    results['analysis_sf2d_lags'] = lags
    results['analysis_sf2d_values'] = vals

    lags, vals = si.haar_fluctuation(test_signal_2d, order=1, axis=0)
    results['analysis_haar2d_lags'] = lags
    results['analysis_haar2d_values'] = vals

    freq, psd = si.power_spectrum_binned(test_signal_2d, axis=0)
    results['analysis_spec2d_freq'] = freq
    results['analysis_spec2d_psd'] = psd

    # --- Haar with NaN data ---
    test_nan = test_signal_1d.copy()
    test_nan[100:110] = np.nan
    lags, vals = si.haar_fluctuation(test_nan, order=1, axis=0, nan_behavior='ignore')
    results['analysis_haar_nan_lags'] = lags
    results['analysis_haar_nan_values'] = vals

    return results


def generate():
    results = {}

    # =========================================================================
    # 1. Numpy backend simulations
    # =========================================================================
    si.set_backend('numpy')
    si.set_numerical_precision('float64')
    results.update(_run_simulations('np_sim'))

    # =========================================================================
    # 2. Torch backend simulations
    # =========================================================================
    si.set_backend('torch')
    si.set_numerical_precision('float64')
    results.update(_run_simulations('torch_sim'))

    # =========================================================================
    # 3. Analysis reference (deterministic, uses numpy sim data as canonical input)
    #    Run with numpy backend to get the "ground truth" analysis values.
    #    After refactor, ALL backends should match these within tolerance.
    # =========================================================================
    si.set_backend('numpy')
    si.set_numerical_precision('float64')
    results.update(_run_analysis(
        test_signal_1d=results['np_sim_fbm_1d_circulant'],
        test_signal_2d=results['np_sim_fbm_2d_circulant'],
        fif_signal=results['np_sim_fif_1d'],
    ))

    # =========================================================================
    # Save
    # =========================================================================
    outpath = os.path.join(os.path.dirname(__file__), 'backend_refactor_reference.npz')
    np.savez(outpath, **results)
    print(f"Reference data saved to {outpath}")
    print(f"Keys ({len(results)}): {sorted(results.keys())}")

    # Sanity checks
    for k, v in results.items():
        arr = np.asarray(v)
        assert np.all(np.isfinite(arr)) or 'nan' in k, f"{k} has non-finite values"
    print("All sanity checks passed.")


if __name__ == '__main__':
    generate()
