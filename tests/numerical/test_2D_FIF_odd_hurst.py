#!/usr/bin/env python3
"""
Hurst-exponent recovery test for FIF_ND (2D) with the new
``observable_kernel_odd_axes`` and per-axis ``causal`` parameters.

Mirrors ``test_2D_FIF_hurst.py`` but exercises:
  - ``observable_kernel_odd_axes=True`` (both axes)
  - ``observable_kernel_odd_axes=(True, False)`` (single axis)
  - ``causal=True`` (both axes, LS2010 observable)
  - ``causal=(True, False)`` (single axis, LS2010 observable)

Odd kernels preserve the magnitude spectrum, so spectral Hurst should
recover H within the same tolerance as the baseline test. Causal kernels
also preserve the magnitude scaling (the variance reduction is
compensated by the ``2^k`` factor), so Hurst estimates should likewise
be recovered.
"""
import numpy as np
from scaleinvariance import FIF_ND, spectral_hurst

N_REALIZATIONS = 6
SIZE = 2**10          # 1024x1024 — keeps runtime reasonable
TOLERANCE = 0.07

H_VALUES = [0.3, 0.7]
C1_VALUES = [0.001, 0.1]
ALPHA_VALUES = [2.0, 1.5]

SCENARIOS = [
    # (label, kwargs to FIF_ND, observable_kwargs)
    # Odd kernels are only supported on the spectral observable path.
    ('odd-both-spectral',  dict(observable_kernel_odd_axes=True),
     dict(kernel_construction_method_observable='spectral')),
    ('odd-x-spectral',     dict(observable_kernel_odd_axes=(True, False)),
     dict(kernel_construction_method_observable='spectral')),
    # Causal kernels are only supported on the LS2010 observable path.
    ('causal-both-LS2010', dict(causal=True),
     dict(kernel_construction_method_observable='LS2010')),
    ('causal-x-LS2010',    dict(causal=(True, False)),
     dict(kernel_construction_method_observable='LS2010')),
]

SCALING_RANGES = [
    (1, 10),
    (10, 100),
    (100, 1000),
]


def run_combination(H, C1, alpha, extra_kwargs, observable_kwargs):
    results = {}

    if C1 > 0:
        K_2 = C1 * (2**alpha - 2) / (alpha - 1)
        intermittency_correction = 0.5 * K_2
    else:
        intermittency_correction = 0.0

    realizations = []
    for _ in range(N_REALIZATIONS):
        fif = FIF_ND(
            (SIZE, SIZE), alpha, C1, H,
            periodic=True,
            kernel_construction_method_flux='LS2010',
            **observable_kwargs, **extra_kwargs,
        )
        realizations.append(fif)
    data_stack = np.stack(realizations, axis=0)

    # Analyze along axis 1 (first spatial axis of each 2D realization).
    for min_sep, max_sep in SCALING_RANGES:
        H_est, _ = spectral_hurst(
            data_stack, min_wavelength=min_sep, max_wavelength=max_sep, axis=1
        )
        H_est += intermittency_correction
        results[(min_sep, max_sep)] = {
            'mean_H': H_est,
            'true_H': H,
            'passed': abs(H_est - H) < TOLERANCE,
        }
    return results


def format_row(scenario, H, C1, alpha, results):
    statuses = []
    for r in SCALING_RANGES:
        rec = results.get(r)
        if rec is None:
            statuses.append("N/A")
        else:
            statuses.append(f"{'✓' if rec['passed'] else '✗'} {rec['mean_H']:.3f}")
    row = (f"  {scenario:<22s}  {H:.1f}  {C1:.3f}  {alpha:.1f}   "
           + "  ".join(f"{s:^11}" for s in statuses))
    return row


def main():
    print("=" * 110)
    print("2D FIF Hurst Recovery — new odd / causal kernels")
    print(f"Realizations: {N_REALIZATIONS}   Size: {SIZE}x{SIZE}   Tolerance: ±{TOLERANCE}")
    print("=" * 110)
    header = f"  {'scenario':<22s}   H    C1     α     " + \
        "  ".join(f"{f'{r[0]}-{r[1]}':^11}" for r in SCALING_RANGES)
    print(header)
    print("-" * 110)

    all_passed = True
    test_count = 0
    passed_count = 0

    for label, extra_kwargs, observable_kwargs in SCENARIOS:
        for H in H_VALUES:
            for C1 in C1_VALUES:
                for alpha in ALPHA_VALUES:
                    test_count += 1
                    results = run_combination(H, C1, alpha, extra_kwargs, observable_kwargs)
                    if all(r['passed'] for r in results.values()):
                        passed_count += 1
                    else:
                        all_passed = False
                    print(format_row(label, H, C1, alpha, results))
        print()

    print("-" * 110)
    print(f"Tests passed: {passed_count}/{test_count}")
    print("=" * 110)
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
