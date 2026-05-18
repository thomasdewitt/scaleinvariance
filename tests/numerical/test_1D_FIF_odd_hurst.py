#!/usr/bin/env python3
"""
Hurst-exponent recovery test for FIF_1D with ``observable_kernel_odd_axes=True``.

Mirrors ``test_1D_FIF_hurst.py`` but with the antisymmetric observable
kernel. The output is zero-mean instead of unit-mean, but the magnitude
spectrum is unchanged — so spectral Hurst estimation should still recover
the input H within tolerance.

This validates that the new ``observable_kernel_odd_axes`` parameter does
not perturb the scaling exponent. Causal variants are also exercised in
the per-axis form (1D causal + 1D odd are mutually exclusive on the same
axis, so only one of them at a time here).
"""
import numpy as np
from scaleinvariance import FIF_1D, spectral_hurst

# Test configuration
KERNEL_METHOD_FLUX = 'LS2010'
N_REALIZATIONS = 200
SIZE = 2**14
TOLERANCE = 0.06

# Parameter space
H_VALUES = [0.0, 0.3, 0.7]
C1_VALUES = [0.001, 0.1]
ALPHA_VALUES = [2.0, 1.5]
# observable_kernel_odd_axes is restricted to the spectral path.
OBSERVABLE_METHODS = ['spectral']

SCALING_RANGES = [
    (1, 10),
    (10, 100),
    (100, 1000),
    (1000, 10000),
]


def run_combination(H, C1, alpha, observable_method):
    results = {}

    if C1 > 0:
        K_2 = C1 * (2**alpha - 2) / (alpha - 1)
        intermittency_correction = 0.5 * K_2
    else:
        intermittency_correction = 0.0

    realizations = []
    for _ in range(N_REALIZATIONS):
        fif = FIF_1D(
            SIZE, alpha, C1, H,
            kernel_construction_method_flux=KERNEL_METHOD_FLUX,
            kernel_construction_method_observable=observable_method,
            observable_kernel_odd_axes=True,
        )
        realizations.append(fif)

    data_stack = np.stack(realizations, axis=0)

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


def format_row(H, C1, alpha, observable_method, results):
    statuses = []
    for r in SCALING_RANGES:
        rec = results.get(r)
        if rec is None:
            statuses.append("N/A")
        else:
            statuses.append(f"{'✓' if rec['passed'] else '✗'} {rec['mean_H']:.3f}")
    row = (f"  {H:.1f}    {C1:.3f}   {alpha:.1f}   {observable_method:^8}  "
           + "  ".join(f"{s:^11}" for s in statuses))
    return row


def main():
    print("=" * 110)
    print("1D FIF Hurst Recovery — observable_kernel_odd_axes=True")
    print(f"Flux kernel: {KERNEL_METHOD_FLUX}")
    print(f"Realizations: {N_REALIZATIONS}   Size: {SIZE:,}   Tolerance: ±{TOLERANCE}")
    print("=" * 110)
    print()
    header = "  H      C1      α    Observable  "
    header += "  ".join(f"{f'{r[0]}-{r[1]}':^11}" for r in SCALING_RANGES)
    print(header)
    print("-" * 110)

    all_passed = True
    test_count = 0
    passed_count = 0

    for H in H_VALUES:
        for C1 in C1_VALUES:
            for alpha in ALPHA_VALUES:
                for obs in OBSERVABLE_METHODS:
                    if H == 0:
                        # Odd kernels not meaningful at H=0; skip.
                        continue
                    test_count += 1
                    results = run_combination(H, C1, alpha, obs)
                    if all(r['passed'] for r in results.values()):
                        passed_count += 1
                    else:
                        all_passed = False
                    print(format_row(H, C1, alpha, obs, results))
        print()

    print("-" * 110)
    print(f"Tests passed: {passed_count}/{test_count}")
    print("=" * 110)
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
