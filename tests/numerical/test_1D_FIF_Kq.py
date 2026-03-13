#!/usr/bin/env python3
"""
Comprehensive test for K(q) parameter recovery via compute_K_q_function.

Tests combinations of H, C1, alpha, verifying that structure function / Haar
analysis correctly recovers the input parameters within tolerances.
"""
import numpy as np
from scaleinvariance import FIF_1D, compute_K_q_function

# Test configuration
KERNEL_METHOD_FLUX       = 'LS2010'
KERNEL_METHOD_OBSERVABLE = 'LS2010'
N_REALIZATIONS           = 50
SIZE                     = 2**14
SCALING_METHOD           = 'structure_function'   # 'structure_function' or 'haar_fluctuation'
HURST_FIT_METHOD         = 'fixed'                # 'fixed' or 'joint'
MIN_SEP                  = 10
MAX_SEP                  = 8000

# Pass/fail tolerances
H_TOL     = 0.10
C1_TOL    = 0.05
ALPHA_TOL = 0.30

# Parameter space
H_VALUES     = [0.3, 0.5, 0.7]
C1_VALUES    = [0.05, 0.1, 0.2]
ALPHA_VALUES = [2.0, 1.7, 1.5]


def test_combination(H, C1, alpha):
    data = np.stack([
        FIF_1D(SIZE, alpha, C1, H,
               kernel_construction_method_flux=KERNEL_METHOD_FLUX,
               kernel_construction_method_observable=KERNEL_METHOD_OBSERVABLE)
        for _ in range(N_REALIZATIONS)
    ], axis=0)

    _, _, H_fit, C1_fit, alpha_fit = compute_K_q_function(
        data, scaling_method=SCALING_METHOD, hurst_fit_method=HURST_FIT_METHOD,
        min_sep=MIN_SEP, max_sep=MAX_SEP, axis=1
    )
    return float(H_fit), float(C1_fit), float(alpha_fit)


def main():
    W = 115
    print("=" * W)
    print("1D FIF K(q) Parameter Recovery Test")
    print(f"  Flux kernel: {KERNEL_METHOD_FLUX}  |  Observable kernel: {KERNEL_METHOD_OBSERVABLE}")
    print(f"  Scaling method: {SCALING_METHOD}  |  Hurst fit method: {HURST_FIT_METHOD}")
    print(f"  Realizations: {N_REALIZATIONS}  |  Size: {SIZE:,}  |  Sep range: {MIN_SEP}–{MAX_SEP}")
    print(f"  Tolerances — H: ±{H_TOL}  C1: ±{C1_TOL}  α: ±{ALPHA_TOL}")
    print("=" * W)

    col = "  True →  Fit   "
    header = f"  {'H':>4}  {'C1':>5}  {'α':>4}  |" + col*3 + "|  Pass"
    print(header)
    sep = (
        f"  {'':4}  {'':5}  {'':4}  |"
        f"  {'H':^14}  {'C1':^14}  {'α':^14}  |"
    )
    print(sep)
    print("-" * W)

    all_passed = True
    passed_count = 0
    test_count   = 0

    for H in H_VALUES:
        for C1 in C1_VALUES:
            for alpha in ALPHA_VALUES:
                test_count += 1
                H_fit, C1_fit, alpha_fit = test_combination(H, C1, alpha)

                H_ok     = abs(H_fit     - H)     < H_TOL
                C1_ok    = abs(C1_fit    - C1)    < C1_TOL
                alpha_ok = abs(alpha_fit - alpha) < ALPHA_TOL
                ok = H_ok and C1_ok and alpha_ok

                if ok:
                    passed_count += 1
                else:
                    all_passed = False

                t = lambda v, fit, ok: f"{'✓' if ok else '✗'} {v:.2f} → {fit:.2f}"

                print(
                    f"  {H:>4.1f}  {C1:>5.3f}  {alpha:>4.1f}  |"
                    f"  {t(H, H_fit, H_ok):^14}"
                    f"  {t(C1, C1_fit, C1_ok):^14}"
                    f"  {t(alpha, alpha_fit, alpha_ok):^14}  |"
                    f"  {'✓' if ok else '✗'}"
                )
        print()

    print("-" * W)
    print(f"Tests passed: {passed_count}/{test_count}")
    print("=" * W)

    if all_passed:
        print("\n✓ ALL TESTS PASSED")
    else:
        print(f"\n✗ SOME TESTS FAILED ({test_count - passed_count} combinations)")

    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
