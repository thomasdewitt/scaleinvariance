#!/usr/bin/env python3
"""
Test fBm_ND_circulant Hurst exponent estimation along each dimension.

Verifies that Haar fluctuation analysis correctly recovers the input Hurst
exponent along all axes for isotropic fBm fields.
"""
import numpy as np
from scaleinvariance import fBm_ND_circulant, haar_fluctuation_hurst

# Test configuration
N_REALIZATIONS_2D = 20
N_REALIZATIONS_3D = 10  # More realizations for better statistics
SIZE_2D = 256
SIZE_3D = 128  # 128^3 is manageable
MIN_SEP = 2
MAX_SEP = 16
TOLERANCE = 0.09
PERIODIC = True

H_VALUES = [0.3, 0.5, 0.7]


def test_fBm_2D_circulant():
    """Test 2D fBm Haar fluctuation along each axis."""
    print("=" * 70)
    print("Testing fBm_ND_circulant (2D) Hurst estimation (Haar fluctuation)")
    print(f"Size: {SIZE_2D}x{SIZE_2D}, Realizations: {N_REALIZATIONS_2D}, periodic={PERIODIC}")
    print(f"min_sep={MIN_SEP}, max_sep={MAX_SEP}, tolerance={TOLERANCE}")
    print("=" * 70)
    print()

    print(f"{'H_true':^8} {'Axis':^6} {'H_est':^10} {'Error':^10} {'Status':^8}")
    print("-" * 50)

    results = []

    for H_true in H_VALUES:
        # Generate stack of realizations
        realizations = []
        for _ in range(N_REALIZATIONS_2D):
            fBm = fBm_ND_circulant((SIZE_2D, SIZE_2D), H_true, periodic=PERIODIC)
            realizations.append(fBm)

        # Stack: shape = (N_REALIZATIONS, SIZE_2D, SIZE_2D)
        data_stack = np.stack(realizations, axis=0)

        # Test each axis
        for axis_to_test in [1, 2]:  # Skip axis 0 (realizations)
            H_est, H_err = haar_fluctuation_hurst(
                data_stack, min_sep=MIN_SEP, max_sep=MAX_SEP, axis=axis_to_test
            )

            error = abs(H_est - H_true)
            passed = error < TOLERANCE
            status = "✓" if passed else "✗"

            print(f"{H_true:^8.1f} {axis_to_test:^6d} {H_est:^10.4f} {error:^10.4f} {status:^8}")

            results.append({
                'H_true': H_true,
                'axis': axis_to_test,
                'H_est': H_est,
                'error': error,
                'passed': passed
            })

    print("-" * 50)

    # Summary
    all_passed = all(r['passed'] for r in results)
    n_passed = sum(r['passed'] for r in results)

    print(f"\nPassed: {n_passed}/{len(results)}")

    if all_passed:
        print("\n✓ ALL 2D TESTS PASSED")
    else:
        print(f"\n✗ SOME 2D TESTS FAILED")
        max_error = max(r['error'] for r in results)
        print(f"  Maximum error: {max_error:.4f}")
        print(f"  Suggested tolerance: {max_error * 1.2:.4f}")

    return all_passed, results


def test_fBm_ND_circulant_3D():
    """Test 3D fBm Haar fluctuation along each axis."""
    print("=" * 70)
    print("Testing fBm_ND_circulant (3D) Hurst estimation (Haar fluctuation)")
    print(f"Size: {SIZE_3D}x{SIZE_3D}x{SIZE_3D}, Realizations: {N_REALIZATIONS_3D}, periodic={PERIODIC}")
    print(f"min_sep={MIN_SEP}, max_sep={MAX_SEP}, tolerance={TOLERANCE}")
    print("=" * 70)
    print()

    print(f"{'H_true':^8} {'Axis':^6} {'H_est':^10} {'Error':^10} {'Status':^8}")
    print("-" * 50)

    results = []

    for H_true in H_VALUES:
        # Generate stack of realizations
        realizations = []
        for _ in range(N_REALIZATIONS_3D):
            fBm = fBm_ND_circulant((SIZE_3D, SIZE_3D, SIZE_3D), H_true, periodic=PERIODIC)
            realizations.append(fBm)

        # Stack: shape = (N_REALIZATIONS, SIZE_3D, SIZE_3D, SIZE_3D)
        data_stack = np.stack(realizations, axis=0)

        # Test each spatial axis
        for axis_to_test in [1, 2, 3]:  # Skip axis 0 (realizations)
            H_est, H_err = haar_fluctuation_hurst(
                data_stack, min_sep=MIN_SEP, max_sep=MAX_SEP, axis=axis_to_test
            )

            error = abs(H_est - H_true)
            passed = error < TOLERANCE
            status = "✓" if passed else "✗"

            print(f"{H_true:^8.1f} {axis_to_test:^6d} {H_est:^10.4f} {error:^10.4f} {status:^8}")

            results.append({
                'H_true': H_true,
                'axis': axis_to_test,
                'H_est': H_est,
                'error': error,
                'passed': passed
            })

    print("-" * 50)

    # Summary
    all_passed = all(r['passed'] for r in results)
    n_passed = sum(r['passed'] for r in results)

    print(f"\nPassed: {n_passed}/{len(results)}")

    if all_passed:
        print("\n✓ ALL 3D TESTS PASSED")
    else:
        print(f"\n✗ SOME 3D TESTS FAILED")
        max_error = max(r['error'] for r in results)
        print(f"  Maximum error: {max_error:.4f}")
        print(f"  Suggested tolerance: {max_error * 1.2:.4f}")

    return all_passed, results


if __name__ == '__main__':
    passed_2d, results_2d = test_fBm_2D_circulant()
    print()
    passed_3d, results_3d = test_fBm_ND_circulant_3D()

    # Print max errors for tolerance calibration
    print("\n" + "=" * 70)
    print("SUMMARY FOR TOLERANCE CALIBRATION")
    print("=" * 70)

    max_error_2d = max(r['error'] for r in results_2d)
    max_error_3d = max(r['error'] for r in results_3d)
    print(f"2D max error: {max_error_2d:.4f}")
    print(f"3D max error: {max_error_3d:.4f}")

    all_passed = passed_2d and passed_3d
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
    else:
        print("\n✗ SOME TESTS FAILED")

    exit(0 if all_passed else 1)
