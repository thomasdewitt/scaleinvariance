#!/usr/bin/env python3
"""
Comprehensive test for 2D FIF Hurst exponent estimation across parameter space.

Tests combinations of H, C1, and alpha, verifying that structure function
analysis correctly recovers the input Hurst exponent across different scaling
ranges in 2D fields.
"""
import numpy as np
from scaleinvariance import FIF_ND, structure_function_hurst, spectral_hurst

# Test configuration
KERNEL_METHOD = 'LS2010'  # Currently only 'LS2010' supported for 2D
HURST_METHOD = 'structure_function'  # 'structure_function' or 'spectral'
N_REALIZATIONS = 10
SIZE = 2**11  # 2048x2048 (reasonable for 2D)
TOLERANCE = 0.05

# Parameter space
H_VALUES = [0., 0.3, 0.7]
C1_VALUES = [0.0, 0.001, 0.1]
ALPHA_VALUES = [2.0, 1.5]

# Scaling ranges to test
SCALING_RANGES = [
    (1, 10),
    (10, 100),
    (100,1000)
]

def test_combination(H, C1, alpha):
    """
    Test a single parameter combination.

    Returns dict with results for each scaling range.
    """
    results = {}

    # Calculate K(2) intermittency correction for spectral method
    if C1 > 0 and HURST_METHOD == 'spectral':
        K_2 = C1 * (2**alpha - 2) / (alpha - 1)
        intermittency_correction = 0.5 * K_2
    else:
        intermittency_correction = 0.0

    # Generate stack of realizations
    realizations = []
    for i in range(N_REALIZATIONS):
        fif = FIF_ND((SIZE, SIZE), alpha, C1, H,
                     kernel_construction_method=KERNEL_METHOD, periodic=False)
        realizations.append(fif)

    # Stack along axis 0: shape = (N_REALIZATIONS, SIZE, SIZE)
    data_stack = np.stack(realizations, axis=0)

    # Test each scaling range with single call per range (analyze along axis 1)
    for min_sep, max_sep in SCALING_RANGES:
        if HURST_METHOD == 'structure_function':
            # Analyze along axis 1 (one spatial dimension)
            H_est, H_err = structure_function_hurst(
                data_stack, min_sep=min_sep, max_sep=max_sep, axis=1
            )
        elif HURST_METHOD == 'spectral':
            # Convert separation range to wavelength range
            min_wavelength = min_sep
            max_wavelength = max_sep
            H_est, H_err = spectral_hurst(
                data_stack, min_wavelength=min_wavelength,
                max_wavelength=max_wavelength, axis=1
            )
            # Apply intermittency correction
            H_est += intermittency_correction
        else:
            raise ValueError(f"Unknown HURST_METHOD: {HURST_METHOD}")

        passed = abs(H_est - H) < TOLERANCE

        results[(min_sep, max_sep)] = {
            'mean_H': H_est,
            'true_H': H,
            'passed': passed
        }

    return results

def format_table_row(H, C1, alpha, results):
    """Format a single row of the results table."""
    # For C1=0, alpha is N/A (fBm case)
    if C1 == 0.0:
        alpha_str = "N/A"
    else:
        alpha_str = f"{alpha:.1f}"

    # Create status indicators for each range
    statuses = []
    for range_tuple in SCALING_RANGES:
        if range_tuple in results:
            passed = results[range_tuple]['passed']
            mean_H = results[range_tuple]['mean_H']
            statuses.append(f"{'✓' if passed else '✗'} {mean_H:.3f}")
        else:
            statuses.append("N/A")

    row = f"  {H:.1f}    {C1:.3f}   {alpha_str:^5}   "
    row += "  ".join(f"{s:^11}" for s in statuses)

    return row

def main():
    print("="*90)
    print("2D FIF Hurst Exponent Estimation Test")
    print(f"Kernel method: {KERNEL_METHOD}")
    print(f"Hurst estimation method: {HURST_METHOD}")
    print(f"Realizations per test: {N_REALIZATIONS}")
    print(f"Size: {SIZE}x{SIZE}")
    print(f"Tolerance: ±{TOLERANCE}")
    print("="*90)
    print()

    # Table header
    print("Test Parameters      " + " "*5 + "Scaling Ranges (min_sep, max_sep)")
    print("-"*90)
    header = "  H      C1      α      "
    header += "  ".join(f"{f'{r[0]}-{r[1]}':^11}" for r in SCALING_RANGES)
    print(header)
    print("-"*90)

    all_passed = True
    test_count = 0
    passed_count = 0

    # Run all combinations
    for H in H_VALUES:
        for C1 in C1_VALUES:
            # For C1=0 (fBm case), skip alpha variations
            if C1 == 0.0:
                test_count += 1
                # Use dummy value for alpha (not used when C1=0)
                results = test_combination(H, C1, alpha=2.0)

                combo_passed = all(r['passed'] for r in results.values())
                if combo_passed:
                    passed_count += 1
                else:
                    all_passed = False

                row = format_table_row(H, C1, alpha=None, results=results)
                print(row)
            else:
                # For C1>0, test all alpha combinations
                for alpha in ALPHA_VALUES:
                    test_count += 1

                    # Run test
                    results = test_combination(H, C1, alpha)

                    # Check if all ranges passed
                    combo_passed = all(r['passed'] for r in results.values())
                    if combo_passed:
                        passed_count += 1
                    else:
                        all_passed = False

                    # Print row
                    row = format_table_row(H, C1, alpha, results)
                    print(row)
        print()
    print("-"*90)
    print(f"Tests passed: {passed_count}/{test_count}")
    print("="*90)

    if all_passed:
        print("\n✓ ALL TESTS PASSED")
    else:
        print(f"\n✗ SOME TESTS FAILED ({test_count - passed_count} combinations)")

    return all_passed

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
