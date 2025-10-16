import numpy as np
from scaleinvariance import FIF_1D, two_point_intermittency_exponent

# Test parameters
size = 2**25
n_realizations = 2
C1_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.6]
alpha = 2.0
H = 1/3
order = 2
min_sep = 2**9
max_sep = 2**20

print(f"Testing C1 estimation on FIF_1D (size={size}, n={n_realizations} realizations each)")
print(f"Scaling range: {min_sep} to {max_sep}, order={order}\n")

all_pass = True

for C1_true in C1_values:
    # Generate realizations
    realizations = []
    for i in range(n_realizations):
        fif = FIF_1D(size, alpha=alpha, C1=C1_true, H=H)
        realizations.append(fif)

    # Stack along axis 0
    data = np.stack(realizations, axis=0)

    # Estimate C1
    C1_est, sigma_C1 = two_point_intermittency_exponent(
        data,
        order=order,
        assumed_alpha=alpha,
        scaling_method='structure_function',
        min_sep=min_sep,
        max_sep=max_sep,
        axis=1
    )

    # Check if within acceptable range (factor of 2, or < 0.002 for smallest case)
    if C1_true == 0.001:
        passes = C1_est < 0.002
    else:
        passes = (0.5 * C1_true < C1_est < 2.0 * C1_true)
    all_pass = all_pass and passes

    status = "PASS" if passes else "FAIL"
    print(f"C1_true={C1_true:.3f}  C1_est={C1_est:.3f}  unc={sigma_C1:.3f}  [{status}]")

print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
