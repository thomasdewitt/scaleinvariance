"""Heavy FIF non-NaN sweep kept separate from basic functionality tests."""

import numpy as np

from scaleinvariance import FIF_1D, FIF_ND


def test_fif_non_nan_parameter_sweep():
    """Check FIF outputs remain non-NaN over a broad parameter range."""
    size_1d = 4096 * 16 * 128
    n_loop = 10
    size_2d = (512, 512)

    alpha_1d = [0.5, 0.99, 1.01]
    C1_1d = [0.01, 0.2]
    H_1d = [0.8]
    causal_1d = [True, False]

    for alpha in alpha_1d:
        for C1 in C1_1d:
            for H in H_1d:
                for causal in causal_1d:
                    for _ in range(n_loop):
                        result = FIF_1D(size_1d, alpha, C1, H, causal=causal, periodic=True)
                        assert not np.any(np.isnan(result)), (
                            f"NaN in FIF_1D(alpha={alpha}, C1={C1}, H={H}, causal={causal})"
                        )

    # 2D sweep covers low/high intermittency and smooth/rough fields.
    alpha_2d = [1.1, 1.6, 1.95]
    C1_2d = [0.01, 0.2]
    H_2d = [0.1, 0.9]

    for alpha in alpha_2d:
        for C1 in C1_2d:
            for H in H_2d:
                result = FIF_ND(size_2d, alpha, C1, H, periodic=False)
                assert not np.any(np.isnan(result)), (
                    f"NaN in FIF_ND(alpha={alpha}, C1={C1}, H={H})"
                )
