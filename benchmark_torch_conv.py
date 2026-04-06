"""
Benchmark: scipy.signal.convolve vs torch.nn.functional.conv1d
for Haar fluctuation analysis on 1D (2^24) and 2D (4096^2) arrays.

Both have 2^24 = 16,777,216 total elements. The question is whether
the 2D layout (4096 short signals batched) benefits more from torch's
parallel conv1d than the 1D layout (one long signal).
"""
import numpy as np
import time
import gc
import torch
import torch.nn.functional as F
from scipy.signal import convolve as scipy_convolve

from scaleinvariance.simulation.fbm import fBm_1D_circulant, fBm_ND_circulant
from scaleinvariance.utils import process_lags, estimate_hurst_from_scaling


def measure_ram_bandwidth(size_mb=256, iters=5):
    n = size_mb * 1024 * 1024 // 8
    a = np.random.randn(n)
    b = np.empty_like(a)
    np.copyto(b, a)  # warmup

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        np.copyto(b, a)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    copy_bw = (2 * n * 8) / min(times) / 1e9

    times2 = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = np.sum(a)
        t1 = time.perf_counter()
        times2.append(t1 - t0)
    read_bw = (n * 8) / min(times2) / 1e9

    del a, b
    return copy_bw, read_bw


def haar_hurst_scipy(data, axis=0, max_sep=None):
    L = data.shape[axis]
    if max_sep is None:
        max_sep = L - 1
    lags = process_lags('powers of 1.2', max_sep, even_only=True)

    haar_flucs = []
    for lag in lags:
        if lag % 2 != 0 or lag > max_sep:
            haar_flucs.append(np.nan)
            continue
        kernel = np.ones(lag, dtype=np.float64) / (lag / 2)
        kernel[:lag // 2] *= -1
        kernel_shape = [1] * data.ndim
        kernel_shape[axis] = lag
        kernel = kernel.reshape(kernel_shape)
        abs_conv = np.abs(scipy_convolve(data, kernel, mode='valid', method='auto'))
        haar_flucs.append(np.nanmean(abs_conv))

    lags_arr = np.array(lags, dtype=np.int64)
    haar_arr = np.array(haar_flucs, dtype=np.float64)
    min_sep = 4 if L > 512 else 1
    return estimate_hurst_from_scaling(lags_arr, haar_arr, min_sep, max_sep, False)


def haar_hurst_torch(data, axis=0, max_sep=None):
    L = data.shape[axis]
    if max_sep is None:
        max_sep = L - 1
    lags = process_lags('powers of 1.2', max_sep, even_only=True)

    # Reshape for conv1d: (batch, 1, length)
    if data.ndim == 1:
        x = torch.from_numpy(data).to(torch.float32).unsqueeze(0).unsqueeze(0)
    elif data.ndim == 2:
        if axis == 0:
            # columns as batch: (4096, 1, 4096)
            x = torch.from_numpy(np.ascontiguousarray(data.T)).to(torch.float32).unsqueeze(1)
        else:
            x = torch.from_numpy(np.ascontiguousarray(data)).to(torch.float32).unsqueeze(1)
    else:
        raise ValueError("Only 1D/2D")

    haar_flucs = []
    for lag in lags:
        if lag % 2 != 0 or lag > max_sep:
            haar_flucs.append(np.nan)
            continue
        kernel = torch.ones(lag, dtype=torch.float32) / (lag / 2)
        kernel[:lag // 2] *= -1
        weight = kernel.reshape(1, 1, -1)
        conv_out = F.conv1d(x, weight)
        haar_flucs.append(conv_out.abs().mean().item())

    del x
    lags_arr = np.array(lags, dtype=np.int64)
    haar_arr = np.array(haar_flucs, dtype=np.float64)
    min_sep = 4 if L > 512 else 1
    return estimate_hurst_from_scaling(lags_arr, haar_arr, min_sep, max_sep, False)


def bench(func, data, axis=0, max_sep=None, n_runs=3, warmup=1, label=""):
    print(f"\n  {label} (warmup={warmup}, runs={n_runs})...")
    for _ in range(warmup):
        result = func(data, axis=axis, max_sep=max_sep)
    gc.collect()

    times = []
    for i in range(n_runs):
        gc.collect()
        t0 = time.perf_counter()
        result = func(data, axis=axis, max_sep=max_sep)
        t1 = time.perf_counter()
        dt = t1 - t0
        times.append(dt)
        print(f"    run {i+1}: {dt:.3f}s")

    best = min(times)
    med = np.median(times)
    print(f"    => best={best:.3f}s  median={med:.3f}s  H={result[0]:.4f}±{result[1]:.4f}")
    return best, med, result


if __name__ == '__main__':
    torch.set_num_threads(4)

    print("=" * 70)
    print("SYSTEM INFO")
    print("=" * 70)
    print(f"CPU: Intel Xeon @ 2.10GHz, 4 cores (Emerald Rapids)")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch threads: {torch.get_num_threads()}")

    print("\n--- RAM Bandwidth ---")
    copy_bw, read_bw = measure_ram_bandwidth()
    print(f"  Copy (read+write): {copy_bw:.1f} GB/s")
    print(f"  Read-only (sum):   {read_bw:.1f} GB/s")

    # ── Generate data ────────────────────────────────────────────────────
    H = 0.7
    from scaleinvariance import set_backend
    set_backend('numpy')

    print("\n" + "=" * 70)
    print("GENERATING DATA (H=0.7)")
    print("=" * 70)

    t0 = time.perf_counter()
    data_1d = fBm_1D_circulant(2**24, H)
    print(f"  1D fBm: shape={data_1d.shape}, {data_1d.nbytes/1e6:.0f} MB, "
          f"{time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    data_2d = fBm_ND_circulant((4096, 4096), H)
    print(f"  2D fBm: shape={data_2d.shape}, {data_2d.nbytes/1e6:.0f} MB, "
          f"{time.perf_counter()-t0:.2f}s")

    # Both arrays: 2^24 elements = 128 MB (float64)
    # For a fair comparison, use max_sep=4095 for both (matching 2D axis length)
    MAX_SEP = 4095
    N_RUNS = 3

    # ── 1D BENCHMARK ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"1D BENCHMARK: 2^24 points, max_sep={MAX_SEP}")
    print(f"  (one signal of length 16,777,216)")
    print(f"{'='*70}")

    b_s, m_s, r_s = bench(haar_hurst_scipy, data_1d, max_sep=MAX_SEP,
                           n_runs=N_RUNS, label="Scipy")
    b_t, m_t, r_t = bench(haar_hurst_torch, data_1d, max_sep=MAX_SEP,
                           n_runs=N_RUNS, label="Torch conv1d")
    print(f"\n  >>> 1D Speedup: {b_s/b_t:.2f}x (best)  {m_s/m_t:.2f}x (median)")

    # ── 2D BENCHMARK ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"2D BENCHMARK: 4096x4096, axis=0")
    print(f"  (4096 signals of length 4096, batched in torch)")
    print(f"{'='*70}")

    b_s, m_s, r_s = bench(haar_hurst_scipy, data_2d, axis=0,
                           n_runs=N_RUNS, label="Scipy")
    b_t, m_t, r_t = bench(haar_hurst_torch, data_2d, axis=0,
                           n_runs=N_RUNS, label="Torch conv1d")
    print(f"\n  >>> 2D Speedup: {b_s/b_t:.2f}x (best)  {m_s/m_t:.2f}x (median)")

    # ── DATA SIZE ANALYSIS ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("MEMORY / BANDWIDTH ANALYSIS")
    print(f"{'='*70}")
    n_lags = len(process_lags('powers of 1.2', MAX_SEP, even_only=True))
    print(f"  Number of lags computed: {n_lags}")
    print(f"  Data size: {data_1d.nbytes/1e6:.0f} MB (float64)")
    print(f"  Torch input: {data_1d.nbytes/2/1e6:.0f} MB (float32)")
    print(f"  Per-lag data touched (1D): ~{data_1d.nbytes/1e6:.0f} MB read + write")
    print(f"  Per-lag data touched (2D): same total, but 4096 short conv ops")
    print(f"  Total data touched: ~{n_lags * data_1d.nbytes * 2 / 1e9:.1f} GB")
    print(f"  RAM bandwidth: ~{read_bw:.1f} GB/s read")
    print(f"  Bandwidth-limited floor: ~{n_lags * data_1d.nbytes * 2 / 1e9 / read_bw:.1f}s")
