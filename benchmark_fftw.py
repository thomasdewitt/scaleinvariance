"""Benchmark FFTW vs NumPy backends for 2048^2 FIF simulations."""

import time
import os
import numpy as np
import psutil
import threading
import scaleinvariance.backend as B
from scaleinvariance import FIF_ND

SIZE = (2048, 2048)
PARAMS = dict(alpha=1.8, C1=0.1, H=0.3)
N_RUNS = 3


class ResourceMonitor:
    """Monitor CPU and memory usage in a background thread."""
    def __init__(self, interval=0.1):
        self.interval = interval
        self.cpu_samples = []
        self.rss_samples = []
        self._stop = threading.Event()
        self._thread = None
        self.process = psutil.Process(os.getpid())

    def start(self):
        self.cpu_samples = []
        self.rss_samples = []
        self._stop.clear()
        self.process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self):
        while not self._stop.is_set():
            self.cpu_samples.append(self.process.cpu_percent(interval=None))
            self.rss_samples.append(self.process.memory_info().rss)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self._thread.join()

    @property
    def max_cpu_percent(self):
        # Filter out obvious glitches (>n_cpus*100%)
        n_cpus = os.cpu_count() or 4
        limit = n_cpus * 100
        valid = [s for s in self.cpu_samples if s <= limit]
        return max(valid) if valid else 0.0

    @property
    def avg_cpu_percent(self):
        n_cpus = os.cpu_count() or 4
        limit = n_cpus * 100
        valid = [s for s in self.cpu_samples if s <= limit]
        return np.mean(valid) if valid else 0.0

    @property
    def max_rss_mb(self):
        return max(self.rss_samples) / (1024 * 1024) if self.rss_samples else 0.0


def benchmark_backend(name, warmup=True):
    n_cpus = os.cpu_count()
    B.set_backend(name)
    print(f"\n{'='*70}")
    print(f"Backend: {name}  |  threads: {B._num_threads}  |  size: {SIZE}  |  cpus: {n_cpus}")
    print(f"{'='*70}")

    if warmup and name == 'fftw':
        print("  (warmup: running one sim to build FFTW plans...)")
        np.random.seed(0)
        t0 = time.perf_counter()
        _ = FIF_ND(SIZE, **PARAMS)
        plan_time = time.perf_counter() - t0
        print(f"  warmup done in {plan_time:.2f}s (includes FFTW_MEASURE planning)")

    monitor = ResourceMonitor(interval=0.05)
    times = []
    all_max_cpu = []
    all_avg_cpu = []
    all_max_rss = []

    for i in range(N_RUNS):
        np.random.seed(42 + i)
        monitor.start()
        t0 = time.perf_counter()
        result = FIF_ND(SIZE, **PARAMS)
        t1 = time.perf_counter()
        monitor.stop()
        elapsed = t1 - t0
        times.append(elapsed)

        max_cpu = monitor.max_cpu_percent
        avg_cpu = monitor.avg_cpu_percent
        max_rss = monitor.max_rss_mb
        all_max_cpu.append(max_cpu)
        all_avg_cpu.append(avg_cpu)
        all_max_rss.append(max_rss)

        max_cores = max_cpu / 100.0
        print(f"  run {i+1}: {elapsed:6.2f}s  |  peak CPU: {max_cpu:5.1f}% ({max_cores:.1f} cores)"
              f"  |  avg CPU: {avg_cpu:5.1f}%"
              f"  |  peak RSS: {max_rss:7.1f} MB")

    avg = np.mean(times)
    print(f"  --> average: {avg:.2f}s  |  avg peak CPU: {np.mean(all_max_cpu):.1f}%"
          f"  |  avg peak RSS: {np.mean(all_max_rss):.1f} MB")
    return avg, np.mean(all_max_cpu), np.mean(all_avg_cpu), np.mean(all_max_rss)


if __name__ == '__main__':
    print(f"FIF_ND benchmark: {SIZE[0]}x{SIZE[1]}, alpha={PARAMS['alpha']}, C1={PARAMS['C1']}, H={PARAMS['H']}")
    print(f"Runs per backend: {N_RUNS}, CPUs: {os.cpu_count()}")

    results = {}

    for backend in ['numpy', 'fftw']:
        try:
            results[backend] = benchmark_backend(backend)
        except Exception as e:
            print(f"\n{backend} FAILED: {e}")
            import traceback; traceback.print_exc()

    try:
        results['torch'] = benchmark_backend('torch')
    except (ImportError, Exception) as e:
        print(f"\ntorch skipped: {e}")

    print(f"\n{'='*70}")
    print("SUMMARY (post-warmup timings)")
    print(f"{'='*70}")
    print(f"  {'Backend':>8}  {'Avg Time':>8}  {'Speedup':>8}  {'Peak CPU':>9}  {'Avg CPU':>9}  {'Peak RSS':>10}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*10}")
    baseline_time = results.get('numpy', (0,))[0]
    for name, (avg, peak_cpu, avg_cpu, rss) in sorted(results.items(), key=lambda x: x[1][0]):
        speedup = f"{baseline_time/avg:.2f}x" if baseline_time and name != 'numpy' else "1.00x"
        print(f"  {name:>8}  {avg:7.2f}s  {speedup:>8}  {peak_cpu:7.1f}%  {avg_cpu:7.1f}%  {rss:7.1f} MB")
