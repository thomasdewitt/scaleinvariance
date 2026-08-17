# Changelog

Notable changes to scaleinvariance. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Entries before 0.14.0 are reconstructed retrospectively from commit history.

## [0.14.0] - 2026-08-17

First release since 0.11.0. Versions 0.12.0 and 0.13.x exist as commits but were never published, so everything below is new relative to the last PyPI release.

### Fixed

- **`FIF_ND` produced fields ~1.57x more intermittent than requested, at every alpha, since inception.** `FIF_ND` scaled the flux generator by `(2**n_causal * C1)**(1/alpha)` — the 1-D amplitude formula — in every dimension. The intermittency a cascade accumulates per e-folding of scale goes as `A**alpha` times the log-rate `d(sum |g|^alpha)/d(ln lam)` of the flux kernel, and on the `dx=2` lattice that rate carries an angular factor `Omega_d / 2**d`: 1 in 1-D, `pi/2` in 2-D and 3-D. The fix is to divide `C1` by that factor. This is the `NDf` constant in Lovejoy's `eps2D.m`, found by cross-validating against it.

  **Every N-D FIF field produced by 0.11.0 or earlier has an effective `C1 ~ 1.57 * C1_requested`.** Fields simulated on older versions are not reproducible on this one and should not be compared against it. `FIF_1D` output is unchanged — it is the calibrated reference convention.

  Verification: the measured alpha-power log-rate ratio of the discrete kernels matches the analytic constant to 1.2e-4 in 2-D (1024², alpha = 2.0/1.8/1.4, with and without an outer scale) and 6.5e-3 in 3-D (256³). End to end on identical noise (2048², alpha=2, C1=0.1, 6 realizations), trace moments on the box-averaged flux read `C1_eff` = 0.143 before and 0.093 after, against an estimator that itself runs a few percent low (Lovejoy's own fields read 0.097 on it).

- **`extremal_levy` raised `TypeError` for tuple `size` on the numpy backend.** `backend.rand`/`randn` took only the varargs form (`rand(2, 3)`), so the tuple form worked on torch — where `torch.rand` accepts either — and raised on numpy. Both now accept both forms. Internal FIF paths were unaffected (they pass an int and reshape); this hit `extremal_levy(alpha, size=(n, n))` from user code.

- **`FIF_ND` returned an all-NaN field for incoherent GSI parameters, silently.** An `elliptical_dim` outside `[1, ndim]` sends the LS2010 correction term negative near the origin, and the flux kernel's `|.|**(1/(alpha-1))` yields NaN, which the FFT then spreads over the whole field. `FIF_ND` now raises. The range check is necessary but not sufficient — an in-range `elliptical_dim` paired with a metric implying a different one (`Hz=0.5` with `D_el=2.0`) produces the same NaN — so the constructed flux kernel is also checked for finiteness before use. The `FIF_ND` docstring's own GSI example was one of the incoherent cases and returned an all-NaN field when run verbatim; it now uses `canonical_scale_metric` with `elliptical_dim = 1 + Hz`.

### Added

- **`wavelet_fluctuation()` / `wavelet_fluctuation_hurst()`** — order-`q` fluctuation analysis over a registry of four wavelets, selected with `wavelet=`:
  - `'haar'` — difference of half-window means (1 vanishing moment)
  - `'structure_function'` — first-difference "poor man's wavelet"
  - `'mexican_hat'` — Ricker, 2 vanishing moments, `sigma = r/sqrt(3)`, cut ±5σ
  - `'morlet'` — complex (ω₀=6), modulus fluctuation, `sigma = 6r/pi`, cut ±4σ

  Kernels are L1-normalized (`sum|g_r|` fixed across scales, mean removed) so `F_q(r) ~ r^{qH-K(q)}` with the same slope for every wavelet and a different amplitude. The Mexican hat and Morlet have more vanishing moments and so reach higher `H` than Haar. `haar_fluctuation`/`haar_fluctuation_hurst` are now thin wrappers over this path and are bit-identical to their previous implementation; `wavelet='structure_function'` reproduces `structure_function()` to numerical precision, with the dedicated function remaining the fast path. Lag is anchored peak-to-first-trough, so at `r=1` the first trough sits on the grid neighbour; lags whose kernel exceeds the domain return NaN. On gappy real data prefer the dedicated `structure_function`/`haar_fluctuation` — the wide wavelets shed large-lag points.

- **`periodic=` on the increment and fluctuation analysis functions**, for data on a periodic (toroidal) domain — e.g. the output of the `periodic=True` simulators. Increments and Haar windows wrap around the array end along `axis`, so every lag uses all `L` samples instead of dropping the `L - r` edge pairs. Threaded through `structure_function`, `costructure_function`, `haar_fluctuation`, `wavelet_fluctuation`, their `_hurst` variants, `K_empirical`, and `two_point_C1`. On a periodic domain `S(r) = S(L - r)` exactly, so structure-function lags are capped at `L // 2` and an explicit larger `max_sep` raises `ValueError`; the Haar fluctuation has no such symmetry and keeps the full lag range. `spectral_*` are FFT-based and already periodic.

### Changed

- **`outer_scale=None` now means "apply no outer-scale cutoff"** in `FIF_1D`, `FIF_ND`, and `fBm_1D`. It was previously remapped to the domain size, which applied a Hanning-window taper to the real-space kernels (and to the always-LS2010 flux kernel) by default. An explicit numeric `outer_scale` behaves exactly as before. H recovery is statistically unchanged against the old default (Δ ~0.002–0.003); the taper removal affects only the largest scales. Passing a non-default `outer_scale_width_factor` alongside `outer_scale=None` now warns, since it is inert.

### Known limitations

- **`C1` is not calibrated under GSI.** The shipped `Omega_d / 2**d` is the Euclidean answer; the correct factor for an anisotropic metric is that metric's own unit-ball measure, which has no closed form. At `Hz = 5/9` the realized `C1` is around 4x the requested value. `Omega_{D_el} / 2**{D_el}` is not the fix — it moves the constant downward, where the true rate is above the isotropic one. Exponents that do not involve the cascade amplitude are unaffected.

## [0.11.0] - 2026-05-18

Last release published to PyPI before 0.14.0.

### Added

- N-D causal FIF and `observable_kernel_odd_axes`.
- `costructure_function`; vectorized `order` in `structure_function` and `haar_fluctuation`.

### Changed

- `scale_metric_dim` renamed to `elliptical_dim` (deprecation alias retained).
- Generalized scale invariance documented in the README and API docs.

## [0.10.0] - 2026-04-22

### Changed

- Beta development classifier; `requires-python >= 3.10`.

## [0.9.0] - 2026-04-04

### Removed

- Spectral kernel support for the **flux** (retained for the observable) — nothing enforced positivity on an IFFT'd spectral flux kernel.

### Changed

- Spectral kernels are returned in Fourier space directly rather than round-tripped through real space; `periodic_convolve` gained `kernel_is_fourier=True`.
- Defaults for `FIF_1D`/`FIF_ND`: `causal=False`, `kernel_construction_method_observable='spectral'`.
- Silent fallbacks to LS2010 (spectral kernel with `causal=True`, or with a `scale_metric`) now raise `ValueError`.

## [0.8.0] - 2026-04-02

Earlier releases (0.1.0, 2025-07-24, through 0.8.1) are not itemized here; see the git history.
