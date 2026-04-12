"""
Backend module for scaleinvariance - provides unified numpy/torch interface.

Automatically detects PyTorch availability and provides wrapper functions
that work with either numpy or torch. All functions return numpy arrays.

Numerical precision is controlled by ``set_numerical_precision('float32'|'float64')``
and applies to both numpy and torch backends. The default is ``'float64'``.
"""

import numpy as np
import os

# Try to import torch
try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False

# Global state
_backend = 'numpy'  # Default backend
_num_threads = int(os.cpu_count() * 0.9) if os.cpu_count() else 4
_numerical_precision = 'float64'  # Default precision

# Auto-detect torch if available
if _torch_available:
    _backend = 'torch'


def set_backend(backend='numpy'):
    """
    Set computational backend.

    Parameters
    ----------
    backend : str
        Backend to use: 'numpy' or 'torch'

    Raises
    ------
    ValueError
        If backend is not recognized
    ImportError
        If torch backend requested but torch not installed
    """
    global _backend

    if backend not in ('numpy', 'torch'):
        raise ValueError(f"Unknown backend: {backend}. Must be 'numpy' or 'torch'")

    if backend == 'torch' and not _torch_available:
        raise ImportError("PyTorch not available. Install with: pip install torch")

    _backend = backend

    # Set threading for the backend
    if _backend == 'torch':
        torch.set_num_threads(_num_threads)
    else:
        os.environ['OMP_NUM_THREADS'] = str(_num_threads)
        os.environ['MKL_NUM_THREADS'] = str(_num_threads)


def get_backend():
    """Get current backend name."""
    return _backend


def set_num_threads(n):
    """
    Set number of threads for parallel operations.

    Parameters
    ----------
    n : int
        Number of threads to use
    """
    global _num_threads
    _num_threads = int(n)

    if _backend == 'torch':
        torch.set_num_threads(_num_threads)
    else:
        os.environ['OMP_NUM_THREADS'] = str(_num_threads)
        os.environ['MKL_NUM_THREADS'] = str(_num_threads)


def set_numerical_precision(precision='float64'):
    """
    Set numerical precision for all simulation computations.

    Controls the default dtype for arrays created by backend primitives
    (array creators, random draws, FFTs) and is honored by the simulation
    methods. Applies to both numpy and torch backends.

    Parameters
    ----------
    precision : str
        Numerical precision: ``'float32'`` or ``'float64'``. Default ``'float64'``.

    Raises
    ------
    ValueError
        If precision is not one of ``'float32'`` or ``'float64'``.
    """
    global _numerical_precision
    if precision not in ('float32', 'float64'):
        raise ValueError(
            f"Unknown precision: {precision}. Must be 'float32' or 'float64'"
        )
    _numerical_precision = precision


def get_numerical_precision():
    """Get current numerical precision ('float32' or 'float64')."""
    return _numerical_precision


# ============================================================================
# Dtype helpers
# ============================================================================

def _active_real_dtype_np():
    """Active real numpy dtype for the current numerical precision."""
    return np.float32 if _numerical_precision == 'float32' else np.float64


def _active_complex_dtype_np():
    """Active complex numpy dtype for the current numerical precision."""
    return np.complex64 if _numerical_precision == 'float32' else np.complex128


def _active_real_dtype_torch():
    """Active real torch dtype for the current numerical precision."""
    return torch.float32 if _numerical_precision == 'float32' else torch.float64


def _active_complex_dtype_torch():
    """Active complex torch dtype for the current numerical precision."""
    return torch.complex64 if _numerical_precision == 'float32' else torch.complex128


def _np_to_torch_dtype(dtype):
    """Map a numpy dtype to the corresponding torch dtype.

    Returns the active real torch dtype for unrecognized inputs.
    """
    if dtype is None:
        return _active_real_dtype_torch()
    key = np.dtype(dtype).type
    if key is np.float32:
        return torch.float32
    if key is np.float64:
        return torch.float64
    if key is np.int32:
        return torch.int32
    if key is np.int64:
        return torch.int64
    if key is np.complex64:
        return torch.complex64
    if key is np.complex128:
        return torch.complex128
    return _active_real_dtype_torch()


def exp_clip_limits():
    """Return ``(lo, hi)`` safe exponent range for ``exp`` at active precision.

    For float32: ``(-88.0, 88.0)``, just below ``log(float32_max) ≈ 88.72``.
    For float64: ``(-700.0, 700.0)``, just below ``log(float64_max) ≈ 709.78``.
    """
    if _numerical_precision == 'float32':
        return (-88.0, 88.0)
    return (-700.0, 700.0)


def eps():
    """Return a small clamp epsilon appropriate for the active precision.

    Approximately 100× machine epsilon: ``1e-6`` for float32 and ``1e-12``
    for float64.
    """
    return 1e-6 if _numerical_precision == 'float32' else 1e-12


# ============================================================================
# FFT Operations
# ============================================================================

def fftfreq(n, d=1.0):
    """FFT frequency array."""
    if _backend == 'numpy':
        return np.fft.fftfreq(n, d=d).astype(_active_real_dtype_np(), copy=False)
    else:
        result = torch.fft.fftfreq(n, d=d, dtype=_active_real_dtype_torch())
        return result.numpy()


def rfftfreq(n, d=1.0):
    """Real FFT frequency array (non-negative half)."""
    if _backend == 'numpy':
        return np.fft.rfftfreq(n, d=d).astype(_active_real_dtype_np(), copy=False)
    else:
        result = torch.fft.rfftfreq(n, d=d, dtype=_active_real_dtype_torch())
        return result.numpy()


def fft(x, axis=-1):
    """1D FFT."""
    if _backend == 'numpy':
        return np.fft.fft(x, axis=axis)
    else:
        x_torch = torch.as_tensor(x, dtype=_active_complex_dtype_torch())
        result = torch.fft.fft(x_torch, dim=axis)
        return result.numpy()


def ifft(x, axis=-1):
    """1D inverse FFT."""
    if _backend == 'numpy':
        return np.fft.ifft(x, axis=axis)
    else:
        x_torch = torch.as_tensor(x, dtype=_active_complex_dtype_torch())
        result = torch.fft.ifft(x_torch, dim=axis)
        return result.numpy()


def fft2(x):
    """2D FFT."""
    if _backend == 'numpy':
        return np.fft.fft2(x)
    else:
        x_torch = torch.as_tensor(x, dtype=_active_complex_dtype_torch())
        result = torch.fft.fft2(x_torch)
        return result.numpy()


def ifft2(x):
    """2D inverse FFT."""
    if _backend == 'numpy':
        return np.fft.ifft2(x)
    else:
        x_torch = torch.as_tensor(x, dtype=_active_complex_dtype_torch())
        result = torch.fft.ifft2(x_torch)
        return result.numpy()


def fftn(x, axes=None):
    """N-D FFT."""
    if _backend == 'numpy':
        return np.fft.fftn(x, axes=axes)
    else:
        x_torch = torch.as_tensor(x, dtype=_active_complex_dtype_torch())
        result = torch.fft.fftn(x_torch, dim=axes)
        return result.numpy()


def ifftn(x, axes=None):
    """N-D inverse FFT."""
    if _backend == 'numpy':
        return np.fft.ifftn(x, axes=axes)
    else:
        x_torch = torch.as_tensor(x, dtype=_active_complex_dtype_torch())
        result = torch.fft.ifftn(x_torch, dim=axes)
        return result.numpy()


def rfft(x, axis=-1):
    """1D real FFT."""
    if _backend == 'numpy':
        return np.fft.rfft(x, axis=axis)
    else:
        x_torch = torch.as_tensor(x, dtype=_active_real_dtype_torch())
        result = torch.fft.rfft(x_torch, dim=axis)
        return result.numpy()


def irfft(x, n, axis=-1):
    """1D inverse real FFT; ``n`` is the output length along ``axis``."""
    if _backend == 'numpy':
        return np.fft.irfft(x, n=n, axis=axis)
    else:
        x_torch = torch.as_tensor(x, dtype=_active_complex_dtype_torch())
        result = torch.fft.irfft(x_torch, n=n, dim=axis)
        return result.numpy()


def rfftn(x, axes=None):
    """N-D real FFT (real input, last axis halved)."""
    if _backend == 'numpy':
        return np.fft.rfftn(x, axes=axes)
    else:
        x_torch = torch.as_tensor(x, dtype=_active_real_dtype_torch())
        result = torch.fft.rfftn(x_torch, dim=axes)
        return result.numpy()


def irfft2(x, s):
    """2D inverse real FFT."""
    if _backend == 'numpy':
        return np.fft.irfft2(x, s=s)
    else:
        x_torch = torch.as_tensor(x, dtype=_active_complex_dtype_torch())
        result = torch.fft.irfft2(x_torch, s=s)
        return result.numpy()


def irfftn(x, s):
    """N-D inverse real FFT."""
    axes = list(range(-len(s), 0))
    if _backend == 'numpy':
        return np.fft.irfftn(x, s=s, axes=axes)
    else:
        x_torch = torch.as_tensor(x, dtype=_active_complex_dtype_torch())
        result = torch.fft.irfftn(x_torch, s=s, dim=axes)
        return result.numpy()


def ifftshift(x, axes=None):
    """Inverse FFT shift."""
    if _backend == 'numpy':
        return np.fft.ifftshift(x, axes=axes)
    else:
        x_torch = torch.as_tensor(x)
        result = torch.fft.ifftshift(x_torch, dim=axes)
        return result.numpy()


# ============================================================================
# Random Number Generation
# ============================================================================

def randn(*shape):
    """Standard normal random numbers at the active real precision."""
    target = _active_real_dtype_np()
    if _backend == 'numpy':
        result = np.random.randn(*shape)
        if result.dtype != target:
            result = result.astype(target, copy=False)
        return result
    else:
        result = torch.randn(*shape, dtype=_active_real_dtype_torch())
        return result.numpy()


def rand(*shape):
    """Uniform random numbers [0, 1) at the active real precision."""
    target = _active_real_dtype_np()
    if _backend == 'numpy':
        result = np.random.rand(*shape)
        if result.dtype != target:
            result = result.astype(target, copy=False)
        return result
    else:
        result = torch.rand(*shape, dtype=_active_real_dtype_torch())
        return result.numpy()


def exponential(scale, size=None):
    """
    Exponential random numbers at the active real precision.

    Parameters
    ----------
    scale : float
        Scale parameter (1/lambda)
    size : int or tuple, optional
        Output shape
    """
    target = _active_real_dtype_np()
    if _backend == 'numpy':
        if size is None:
            result = np.asarray(np.random.exponential(scale))
        else:
            result = np.random.exponential(scale, size=size)
        if result.dtype != target:
            result = result.astype(target, copy=False)
        return result
    else:
        if size is None:
            size = 1
        rate = torch.tensor(1.0 / scale, dtype=_active_real_dtype_torch())
        dist = torch.distributions.Exponential(rate)
        result = dist.sample(size if isinstance(size, tuple) else (size,))
        return result.numpy()


# ============================================================================
# Array Creation
# ============================================================================

def zeros(shape, dtype=None):
    """Create zero array. Defaults to active real dtype."""
    if dtype is None:
        dtype = _active_real_dtype_np()
    if _backend == 'numpy':
        return np.zeros(shape, dtype=dtype)
    else:
        torch_dtype = _np_to_torch_dtype(dtype)
        result = torch.zeros(shape, dtype=torch_dtype)
        return result.numpy()


def ones(shape, dtype=None):
    """Create ones array. Defaults to active real dtype."""
    if dtype is None:
        dtype = _active_real_dtype_np()
    if _backend == 'numpy':
        return np.ones(shape, dtype=dtype)
    else:
        torch_dtype = _np_to_torch_dtype(dtype)
        result = torch.ones(shape, dtype=torch_dtype)
        return result.numpy()


def arange(start, stop=None, step=1, dtype=None):
    """Create array with evenly spaced values. Defaults to active real dtype."""
    if dtype is None:
        dtype = _active_real_dtype_np()
    if _backend == 'numpy':
        return np.arange(start, stop, step, dtype=dtype)
    else:
        torch_dtype = _np_to_torch_dtype(dtype)
        if stop is None:
            stop = start
            start = 0
        result = torch.arange(start, stop, step, dtype=torch_dtype)
        return result.numpy()


def asarray(x, dtype=None):
    """Convert to array. If ``dtype`` is None, preserves the input dtype.

    Unlike earlier versions, ``asarray`` no longer silently promotes every
    input to float64. Pass an explicit ``dtype=`` if you need a cast.
    """
    if _backend == 'numpy':
        if dtype is None:
            return np.asarray(x)
        return np.asarray(x, dtype=dtype)
    else:
        if dtype is None:
            x_torch = torch.as_tensor(x)
        else:
            torch_dtype = _np_to_torch_dtype(dtype)
            x_torch = torch.as_tensor(x, dtype=torch_dtype)
        return x_torch.numpy()


def full_like(x, fill_value):
    """Create array with same shape/dtype as x, filled with value."""
    if _backend == 'numpy':
        return np.full_like(x, fill_value)
    else:
        x_torch = torch.as_tensor(x)
        result = torch.full_like(x_torch, fill_value, dtype=x_torch.dtype)
        return result.numpy()


def concatenate(arrays, axis=0):
    """Concatenate arrays along axis."""
    if _backend == 'numpy':
        return np.concatenate(arrays, axis=axis)
    else:
        arrays_torch = [torch.as_tensor(a) for a in arrays]
        result = torch.cat(arrays_torch, dim=axis)
        return result.numpy()


# ============================================================================
# Array Manipulation
# ============================================================================

def meshgrid(*arrays, indexing='xy'):
    """Create mesh grids."""
    if _backend == 'numpy':
        return np.meshgrid(*arrays, indexing=indexing)
    else:
        arrays_torch = [torch.as_tensor(a, dtype=_active_real_dtype_torch())
                        for a in arrays]
        result = torch.meshgrid(*arrays_torch, indexing=indexing)
        return tuple(r.numpy() for r in result)


def flip(x, axis):
    """Flip array along axis."""
    if _backend == 'numpy':
        return np.flip(x, axis=axis)
    else:
        x_torch = torch.as_tensor(x)
        result = torch.flip(x_torch, dims=(axis,))
        return result.numpy()


def conj(x):
    """Complex conjugate. Preserves input dtype."""
    if _backend == 'numpy':
        return np.conj(x)
    else:
        x_torch = torch.as_tensor(x)
        result = torch.conj(x_torch)
        # resolve_conj() is needed before calling numpy() on conjugated tensor
        return result.resolve_conj().numpy()


def where(condition, x, y):
    """Element-wise selection."""
    if _backend == 'numpy':
        return np.where(condition, x, y)
    else:
        cond_torch = torch.as_tensor(np.asarray(condition, dtype=np.bool_))
        x_torch = torch.as_tensor(x)
        y_torch = torch.as_tensor(y)
        result = torch.where(cond_torch, x_torch, y_torch)
        return result.numpy()


def clip(x, x_min, x_max):
    """Clip values to range. ``x_min`` or ``x_max`` may be ``None``."""
    if _backend == 'numpy':
        return np.clip(x, x_min, x_max)
    else:
        x_torch = torch.as_tensor(x)
        result = torch.clamp(x_torch, min=x_min, max=x_max)
        return result.numpy()


def maximum(a, b):
    """Element-wise maximum of two arrays (or array and scalar)."""
    if _backend == 'numpy':
        return np.maximum(a, b)
    else:
        a_torch = torch.as_tensor(a)
        if isinstance(b, torch.Tensor):
            b_torch = b
        else:
            b_torch = torch.as_tensor(b, dtype=a_torch.dtype)
        result = torch.maximum(a_torch, b_torch)
        return result.numpy()


# ============================================================================
# Mathematical Operations
# ============================================================================

def abs(x):
    """Absolute value."""
    if _backend == 'numpy':
        return np.abs(x)
    else:
        x_torch = torch.as_tensor(x)
        result = torch.abs(x_torch)
        return result.numpy()


def sqrt(x):
    """Square root."""
    if _backend == 'numpy':
        return np.sqrt(x)
    else:
        x_torch = torch.as_tensor(x)
        result = torch.sqrt(x_torch)
        return result.numpy()


def sign(x):
    """Sign of values."""
    if _backend == 'numpy':
        return np.sign(x)
    else:
        x_torch = torch.as_tensor(x)
        result = torch.sign(x_torch)
        return result.numpy()


def exp(x):
    """Exponential function."""
    if _backend == 'numpy':
        return np.exp(x)
    else:
        x_torch = torch.as_tensor(x)
        result = torch.exp(x_torch)
        return result.numpy()


def cos(x):
    """Cosine function."""
    if _backend == 'numpy':
        return np.cos(x)
    else:
        x_torch = torch.as_tensor(x)
        result = torch.cos(x_torch)
        return result.numpy()


def sin(x):
    """Sine function."""
    if _backend == 'numpy':
        return np.sin(x)
    else:
        x_torch = torch.as_tensor(x)
        result = torch.sin(x_torch)
        return result.numpy()


# ============================================================================
# Reductions
# ============================================================================

def mean(x, axis=None):
    """Mean of array."""
    if _backend == 'numpy':
        return np.mean(x, axis=axis)
    else:
        x_torch = torch.as_tensor(x)
        if axis is None:
            result = torch.mean(x_torch)
        else:
            result = torch.mean(x_torch, dim=axis)
        return result.numpy()


def std(x, axis=None):
    """Standard deviation (population, ddof=0 — matches numpy default)."""
    if _backend == 'numpy':
        return np.std(x, axis=axis)
    else:
        x_torch = torch.as_tensor(x)
        if axis is None:
            result = torch.std(x_torch, correction=0)
        else:
            result = torch.std(x_torch, dim=axis, correction=0)
        return result.numpy()


def sum(x, axis=None):
    """Sum of array."""
    if _backend == 'numpy':
        return np.sum(x, axis=axis)
    else:
        x_torch = torch.as_tensor(x)
        if axis is None:
            result = torch.sum(x_torch)
        else:
            result = torch.sum(x_torch, dim=axis)
        return result.numpy()


def cumsum(x, axis=0):
    """Cumulative sum."""
    if _backend == 'numpy':
        return np.cumsum(x, axis=axis)
    else:
        x_torch = torch.as_tensor(x)
        result = torch.cumsum(x_torch, dim=axis)
        return result.numpy()


def diff(x, axis=-1):
    """Difference along axis."""
    if _backend == 'numpy':
        return np.diff(x, axis=axis)
    else:
        x_torch = torch.as_tensor(x)
        result = torch.diff(x_torch, dim=axis)
        return result.numpy()


# ============================================================================
# Constants and Utilities
# ============================================================================

pi = np.pi


def real(x):
    """Real part of array."""
    if _backend == 'numpy':
        return np.real(x)
    else:
        x_torch = torch.as_tensor(x)
        result = torch.real(x_torch)
        return result.numpy()


def imag(x):
    """Imaginary part of array."""
    if _backend == 'numpy':
        return np.imag(x)
    else:
        x_torch = torch.as_tensor(x)
        result = torch.imag(x_torch)
        return result.numpy()
