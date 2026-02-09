"""
Backend module for scaleinvariance - provides unified numpy/torch interface.

Automatically detects PyTorch availability and provides wrapper functions
that work with either numpy or torch. All functions return numpy arrays.
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


# ============================================================================
# FFT Operations
# ============================================================================

def fftfreq(n, d=1.0):
    """FFT frequency array."""
    if _backend == 'numpy':
        return np.fft.fftfreq(n, d=d)
    else:
        return torch.fft.fftfreq(n, d=d).numpy()


def fft(x, axis=-1):
    """1D FFT."""
    if _backend == 'numpy':
        return np.fft.fft(x, axis=axis)
    else:
        x_torch = torch.as_tensor(x, dtype=torch.complex128)
        result = torch.fft.fft(x_torch, axis=axis)
        return result.numpy()


def ifft(x, axis=-1):
    """1D inverse FFT."""
    if _backend == 'numpy':
        return np.fft.ifft(x, axis=axis)
    else:
        x_torch = torch.as_tensor(x, dtype=torch.complex128)
        result = torch.fft.ifft(x_torch, axis=axis)
        return result.numpy()


def fft2(x):
    """2D FFT."""
    if _backend == 'numpy':
        return np.fft.fft2(x)
    else:
        x_torch = torch.as_tensor(x, dtype=torch.complex128)
        result = torch.fft.fft2(x_torch)
        return result.numpy()


def ifft2(x):
    """2D inverse FFT."""
    if _backend == 'numpy':
        return np.fft.ifft2(x)
    else:
        x_torch = torch.as_tensor(x, dtype=torch.complex128)
        result = torch.fft.ifft2(x_torch)
        return result.numpy()


def fftn(x, axes=None):
    """N-D FFT."""
    if _backend == 'numpy':
        return np.fft.fftn(x, axes=axes)
    else:
        x_torch = torch.as_tensor(x, dtype=torch.complex128)
        # torch uses 'dim' instead of 'axes'
        result = torch.fft.fftn(x_torch, dim=axes)
        return result.numpy()


def ifftn(x, axes=None):
    """N-D inverse FFT."""
    if _backend == 'numpy':
        return np.fft.ifftn(x, axes=axes)
    else:
        x_torch = torch.as_tensor(x, dtype=torch.complex128)
        # torch uses 'dim' instead of 'axes'
        result = torch.fft.ifftn(x_torch, dim=axes)
        return result.numpy()


def rfft(x, axis=-1):
    """1D real FFT."""
    if _backend == 'numpy':
        return np.fft.rfft(x, axis=axis)
    else:
        x_torch = torch.as_tensor(x, dtype=torch.float64)
        result = torch.fft.rfft(x_torch, axis=axis)
        return result.numpy()


def irfft2(x, s):
    """2D inverse real FFT."""
    if _backend == 'numpy':
        return np.fft.irfft2(x, s=s)
    else:
        x_torch = torch.as_tensor(x, dtype=torch.complex128)
        result = torch.fft.irfft2(x_torch, s=s)
        return result.numpy()


def irfftn(x, s):
    """N-D inverse real FFT."""
    if _backend == 'numpy':
        return np.fft.irfftn(x, s=s)
    else:
        x_torch = torch.as_tensor(x, dtype=torch.complex128)
        result = torch.fft.irfftn(x_torch, s=s)
        return result.numpy()


def ifftshift(x, axes=None):
    """Inverse FFT shift."""
    if _backend == 'numpy':
        return np.fft.ifftshift(x, axes=axes)
    else:
        x_torch = torch.as_tensor(x)
        # torch uses 'dims' instead of 'axes'
        result = torch.fft.ifftshift(x_torch, dim=axes)
        return result.numpy()


# ============================================================================
# Random Number Generation
# ============================================================================

def randn(*shape):
    """Standard normal random numbers."""
    if _backend == 'numpy':
        return np.random.randn(*shape)
    else:
        result = torch.randn(*shape, dtype=torch.float64)
        return result.numpy()


def rand(*shape):
    """Uniform random numbers [0, 1)."""
    if _backend == 'numpy':
        return np.random.rand(*shape)
    else:
        result = torch.rand(*shape, dtype=torch.float64)
        return result.numpy()


def exponential(scale, size=None):
    """
    Exponential random numbers.

    Parameters
    ----------
    scale : float
        Scale parameter (1/lambda)
    size : int or tuple, optional
        Output shape

    Returns
    -------
    ndarray
        Exponential random samples
    """
    if _backend == 'numpy':
        if size is None:
            return np.random.exponential(scale)
        else:
            return np.random.exponential(scale, size=size)
    else:
        if size is None:
            size = 1
        # torch uses 1/scale as parameter, numpy uses scale
        dist = torch.distributions.Exponential(1.0 / scale)
        result = dist.sample(size if isinstance(size, tuple) else (size,))
        return result.numpy()


# ============================================================================
# Array Creation
# ============================================================================

def zeros(shape, dtype=np.float64):
    """Create zero array."""
    if _backend == 'numpy':
        return np.zeros(shape, dtype=dtype)
    else:
        # Convert numpy dtype to torch dtype
        if dtype == np.float64 or dtype is None:
            torch_dtype = torch.float64
        elif dtype == np.float32:
            torch_dtype = torch.float32
        elif dtype == np.int32:
            torch_dtype = torch.int32
        elif dtype == np.int64:
            torch_dtype = torch.int64
        elif dtype == np.complex128:
            torch_dtype = torch.complex128
        else:
            torch_dtype = torch.float64  # Default fallback

        result = torch.zeros(shape, dtype=torch_dtype)
        return result.numpy()


def ones(shape, dtype=np.float64):
    """Create ones array."""
    if _backend == 'numpy':
        return np.ones(shape, dtype=dtype)
    else:
        # Convert numpy dtype to torch dtype
        if dtype == np.float64 or dtype is None:
            torch_dtype = torch.float64
        elif dtype == np.float32:
            torch_dtype = torch.float32
        elif dtype == np.int32:
            torch_dtype = torch.int32
        elif dtype == np.int64:
            torch_dtype = torch.int64
        elif dtype == np.complex128:
            torch_dtype = torch.complex128
        else:
            torch_dtype = torch.float64  # Default fallback

        result = torch.ones(shape, dtype=torch_dtype)
        return result.numpy()


def arange(start, stop=None, step=1, dtype=np.float64):
    """Create array with evenly spaced values."""
    if _backend == 'numpy':
        return np.arange(start, stop, step, dtype=dtype)
    else:
        # Convert numpy dtype to torch dtype
        if dtype == np.float64 or dtype is None:
            torch_dtype = torch.float64
        elif dtype == np.float32:
            torch_dtype = torch.float32
        elif dtype == np.int32:
            torch_dtype = torch.int32
        elif dtype == np.int64:
            torch_dtype = torch.int64
        else:
            torch_dtype = torch.float64  # Default fallback

        if stop is None:
            stop = start
            start = 0
        result = torch.arange(start, stop, step, dtype=torch_dtype)
        return result.numpy()


def asarray(x, dtype=np.float64):
    """Convert to array."""
    if _backend == 'numpy':
        return np.asarray(x, dtype=dtype)
    else:
        # Convert numpy dtype to torch dtype
        if dtype == np.float64 or dtype is None:
            torch_dtype = torch.float64
        elif dtype == np.float32:
            torch_dtype = torch.float32
        elif dtype == np.int32:
            torch_dtype = torch.int32
        elif dtype == np.int64:
            torch_dtype = torch.int64
        elif dtype == np.complex128:
            torch_dtype = torch.complex128
        else:
            torch_dtype = torch.float64  # Default fallback

        x_torch = torch.as_tensor(x, dtype=torch_dtype)
        return x_torch.numpy()


def full_like(x, fill_value):
    """Create array with same shape as x, filled with value."""
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
        arrays_torch = [torch.as_tensor(a, dtype=torch.float64) for a in arrays]
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
    """Complex conjugate."""
    if _backend == 'numpy':
        return np.conj(x)
    else:
        x_torch = torch.as_tensor(x, dtype=torch.complex128)
        result = torch.conj(x_torch)
        # resolve_conj() is needed before calling numpy() on conjugated tensor
        return result.resolve_conj().numpy()


def where(condition, x, y):
    """Element-wise selection."""
    if _backend == 'numpy':
        return np.where(condition, x, y)
    else:
        cond_torch = torch.as_tensor(condition)
        x_torch = torch.as_tensor(x)
        y_torch = torch.as_tensor(y)
        result = torch.where(cond_torch, x_torch, y_torch)
        return result.numpy()


def clip(x, x_min, x_max):
    """Clip values to range."""
    if _backend == 'numpy':
        return np.clip(x, x_min, x_max)
    else:
        x_torch = torch.as_tensor(x)
        result = torch.clamp(x_torch, min=x_min, max=x_max)
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
    """Standard deviation."""
    if _backend == 'numpy':
        return np.std(x, axis=axis)
    else:
        x_torch = torch.as_tensor(x)
        if axis is None:
            result = torch.std(x_torch)
        else:
            result = torch.std(x_torch, dim=axis)
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

@property
def pi():
    """Pi constant."""
    return np.pi


# Make pi accessible as attribute
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
