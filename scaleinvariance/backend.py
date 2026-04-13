"""
Backend module for scaleinvariance - provides unified numpy/torch interface.

Automatically detects PyTorch availability and provides wrapper functions
that work with either numpy or torch. When ``backend='numpy'``, all functions
return numpy arrays. When ``backend='torch'``, functions return native torch
tensors (on the active device); call :func:`to_numpy` at public API boundaries
to convert back.

Numerical precision is controlled by ``set_numerical_precision('float32'|'float64')``
and applies to both numpy and torch backends. The default is ``'float32'``.
"""

import numpy as np
import os

# Try to import torch
try:
    import torch
    import torch.nn.functional as F
    _torch_available = True
except ImportError:
    _torch_available = False

# Global state
_backend = 'numpy'  # Default backend
_num_threads = int(os.cpu_count() * 0.9) if os.cpu_count() else 4
_numerical_precision = 'float32'  # Default precision
_device = 'cpu'
_device_obj = torch.device('cpu') if _torch_available else None

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


def set_device(device='cpu'):
    """Set torch compute device.

    Only meaningful when ``backend='torch'``. When ``backend='numpy'``,
    the device setting is stored but has no effect.

    Parameters
    ----------
    device : str
        Device string: ``'cpu'``, ``'cuda'``, ``'cuda:0'``, etc.

    Raises
    ------
    RuntimeError
        If CUDA is requested but not available.
    """
    global _device, _device_obj
    if device != 'cpu' and _torch_available:
        if 'cuda' in device and not torch.cuda.is_available():
            raise RuntimeError(
                f"Device '{device}' requested but CUDA is not available."
            )
    _device = device
    if _torch_available:
        _device_obj = torch.device(device)


def get_device():
    """Get current device string."""
    return _device


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


def _to_torch_dtype(dtype):
    """Map a dtype to the corresponding torch dtype.

    Accepts numpy dtypes, torch dtypes, and None. Returns the active real
    torch dtype for None or unrecognized inputs.
    """
    if dtype is None:
        return _active_real_dtype_torch()
    if _torch_available and isinstance(dtype, torch.dtype):
        return dtype
    try:
        key = np.dtype(dtype).type
    except TypeError:
        return _active_real_dtype_torch()
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
    if key is np.bool_:
        return torch.bool
    return _active_real_dtype_torch()


# Keep old name as internal alias
_np_to_torch_dtype = _to_torch_dtype


def _to_torch(x, dtype=None):
    """Convert input to a torch tensor on the active device.

    If *x* is already a tensor with the correct dtype and device, this is
    close to a no-op. Numpy arrays on CPU share memory via ``as_tensor``.
    """
    if isinstance(x, torch.Tensor):
        target_dtype = dtype if dtype is not None else x.dtype
        if x.dtype != target_dtype or x.device != _device_obj:
            return x.to(dtype=target_dtype, device=_device_obj)
        return x
    if dtype is not None:
        return torch.as_tensor(x, dtype=dtype).to(device=_device_obj)
    return torch.as_tensor(x).to(device=_device_obj)


def to_numpy(x):
    """Convert to numpy array. Call at public API boundaries.

    No-op if already a numpy array. For torch tensors, handles both
    CPU (zero-copy) and GPU (transfer + copy) cases.
    """
    if isinstance(x, np.ndarray):
        return x
    if _torch_available and isinstance(x, torch.Tensor):
        t = x.detach()
        if t.device.type != 'cpu':
            t = t.cpu()
        return t.numpy()
    return np.asarray(x)


def exp_clip_limits():
    """Return ``(lo, hi)`` safe exponent range for ``exp`` at active precision.

    For float32: ``(-88.7, 88.7)``, just below ``log(float32_max) ≈ 88.72``.
    For float64: ``(-709.0, 709.0)``, just below ``log(float64_max) ≈ 709.78``.
    """
    if _numerical_precision == 'float32':
        return (-88.7, 88.7)
    return (-709.0, 709.0)


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
        return torch.fft.fftfreq(n, d=d, dtype=_active_real_dtype_torch(),
                                 device=_device_obj)


def rfftfreq(n, d=1.0):
    """Real FFT frequency array (non-negative half)."""
    if _backend == 'numpy':
        return np.fft.rfftfreq(n, d=d).astype(_active_real_dtype_np(), copy=False)
    else:
        return torch.fft.rfftfreq(n, d=d, dtype=_active_real_dtype_torch(),
                                  device=_device_obj)


def fft(x, axis=-1):
    """1D FFT."""
    if _backend == 'numpy':
        return np.fft.fft(x, axis=axis)
    else:
        return torch.fft.fft(_to_torch(x, _active_complex_dtype_torch()), dim=axis)


def ifft(x, axis=-1):
    """1D inverse FFT."""
    if _backend == 'numpy':
        return np.fft.ifft(x, axis=axis)
    else:
        return torch.fft.ifft(_to_torch(x, _active_complex_dtype_torch()), dim=axis)


def fft2(x):
    """2D FFT."""
    if _backend == 'numpy':
        return np.fft.fft2(x)
    else:
        return torch.fft.fft2(_to_torch(x, _active_complex_dtype_torch()))


def ifft2(x):
    """2D inverse FFT."""
    if _backend == 'numpy':
        return np.fft.ifft2(x)
    else:
        return torch.fft.ifft2(_to_torch(x, _active_complex_dtype_torch()))


def fftn(x, axes=None):
    """N-D FFT."""
    if _backend == 'numpy':
        return np.fft.fftn(x, axes=axes)
    else:
        return torch.fft.fftn(_to_torch(x, _active_complex_dtype_torch()), dim=axes)


def ifftn(x, axes=None):
    """N-D inverse FFT."""
    if _backend == 'numpy':
        return np.fft.ifftn(x, axes=axes)
    else:
        return torch.fft.ifftn(_to_torch(x, _active_complex_dtype_torch()), dim=axes)


def rfft(x, axis=-1):
    """1D real FFT."""
    if _backend == 'numpy':
        return np.fft.rfft(x, axis=axis)
    else:
        return torch.fft.rfft(_to_torch(x, _active_real_dtype_torch()), dim=axis)


def irfft(x, n, axis=-1):
    """1D inverse real FFT; ``n`` is the output length along ``axis``."""
    if _backend == 'numpy':
        return np.fft.irfft(x, n=n, axis=axis)
    else:
        return torch.fft.irfft(_to_torch(x, _active_complex_dtype_torch()),
                               n=n, dim=axis)


def rfftn(x, axes=None):
    """N-D real FFT (real input, last axis halved)."""
    if _backend == 'numpy':
        return np.fft.rfftn(x, axes=axes)
    else:
        return torch.fft.rfftn(_to_torch(x, _active_real_dtype_torch()), dim=axes)


def irfft2(x, s):
    """2D inverse real FFT."""
    if _backend == 'numpy':
        return np.fft.irfft2(x, s=s)
    else:
        return torch.fft.irfft2(_to_torch(x, _active_complex_dtype_torch()), s=s)


def irfftn(x, s):
    """N-D inverse real FFT."""
    axes = list(range(-len(s), 0))
    if _backend == 'numpy':
        return np.fft.irfftn(x, s=s, axes=axes)
    else:
        return torch.fft.irfftn(_to_torch(x, _active_complex_dtype_torch()),
                                s=s, dim=axes)


def ifftshift(x, axes=None):
    """Inverse FFT shift."""
    if _backend == 'numpy':
        return np.fft.ifftshift(x, axes=axes)
    else:
        return torch.fft.ifftshift(_to_torch(x), dim=axes)


# ============================================================================
# Random Number Generation
# ============================================================================

def randn(*shape):
    """Standard normal random numbers at the active real precision."""
    if _backend == 'numpy':
        target = _active_real_dtype_np()
        result = np.random.randn(*shape)
        if result.dtype != target:
            result = result.astype(target, copy=False)
        return result
    else:
        return torch.randn(*shape, dtype=_active_real_dtype_torch(),
                           device=_device_obj)


def rand(*shape):
    """Uniform random numbers [0, 1) at the active real precision."""
    if _backend == 'numpy':
        target = _active_real_dtype_np()
        result = np.random.rand(*shape)
        if result.dtype != target:
            result = result.astype(target, copy=False)
        return result
    else:
        return torch.rand(*shape, dtype=_active_real_dtype_torch(),
                          device=_device_obj)


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
    if _backend == 'numpy':
        target = _active_real_dtype_np()
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
        rate = torch.tensor(1.0 / scale, dtype=_active_real_dtype_torch(),
                            device=_device_obj)
        dist = torch.distributions.Exponential(rate)
        return dist.sample(size if isinstance(size, tuple) else (size,))


# ============================================================================
# Array Creation
# ============================================================================

def zeros(shape, dtype=None):
    """Create zero array. Defaults to active real dtype."""
    if _backend == 'numpy':
        if dtype is None:
            dtype = _active_real_dtype_np()
        return np.zeros(shape, dtype=dtype)
    else:
        torch_dtype = _to_torch_dtype(dtype)
        return torch.zeros(shape, dtype=torch_dtype, device=_device_obj)


def ones(shape, dtype=None):
    """Create ones array. Defaults to active real dtype."""
    if _backend == 'numpy':
        if dtype is None:
            dtype = _active_real_dtype_np()
        return np.ones(shape, dtype=dtype)
    else:
        torch_dtype = _to_torch_dtype(dtype)
        return torch.ones(shape, dtype=torch_dtype, device=_device_obj)


def empty(shape, dtype=None):
    """Create uninitialized array. Defaults to active real dtype."""
    if _backend == 'numpy':
        if dtype is None:
            dtype = _active_real_dtype_np()
        return np.empty(shape, dtype=dtype)
    else:
        torch_dtype = _to_torch_dtype(dtype)
        return torch.empty(shape, dtype=torch_dtype, device=_device_obj)


def arange(start, stop=None, step=1, dtype=None):
    """Create array with evenly spaced values. Defaults to active real dtype."""
    if _backend == 'numpy':
        if dtype is None:
            dtype = _active_real_dtype_np()
        return np.arange(start, stop, step, dtype=dtype)
    else:
        torch_dtype = _to_torch_dtype(dtype)
        if stop is None:
            stop = start
            start = 0
        return torch.arange(start, stop, step, dtype=torch_dtype,
                            device=_device_obj)


def logspace(start, stop, num, dtype=None):
    """Logarithmically spaced values from 10**start to 10**stop."""
    if _backend == 'numpy':
        if dtype is None:
            dtype = _active_real_dtype_np()
        return np.logspace(start, stop, num, dtype=dtype)
    else:
        torch_dtype = _to_torch_dtype(dtype)
        return torch.logspace(start, stop, num, dtype=torch_dtype,
                              device=_device_obj)


def asarray(x, dtype=None):
    """Convert to array. If ``dtype`` is None, preserves the input dtype.

    Unlike earlier versions, ``asarray`` no longer silently promotes every
    input to float64. Pass an explicit ``dtype=`` if you need a cast.
    """
    if _backend == 'numpy':
        if dtype is None:
            return np.asarray(x)
        if _torch_available and isinstance(dtype, torch.dtype):
            # Convert torch dtype to numpy dtype for numpy backend
            dtype = torch.zeros(0, dtype=dtype).numpy().dtype
        return np.asarray(x, dtype=dtype)
    else:
        if dtype is None:
            return _to_torch(x)
        return _to_torch(x, _to_torch_dtype(dtype))


def full_like(x, fill_value):
    """Create array with same shape/dtype as x, filled with value."""
    if _backend == 'numpy':
        return np.full_like(x, fill_value)
    else:
        return torch.full_like(_to_torch(x), fill_value)


def concatenate(arrays, axis=0):
    """Concatenate arrays along axis."""
    if _backend == 'numpy':
        return np.concatenate(arrays, axis=axis)
    else:
        arrays_torch = [_to_torch(a) for a in arrays]
        return torch.cat(arrays_torch, dim=axis)


# ============================================================================
# Array Manipulation
# ============================================================================

def meshgrid(*arrays, indexing='xy'):
    """Create mesh grids."""
    if _backend == 'numpy':
        return np.meshgrid(*arrays, indexing=indexing)
    else:
        arrays_torch = [_to_torch(a, _active_real_dtype_torch()) for a in arrays]
        return tuple(torch.meshgrid(*arrays_torch, indexing=indexing))


def flip(x, axis):
    """Flip array along axis."""
    if _backend == 'numpy':
        return np.flip(x, axis=axis)
    else:
        return torch.flip(_to_torch(x), dims=(axis,))


def conj(x):
    """Complex conjugate. Preserves input dtype."""
    if _backend == 'numpy':
        return np.conj(x)
    else:
        result = torch.conj(_to_torch(x))
        return result.resolve_conj()


def where(condition, x, y):
    """Element-wise selection."""
    if _backend == 'numpy':
        return np.where(condition, x, y)
    else:
        cond_torch = _to_torch(condition).bool()
        return torch.where(cond_torch, _to_torch(x), _to_torch(y))


def clip(x, x_min, x_max):
    """Clip values to range. ``x_min`` or ``x_max`` may be ``None``."""
    if _backend == 'numpy':
        return np.clip(x, x_min, x_max)
    else:
        return torch.clamp(_to_torch(x), min=x_min, max=x_max)


def maximum(a, b):
    """Element-wise maximum of two arrays (or array and scalar)."""
    if _backend == 'numpy':
        return np.maximum(a, b)
    else:
        a_torch = _to_torch(a)
        if isinstance(b, (int, float)):
            b_torch = torch.tensor(b, dtype=a_torch.dtype, device=_device_obj)
        elif isinstance(b, torch.Tensor):
            b_torch = b
        else:
            b_torch = _to_torch(b, dtype=a_torch.dtype)
        return torch.maximum(a_torch, b_torch)


def pad(x, before, after, axis=-1, value=0):
    """Pad array along a single axis with constant value.

    Parameters
    ----------
    x : array
        Input array.
    before : int
        Number of values to pad before.
    after : int
        Number of values to pad after.
    axis : int
        Axis along which to pad.
    value : scalar
        Pad value (default 0).
    """
    if _backend == 'numpy':
        pad_width = [(0, 0)] * x.ndim
        ndim_axis = axis % x.ndim
        pad_width[ndim_axis] = (before, after)
        return np.pad(x, pad_width, mode='constant', constant_values=value)
    else:
        x_torch = _to_torch(x)
        ndim = x_torch.ndim
        ndim_axis = axis % ndim
        # torch.nn.functional.pad expects padding in reversed axis order:
        # (last_dim_before, last_dim_after, ..., first_dim_before, first_dim_after)
        pad_list = [0] * (2 * ndim)
        # Position for ndim_axis: distance from last = (ndim - 1 - ndim_axis)
        idx = 2 * (ndim - 1 - ndim_axis)
        pad_list[idx] = before
        pad_list[idx + 1] = after
        return F.pad(x_torch, pad_list, mode='constant', value=value)


# ============================================================================
# Mathematical Operations
# ============================================================================

def abs(x):
    """Absolute value."""
    if _backend == 'numpy':
        return np.abs(x)
    else:
        return torch.abs(_to_torch(x))


def abs_squared(x):
    """Squared absolute value. For complex input, computes real^2 + imag^2
    without the intermediate sqrt that ``abs(x)**2`` would require."""
    if _backend == 'numpy':
        x = np.asarray(x)
        if np.iscomplexobj(x):
            return x.real ** 2 + x.imag ** 2
        return x ** 2
    else:
        x_torch = _to_torch(x)
        if x_torch.is_complex():
            return x_torch.real ** 2 + x_torch.imag ** 2
        return x_torch ** 2


def sqrt(x):
    """Square root."""
    if _backend == 'numpy':
        return np.sqrt(x)
    else:
        return torch.sqrt(_to_torch(x))


def sign(x):
    """Sign of values."""
    if _backend == 'numpy':
        return np.sign(x)
    else:
        return torch.sign(_to_torch(x))


def exp(x):
    """Exponential function."""
    if _backend == 'numpy':
        return np.exp(x)
    else:
        return torch.exp(_to_torch(x))


def log(x):
    """Natural logarithm."""
    if _backend == 'numpy':
        return np.log(x)
    else:
        return torch.log(_to_torch(x))


def log10(x):
    """Base-10 logarithm."""
    if _backend == 'numpy':
        return np.log10(x)
    else:
        return torch.log10(_to_torch(x))


def cos(x):
    """Cosine function."""
    if _backend == 'numpy':
        return np.cos(x)
    else:
        return torch.cos(_to_torch(x))


def sin(x):
    """Sine function."""
    if _backend == 'numpy':
        return np.sin(x)
    else:
        return torch.sin(_to_torch(x))


# ============================================================================
# Reductions
# ============================================================================

def mean(x, axis=None):
    """Mean of array. *axis* may be an int, tuple of ints, or None."""
    if _backend == 'numpy':
        return np.mean(x, axis=axis)
    else:
        x_torch = _to_torch(x)
        if axis is None:
            return torch.mean(x_torch)
        return torch.mean(x_torch, dim=axis)


def nanmean(x, axis=None):
    """Mean of array, ignoring NaNs."""
    if _backend == 'numpy':
        return np.nanmean(x, axis=axis)
    else:
        x_torch = _to_torch(x)
        if axis is None:
            return torch.nanmean(x_torch)
        return torch.nanmean(x_torch, dim=axis)


def std(x, axis=None):
    """Standard deviation (population, ddof=0 — matches numpy default)."""
    if _backend == 'numpy':
        return np.std(x, axis=axis)
    else:
        x_torch = _to_torch(x)
        if axis is None:
            return torch.std(x_torch, correction=0)
        return torch.std(x_torch, dim=axis, correction=0)


def max(x):
    """Maximum value of array (scalar)."""
    if _backend == 'numpy':
        return np.max(x)
    else:
        return torch.max(_to_torch(x))


def sum(x, axis=None):
    """Sum of array."""
    if _backend == 'numpy':
        return np.sum(x, axis=axis)
    else:
        x_torch = _to_torch(x)
        if axis is None:
            return torch.sum(x_torch)
        return torch.sum(x_torch, dim=axis)


def cumsum(x, axis=0):
    """Cumulative sum."""
    if _backend == 'numpy':
        return np.cumsum(x, axis=axis)
    else:
        return torch.cumsum(_to_torch(x), dim=axis)


def diff(x, axis=-1):
    """Difference along axis."""
    if _backend == 'numpy':
        return np.diff(x, axis=axis)
    else:
        return torch.diff(_to_torch(x), dim=axis)


# ============================================================================
# Boolean / inspection
# ============================================================================

def isnan(x):
    """Element-wise NaN test."""
    if _backend == 'numpy':
        return np.isnan(x)
    else:
        return torch.isnan(_to_torch(x))


def any(x):
    """True if any element is truthy. Returns Python bool."""
    if _backend == 'numpy':
        return bool(np.any(x))
    else:
        return bool(torch.any(_to_torch(x)))


def all(x):
    """True if all elements are truthy. Returns Python bool."""
    if _backend == 'numpy':
        return bool(np.all(x))
    else:
        return bool(torch.all(_to_torch(x)))


def numel(x):
    """Total number of elements in array."""
    if _backend == 'numpy':
        return np.asarray(x).size
    else:
        if isinstance(x, torch.Tensor):
            return x.numel()
        return np.asarray(x).size


def iscomplexobj(x):
    """Check if x has a complex dtype."""
    if _torch_available and isinstance(x, torch.Tensor):
        return x.is_complex()
    return np.iscomplexobj(x)


# ============================================================================
# Constants and Utilities
# ============================================================================

pi = np.pi


def real(x):
    """Real part of array."""
    if _backend == 'numpy':
        return np.real(x)
    else:
        return torch.real(_to_torch(x))


def imag(x):
    """Imaginary part of array."""
    if _backend == 'numpy':
        return np.imag(x)
    else:
        return torch.imag(_to_torch(x))


# ============================================================================
# Convolution
# ============================================================================

def convolve1d(signal, kernel, axis=-1, nan_safe=False):
    """1-D aperiodic convolution along *axis*, mode='valid'.

    Parameters
    ----------
    signal : array
        Input signal (N-D).
    kernel : 1-D array
        Convolution kernel.
    axis : int
        Axis of *signal* along which to convolve.
    nan_safe : bool
        If True, use direct (non-FFT) convolution that correctly propagates
        NaNs. Falls back to scipy regardless of backend.

    Returns
    -------
    array
        Convolution result with size ``signal.shape[axis] - len(kernel) + 1``
        along *axis*.
    """
    if nan_safe:
        return _convolve1d_direct(signal, kernel, axis)
    return _convolve1d_fft(signal, kernel, axis)


def _convolve1d_fft(signal, kernel, axis):
    """FFT-based aperiodic convolution, mode='valid'."""
    n_sig = signal.shape[axis]
    n_ker = len(kernel)
    if n_ker > n_sig:
        raise ValueError(
            f"Kernel length ({n_ker}) exceeds signal length ({n_sig}) along axis {axis}; "
            "mode='valid' requires kernel <= signal."
        )
    n_fft = n_sig + n_ker - 1
    n_out = n_sig - n_ker + 1

    # Pad signal along axis
    sig_padded = pad(signal, 0, n_ker - 1, axis=axis)

    # Build N-D kernel: shape is 1 everywhere except along axis
    kernel_shape = [1] * signal.ndim if hasattr(signal, 'ndim') else [1]
    ndim = len(kernel_shape)
    kernel_shape[axis % ndim] = n_ker
    if _backend == 'numpy':
        kernel_nd = np.asarray(kernel).reshape(kernel_shape)
    else:
        kernel_nd = _to_torch(kernel).reshape(kernel_shape)
    kernel_padded = pad(kernel_nd, 0, n_fft - n_ker, axis=axis)

    # FFT, multiply, IFFT
    sig_fft = rfft(sig_padded, axis=axis)
    ker_fft = rfft(kernel_padded, axis=axis)
    product = sig_fft * ker_fft
    del sig_fft, ker_fft, sig_padded, kernel_padded
    conv_full = irfft(product, n=n_fft, axis=axis)
    del product

    # Extract valid region
    slices = [slice(None)] * conv_full.ndim
    slices[axis % conv_full.ndim] = slice(n_ker - 1, n_ker - 1 + n_out)
    return conv_full[tuple(slices)]


def _convolve1d_direct(signal, kernel, axis):
    """Direct convolution via scipy (handles NaNs). Always uses numpy."""
    from scipy.signal import convolve
    sig_np = to_numpy(signal) if not isinstance(signal, np.ndarray) else signal
    ker_np = to_numpy(kernel) if not isinstance(kernel, np.ndarray) else kernel

    # Reshape kernel for N-D
    kernel_shape = [1] * sig_np.ndim
    kernel_shape[axis % sig_np.ndim] = len(ker_np)
    ker_nd = ker_np.reshape(kernel_shape)

    result = convolve(sig_np, ker_nd, mode='valid', method='direct')

    # Convert back to backend type if needed
    if _backend != 'numpy':
        return _to_torch(result)
    return result
