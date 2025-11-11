"""
Generalized Scale Invariance (GSI) utilities for multifractal simulations.

This module provides functions for constructing anisotropic scale metrics
used in GSI multifractal models, particularly for use with FIF_ND.
"""

import numpy as np
from .. import backend as B


def canonical_scale_metric(size, ls, Hz=5/9):
    """
    Compute the canonical anisotropic scale metric for GSI multifractal models.

    This metric introduces anisotropy in the vertical (z) direction, allowing
    different scaling behavior along different axes. The metric is defined as:

    For 3D:
        r = ls * sqrt((dx/ls)^2 + (dy/ls)^2 + (dz/ls)^(2/Hz))

    For 2D (x-z plane, where axis 1 is vertical):
        r = ls * sqrt((dx/ls)^2 + (dz/ls)^(2/Hz))

    where ls is the spheroscale (characteristic horizontal scale) and Hz controls
    the degree of anisotropy in the vertical direction.

    **IMPORTANT**: Uses dx=2 grid spacing to match LS2010 kernel construction.
    Coordinate arrays are created as arange(-(size[i]-1), size[i], 2) for each dimension.

    Parameters
    ----------
    size : tuple of ints
        Size of the simulation domain (e.g., (512, 512) for 2D, (256, 256, 128) for 3D).
        Must be a 2D or 3D tuple.
    ls : float
        Spheroscale - characteristic horizontal scale length. Must be > 0.
        This sets the fundamental length scale for the metric.
    Hz : float, optional
        Anisotropy exponent for vertical (z) direction. Default is 5/9 ≈ 0.556
        (empirical value for atmospheric stratification).
        - Hz = 1.0: Isotropic (standard Euclidean metric)
        - Hz < 1.0: Vertical direction scales slower (stratified/layered structures)
        - Hz > 1.0: Vertical direction scales faster
        Applied to the last axis in 2D and 3D cases.

    Returns
    -------
    numpy.ndarray
        N-D array of scale metric values with shape matching size.
        Always returns a NumPy array regardless of backend.

    Raises
    ------
    ValueError
        If size is not a 2D or 3D tuple
        If ls <= 0
        If Hz <= 0

    Examples
    --------
    >>> # 2D x-z plane with stratification (Hz < 1, vertical scales slower)
    >>> metric_2d = canonical_scale_metric((512, 256), ls=100.0, Hz=0.556)
    >>>
    >>> # 2D isotropic (set Hz=1.0)
    >>> metric_2d_iso = canonical_scale_metric((512, 512), ls=100.0, Hz=1.0)
    >>>
    >>> # 3D with stratification (Hz < 1, vertical scales slower)
    >>> metric_3d = canonical_scale_metric((256, 256, 128), ls=50.0, Hz=0.5)
    >>>
    >>> # Use with FIF_ND for GSI multifractal simulation
    >>> from scaleinvariance import FIF_ND
    >>> size = (256, 256, 128)
    >>> ls = 50.0
    >>> Hz = 0.6
    >>> metric = canonical_scale_metric(size, ls, Hz)
    >>> # For non-periodic simulation, need to double the domain
    >>> sim_size = tuple(s * 2 for s in size)
    >>> metric_doubled = canonical_scale_metric(sim_size, ls, Hz)
    >>> fif = FIF_ND(size, alpha=1.7, C1=0.1, H=0.3, periodic=False,
    ...              scale_metric=metric_doubled, scale_metric_dim=2.5)

    Notes
    -----
    - The metric reduces to standard Euclidean distance when Hz=1.0 and ls=1.0
    - For atmospheric applications, Hz < 1 is typical due to stratification
    - The ls parameter can be tuned to match physical scales in the system
    - Grid spacing is dx=2 to match LS2010 kernel construction in FIF_ND

    References
    ----------
    Lovejoy, S., & Schertzer, D. (2007). Scaling and multifractal fields in the
    solid earth and topography. Nonlinear Processes in Geophysics, 14(4), 465-502.
    """
    # Validate inputs
    if not isinstance(size, tuple) or len(size) not in (2, 3):
        raise ValueError("size must be a 2D or 3D tuple (e.g., (512, 512) or (256, 256, 128))")

    if ls <= 0:
        raise ValueError("ls (spheroscale) must be positive")

    if Hz <= 0:
        raise ValueError("Hz (anisotropy exponent) must be positive")

    ndim = len(size)

    # Create coordinate arrays with dx=2 spacing (to match LS2010)
    coord_arrays = [B.arange(-(dim - 1), dim, 2, dtype=float) for dim in size]

    # Create meshgrid
    coord_grids = B.meshgrid(*coord_arrays, indexing='ij')

    if ndim == 2:
        # 2D case (x-z plane): r = ls * sqrt((dx/ls)^2 + (dz/ls)^(2/Hz))
        # Axis 0 is horizontal (x), axis 1 is vertical (z)
        dx, dz = coord_grids

        # Compute normalized squared distances
        dx_norm_sq = (dx / ls)**2
        dz_norm_aniso = (B.abs(dz) / ls)**(2.0 / Hz)

        # Combine and scale
        metric = ls * B.sqrt(dx_norm_sq + dz_norm_aniso)

    elif ndim == 3:
        # 3D case: r = ls * sqrt((dx/ls)^2 + (dy/ls)^2 + (dz/ls)^(2/Hz))
        dx, dy, dz = coord_grids

        # Compute normalized squared distances
        dx_norm_sq = (dx / ls)**2
        dy_norm_sq = (dy / ls)**2
        dz_norm_aniso = (B.abs(dz) / ls)**(2.0 / Hz)

        # Combine and scale
        metric = ls * B.sqrt(dx_norm_sq + dy_norm_sq + dz_norm_aniso)

    # Backend functions already return numpy arrays
    return metric
