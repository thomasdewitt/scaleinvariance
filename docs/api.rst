API Reference
=============

Notes
-----

`FIF_1D` and `FIF_ND` no longer accept the old combined
`kernel_construction_method` argument. Use
`kernel_construction_method_flux` and
`kernel_construction_method_observable` instead.

Naive FIF kernels remain available for comparison only and emit a warning
because their outputs are not remotely accurate.

Simulation
----------

.. automodule:: scaleinvariance.simulation
   :members:

.. autofunction:: scaleinvariance.fBm_1D_circulant

.. autofunction:: scaleinvariance.fBm_ND_circulant

.. autofunction:: scaleinvariance.fBm_1D

.. autofunction:: scaleinvariance.FIF_1D

.. autofunction:: scaleinvariance.FIF_ND

Generalized Scale Invariance (GSI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``FIF_ND`` supports anisotropic GSI scaling via a user-supplied
``scale_metric`` (an N-D array of generalized distances) together with
the ``elliptical_dim`` parameter (the GSI elliptical dimension
``D_el``). For the canonical self-affine anisotropy with stratification
exponent ``Hz``, ``D_el = d - 1 + Hz`` in ``d`` dimensions. The
``canonical_scale_metric`` helper builds the standard stratified
metric (last axis anisotropic, all others isotropic) on the dx=2 grid
expected by the LS2010 kernel construction.

GSI in ``FIF_ND`` currently requires
``kernel_construction_method_observable='LS2010'`` — the spectral
observable path does not accept a custom ``scale_metric``.

.. autofunction:: scaleinvariance.canonical_scale_metric

Analysis
--------

.. automodule:: scaleinvariance.analysis
   :members:

Hurst Estimation
~~~~~~~~~~~~~~~~

.. autofunction:: scaleinvariance.haar_fluctuation_hurst

.. autofunction:: scaleinvariance.structure_function_hurst

.. autofunction:: scaleinvariance.spectral_hurst

Structure Functions
~~~~~~~~~~~~~~~~~~~

.. autofunction:: scaleinvariance.structure_function

.. autofunction:: scaleinvariance.costructure_function

Analysis Functions
~~~~~~~~~~~~~~~~~~

.. automodule:: scaleinvariance.analysis.haar_fluctuation
   :members:

.. automodule:: scaleinvariance.analysis.structure_function
   :members:

.. automodule:: scaleinvariance.analysis.spectral
   :members:

Utilities
---------

.. automodule:: scaleinvariance.utils
   :members:
