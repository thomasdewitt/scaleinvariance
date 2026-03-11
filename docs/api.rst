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

Analysis
--------

.. automodule:: scaleinvariance.analysis
   :members:

Hurst Estimation
~~~~~~~~~~~~~~~~

.. autofunction:: scaleinvariance.haar_fluctuation_hurst

.. autofunction:: scaleinvariance.structure_function_hurst

.. autofunction:: scaleinvariance.spectral_hurst

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
