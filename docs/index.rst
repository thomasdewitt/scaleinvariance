ScaleInvariance Documentation
=============================

.. warning::
   This package is under active development. Current functionality is limited to Hurst exponent estimation and fractional Brownian motion simulation.

ScaleInvariance provides simulation and analysis tools for scale-invariant processes and multifractal fields.

Features
--------

**Simulation**
   - 1D fractional Brownian motion via spectral synthesis
   - 2D fractional Brownian motion with proper isotropy

**Hurst Exponent Estimation**
   - Haar fluctuation method
   - Structure function method  
   - Spectral method

All methods support multi-dimensional arrays and return uncertainty estimates.

Quick Start
-----------

.. code-block:: python

   from scaleinvariance import acausal_fBm_1D, acausal_fBm_2D, haar_fluctuation_hurst

   # Generate 1D fractional Brownian motion
   fBm_1d = acausal_fBm_1D(1024, H=0.7)

   # Generate 2D fractional Brownian motion  
   fBm_2d = acausal_fBm_2D((512, 1024), H=0.7)

   # Estimate Hurst exponent
   H_est, H_err = haar_fluctuation_hurst(fBm_1d)
   print(f"Estimated H = {H_est:.3f} ± {H_err:.3f}")

Installation
------------

.. code-block:: bash

   pip install scaleinvariance

Contents
--------

.. toctree::
   :maxdepth: 2

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`