Changelog
=========

pycaputo 0.2 (TBD)
------------------

Features
^^^^^^^^

* Bumped minimum Python version to 3.9 (to match latest Numpy).
* Added an example with the fractional Lorenz system.
* Support setting a constant for
  :meth:`~pycaputo.fode.FractionalDifferentialEquationMethod.predict_time_step`.
* Add a guess for the number of corrector iterations
  for :class:`~pycaputo.fode.CaputoPECEMethod` from [Garrappa2010]_.

Fixes
^^^^^

* Fix :class:`~pycaputo.quad.RiemannLiouvilleTrapezoidalMethod` on uniform grids.
* Fix Jacobian construction for :class:`~pycaputo.fode.CaputoWeightedEulerMethod`
  which gave incorrect results for systems of equations.
* Add dark variants of plots to the documentation for nicer results.

pycaputo 0.1 (June 12, 2023)
----------------------------

This is the initial release of the project and has some basic functionality
implemented already.

* Evaluate Caputo derivatives of arbitrary real orders; several numerical methods
  are implemented (L1, L2, spectral) in :ref:`sec-differentiation`.
* Evaluate Riemann-Liouville integrals of arbitrary real orders; several numerical
  methods are implemented (rectangular, trapezoidal, spectral) in
  :ref:`sec-quadrature`.
* Solve single-term fractional ordinary differential equations; several numerical
  methods are implemented (forward and backward Euler, PECE) in
  :ref:`sec-fode`.

The library is not stable in any way. Performance work will likely require
changes to some interfaces.
