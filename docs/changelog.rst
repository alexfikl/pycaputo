Changelog
=========

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
