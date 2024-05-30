Changelog
=========

pycaputo 0.6.0 (May 30, 2024)
-----------------------------

Features
^^^^^^^^

* Implement Riemann-Lioville quadrature based on high-order Lagrange polynomials
  in :class:`~pycaputo.quadrature.riemann_liouville.SplineLagrange`. These methods
  require knowledge of the function :math:`f` being integrated, but can obtain
  high order :math:`> 3`.
* Implement the implicit :class:`~pycaputo.fode.caputo.Trapezoidal` and
  :class:`~pycaputo.fode.caputo.ExplicitTrapezoidal` methods. These methods are
  closely related to the standard :class:`~pycaputo.fode.caputo.PECE` method.
  The implicit method has better stability.
* Implement the Mittag-Leffler algorithm from [Garrappa2015]_.
* Added the :mod:`pycaputo.fode.special` module with some exact solutions. This
  is subject to a lot of change, but in general it is desired to have some
  examples for testing and demonstration.

Changes
^^^^^^^

* The base :class:`~pycaputo.derivatives.FractionalOperator` no longer defines
  an ``order`` attribute. This does not make sense for more complex operators
  with multiple parameters.
* :mod:`pycaputo.differentiation` no longer exports all the underlying methods.
  It is not required to do e.g. ``from pycaputo.differentiation.caputo import L1``.
* All the methods in :mod:`pycaputo.differentiation` have been renamed without the
  derivative type, e.g ``CaputoL1Method`` becomes simply ``L1``.
* The methods in :mod:`pycaputo.differentiation` no longer provide an order. This
  was not well-defined, since e.g. the L1 method has different orders depending
  on the smoothness of the operand.
* The :mod:`pycaputo.quadrature` module went through similar changes to the
  differentiation one.
* The :mod:`pycaputo.fode.caputo` module went through similar changes to the
  differentiation and quadrature ones.

pycaputo 0.5.0 (April 19, 2024)
-------------------------------

Features
^^^^^^^^

* Implement an implicit adaptive L1 type method for Integrate-and-Fire models
  in :mod:`pycaputo.integrate_fire`.
* Implement PIF, LIF, EIF and AdEx models explicitly. These can be used to model
  neurons using fractional dynamics.

Maintenance
^^^^^^^^^^^

* Use `uv <https://github.com/astral-sh/uv>`__ to handled pinning dependencies.
* Use `hatchling <https://hatch.pypa.io>`__ as the build backend instead of ``setuptools``.
* Start using `pytest-benchmark <https://pytest-benchmark.readthedocs.io>`__ to
  benchmark the code. This still needs a lot of work.

pycaputo 0.4.0 (February 6, 2024)
---------------------------------

Features
^^^^^^^^

* Implement an implicit L1 method in :class:`pycaputo.fode.caputo.L1`.
* Store an :class:`~numpy.ndarray` for the orders so that they are not recomputed
  at each time step in :class:`~pycaputo.stepping.FractionalDifferentialEquationMethod`.
  Several functions using :func:`~pycaputo.utils.cached_on_first_arg`, e.g.
  :func:`~pycaputo.stepping.gamma1p` are also cached.
* Rework the hierarchy for the product integration methods and update their
  names. They are now available in :mod:`pycaputo.fode.caputo` only and called
  directly ``ForwardEuler`` (before it was ``CaputoForwardEulerMethod``).
* Promote events to :mod:`pycaputo.events`. Specific methods can then inherit
  from there to return additional information, as required.
* Add some dataclass helpers, e.g. :func:`~pycaputo.utils.dc_stringify`. All
  numerical methods store their parameters in a dataclass, so these are used
  all over.

Fixes
^^^^^

* Add more extensive tests for the Mittag-Leffler function.
* Add a ``py.typed`` file for upstream projects.
* Updated and fixed Lorenz example with
  :class:`~pycaputo.fode.caputo.WeightedEuler` (:ghpr:`19`).
* Use :func:`numpy.einsum` to compute weights for faster evaluation.

Maintenance
^^^^^^^^^^^

* Use ``ruff format`` for our formatting needs.
* Switch to a ``src`` based layout.

pycaputo 0.3.1 (December 29, 2023)
----------------------------------

Features
^^^^^^^^

* Released on PyPI!

pycaputo 0.3.0 (December 28, 2023)
----------------------------------

Features
^^^^^^^^

* Add support for adaptive time stepping (:ghpr:`32`). This functionality is
  very nice and there isn't much literature on the matter so it will likely
  need substantial improvements in the future. For the moment, the work of
  [Jannelli2020]_ is implement and seems to work reasonably well.

Fixes
^^^^^

* Make all methods use a vector of orders ``alpha`` when solving systems to be
  more future proof.

pycaputo 0.2.0 (December 25, 2023)
----------------------------------

Dependency changes
^^^^^^^^^^^^^^^^^^

* Bumped minimum Python version to 3.9 (to match latest Numpy).

Features
^^^^^^^^

* Added an example with the fractional Lorenz system (:ghpr:`13`).
* Add a guess for the number of corrector iterations
  for :class:`~pycaputo.fode.caputo.PECE` from [Garrappa2010]_.
* Added a modified PECE method from [Garrappa2010]_ in the form of
  :class:`~pycaputo.fode.caputo.ModifiedPECE`.
* Implement :class:`~pycaputo.quadrature.riemann_liouville.Simpson`, a
  standard 3rd order method.
* Implement :class:`~pycaputo.quadrature.riemann_liouville.CubicHermite`, a
  standard 4th order method.
* Implement differentiation methods for the Riemann-Liouville derivatives based
  on the Caputo derivative in
  :class:`~pycaputo.differentiation.riemann_liouville.RiemannLiouvilleFromCaputoDerivativeMethod`.
* Support different fractional orders for FODE systems in
  :class:`~pycaputo.fode.caputo.ForwardEuler`,
  :class:`~pycaputo.fode.caputo.WeightedEuler` and others.
* Add approximation for the Lipschitz constant (:ghpr:`18`).
* Add a (rather slow) wrapper to compute a fractional gradient (:ghpr:`35`).

Fixes
^^^^^

* Fix :class:`~pycaputo.quadrature.riemann_liouville.Trapezoidal` on
  uniform grids (:ghissue:`3`).
* Fix Jacobian construction for :class:`~pycaputo.fode.caputo.WeightedEuler`
  which gave incorrect results for systems of equations (:ghissue:`11`).
* Add dark variants of plots to the documentation for nicer results.
* Promoto history management to :mod:`pycaputo.history`.

pycaputo 0.1.0 (June 12, 2023)
------------------------------

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
