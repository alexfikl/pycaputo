Changelog
=========

pycaputo 0.10.0 (October 13, 2025)
----------------------------------

Dependencies
^^^^^^^^^^^^

* Add official support for Python 3.14.

Features
^^^^^^^^

* Implement an FODE solver for the Caputo-Fabrizio operator using
  :class:`~pycaputo.derivatives.CaputoFabrizioOperator`. The discretizations are
  :class:`~pycaputo.fode.caputo_fabrizio.AtanganaSeda2` and
  :class:`~pycaputo.fode.caputo_fabrizio.AtanganaSeda3`.
* Add example for using ``numba`` and ``jax`` (through ``array_api_compat``)
  with some of the simpler fractional derivative evaluation methods.
* Add a sinusoidal grid spacing :func:`~pycaputo.grid.make_sine_points`.
* Refactor fixed step size controllers. They now all inherit from a base class
  that just stores all the time steps (:class:`~pycaputo.controller.GivenStepController`).

Fixes
^^^^^

* Make :func:`~pycaputo.special.sine` work at ``t = 0``.

pycaputo 0.9.0 (March 6, 2025)
------------------------------

Dependencies
^^^^^^^^^^^^

* Switch to the `pymittagleffler <https://github.com/alexfikl/mittagleffler>`__
  library to evaluate the Mittag-Leffler function. This should be a lot faster
  and generally cleaner. The code in ``pycaputo.mittagleffler`` has been completely
  removed.

Features
^^^^^^^^

* Add some more chaotic systems examples: Labyrinth, Chua, Qi, Lorenz-84. They
  are all listed in :mod:`pycaputo.fode.gallery`.
* Add some neuron models: FtizHugh-Nagumo, FitzHugh-Rinzel, Morris-Lecar,
  Hindmarsh-Rose, Hodgkin-Huxley. This is on top of the existing Integrate-and-Fire
  models from :mod:`pycaputo.integrate_fire`. They can be found in
  :mod:`pycaputo.fode.gallery`.
* Made :class:`~pycaputo.fode.caputo.Trapezoidal` a bit faster by avoiding
  multiple ``einsum``.
* Added an implementation for variable-order fractional operators based on
  [Garrappa2023]_. We implement a variable-order Riemann-Liouville integral in
  :class:`~pycaputo.quadrature.variable_riemann_liouville.ExponentialRectangular`
  and an FODE solver for a variable-order Caputo derivative in
  :class:`~pycaputo.fode.variable_caputo.VariableExponentialBackwardEuler`.
  As the name suggest, this only supports the exponential decaying order
  :math:`\alpha(t) = \alpha_2 + (\alpha_1 - \alpha_2) e^{-c t}` from the paper.

Breaking Changes
^^^^^^^^^^^^^^^^

* :class:`~pycaputo.stepping.FractionalDifferentialEquationMethod` now takes
  a tuple of :class:`~pycaputo.derivatives.FractionalOperator` instances, not
  just derivative orders. This requires changing ``derivatives=(alpha, ...)``
  to ``ds=(CaputoDerivative(alpha), ...)``.
* Removed the ``gamma1p``, ``gamma2p``, ``gamma2m`` functions from
  :mod:`pycaputo.stepping`. Maybe the Caputo classes will cache them in the
  future.
* All :class:`~pycaputo.stepping.FractionalDifferentialEquationMethod` need to
  also define :meth:`~pycaputo.stepping.FractionalDifferentialEquationMethod.make_default_history`.

Maintenance
^^^^^^^^^^^

* Switched to `just <https://just.systems/man/en/>`__ for running all the simple
  development and CI commands. ``just`` is less available than ``make``, but it
  is much nicer and self-documenting.

pycaputo 0.8.1 (October 13, 2024)
---------------------------------

Dependencies
^^^^^^^^^^^^

* Add official support for Python 3.13.

Features
^^^^^^^^

* Add :class:`~pycaputo.fode.caputo.BackwardEuler` as a special case of
  :class:`~pycaputo.fode.caputo.WeightedEuler` (that should also be slightly
  faster).

Fixes
^^^^^

* Make :class:`~pycaputo.controller.FixedController` work correctly when doubling
  the time step. Now, when calling :func:`~pycaputo.controller.make_fixed_controller`
  with ``dt1`` and ``dt2 = dt1/2``, you'll get exactly double the time steps and they
  interlock in such a way that ``t1 == t2[::2]``.

pycaputo 0.8.0 (September 1, 2024)
----------------------------------

Dependencies
^^^^^^^^^^^^

* Bumped minimum Python version to 3.10. This is a hard requirement, as it
  comes with some changes to the typing syntax and other functions
  (e.g. we make use of ``zip(..., strict=True)``).

Features
^^^^^^^^

* Add a :mod:`pycaputo.typing` module containing some helpful typing definitions
  (mostly previously in :mod:`pycaputo.utils`).
* Reworked :func:`~functools.singledispatch` functions for
  :class:`~pycaputo.differentiation.DerivativeMethod`. We now have
  :func:`~pycaputo.differentiation.quadrature_weights`,
  :func:`~pycaputo.differentiation.differentiation_matrix`,
  :func:`~pycaputo.differentiation.diffs`, and
  :func:`~pycaputo.differentiation.diff`. Not all of these need to be implemented
  and most methods are not ported yet. Currently only the
  :class:`~pycaputo.differentiation.caputo.L1` and
  :class:`~pycaputo.differentiation.caputo.L2` methods implement these new functions.
* Introduce some more specific methods for the Caputo derivative. The
  :class:`~pycaputo.differentiation.caputo.L2F` uses the L2 method with function
  evaluations outside of the interval of definition. The
  :class:`~pycaputo.differentiation.caputo.LXD` allows evaluating arbitrary
  Caputo derivatives when the integer derivatives are known.
* Added a lot of fancy chaotic systems to :mod:`pycaputo.fode.gallery`. These
  are mostly used to showcase the library.

Breaking Changes
^^^^^^^^^^^^^^^^

* Renamed `pycaputo.differentiation.caputo.SpectralJacobi` to
  :class:`~pycaputo.differentiation.caputo.Jacobi`.

pycaputo 0.7.0 (July 13, 2024)
------------------------------

Dependencies
^^^^^^^^^^^^

* Official support for numpy 2.0. Everything works and is tested on the CI!

Features
^^^^^^^^

* Extend :class:`~pycaputo.fode.caputo.PECE`, :class:`~pycaputo.fode.caputo.PEC`
  and :class:`~pycaputo.fode.caputo.ModifiedPECE` to support systems with different
  orders (:ghpr:`46` and :ghissue:`17`).
* Implement diffusive methods for evaluating the Riemann-Liouville integral (:ghpr:`48`):
  :class:`~pycaputo.quadrature.riemann_liouville.YuanAgrawal`,
  :class:`~pycaputo.quadrature.riemann_liouville.Diethelm`, and
  :class:`~pycaputo.quadrature.riemann_liouville.BirkSong`.
* Implement diffusive methods for evaluating the Caputo derivative (:ghpr:`49`):
  :class:`~pycaputo.differentiation.caputo.YuanAgrawal`,
  :class:`~pycaputo.differentiation.caputo.Diethelm`, and
  :class:`~pycaputo.differentiation.caputo.BirkSong`.
* Implement approximations of the Grünwald-Letnikov derivative (:ghpr:`51`):
  :class:`~pycaputo.differentiation.grunwald_letnikov.GrunwaldLetnikov`,
  :class:`~pycaputo.differentiation.grunwald_letnikov.ShiftedGrunwaldLetnikov`,
  :class:`~pycaputo.differentiation.grunwald_letnikov.TianZhouDeng2`, and
  :class:`~pycaputo.differentiation.grunwald_letnikov.TianZhouDeng3`.
* Add derivatives of some known elementary functions in :mod:`pycaputo.special`
  (:ghpr:`50`): ``exp``, ``sin``, ``cos`` and ``pow``. They're mainly based on
  the Mittag-Leffler function.

Breaking Changes
^^^^^^^^^^^^^^^^

* Renamed ``RiemannLiouvilleFromCaputoDerivativeMethod`` to
  :class:`~pycaputo.differentiation.riemann_liouville.RiemannLiouvilleFromCaputoMethod`.
* Renamed ``CaputoDerivativeMethod`` to
  :class:`~pycaputo.differentiation.caputo.CaputoMethod`.

Fixes
^^^^^

* Fix convergence of :class:`~pycaputo.fode.caputo.ModifiedPECE` (:ghissue:`15`).

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
* Implement the Mittag-Leffler algorithm from `Garrappa2015 <https://doi.org/10.1137/140971191>`__.
* Added the :mod:`pycaputo.fode.special` module with some exact solutions. This
  is subject to a lot of change, but in general it is desired to have some
  examples for testing and demonstration.

Breaking Changes
^^^^^^^^^^^^^^^^

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
  ``gamma1p`` are also cached.
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
  :class:`~pycaputo.differentiation.riemann_liouville.RiemannLiouvilleFromCaputoMethod`.
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
