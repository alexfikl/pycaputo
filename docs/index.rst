pycaputo documentation
======================

.. toctree::
    :maxdepth: 2
    :hidden:

    tutorials
    operator
    quadrature
    differentiation
    history
    controllers
    timestepping
    fode
    misc
    references
    changelog

.. warning::

   This package is currently in development and very experimental (the API
   can and will change frequently for the foreseeable future). For more mature
   libraries see `differint <https://github.com/differint/differint>`__ in
   Python or `FractionalDiffEq.jl <https://github.com/SciFracX/FractionalDiffEq.jl>`__
   in Julia.

This package provides tools to (numerically) compute fractional order
derivatives and integrals. It offers functionality to (non-exhaustive)

* Evaluate Caputo fractional-order derivatives of real orders.
* Evaluate Riemann-Liouville fractional-order integrals of arbitrary real orders.
* Solve single-term fractional-order ordinary differential equations.
* Easily extend this functionality with new numerical methods.

At the moment, performance is not an important focus, but fractional-order
operators are generally more computationally intensive than their integer-order
counterparts. Once a solid framework is worked out, more care will be given to
this aspect as well.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
