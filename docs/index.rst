PYCAPUTO: Fractional Calculus Toolkit
=====================================

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting started

    installation
    tutorials
    gallery_chaos
    gallery_neural
    literature
    changelog

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Developer Guides

    guide_derivative
    guide_quadrature
    guide_differentiation
    guide_timestepping

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: API Reference

    operator
    quadrature
    differentiation
    timestepping
    fode
    integrate_fire
    misc

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
* Solve single-term systems of fractional-order ordinary differential equation.
* Easily extend this functionality with new numerical methods.

At the moment, performance is not an important focus, but fractional-order
operators are generally more computationally intensive than their integer-order
counterparts. Once a solid framework is worked out, more care will be given to
this aspect as well.

Acknowledgments
===============

Work on ``pycaputo`` was sponsored, in part, by

* the West University of Timișoara (Romania) under START Grant No. 33580/25.05.2023,
* the CNCS-UEFISCDI (Romania), under Project No. ROSUA-2024-0002,
* the "Romanian Hub for Artificial Intelligence - HRIA", Smart Growth,
  Digitization and Financial Instruments Program, 2021-2027, MySMIS no. 351416.

The views and opinions expressed herein do not necessarily reflect those of the
funding agencies.
