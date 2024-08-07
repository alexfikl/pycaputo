.. _sec-step-size-selection:

Step Size Selection
===================

This module implements a simple time step selection interface that can be used
by fractional order differential equation solvers. It is largely based on the
ideas from [Hairer2008]_ and [Hairer2010]_ for integer order methods and
uses the implementation from
`OrdinaryDiffEq.jl <https://docs.sciml.ai/DiffEqDocs/latest/extras/timestepping/>`__
as an inspiration.

A controller must inherit from :class:`~pycaputo.controller.Controller` and
implement the 4 following functions in an appropriate way:

* :func:`~pycaputo.controller.evaluate_error_estimate`: evaluates a scaled
  error estimate that can be used for time step control. The error estimate
  should be :math:`E_{\text{est}} \in [0, 1]` if the step is meant to be accepted
  and in :math:`(1, \infty)` otherwise.
* :func:`~pycaputo.controller.evaluate_timestep_factor`: computes a factor to
  increase or decrease the time step based on :math:`E_{\text{est}}`.
* :func:`~pycaputo.controller.evaluate_timestep_accept` and
  :func:`~pycaputo.controller.evaluate_timestep_reject`: evaluates the time
  step for the next iteration.

In broad strokes, an algorithm using the adaptive step size controllers should
be structured as follows

.. code:: python

    from pycaputo.controller import (
        evaluate_error_estimate,
        evaluate_timestep_factor,
        evaluate_timestep_accept,
        evaluate_timestep_reject)

    while t < tfinal:
        # Evaluate the solution and the truncation error at :math:`t_{n + 1}`
        # with the current estimate of the time step
        ynext, ytrunc = advance(...)

        # Use the controller to appropriately scale the truncation error and
        # obtain an error estimator that can be used to scale the time step
        eest = evaluate_error_estimate(c, ytrunc, ...)
        accepted = eest < 1.0

        if accepted:
            # Step is accepted
            q = evaluate_timestep_factor(c, eest, ...)
            dtnext = evaluate_timestep_accept(c, q, ...)
        else:
            # Step is rejected
            q = evaluate_timestep_factor(c, eest, ...)
            dtnext = evaluate_timestep_reject(c, q, ...)

        if accepted:
            # Update iteration counters if the step is accepted
            n += 1
            t += dt
            dt = dtnext
        else:
            # Only update the time step if the step is rejected
            assert dt >= dtnext
            dt = dtnext

The methods used to define the control algorithm use the :func:`~functools.singledispatch`
mechanism and must be implemented for every new controller. Note that not all
time stepping methods will accept or handle adaptive time step selection in
this fashion.

.. warning::

    This is an experimental interface and more research is needed into time
    step selection for fractional methods. Currently only the
    :class:`~pycaputo.controller.JannelliIntegralController` has been tested
    and published in the literature.

    If in doubt, use one of the fixed step size controllers, such as
    :class:`~pycaputo.controller.FixedController` or
    :class:`~pycaputo.controller.GradedController`, to compare your results
    against.

API Reference
-------------

.. automodule:: pycaputo.controller

.. class:: Array

   See :class:`pycaputo.typing.Array`.
