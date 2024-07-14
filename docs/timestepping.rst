.. _sec-fode:

Time Stepping
=============

This section provides an API reference for the time stepping interface of
``pycaputo``. The time stepping has three components

* An **event-based** stepping interface. This consists of a subclass of
  :class:`~pycaputo.stepping.FractionalDifferentialEquationMethod` that implements
  :func:`~pycaputo.stepping.evolve` to iterate through the time steps. Each
  time step returns an :class:`~pycaputo.events.Event` depending on its state.
* A **history** (checkpointing) interface. This consists of a subclass of
  :class:`~pycaputo.history.History` that handles retrieving and storing the
  history for a fractional differential equation. The interface is intentionally
  general to allow retrieving data from disk for large problems.
* A **step size selection** (controller) interface. This consists of a subclass of
  :class:`~pycaputo.controller.Controller` that can be used to estimate the next
  step size and accept or reject a time step.

.. toctree::
    :maxdepth: 1
    :caption: Table of Contents

    timestepping_interface
    timestepping_history
    timestepping_controller
