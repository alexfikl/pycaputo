.. _sec-fode:

Single-Term Fractional Ordinary Differential Equations
======================================================

.. currentmodule:: pycaputo.fode

Time Stepping Events
--------------------

.. autoclass:: Event
.. autoclass:: StepFailed
.. autoclass:: StepCompleted

Time Stepping Interface
-----------------------

.. autoexception:: AdvanceFailedError

.. autoclass:: FractionalDifferentialEquationMethod
.. autoclass:: ProductIntegrationMethod

.. data:: AdvanceResult

    A tuple containing the result of :func:`advance`. It should contain
    :math:`(y_{n + 1}, \tau_{n + 1}, h_{n + 1})`, where :math:`y_{n + 1}` is the updated
    solution, :math:`\tau_{n + 1}` is an estimate of the truncation error, and
    :math:`h_{n + 1}` is the value that should be stored in the history, if the
    step was successful.

    Alias of :class:`tuple` [:class:`Array`, :class:`Array`, :class:`Array`].

.. autofunction:: evolve
.. autofunction:: advance
.. autofunction:: make_initial_condition

Caputo Derivative FODEs
=======================

.. autoclass:: CaputoProductIntegrationMethod

.. autoclass:: CaputoForwardEulerMethod
.. autoclass:: CaputoWeightedEulerMethod
.. autoclass:: CaputoPredictorCorrectorMethod
.. autoclass:: CaputoPECEMethod
.. autoclass:: CaputoPECMethod
.. autoclass:: CaputoModifiedPECEMethod

.. class:: Array

   See :class:`pycaputo.utils.Array`.
