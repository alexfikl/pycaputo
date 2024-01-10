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

.. autoclass:: FractionalDifferentialEquationMethod
.. autoclass:: ProductIntegrationMethod

.. autofunction:: evolve
.. autofunction:: advance
.. autofunction:: make_initial_condition

Caputo Derivative FODEs
=======================

.. autoclass:: AdvanceResult
.. autoclass:: CaputoProductIntegrationMethod

.. autoclass:: CaputoForwardEulerMethod
.. autoclass:: CaputoWeightedEulerMethod
.. autoclass:: CaputoPredictorCorrectorMethod
.. autoclass:: CaputoPECEMethod
.. autoclass:: CaputoPECMethod
.. autoclass:: CaputoModifiedPECEMethod

.. class:: Array

   See :class:`pycaputo.utils.Array`.
