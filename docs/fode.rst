.. _sec-fode:

Single-Term Fractional Ordinary Differential Equations
======================================================

.. currentmodule:: pycaputo.fode

History Handling
----------------

.. autoclass:: State
.. autoclass:: History

.. autoclass:: ProductIntegrationState
.. autoclass:: VariableProductIntegrationHistory

.. autoclass:: FixedState
.. autoclass:: FixedSizeHistory

Time Interval Handling
----------------------

.. exception:: StepEstimateError

.. autoclass:: TimeSpan
.. autoclass:: FixedTimeSpan
.. autoclass:: GradedTimeSpan
.. autoclass:: LipschitzTimeSpan

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

.. autofunction:: solve

Caputo Derivative FODEs
=======================

.. autoclass:: CaputoProductIntegrationMethod

.. autoclass:: CaputoForwardEulerMethod
.. autoclass:: CaputoWeightedEulerMethod
.. autoclass:: CaputoPredictorCorrectorMethod
.. autoclass:: CaputoPECEMethod
.. autoclass:: CaputoPECMethod
.. autoclass:: CaputoModifiedPECEMethod

Integrate and Fire FODEs
========================

.. autoclass:: CaputoIntegrateFireL1Method

.. class:: Array

   See :class:`pycaputo.utils.Array`.
