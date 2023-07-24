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

Interface
---------

.. autoclass:: Event
.. autoclass:: StepFailed
.. autoclass:: StepCompleted

.. autoclass:: FractionalDifferentialEquationMethod
.. autoclass:: ProductIntegrationMethod

.. autofunction:: evolve
.. autofunction:: advance
.. autofunction:: make_initial_condition

.. autofunction:: make_predict_time_step_fixed
.. autofunction:: make_predict_time_step_graded

Caputo Derivative FODEs
=======================

.. autoclass:: CaputoProductIntegrationMethod

.. autoclass:: CaputoForwardEulerMethod
.. autoclass:: CaputoWeightedEulerMethod
.. autoclass:: CaputoPredictorCorrectorMethod
.. autoclass:: CaputoPECEMethod
.. autoclass:: CaputoPECMethod
.. autoclass:: CaputoModifiedPECEMethod
