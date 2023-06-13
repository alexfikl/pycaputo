.. _sec-fode:

Single-Term Fractional Ordinary Differential Equations
======================================================

.. currentmodule:: pycaputo.fode

History Handling
----------------

.. autoclass:: StateHistory
.. autoclass:: SourceHistory
.. autoclass:: History

Interface
---------

.. autoclass:: Event
.. autoclass:: StepFailed
.. autoclass:: StepCompleted

.. autoclass:: FractionalDifferentialEquationMethod

.. autofunction:: make_initial_condition
.. autofunction:: evolve
.. autofunction:: advance

.. autofunction:: make_predict_time_step_fixed
.. autofunction:: make_predict_time_step_graded

Caputo Derivative FODEs
=======================

.. autoclass:: CaputoDifferentialEquationMethod
.. autoclass:: CaputoForwardEulerMethod
.. autoclass:: CaputoCrankNicolsonMethod
.. autoclass:: CaputoPredictorCorrectorMethod
.. autoclass:: CaputoPECEMethod
.. autoclass:: CaputoPECMethod
