# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, overload

import numpy as np

from pycaputo.history import ProductIntegrationHistory
from pycaputo.integrate_fire.base import (
    AdvanceResult,
    IntegrateFireMethod,
    IntegrateFireModel,
)
from pycaputo.logging import get_logger
from pycaputo.stepping import advance
from pycaputo.typing import Array

log = get_logger(__name__)

# {{{ model


class EIFReference(NamedTuple):
    """Reference variables used to non-dimensionalize the EIF model."""

    alpha: float
    """Fractional order used to non-dimensionalize."""

    T_ref: float
    """Time scale (in milliseconds: *ms*)."""
    V_off: float
    """Voltage offset (in millivolts: *mV*)."""
    V_ref: float
    """Voltage scale (in millivolts: *mV*)."""
    I_ref: float
    """Current scale (in picoamperes: *pA*)."""

    @overload
    def time(self, t: float) -> float: ...

    @overload
    def time(self, t: Array) -> Array: ...

    def time(self, t: float | Array) -> float | Array:
        """Add dimensions to the non-dimensional time *t*."""
        return self.T_ref * t

    def var(self, y: Array) -> Array:
        """Add dimensions to the non-dimensional potential *y*."""
        return self.V_ref * y + self.V_off


class EIFDim(NamedTuple):
    """Dimensional parameters for the Exponential Integrate-and-Fire (EIF) model."""

    current: float
    """Added current :math:`I` (in picoamperes *pA*)."""
    C: float
    """Total capacitance :math:`C` (in picofarad per ms *pF / ms^(alpha - 1)*)."""
    gl: float
    """Total leak conductance :math:`g_L` (in nanosiemens *nS*)."""
    e_leak: float
    """Equilibrium potential leak :math:`E_L` (in microvolts *mV*)."""
    delta_t: float
    r"""Threshold slope factor :math:`\Delta_T` (in microvolts *mV*)."""
    vt: float
    """Effective threshold potential :math:`V_T` (in microvolts *mV*)."""

    v_peak: float
    """Peak potential :math:`V_{peak}` (in millivolts: *mV*)."""
    v_reset: float
    """Reset potential :math:`V_r` (in millivolts: *mV*)."""

    def __str__(self) -> str:
        from pycaputo.utils import dc_stringify

        return dc_stringify(
            {
                "I      (current / pA)": self.current,
                "C      (total capacitance / pF/ms^alpha)": self.C,
                "g_L    (total leak conductance / nS)": self.gl,
                "E_leak (equilibrium potential leak / mV)": self.e_leak,
                "delta_T(threshold slope factor / mV)": self.delta_t,
                "V_T    (effective threshold potential / mV)": self.vt,
                "V_peak (peak potential / mV)": self.v_peak,
                "V_r    (reset potential / mV)": self.v_reset,
            },
            header=("model", type(self).__name__),
        )

    def ref(self, alpha: float) -> EIFReference:
        r"""Construct reference variables used in non-dimensionalizating the PIF model.

        The non-dimensionalization is performed using the following rescaling

        .. math::

            \hat{t} = \sqrt[\alpha]{\frac{g_L}{C}} t,
            \qquad
            \hat{V} = \frac{V - V_T}{\Delta_T},
            \qquad
            \hat{I} = \frac{I}{g_L \Delta_T},

        :arg alpha: the order of the fractional derivative.
        """
        return EIFReference(
            alpha=alpha,
            T_ref=1.0 / (self.gl / self.C) ** (1 / alpha),
            V_off=self.vt,
            V_ref=self.delta_t,
            I_ref=self.gl * self.delta_t,
        )

    def nondim(self, alpha: float) -> EIF:
        r"""Construct a non-dimensional set of parameters for the EIF model.

        This uses the reference variables from :meth:`ref` to reduce the parameter
        space to only the non-dimensional threshold values
        :math:`(\hat{V}_{peak}, \hat{V}_r)` and :math:`(\hat{I}, \hat{E}_L)`.
        """
        ref = self.ref(alpha)
        return EIF(
            ref=ref,
            current=self.current / ref.I_ref,
            e_leak=(self.e_leak - ref.V_off) / ref.V_ref,
            v_peak=(self.v_peak - ref.V_off) / ref.V_ref,
            v_reset=(self.v_reset - ref.V_off) / ref.V_ref,
        )


class EIF(NamedTuple):
    """Non-dimensional parameters for the EIF model (see :class:`EIFDim`)."""

    ref: EIFReference
    """Reference values used in non-dimensionalization."""

    current: float
    """Current :math:`I`."""
    e_leak: float
    """Equilibrium potential leak :math:`E_L`."""

    v_peak: float
    """Peak potential :math:`V_{peak}`."""
    v_reset: float
    """Reset potential :math:`V_r`."""

    def __str__(self) -> str:
        from pycaputo.utils import dc_stringify

        return dc_stringify(
            {
                "I      (current)": self.current,
                "E_leak (equilibrium potential leak)": self.e_leak,
                "V_peak (peak potential)": self.v_peak,
                "V_r    (reset potential)": self.v_reset,
            },
            header=("model", type(self).__name__),
        )


@dataclass(frozen=True)
class EIFModel(IntegrateFireModel):
    r"""Functionals for the EIF model with parameters :class:`EIF`.

    .. math::

        D_C^\alpha[V](t) = I(t) - (V - E_L) + \exp V,

    where :math:`I` is taken to be a constant by default, but can be
    easily overwritten by subclasses. The reset condition is given by

    .. math::

        \text{if } V > V_{peak} \qquad \text{then} \qquad V \gets V_r.
    """

    param: EIF
    """Non-dimensional parameters of the model."""

    if __debug__:

        def __post_init__(self) -> None:
            if not isinstance(self.param, EIF):
                raise TypeError(
                    f"Invalid parameter type: '{type(self.param).__name__}'"
                )

    def source(self, t: float, y: Array) -> Array:
        """Evaluate right-hand side of the LIF model."""
        return np.array(self.param.current - (y - self.param.e_leak) + np.exp(y))

    def source_jac(self, t: float, y: Array) -> Array:
        """Evaluate the Jacobian of the right-hand side of the LIF model."""
        return -1.0 + np.exp(y)

    def spiked(self, t: float, y: Array) -> float:
        """Compute a delta from the peak threshold :math:`V_{peak}`.

        :returns: a delta of :math:`V - V_{peak}` that can be used to determine
            if the neuron spiked.
        """

        (V,) = np.real_if_close(y)
        return float(V - self.param.v_peak)

    def reset(self, t: float, y: Array) -> Array:
        """Evaluate the reset values for the LIF model.

        This function assumes that the neuron has spiked, i.e. that :meth:`spiked`
        returns a non-negative value. If this is not the case, the reset should
        not be applied.
        """
        # TODO: should this be a proper check?
        assert self.spiked(t, y) > -10.0 * np.finfo(y.dtype).eps

        return np.full_like(y, self.param.v_reset)


# }}}


# {{{ CaputoExponentialIntegrateFireL1Method


def _evaluate_lambert_coefficients(
    eif: EIFModel, t: float, y: Array, h: Array, r: Array
) -> tuple[float, float, Array]:
    d0 = 1 + h
    d1 = h
    # NOTE: we need the terms that do not depend on V, so we do
    #   f(t, y) + y - exp(y) = I(t) + E_L
    # since the current may be time-dependent
    d2 = r + h * (eif.source(t, np.array([0.0])) - 1.0)

    return float(d0), float(d1), np.array(d2)


def find_maximum_time_step_lambert(
    eif: EIFModel, t: float, tprev: float, yprev: Array, r: Array
) -> float:
    """Find a maximum time step such that the Lambert W function is real.

    This function looks at the argument of the Lambert W function for the EIF
    model and ensures that it is :math:`> -1/e`. Note that this is not done
    exactly, as we assume that the memory terms *r* are fixed.

    :arg t: initial guess for the spike time.
    :arg tprev: previous time step that was successful, i.e. that resulted in a
        real valued membrane potential.
    :arg yprev: solution at the previous time step.
    :arg r: memory terms, considered fixed for this solution.
    """
    from math import gamma

    def func(tspike: float) -> float:
        alpha = eif.param.ref.alpha
        h = gamma(2 - alpha) * (tspike - tprev) ** alpha

        d0, d1, d2 = _evaluate_lambert_coefficients(
            eif, tspike, yprev, h, yprev - h * r
        )
        return float(d1 / d0 * np.exp(d2 / d0 + 1) - 1)

    import scipy.optimize as so

    result = so.root_scalar(f=func, x0=t, bracket=[tprev, t])
    t = float(result.root)

    return t - tprev


@dataclass(frozen=True)
class CaputoExponentialIntegrateFireL1Method(IntegrateFireMethod[EIFModel]):
    r"""Implementation of the L1 method for the Exponential Integrate-and-Fire model.

    The model is described by :class:`EIFModel` with parameters :class:`EIF`.
    """

    @property
    def order(self) -> float:
        # NOTE: this is currently not tested, but it should match the PIF/LIF
        # estimates for the time step even though it does not do the interpolation
        return 1.0

    def solve(self, t: float, y: Array, h: Array, r: Array) -> Array:
        r"""Solve the implicit equation for the EIF model.

        In this case, since the right-hand side is nonlinear, but we can solve
        the implicit equation using the Lambert W function, a special function
        that is a solution to

        .. math::

            z = w e^w.
        """
        from scipy.special import lambertw

        d0, d1, d2 = _evaluate_lambert_coefficients(self.source, t, y, h, r)
        V = d2 / d0 - lambertw(-d1 / d0 * np.exp(d2 / d0), tol=1.0e-12)
        V = np.real_if_close(V, tol=100)

        assert np.linalg.norm(V - h * self.source(t, V) - r) < 1.0e-8

        return np.array(V)


@advance.register(CaputoExponentialIntegrateFireL1Method)
def _advance_caputo_eif_l1(  # type: ignore[misc]
    m: CaputoExponentialIntegrateFireL1Method,
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> AdvanceResult:
    from pycaputo.controller import AdaptiveController
    from pycaputo.integrate_fire.base import advance_caputo_integrate_fire_l1
    from pycaputo.integrate_fire.spikes import estimate_spike_time_exp

    c = m.control
    assert isinstance(c, AdaptiveController)

    tprev = history.current_time
    t = tprev + dt
    result, r = advance_caputo_integrate_fire_l1(m, history, y, dt)

    model = m.source
    p = model.param
    if np.any(np.iscomplex(result.y)):
        # NOTE: if the result is complex, it means the Lambert W function is out
        # of range. We try here to find the maximum time step that would put it
        # back in range and use that to mark the spike.
        yprev = np.array([p.v_peak], dtype=y.dtype)
        ynext = np.array([p.v_reset], dtype=y.dtype)

        try:
            dts = find_maximum_time_step_lambert(model, t, tprev, y, r)
            trunc = np.zeros_like(y)
            spiked = np.array(1)
        except ValueError:
            # NOTE: if we can't find a maximum time step, just let the adaptive
            # step controller do its thing until it can't anymore
            dts = float(result.dts)
            trunc = np.full_like(y, 1.0e5)
            spiked = np.array(0)
            spiked = np.array(int(c.nrejects > c.max_rejects))

        result = AdvanceResult(
            y=ynext,
            trunc=trunc,
            storage=np.hstack([yprev, ynext]),
            spiked=spiked,
            dts=np.array(dts),
        )
    elif model.spiked(t, result.y) > 0.0:
        yprev = np.array([p.v_peak], dtype=y.dtype)
        ynext = np.array([p.v_reset], dtype=y.dtype)

        ts = estimate_spike_time_exp(t, result.y[0], tprev, y[0], p.v_peak)
        result = AdvanceResult(
            y=ynext,
            trunc=np.zeros_like(y),
            storage=np.hstack([yprev, ynext]),
            spiked=np.array(1),
            dts=np.array(ts - tprev),
        )

    return result


# }}}
