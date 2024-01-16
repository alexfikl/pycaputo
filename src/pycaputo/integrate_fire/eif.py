# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, overload

import numpy as np

from pycaputo.fode.base import advance
from pycaputo.history import ProductIntegrationHistory
from pycaputo.integrate_fire.base import (
    AdvanceResult,
    CaputoIntegrateFireL1Method,
    IntegrateFireModel,
)
from pycaputo.logging import get_logger
from pycaputo.utils import Array, dc_stringify

logger = get_logger(__name__)

# {{{ model


class EIFReference(NamedTuple):
    """Reference variables used to non-dimensionalize the EIF model."""

    #: Fractional order used to non-dimensionalize.
    alpha: float

    #: Time scale (in milliseconds: *ms*).
    T_ref: float
    #: Voltage offset (in millivolts: *mV*).
    V_off: float
    #: Voltage scale (in millivolts: *mV*).
    V_ref: float
    #: Current scale (in picoamperes: *pA*).
    I_ref: float

    @overload
    def time(self, t: float) -> float: ...

    @overload
    def time(self, t: Array) -> Array: ...

    def time(self, t: float | Array) -> float | Array:
        """Add dimensions to the non-dimensional time *t*."""
        return self.T_ref * t

    def potential(self, V: Array) -> Array:
        """Add dimensions to the non-dimensional potential *V*."""
        return self.V_ref * V + self.V_off


class EIFDim(NamedTuple):
    """Dimensional parameters for the Exponential Integrate-and-Fire (EIF) model."""

    #: Added current :math:`I` (in picoamperes *pA*).
    current: float
    #: Total capacitance :math:`C` (in picofarad per millisecond *pF / ms^(alpha - 1)*).
    C: float
    #: Total leak conductance :math:`g_L` (in nanosiemens *nS*).
    gl: float
    #: Equilibrium potential leak :math:`E_L` (in microvolts *mV*).
    e_leak: float
    #: Threshold slope factor :math:`\Delta_T` (in microvolts *mV*).
    delta_t: float
    #: Effective threshold potential :math:`V_T` (in microvolts *mV*).
    vt: float

    #: Peak potential :math:`V_{peak}` (in millivolts: *mV*).
    v_peak: float
    #: Reset potential :math:`V_r` (in millivolts: *mV*).
    v_reset: float

    def __str__(self) -> str:
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

        which results in a reduction of the parameter space to just the threshold
        values :math:`(V_{peak}, V_r)` and the constants :math:`(I, E_L)`.

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

    #: Reference values used in non-dimensionalization.
    ref: EIFReference

    #: Current :math:`I`.
    current: float
    #: Equilibrium potential leak :math:`E_L`.
    e_leak: float

    #: Peak potential :math:`V_{peak}`.
    v_peak: float
    #: Reset potential :math:`V_r`.
    v_reset: float

    def __str__(self) -> str:
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

    #: Non-dimensional parameters of the model.
    param: EIF

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


@dataclass(frozen=True)
class CaputoExponentialIntegrateFireL1Method(CaputoIntegrateFireL1Method[EIFModel]):
    r"""Implementation of the L1 method for the Exponential Integrate-and-Fire model.

    The model is described by :class:`EIFModel` with parameters :class:`EIF`.
    """

    model: EIFModel

    def solve(self, t: float, y0: Array, c: Array, r: Array) -> Array:
        r"""Solve the implicit equation for the EIF model.

        In this case, since the right-hand side is nonlinear, but we can solve
        the implicit equation using the Lambert W function, a special function
        that is a solution to

        .. math::

            z = w e^w.
        """
        from scipy.special import lambertw

        d0 = 1 + c
        d1 = c
        # NOTE: we need the terms that do not depend on V, so we do
        #   f(t, y) + y - exp(y) = I(t) + E_L
        # since the current may be time-dependent
        d2 = r + c * (self.source(t, y0) + y0 - np.exp(y0))

        V = d2 / d0 - lambertw(-d1 / d0 * np.exp(d2 / d0), tol=1.0e-12)
        V = np.real_if_close(V, tol=100)

        assert np.linalg.norm(V - c * self.source(t, V) - r) < 1.0e-8

        return np.array(V)


@advance.register(CaputoExponentialIntegrateFireL1Method)
def _advance_caputo_eif_l1(
    m: CaputoExponentialIntegrateFireL1Method,
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> AdvanceResult:
    from pycaputo.integrate_fire.base import (
        advance_caputo_integrate_fire_l1,
        advance_caputo_integrate_fire_spike_exp,
    )

    tprev = history.current_time
    t = tprev + dt

    result = advance_caputo_integrate_fire_l1(m, history, y, dt)

    is_complex = np.any(np.iscomplex(result.y))
    if is_complex or m.model.spiked(t, result.y) > 0.0:
        p = m.model.param
        result = advance_caputo_integrate_fire_spike_exp(
            tprev, y, t, result, v_peak=p.v_peak, v_reset=p.v_reset
        )

        from pycaputo.controller import AdaptiveController

        if is_complex and isinstance(m.control, AdaptiveController):
            c = m.control

            if dt < c.dtmin or c.nrejects > c.max_rejects:
                # NOTE: we really tried to reduce the time step until the spike
                # occurred, so now it's time to give up and spike!
                pass
            else:
                result = result._replace(spiked=np.array(0))

    return result


# }}}
