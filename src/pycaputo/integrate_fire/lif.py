# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

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


class LIFDim(NamedTuple):
    """Dimensional parameters for the Leaky Integrate-and-Fire (LIF) model."""

    #: Added current :math:`I` (in picoamperes *pA*).
    current: float
    #: Total capacitance :math:`C` (in picofarad per millisecond *pF / ms^(alpha - 1)*).
    C: float
    #: Total leak conductance :math:`g_L` (in nanosiemens *nS*).
    gl: float
    #: Equilibrium potential leak :math:`E_L` (in microvolts *mV*).
    e_leak: float

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
                "V_peak (peak potential / mV)": self.v_peak,
                "V_r    (reset potential / mV)": self.v_reset,
            },
            header=("model", type(self).__name__),
        )

    def nondim(self, alpha: float, *, v_ref: float | None = None) -> LIF:
        r"""Construct a non-dimensional set of parameters for the LIF model.

        The non-dimensionalization is performed using the following rescaling

        .. math::

            \hat{t} = \sqrt[\alpha]{\frac{g_L}{C}} t,
            \qquad
            \hat{V} = \frac{V}{V_{ref}},
            \qquad
            \hat{I} = \frac{I}{g_L V_{ref}},

        which results in a reduction of the parameter space to just the threshold
        values :math:`(V_{peak}, V_r)` and the constants :math:`(I, E_L)`.

        :arg alpha: the order of the fractional derivative.
        :arg v_ref: a reference potential used in non-dimensionalizing the
            membrane potential.
        """
        if v_ref is None:
            v_ref = max(abs(self.v_peak), abs(self.v_reset))
        v_ref = abs(v_ref)

        return LIF(
            alpha=alpha,
            T=(self.gl / self.C) ** (1 / alpha),
            current=self.current / (self.gl * v_ref),
            e_leak=self.e_leak / v_ref,
            v_peak=self.v_peak / v_ref,
            v_reset=self.v_reset / v_ref,
        )


class LIF(NamedTuple):
    """Non-dimensional parameters for the LIF model (see :class:`LIFDim`)."""

    #: Fractional order used in the non-dimensionalization.
    alpha: float
    #: Fractional time scale.
    T: float

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
                "alpha  (fractional order)": self.alpha,
                "T      (fractional time scale)": self.T,
                "I      (current)": self.current,
                "E_leak (equilibrium potential leak)": self.e_leak,
                "V_peak (peak potential)": self.v_peak,
                "V_r    (reset potential)": self.v_reset,
            },
            header=("model", type(self).__name__),
        )


@dataclass(frozen=True)
class LIFModel(IntegrateFireModel):
    r"""Functionals for the LIF model with parameters :class:`LIF`.

    .. math::

        D_C^\alpha[V](t) = I(t) - (V - E_L),

    where :math:`I` is taken to be a constant by default, but can be
    easily overwritten by subclasses. The reset condition is also given by

    .. math::

        \text{if } V > V_{peak} \qquad \text{then} \qquad V \gets V_r.
    """

    #: Non-dimensional parameters of the model.
    param: LIF

    if __debug__:

        def __post_init__(self) -> None:
            if not isinstance(self.param, LIF):
                raise TypeError(
                    f"Invalid parameter type: '{type(self.param).__name__}'"
                )

    def source(self, t: float, y: Array) -> Array:
        """Evaluate right-hand side of the LIF model."""
        return self.param.current - (y - self.param.e_leak)

    def source_jac(self, t: float, y: Array) -> Array:
        """Evaluate the Jacobian of the right-hand side of the LIF model."""
        return np.full_like(y, -1.0)

    def spiked(self, t: float, y: Array) -> float:
        """Compute a delta from the peak threshold :math:`V_{peak}`.

        :returns: a delta of :math:`V - V_{peak}` that can be used to determine
            if the neuron spiked.
        """

        (V,) = y
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


# {{{ CaputoLeakyIntegrateFireL1Method


@dataclass(frozen=True)
class CaputoLeakyIntegrateFireL1Method(CaputoIntegrateFireL1Method[LIFModel]):
    r"""Implementation of the L1 method for the Leaky Integrate-and-Fire model.

    The model is described by :class:`LIFModel` with parameters :class:`LIF`.
    """

    model: LIFModel

    def solve(self, t: float, y0: Array, c: Array, r: Array) -> Array:
        r"""Solve the implicit equation for the LIF model.

        In this case, since the right-hand side is linear, so we can solve the
        implicit equation exactly as

        .. math::

            y_{n + 1} = \frac{c I(t_{n + 1}) + c E_L + r}{1 + c}
        """
        # NOTE: we need the terms that do not depend on V, so we do
        #   f(t, y) + y - exp(y) = I(t) + E_L
        # since the current may be time-dependent
        current = self.source(t, y0) + y0

        result = (c * current + r) / (1 + c)
        return np.array(result)


@advance.register(CaputoLeakyIntegrateFireL1Method)
def _advance_caputo_lif_l1(
    m: CaputoLeakyIntegrateFireL1Method,
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> AdvanceResult:
    from pycaputo.integrate_fire.base import (
        advance_caputo_integrate_fire_l1,
        advance_caputo_integrate_fire_spike_linear,
    )

    tprev = history.current_time
    t = tprev + dt

    result = advance_caputo_integrate_fire_l1(m, history, y, dt)
    if m.model.spiked(t, result.y) > 0.0:
        p = m.model.param
        result = advance_caputo_integrate_fire_spike_linear(
            tprev, y, t, result, v_peak=p.v_peak, v_reset=p.v_reset
        )

    return result


# }}}
