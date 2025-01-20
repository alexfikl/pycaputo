# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable
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

# {{{ first spike time


def _find_bracket(
    f: Callable[[float], float], a: float = 0.0, b: float = 10.0
) -> tuple[float, float]:
    t = np.linspace(a, b, 32)

    fprev = f(t[0])
    n = 1
    while n < t.size:
        fnext = f(t[n])
        if fprev * fnext < 0.0:
            return t[n - 1], t[n]

        n += 1

    return a, b


def _find_first_spike_time(p: LIF, V0: float = 0.0) -> float:
    from pymittagleffler import mittag_leffler

    alpha = p.ref.alpha

    def func(t: float) -> float:
        a = p.current + p.e_leak
        b = a - V0

        E = mittag_leffler(-(t**alpha), alpha=alpha, beta=1.0)
        result = a - b * E - p.v_peak

        return float(np.real_if_close(result))

    import scipy.optimize as so

    bracket = _find_bracket(func)
    result = so.root_scalar(func, x0=(bracket[1] + bracket[0]) / 2, bracket=bracket)

    return float(result.root)


# }}}


# {{{ model


class LIFReference(NamedTuple):
    """Reference variables used to non-dimensionalize the LIF model."""

    alpha: float
    """Fractional order used to non-dimensionalize."""

    T_ref: float
    """Time scale (in milliseconds: *ms*)."""
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
        return self.V_ref * y


class LIFDim(NamedTuple):
    """Dimensional parameters for the Leaky Integrate-and-Fire (LIF) model."""

    current: float
    """Added current :math:`I` (in picoamperes *pA*)."""
    C: float
    """Total capacitance :math:`C` (in picofarad per ms *pF / ms^(alpha - 1)*)."""

    gl: float
    """Total leak conductance :math:`g_L` (in nanosiemens *nS*)."""
    e_leak: float
    """Equilibrium potential leak :math:`E_L` (in microvolts *mV*)."""

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
                "V_peak (peak potential / mV)": self.v_peak,
                "V_r    (reset potential / mV)": self.v_reset,
            },
            header=("model", type(self).__name__),
        )

    def ref(self, alpha: float, *, V_ref: float | None = None) -> LIFReference:
        r"""Construct reference variables used in non-dimensionalizating the LIF model.

        The non-dimensionalization is performed using the following rescaling

        .. math::

            \hat{t} = \sqrt[\alpha]{\frac{g_L}{C}} t,
            \qquad
            \hat{V} = \frac{V}{V_{ref}},
            \qquad
            \hat{I} = \frac{I}{g_L V_{ref}},

        :arg alpha: the order of the fractional derivative.
        :arg v_ref: a reference potential used in non-dimensionalizing the
            membrane potential.
        """

        if V_ref is None:
            V_ref = max(abs(self.v_peak), abs(self.v_reset))
        V_ref = abs(V_ref)

        return LIFReference(
            alpha=alpha,
            T_ref=1.0 / (self.gl / self.C) ** (1 / alpha),
            V_ref=V_ref,
            I_ref=self.gl * V_ref,
        )

    def nondim(self, alpha: float, *, V_ref: float | None = None) -> LIF:
        r"""Construct a non-dimensional set of parameters for the LIF model.

        This uses the reference variables from :meth:`ref` to reduce the parameter
        space to only the non-dimensional threshold values
        :math:`(\hat{V}_{peak}, \hat{V}_r)` and :math:`(\hat{I}, \hat{E}_L)`.
        """
        ref = self.ref(alpha, V_ref=V_ref)
        return LIF(
            ref=ref,
            current=self.current / ref.I_ref,
            e_leak=self.e_leak / ref.V_ref,
            v_peak=self.v_peak / ref.V_ref,
            v_reset=self.v_reset / ref.V_ref,
        )


class LIF(NamedTuple):
    """Non-dimensional parameters for the LIF model (see :class:`LIFDim`)."""

    ref: LIFReference
    """Reference values used in non-dimensionalization."""

    current: float
    """Current :math:`I`."""
    e_leak: float
    """Equilibrium potential leak :math:`E_L`."""

    v_peak: float
    """Peak potential :math:`V_{peak}`."""
    v_reset: float
    """Reset potential :math:`V_r`."""

    def constant_spike_times(self, tfinal: float, V0: float = 0.0) -> Array:
        """Compute the spike times for a constant current.

        Note that we can only compute the first spike time, since there is no
        known analytic solution for the remaining spikes.

        :arg tfinal: final time for the evolution.
        :arg V0: initial membrane current.
        """
        result = _find_first_spike_time(self, V0=V0)
        return np.array([result])

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
class LIFModel(IntegrateFireModel):
    r"""Functionals for the LIF model with parameters :class:`LIF`.

    .. math::

        D_C^\alpha[V](t) = I(t) - (V - E_L),

    where :math:`I` is taken to be a constant by default, but can be
    easily overwritten by subclasses. The reset condition is also given by

    .. math::

        \text{if } V > V_{peak} \qquad \text{then} \qquad V \gets V_r.
    """

    param: LIF
    """Non-dimensional parameters of the model."""

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
class CaputoLeakyIntegrateFireL1Method(IntegrateFireMethod[LIFModel]):
    r"""Implementation of the L1 method for the Leaky Integrate-and-Fire model.

    The model is described by :class:`LIFModel` with parameters :class:`LIF`.
    """

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
def _advance_caputo_lif_l1(  # type: ignore[misc]
    m: CaputoLeakyIntegrateFireL1Method,
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> AdvanceResult:
    from pycaputo.integrate_fire.base import (
        advance_caputo_integrate_fire_l1,
        advance_caputo_integrate_fire_spike_linear,
    )

    model = m.source
    tprev = history.current_time
    t = tprev + dt

    result, _ = advance_caputo_integrate_fire_l1(m, history, y, dt)
    if model.spiked(t, result.y) > 0.0:
        p = model.param
        result = advance_caputo_integrate_fire_spike_linear(
            t, result.y[0], history, v_peak=p.v_peak, v_reset=p.v_reset
        )

    return result


# }}}
