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


class PIFReference(NamedTuple):
    """Reference variables used to non-dimensionalize the PIF model."""

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


class PIFDim(NamedTuple):
    """Dimensional parameters for the Perfect Integrate-and-Fire (PIF) model."""

    current: float
    """Added current :math:`I` (in picoamperes: *pA*)."""
    C: float
    """Total capacitance :math:`C` (in picofarad per ms: *pF / ms^(alpha - 1)*)."""

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
                "V_peak (peak potential / mV)": self.v_peak,
                "V_r    (reset potential / mV)": self.v_reset,
            },
            header=("model", type(self).__name__),
        )

    def ref(
        self,
        alpha: float,
        *,
        V_ref: float | None = None,
        I_ref: float | None = None,
    ) -> PIFReference:
        r"""Construct reference variables used in non-dimensionalizating the PIF model.

        The non-dimensionalization is performed using the following rescaling

        .. math::

            \hat{t} = \sqrt[\alpha]{\frac{I_{ref}}{C V_{ref}}} t,
            \qquad
            \hat{V} = \frac{V}{V_{ref}},
            \qquad
            \hat{I} = \frac{I}{I_{ref}}.

        :arg alpha: the order of the fractional derivative.
        :arg V_ref: a reference potential used in non-dimensionalizing the
            membrane potential.
        :arg I_ref: a reference current used in non-dimensionalizing the current.
        """

        if V_ref is None:
            V_ref = max(abs(self.v_peak), abs(self.v_reset))
        V_ref = abs(V_ref)

        if I_ref is None:
            I_ref = abs(self.current)
        I_ref = abs(I_ref)

        return PIFReference(
            alpha=alpha,
            T_ref=1.0 / (I_ref / (self.C * V_ref)) ** (1 / alpha),
            V_ref=V_ref,
            I_ref=I_ref,
        )

    def nondim(
        self,
        alpha: float,
        *,
        V_ref: float | None = None,
        I_ref: float | None = None,
    ) -> PIF:
        r"""Construct a non-dimensional set of parameters for the PIF model.

        This uses the reference variables from :meth:`ref` to reduce the parameter
        space to only the non-dimensional threshold values
        :math:`(\hat{V}_{peak}, \hat{V}_r)` and current :math:`\hat{I}`.
        """
        ref = self.ref(alpha, V_ref=V_ref, I_ref=I_ref)

        return PIF(
            ref=ref,
            current=self.current / ref.I_ref,
            v_peak=self.v_peak / ref.V_ref,
            v_reset=self.v_reset / ref.V_ref,
        )


class PIF(NamedTuple):
    """Non-dimensional parameters for the PIF model (see :class:`PIFDim`)."""

    ref: PIFReference
    """Reference values used in non-dimensionalization."""

    current: float
    """Current."""
    v_peak: float
    """Peak potential :math:`V_{peak}`."""
    v_reset: float
    """Reset potential :math:`V_r`."""

    def constant_spike_times(self, tfinal: float, V0: float = 0.0) -> Array:
        """Compute the spike times for a constant current.

        :arg tfinal: final time for the evolution.
        :arg V0: initial membrane current.
        """
        from math import ceil, gamma

        alpha = self.ref.alpha
        gamma1p = gamma(1 + alpha)
        kmax = ceil(
            ((self.v_peak - V0) * gamma1p - self.current * tfinal**alpha)
            / ((self.v_reset - self.v_peak) * gamma1p)
        )
        k = np.arange(kmax)
        ts = (
            gamma1p
            * (self.v_peak - V0 + k * (self.v_peak - self.v_reset))
            / self.current
        ) ** (1.0 / alpha)

        return ts

    def __str__(self) -> str:
        from pycaputo.utils import dc_stringify

        return dc_stringify(
            {
                "I      (current)": self.current,
                "V_peak (peak potential)": self.v_peak,
                "V_r    (reset potential)": self.v_reset,
            },
            header=("model", type(self).__name__),
        )


@dataclass(frozen=True)
class PIFModel(IntegrateFireModel):
    r"""Functionals for the PIF model with parameters :class:`PIF`.

    .. math::

        D_C^\alpha[V](t) = I(t),

    where :math:`I` is taken to be a constant by default, but can be
    easily overwritten by subclasses. The reset condition is also given by

    .. math::

        \text{if } V > V_{peak} \qquad \text{then} \qquad V \gets V_r.
    """

    param: PIF
    """Non-dimensional parameters of the model."""

    if __debug__:

        def __post_init__(self) -> None:
            if not isinstance(self.param, PIF):
                raise TypeError(
                    f"Invalid parameter type: '{type(self.param).__name__}'"
                )

    def source(self, t: float, y: Array) -> Array:
        """Evaluate right-hand side of the PIF model."""
        return np.full_like(y, self.param.current)

    def source_jac(self, t: float, y: Array) -> Array:
        """Evaluate the Jacobian of the right-hand side of the PIF model."""
        return np.zeros_like(y)

    def spiked(self, t: float, y: Array) -> float:
        """Compute a delta from the peak threshold :math:`V_{peak}`.

        :returns: a delta of :math:`V - V_{peak}` that can be used to determine
            if the neuron spiked.
        """

        (V,) = y
        return float(V - self.param.v_peak)

    def reset(self, t: float, y: Array) -> Array:
        """Evaluate the reset values for the PIF model.

        This function assumes that the neuron has spiked, i.e. that :meth:`spiked`
        returns a non-negative value. If this is not the case, the reset should
        not be applied.
        """
        # TODO: should this be a proper check?
        assert self.spiked(t, y) > -10.0 * np.finfo(y.dtype).eps

        return np.full_like(y, self.param.v_reset)


# }}}


# {{{ CaputoPerfectIntegrateFireL1Method


@dataclass(frozen=True)
class CaputoPerfectIntegrateFireL1Method(IntegrateFireMethod[PIFModel]):
    r"""Implementation of the L1 method for the Perfect Integrate-and-Fire model.

    The model is described by :class:`PIFModel` with parameters :class:`PIF`.
    """

    @property
    def spike_order(self) -> int:
        return 1

    def solve(self, t: float, y0: Array, c: Array, r: Array) -> Array:
        r"""Solve the implicit equation for the PIF model.

        In this case, since the right-hand side does not depend on the solution,
        we simply have that

        .. math::

            y_{n + 1} = c f(t_{n + 1}, \cdot) + r
        """
        return c * self.source(t, y0) + r


@advance.register(CaputoPerfectIntegrateFireL1Method)
def _advance_caputo_pif_l1(  # type: ignore[misc]
    m: CaputoPerfectIntegrateFireL1Method,
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> AdvanceResult:
    from pycaputo.integrate_fire.base import (
        advance_caputo_integrate_fire_l1,
        advance_caputo_integrate_fire_spike_linear,
        advance_caputo_integrate_fire_spike_quadratic,
    )

    model = m.source
    tprev = history.current_time
    t = tprev + dt

    result, _ = advance_caputo_integrate_fire_l1(m, history, y, dt)
    if model.spiked(t, result.y) > 0.0:
        p = model.param
        if m.spike_order == 1:
            result = advance_caputo_integrate_fire_spike_linear(
                t, result.y[0], history, v_peak=p.v_peak, v_reset=p.v_reset
            )
        elif m.spike_order == 2:
            result = advance_caputo_integrate_fire_spike_quadratic(
                t, result.y[0], history, v_peak=p.v_peak, v_reset=p.v_reset
            )
        else:
            raise ValueError(
                f"Unsupported spike time approximation order: {m.spike_order}"
            )

    return result


# }}}
