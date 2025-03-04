# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Any, NamedTuple, TypeVar

import numpy as np
from scipy.special import gamma

from pycaputo import events
from pycaputo.derivatives import CaputoDerivative
from pycaputo.history import History, ProductIntegrationHistory
from pycaputo.logging import get_logger
from pycaputo.stepping import (
    FractionalDifferentialEquationMethod,
    advance,
    evolve,
    make_initial_condition,
)
from pycaputo.typing import Array

log = get_logger(__name__)

# {{{ model interface


class IntegrateFireModel(ABC):
    @abstractmethod
    def source(self, t: float, y: Array, /) -> Array:
        """Evaluate the right-hand side of the IF model."""

    @abstractmethod
    def source_jac(self, t: float, y: Array, /) -> Array:
        """Evaluate the Jacobian of the right-hand side of the IF model."""

    @abstractmethod
    def spiked(self, t: float, y: Array, /) -> float:
        """Check if the neuron has spiked.

        In most cases, this will simply be a delta of :math:`V - V_{peak}`, but
        it is generally left to the model itself to define how the spike is
        recognized.

        :returns: a positive value if the neuron spiked, a negative value
            if the neuron has not spiked yet and *0* if the spike threshold is
            achieved exactly.
        """

    @abstractmethod
    def reset(self, t: float, y: Array, /) -> Array:
        """Evaluate the reset values for the IF model.

        This function assumes that the neuron has spiked, i.e. that :meth:`spiked`
        returns a non-negative value. If this is not the case, the reset should
        not be applied.
        """

    # NOTE: this is here to satisfy the StateFunction protocol

    def __call__(self, t: float, y: Array, /) -> Array:
        """Evaluate the right-hand side of the IF model (see :meth:`source`)."""
        return self.source(t, y)


IntegrateFireModelT = TypeVar("IntegrateFireModelT", bound=IntegrateFireModel)
"""An invariant :class:`~typing.TypeVar` bound to :class:`IntegrateFireModel`."""

# }}}


# {{{ method


@dataclass(frozen=True)
class StepFailed(events.StepFailed):
    pass


@dataclass(frozen=True)
class StepAccepted(events.StepAccepted):
    spiked: bool
    """A flag to denote if the current accepted step was due to a spike."""


@dataclass(frozen=True)
class StepRejected(events.StepRejected):
    pass


class AdvanceResult(NamedTuple):
    """Result of :func:`~pycaputo.stepping.advance` for :class:`IntegrateFireMethod`
    subclasses.
    """

    y: Array
    """Estimated solution at the next time step."""
    trunc: Array
    """Estimated truncation error at the next time step."""
    storage: Array
    """Array to add to the history storage."""

    spiked: Array
    """Flag to denote if a spike occurred."""
    dts: Array
    """Time step taken by the method in case a spike occurred."""


@dataclass(frozen=True)
class IntegrateFireMethod(
    FractionalDifferentialEquationMethod[CaputoDerivative, IntegrateFireModelT]
):
    r"""A discretization of Integrate-and-Fire models using a fractional derivative.

    A generic Integrate-and-Fire model is given by

    .. math::

        D_C^\alpha[\mathbf{y}](t) = \mathbf{f}(t, \mathbf{y})

    together with a reset condition of the form

    .. math::

        \text{spiked}(t, \mathbf{y}) \ge 0
        \quad \text{then} \quad
        \mathbf{y} = \text{reset}(t, \mathbf{y}),

    where the ``spiked`` function and the ``reset`` functions are provided
    by the user. For example, a standard adaptive model has two variables
    :math:`(V, w)` and uses the reset condition

    .. math::

        V \ge V_{peak} \quad \text{then} \quad
        \begin{cases}
        V \gets V_r, \\
        w \gets w + b,
        \end{cases}

    so

    .. math::

        \begin{cases}
        \text{spiked}(t, [V, w]) = V - V_{peak}, \\
        \text{reset}(t, [V, w]) = (V_r, w + b)
        \end{cases}

    for certain coefficients :math:`V_{peak}, V_r` and :math:`b`. These can
    be very general conditions. The methods implemented here take into account
    the inherent discontinuity of the solutions to this type of equation.
    """

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()

            from pycaputo.derivatives import CaputoDerivative

            if not all(isinstance(d, CaputoDerivative) for d in self.ds):
                raise TypeError(f"Expected 'CaputoDerivative' operators: {self.ds}")

            if any(d.alpha > 1.0 or d.alpha < 0.0 for d in self.ds):
                raise ValueError(f"Expected 'alpha' in (0, 1): {self.derivative_order}")

    @cached_property
    def derivative_order(self) -> tuple[float, ...]:
        return tuple([d.alpha for d in self.ds])

    @cached_property
    def alpha(self) -> Array:
        return np.array([d.alpha for d in self.ds])

    @property
    def order(self) -> float:
        return 1.0

    def solve(self, t: float, y0: Array, c: Array, r: Array) -> Array:
        """Wrapper around :func:`pycaputo.implicit.solve` to solve the implicit
        equation.

        This function should be overwritten for specific applications if better
        solvers are known. For example, many problems can be solved explicitly
        or approximated to a very good degree to provide a better *y0*.
        """
        raise NotImplementedError(type(self).__name__)

    def make_default_history(self) -> ProductIntegrationHistory:
        # NOTE: all IntegrateFireMethod classes store a `(yn^-, yn^+)` for the
        # values to the left and right of a discontinuity (i.e. spike).
        shape = (2 * self.y0[0].size,)
        dtype = self.y0[0].dtype
        return ProductIntegrationHistory.empty(shape=shape, dtype=dtype)


@make_initial_condition.register(IntegrateFireMethod)
def _make_initial_condition_caputo(  # type: ignore[misc]
    m: IntegrateFireMethod[IntegrateFireModelT],
) -> Array:
    return m.y0[0]


@evolve.register(IntegrateFireMethod)
def _evolve_caputo_integrate_fire_l1(  # type: ignore[misc]
    m: IntegrateFireMethod[IntegrateFireModelT],
    *,
    history: History[Any] | None = None,
    dtinit: float | None = None,
) -> Iterator[events.Event]:
    from pycaputo.controller import (
        StepEstimateError,
        estimate_initial_time_step,
        evaluate_error_estimate,
        evaluate_timestep_accept,
        evaluate_timestep_factor,
        evaluate_timestep_reject,
    )

    if history is None:
        history = m.make_default_history()

    # initialize
    c = m.control
    n = 0
    t = c.tstart

    # determine initial condition
    yprev = make_initial_condition(m)
    history.append(t, np.hstack([yprev, yprev]))

    # determine initial time step
    if dtinit is None:
        dtinit = estimate_initial_time_step(
            t, yprev, m.source, m.smallest_derivative_order, trunc=m.order + 1
        )

    dt = dtinit
    yield StepAccepted(
        t=t,
        iteration=n,
        dt=dt,
        y=yprev,
        eest=0.0,
        q=1.0,
        trunc=np.zeros_like(yprev),
        spiked=False,
    )

    while not c.finished(n, t):
        # evolve solution with current estimate of the time step
        result = advance(m, history, yprev, dt)

        assert isinstance(result, AdvanceResult)
        ynext, trunc, storage = result[:3]

        # determine the next time step
        if result.spiked:
            # NOTE: a spike occurred, so we reset everything
            dtnext = dtinit
            accepted = True
            eest = 0.0
            q = 1.0
        else:
            try:
                # estimate error
                eest = evaluate_error_estimate(c, m, trunc, ynext, yprev)
                accepted = eest <= 1.0

                # estimate time step factor using the error
                q = evaluate_timestep_factor(c, m, eest)

                # finally estimate a good dt for the next step
                tmp_state = {"t": t, "n": n, "y": yprev}
                if accepted:
                    dtnext = evaluate_timestep_accept(c, m, q, dt, tmp_state)
                else:
                    dtnext = evaluate_timestep_reject(c, m, q, dt, tmp_state)
            except StepEstimateError as exc:
                accepted = False
                dtnext = 1.0e-6
                log.error("Failed to estimate timestep.", exc_info=exc)
                yield StepFailed(t=t, iteration=n, reason="Failed to estimate timestep")

        assert dtnext >= 0.0

        # update variables
        assert result.dts >= 0
        if eest <= 1.0:
            dt = float(result.dts)

            n += 1
            t += dt
            yprev = ynext
            history.append(t, storage)

            if result.spiked:
                # NOTE: if we spiked, also yield the value before the
                # discontinuity so that the plots look nice!
                yield StepAccepted(
                    t=t,
                    iteration=n,
                    dt=dt,
                    y=storage[: ynext.size],
                    eest=eest,
                    q=q,
                    trunc=trunc,
                    spiked=bool(result.spiked),
                )

                # NOTE: we need to increment the iteration, but not the time
                # because both values in the spike happen at the same time due
                # to the discontinuity
                n += 1

            yield StepAccepted(
                t=t,
                iteration=n,
                dt=dt,
                y=ynext,
                eest=eest,
                q=q,
                trunc=trunc,
                # NOTE: if we spiked on this step, the previous value will
                # record it so we don't need to do it twice.
                spiked=False,
            )
        else:
            yield StepRejected(
                t=t, iteration=n, dt=dt, y=ynext, eest=eest, q=q, trunc=trunc
            )

        dt = dtnext


def advance_caputo_integrate_fire_l1(
    m: IntegrateFireMethod[IntegrateFireModelT],
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> tuple[AdvanceResult, Array]:
    # set next time step
    d = y.size
    n = len(history)
    t = history.ts[n] = history.ts[n - 1] + dt

    # get time history
    ts = history.ts[n] - history.ts[: n + 1]

    # compute convolution coefficients
    alpha = m.alpha.reshape(-1, 1)
    g2m = gamma(2 - alpha)

    omega = (ts[:-1] ** (1 - alpha) - ts[1:] ** (1 - alpha)) / g2m
    h = (omega / np.diff(history.ts[: n + 1])).T
    assert h.shape == (n, d)

    # NOTE: we store (t_k, [y^-_k, y^+_k]) at every time step and want to compute
    #   h_{n + 1, k} (y^-_{k + 1} - y^+_k)
    dy: Array = history.storage[1 : n + 1, :d] - history.storage[:n, d:]
    r: Array = np.einsum("ij,ij->j", h[:-1], dy[:-1])

    from pycaputo.fode.caputo import _truncation_error

    ynext = m.solve(t, y, 1 / h[-1], y - r / h[-1])
    trunc = _truncation_error(m.control, m.alpha, t, ynext, t - dt, y)

    result = AdvanceResult(
        ynext,
        trunc,
        np.hstack([ynext, ynext]),
        spiked=np.array(0),
        dts=np.array(dt),
    )

    return result, r


# }}}


# {{{ advance helpers


def advance_caputo_integrate_fire_spike_linear(
    t: float,
    y: Array,
    history: ProductIntegrationHistory,
    *,
    v_peak: float,
    v_reset: float,
) -> AdvanceResult:
    from pycaputo.integrate_fire.spikes import estimate_spike_time_linear

    prev = history[-1]

    # we have spiked, so we need to reconstruct our solution
    yprev = np.array([v_peak], dtype=y.dtype)
    ynext = np.array([v_reset], dtype=y.dtype)

    # estimate the time step used to get here
    ts = estimate_spike_time_linear(t, y, prev.t, prev.f[0], v_peak)
    dt = float(ts - prev.t)

    # set the truncation error to zero because we want to reset the next time
    # step to the maximum allowable value.
    trunc = np.zeros_like(y)

    return AdvanceResult(
        ynext, trunc, np.hstack([yprev, ynext]), spiked=np.array(1), dts=np.array(dt)
    )


def advance_caputo_integrate_fire_spike_quadratic(
    t: float,
    y: Array,
    history: ProductIntegrationHistory,
    *,
    v_peak: float,
    v_reset: float,
) -> AdvanceResult:
    from pycaputo.integrate_fire.spikes import estimate_spike_time_quadratic

    prev = history[-1]
    pprev = history[-2]

    # we have spiked, so we need to reconstruct our solution
    yprev = np.array([v_peak], dtype=y.dtype)
    ynext = np.array([v_reset], dtype=y.dtype)

    # estimate the time step used to get here
    ts = estimate_spike_time_quadratic(
        t, y, prev.t, prev.f[0], pprev.t, pprev.f[0], v_peak
    )
    dt = float(ts - prev.t)

    # set the truncation error to zero because we want to reset the next time
    # step to the maximum allowable value.
    trunc = np.zeros_like(y)

    return AdvanceResult(
        ynext, trunc, np.hstack([yprev, ynext]), spiked=np.array(1), dts=np.array(dt)
    )


# }}}
