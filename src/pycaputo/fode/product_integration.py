# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import cached_property
from typing import Any, Iterator

import numpy as np

from pycaputo.derivatives import CaputoDerivative, Side
from pycaputo.fode.base import (
    Event,
    FractionalDifferentialEquationMethod,
    evolve,
    make_initial_condition,
)
from pycaputo.history import History
from pycaputo.logging import get_logger
from pycaputo.utils import Array

logger = get_logger(__name__)

# {{{ ProductIntegrationMethod


@dataclass(frozen=True)
class AdvanceResult:
    """Result of :func:`advance`."""

    __slots__ = ("y", "trunc", "storage")

    #: Estimated solution at the next time step.
    y: Array
    #: Estimated truncation error at the next time step.
    trunc: Array
    #: Array to add to the history storage.
    storage: Array

    def __iter__(self) -> Iterator[Array]:
        for f in fields(self):
            yield getattr(self, f.name)


@dataclass(frozen=True)
class ProductIntegrationMethod(FractionalDifferentialEquationMethod):
    """A generic class of methods based on Product Integration.

    In general, these methods support a variable time step.
    """


@evolve.register(ProductIntegrationMethod)
def _evolve_pi(
    m: FractionalDifferentialEquationMethod,
    *,
    history: History[Any] | None = None,
    dt: float | None = None,
) -> Iterator[Event]:
    if history is None:
        from pycaputo.history import ProductIntegrationHistory

        history = ProductIntegrationHistory.empty_like(m.y0[0])

    from pycaputo.controller import (
        StepEstimateError,
        estimate_initial_time_step,
        evaluate_error_estimate,
        evaluate_timestep_accept,
        evaluate_timestep_factor,
        evaluate_timestep_reject,
    )
    from pycaputo.fode.base import (
        StepAccepted,
        StepFailed,
        StepRejected,
        advance,
    )

    # initialize
    c = m.control
    n = 0
    t = c.tstart

    # determine the initial condition
    yprev = make_initial_condition(m)
    history.append(t, m.source(t, yprev))

    # determine initial time step
    if dt is None:
        dt = estimate_initial_time_step(
            t, yprev, m.source, m.smallest_derivative_order, trunc=m.order + 1
        )

    yield StepAccepted(
        t=t,
        iteration=n,
        dt=dt,
        y=yprev,
        eest=0.0,
        q=1.0,
        trunc=np.zeros_like(yprev),
    )

    while not c.finished(n, t):
        # evolve solution with current estimate of the time step
        ynext, trunc, storage = advance(m, history, yprev, dt)

        # determine the next time step
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
            logger.error("Failed to estimate timestep.", exc_info=exc)
            yield StepFailed(t=t, iteration=n, reason="Failed to estimate timestep")
            continue

        # yield the solution if we got this far
        if accepted:
            n += 1
            t += dt
            yprev = ynext
            history.append(t, storage)

            yield StepAccepted(
                t=t, iteration=n, dt=dt, y=ynext, eest=eest, q=q, trunc=trunc
            )
        else:
            yield StepRejected(
                t=t, iteration=n, dt=dt, y=ynext, eest=eest, q=q, trunc=trunc
            )

        dt = dtnext


# }}}


# {{{ CaputoProductIntegrationMethod


@dataclass(frozen=True)
class CaputoProductIntegrationMethod(ProductIntegrationMethod):
    @cached_property
    def d(self) -> tuple[CaputoDerivative, ...]:
        return tuple([
            CaputoDerivative(order=alpha, side=Side.Left)
            for alpha in self.derivative_order
        ])


@make_initial_condition.register(CaputoProductIntegrationMethod)
def _make_initial_condition_caputo(m: CaputoProductIntegrationMethod) -> Array:
    return m.y0[0]


# }}}