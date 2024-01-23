# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, NamedTuple

import numpy as np

from pycaputo.events import Event
from pycaputo.history import History, ProductIntegrationHistory
from pycaputo.logging import get_logger
from pycaputo.stepping import (
    FractionalDifferentialEquationMethod,
    advance,
    evolve,
    make_initial_condition,
)
from pycaputo.utils import Array, StateFunctionT

logger = get_logger(__name__)


class AdvanceResult(NamedTuple):
    """Result of :func:`~pycaputo.stepping.advance` for
    :class:`ProductIntegrationMethod` subclasses."""

    #: Estimated solution at the next time step.
    y: Array
    #: Estimated truncation error at the next time step.
    trunc: Array
    #: Array to add to the history storage.
    storage: Array


@dataclass(frozen=True)
class ProductIntegrationMethod(FractionalDifferentialEquationMethod[StateFunctionT]):
    """A generic class of methods based on Product Integration.

    In general, these methods support a variable time step.
    """


@evolve.register(ProductIntegrationMethod)
def _evolve_pi(
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    *,
    history: History[Any] | None = None,
    dtinit: float | None = None,
) -> Iterator[Event]:
    from pycaputo.controller import (
        StepEstimateError,
        estimate_initial_time_step,
        evaluate_error_estimate,
        evaluate_timestep_accept,
        evaluate_timestep_factor,
        evaluate_timestep_reject,
    )
    from pycaputo.events import (
        StepAccepted,
        StepFailed,
        StepRejected,
    )

    if history is None:
        history = ProductIntegrationHistory.empty_like(m.y0[0])

    # initialize
    c = m.control
    n = 0
    t = c.tstart

    # determine the initial condition
    yprev = make_initial_condition(m)
    history.append(t, m.source(t, yprev))

    # determine initial time step
    if dtinit is None:
        dt = estimate_initial_time_step(
            t, yprev, m.source, m.smallest_derivative_order, trunc=m.order + 1
        )
    else:
        dt = dtinit

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
