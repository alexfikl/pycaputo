# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from pycaputo.derivatives import FractionalOperatorT
from pycaputo.events import Event
from pycaputo.history import History, ProductIntegrationHistory
from pycaputo.logging import get_logger
from pycaputo.stepping import (
    FractionalDifferentialEquationMethod,
    advance,
    evolve,
    make_initial_condition,
)
from pycaputo.typing import Array, StateFunctionT

log = get_logger(__name__)


class AdvanceResult(NamedTuple):
    """Result of :func:`~pycaputo.stepping.advance` for
    :class:`ProductIntegrationMethod` subclasses."""

    y: Array
    """Estimated solution at the next time step."""
    trunc: Array
    """Estimated truncation error at the next time step."""
    storage: Array
    """Array to add to the history storage."""


@dataclass(frozen=True)
class ProductIntegrationMethod(
    FractionalDifferentialEquationMethod[FractionalOperatorT, StateFunctionT]
):
    """A generic class of methods based on Product Integration.

    In general, these methods support a variable time step.
    """

    def make_default_history(self) -> ProductIntegrationHistory:
        nsteps = self.control.nsteps
        return ProductIntegrationHistory.empty_like(
            # NOTE: All product integration rules just store the right-hand side
            # `f`, which are the same size and dtype as `y0`
            self.y0[0],
            n=512 if nsteps is None else nsteps,
        )


@evolve.register(ProductIntegrationMethod)
def _evolve_pi(
    m: FractionalDifferentialEquationMethod[FractionalOperatorT, StateFunctionT],
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
        history = m.make_default_history()

    # initialize
    c = m.control
    n = 0
    t = c.tstart

    # determine the initial condition
    yprev = make_initial_condition(m)

    # store the initial right-hand side
    fprev = m.source(t, yprev)
    storage = history.stateinfo.zeros()
    storage[tuple(slice(n) for n in fprev.shape)] = fprev
    history.append(t, storage)

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
            log.error("Failed to estimate timestep.", exc_info=exc)
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
