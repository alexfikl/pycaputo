# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Any, NamedTuple

import numpy as np

from pycaputo.derivatives import CaputoDerivative
from pycaputo.events import Event, StepCompleted
from pycaputo.history import History, ProductIntegrationHistory
from pycaputo.logging import get_logger
from pycaputo.stepping import (
    FractionalDifferentialEquationMethod,
    evolve,
    make_initial_condition,
)
from pycaputo.typing import Array, StateFunctionT

logger = get_logger(__name__)


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
class SplineCollocationMethod(
    FractionalDifferentialEquationMethod[CaputoDerivative, StateFunctionT]
):
    """A spline collocation method for the Caputo fractional derivative."""

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()

            if not all(isinstance(d, CaputoDerivative) for d in self.ds):
                raise TypeError(f"Expected 'CaputoDerivative' operators: {self.ds}")

    @cached_property
    def derivative_order(self) -> tuple[float, ...]:
        return tuple([d.alpha for d in self.ds])

    @cached_property
    def alpha(self) -> Array:
        return np.array([d.alpha for d in self.ds])


@make_initial_condition.register(SplineCollocationMethod)
def _make_initial_condition_caputo_spline_collocation(  # type: ignore[misc]
    m: SplineCollocationMethod[StateFunctionT],
) -> Array:
    return m.y0[0]


@evolve.register(SplineCollocationMethod)
def _evolve_caputo_spline_collocation(  # type: ignore[misc]
    m: SplineCollocationMethod[StateFunctionT],
    *,
    history: History[Any] | None = None,
    dtinit: float | None = None,
) -> Iterator[Event]:
    from pycaputo.controller import estimate_initial_time_step

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

    yield StepCompleted(
        t=t,
        iteration=n,
        dt=dt,
        y=yprev,
        eest=0.0,
        q=1.0,
        trunc=np.zeros_like(yprev),
    )
