# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterator

import numpy as np

from pycaputo.derivatives import CaputoDerivative, Side
from pycaputo.fode.base import (
    Event,
    FractionalDifferentialEquationMethod,
    evolve,
    make_initial_condition,
)
from pycaputo.fode.history import History
from pycaputo.logging import get_logger
from pycaputo.utils import Array, CallbackFunction

logger = get_logger(__name__)


# {{{ ProductIntegrationMethod


@dataclass(frozen=True)
class ProductIntegrationMethod(FractionalDifferentialEquationMethod):
    """A generic class of methods based on Product Integration.

    In general, these methods support a variable time step.
    """


@evolve.register(ProductIntegrationMethod)
def _evolve_pi(
    m: FractionalDifferentialEquationMethod,
    *,
    callback: CallbackFunction | None = None,
    history: History | None = None,
    maxit: int | None = None,
    verbose: bool = True,
) -> Iterator[Event]:
    from pycaputo.fode.base import (
        StepCompleted,
        StepFailed,
        advance,
        make_predict_time_step_fixed,
    )

    if callable(m.predict_time_step):
        predict_time_step = m.predict_time_step
    else:
        predict_time_step = make_predict_time_step_fixed(m.predict_time_step)

    n = 0
    t, tfinal = m.tspan
    y = make_initial_condition(m)

    if history is None:
        from pycaputo.fode.history import VariableProductIntegrationHistory

        history = VariableProductIntegrationHistory()

    # NOTE: called to update the history
    y = advance(m, history, t, y)

    yield StepCompleted(t=t, iteration=n, dt=0.0, y=y)

    while True:
        if callback is not None and callback(t, y):
            break

        if tfinal is not None and t >= tfinal:
            break

        if maxit is not None and n >= maxit:
            break

        # next iteration
        n += 1

        # next time step
        try:
            dt = predict_time_step(t, y)

            if not np.isfinite(dt):
                raise ValueError(f"Invalid time step at iteration {n}: {dt!r}")
        except Exception as exc:
            if verbose:
                logger.error("Failed to predict time step.", exc_info=exc)

            yield StepFailed(t=t, iteration=n)

        if tfinal is not None:
            # NOTE: adding eps to ensure that t >= tfinal is true
            dt = min(dt, tfinal - t) + float(5 * np.finfo(y.dtype).eps)

        t += dt

        # advance
        try:
            y = advance(m, history, t, y)
            if not np.all(np.isfinite(y)):
                if verbose:
                    logger.error("Failed to update solution: %s", y)

                yield StepFailed(t=t, iteration=n)
            else:
                yield StepCompleted(t=t, iteration=n, dt=dt, y=y)
        except Exception as exc:
            if verbose:
                logger.error("Failed to advance time step.", exc_info=exc)

            yield StepFailed(t=t, iteration=n)


# }}}


# {{{ CaputoProductIntegrationMethod


@dataclass(frozen=True)
class CaputoProductIntegrationMethod(ProductIntegrationMethod):
    @cached_property
    def d(self) -> tuple[CaputoDerivative, ...]:
        return tuple(
            [
                CaputoDerivative(order=alpha, side=Side.Left)
                for alpha in self.derivative_order
            ]
        )


@make_initial_condition.register(CaputoProductIntegrationMethod)
def _make_initial_condition_caputo(m: CaputoProductIntegrationMethod) -> Array:
    return m.y0[0]


# }}}
