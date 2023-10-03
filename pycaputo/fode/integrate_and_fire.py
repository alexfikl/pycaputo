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
    advance,
    evolve,
    make_initial_condition,
)
from pycaputo.fode.history import History, VariableProductIntegrationHistory
from pycaputo.logging import get_logger
from pycaputo.utils import Array, CallbackFunction, ScalarStateFunction, StateFunction

logger = get_logger(__name__)


@dataclass(frozen=True)
class CaputoIntegrateFireL1Method(FractionalDifferentialEquationMethod):
    r"""An implicit discretization of the Caputo derivative for leaky
    Integrate-and-Fire models based on the L1 method.

    A generic leaky Integrate-and-Fire model is given by

    .. math::

        D_C^\alpha[\mathbf{y}](t) = \mathbf{f}(t, \mathbf{y})

    together with a reset condition of the form

    .. math::

        \text{condition}(t, \mathbf{y}) \ge 0
        \quad \text{then} \quad
        \mathbf{y} = \text{reset}(t, \mathbf{y}),

    where the ``condition`` function and the ``reset`` functions are provided
    by the user. For example, if we assume that our model has two variables
    :math:`(V, w)` and we want to model the reset condition

    .. math::

        V \ge V_{peak} \quad \text{then} \quad
        \begin{cases}
        V \gets V_r, \\
        w \gets w + b,
        \end{cases}

    we can set

    .. math::

        \begin{cases}
        \text{condition}(t, [V, w]) = V - V_{peak}, \\
        \text{reset}(t, [V, w]) = (V_r, w + b)
        \end{cases}

    for certain coefficients :math:`V_{peak}, V_r` and :math:`b`. These can
    be very general conditions. The variant of the L1 method implemented here
    takes into account the inherent discontinuity of the solutions to this
    type of equation (see :class:`~pycaputo.differentiation.CaputoL1Method`
    for a smooth implementation).
    """

    #: Jacobian of :attr:`~pycaputo.fode.FractionalDifferentialEquationMethod.source`.
    #: By default, this method uses :mod:`scipy` for its root finding requirements,
    #: which defines the Jacobian as :math:`J_{ij} = \partial f_i / \partial y_j`.
    source_jac: StateFunction | None

    #: Callable to check the reset condition.
    condition: ScalarStateFunction
    #: Callable to reset the state variables when the reset condition is hit.
    reset: StateFunction

    @cached_property
    def d(self) -> tuple[CaputoDerivative, ...]:
        return tuple(
            [
                CaputoDerivative(order=alpha, side=Side.Left)
                for alpha in self.derivative_order
            ]
        )

    @property
    def order(self) -> float:
        # FIXME: this is the same as the standard L1 method? Unlikely..
        alpha = min(self.derivative_order)
        return 2.0 - alpha

    # NOTE: `_get_kwargs` is meant to be overwritten for testing purposes or
    # some specific application (undocumented for now).

    def _get_kwargs(self) -> dict[str, object]:
        """
        :returns: additional keyword arguments for :func:`scipy.optimize.root`.
        """
        # NOTE: the default hybr does not use derivatives, so use lm instead
        # FIXME: will need to maybe benchmark these a bit?
        return {"method": "lm" if self.source_jac else None}

    def solve(self, t: float, y0: Array, c: Array, r: Array) -> Array:
        """Wrapper around :func:`pycaputo.fode.solve` to solve the implicit equation.

        This function should be overwritten for specific applications if better
        solvers are known. For example, many problems can be solved explicitly
        or approximated to a very good degree to provide a better *y0*.
        """
        from pycaputo.fode.base import solve

        return solve(
            self.source,
            self.source_jac,
            t,
            y0,
            c,
            r,
            **self._get_kwargs(),
        )


@evolve.register(CaputoIntegrateFireL1Method)
def _evolve_caputo_integrate_fire_l1(
    m: CaputoIntegrateFireL1Method,
    *,
    callback: CallbackFunction | None = None,
    history: History | None = None,
    raise_on_fail: bool = False,
) -> Iterator[Event]:
    from pycaputo.fode.base import StepCompleted, StepEstimateError, StepFailed, advance

    n = 0
    t = m.tspan.tstart
    y = make_initial_condition(m)

    if history is None:
        history = VariableProductIntegrationHistory()

    # NOTE: called to update the history
    y = advance(m, history, t, y)

    yield StepCompleted(t=t, iteration=n, dt=0.0, y=y)

    while True:
        if callback is not None and callback(t, y):
            break

        if m.tspan.finished(n, t):
            break

        # next time step
        try:
            dt = m.tspan.get_next_time_step(n, t, y)
        except StepEstimateError as exc:
            logger.error("Failed to predict time step.", exc_info=exc)
            if raise_on_fail:
                raise exc

            yield StepFailed(t=t, iteration=n, reason=str(exc))

        # next iteration
        n += 1
        t += dt

        # advance
        try:
            y = advance(m, history, t, y)
        except Exception as exc:
            logger.error("Failed to advance solution.", exc_info=exc)
            if raise_on_fail:
                raise exc

            yield StepFailed(t=t, iteration=n, reason=str(exc))

        if not np.all(np.isfinite(y)):
            logger.error("Failed to update solution: %s", y)
            if raise_on_fail:
                raise ValueError(f"Predicted solution is not finite: {y}")

            yield StepFailed(t=t, iteration=n, reason="solution is not finite")
        else:
            yield StepCompleted(t=t, iteration=n, dt=dt, y=y)


@advance.register(CaputoIntegrateFireL1Method)
def _advance_caputo_forward_euler(
    m: CaputoIntegrateFireL1Method,
    history: VariableProductIntegrationHistory,
    t: float,
    y: Array,
) -> Array:
    from math import gamma

    history.ts.append(t)
    if not history:
        history.append(t, m.source(t, y))
        return y

    n = len(history)
    ts = history.ts[-1] - np.array(history.ts[: n + 1])

    alpha = np.array(m.derivative_order).reshape(-1, 1)
    gamma2 = np.array([gamma(2 - a) for a in m.derivative_order])
    omega = (ts[:-1] ** (1 - alpha) - ts[1:] ** (1 - alpha)) / gamma2
    c = (history.ts[-1] - history.ts[-2]) / omega[:, -1]

    from itertools import pairwise

    dy = sum(
        w * (yn.f - yp.f) / (yn.t - yp.t)
        for w, (yn, yp) in zip(omega, pairwise(history.history))
    )

    ynext = m.solve(t, y, c, y - dy)
    if m.condition(t, ynext) > 0:
        ynext = m.reset(t, ynext)
        history.append(t, m.source(t, ynext))
        history.append(t, m.source(t, ynext))
    else:
        history.append(t, m.source(t, ynext))

    return ynext
