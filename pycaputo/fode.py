# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import singledispatch
from typing import Iterator

import numpy as np

from pycaputo.derivatives import FractionalOperator
from pycaputo.logging import get_logger
from pycaputo.utils import Array, ScalarStateFunction, StateFunction

logger = get_logger(__name__)


# {{{ interface


@dataclass(frozen=True)
class StepResult:
    """Result of advancing a time step."""


@dataclass(frozen=True)
class StepFailed(StepResult):
    """Result of a failed update to time :attr:`t`."""

    #: Current time.
    t: float
    #: Current iteration.
    iteration: int


@dataclass(frozen=True)
class StepCompleted(StepResult):
    """Result of a successful update to time :attr:`t`."""

    #: Current time.
    t: float
    #: Current iteration.
    iteration: int
    #: Final time of the simulation.
    dt: float
    #: State at the time :attr:`t`.
    y: Array

    def __str__(self) -> str:
        return f"[{self.iteration:5d}] t = {self.t:.5e} dt {self.dt:.5e}"


@dataclass(frozen=True)
class History:
    """A class handling the history checkpointing for the equation."""

    #: History of state variables, required to compute the memory term.
    yhistory: list[Array] = field(default_factory=list, init=False, repr=False)
    #: History of time instantes corresponding to :attr:`yhistory`.
    thistory: list[float] = field(default_factory=list, init=False, repr=False)

    @property
    def nhistory(self) -> int:
        return len(self.yhistory)

    def dump(self, t: float, y: Array) -> None:
        """Save the solution *y* at time *t*.

        :arg t: time at which the checkpoint is taken.
        :arg y: solution at the time *t*.
        """
        self.yhistory.append(y)
        self.thistory.append(t)

    def load(self, k: int) -> tuple[float, Array]:
        """Load solution from the *k*-th time step.

        :returns: a tuple of ``(t, y)`` mirroring the state from :meth:`dump`.
        """
        if k == -1:
            k = self.nhistory - 1

        if not 0 <= k < self.nhistory:
            raise IndexError(f"history index out of range: {k} >= {self.nhistory}")

        return self.thistory[k], self.yhistory[k]


@dataclass(frozen=True)
class FractionalDifferentialEquationMethod(ABC):
    r"""A generic method used to solve fractional ordinary differential
    equations (FODE).

    The simplest such example is a scalar single-term equation

    .. math::

        \begin{cases}
        D^\alpha_C[y](t) = f(t, y), & \quad t \in [0, T], \\
        y(0) = y_0.
        \end{cases}

    where the Caputo derivative :math:`D_C^\alpha` is used with
    :math:`\alpha \in (0, 1)`. In this case, only :math:`y(0)` is required
    as an initial condition. Different derivatives or higher-order derivatives
    will required additional initial data.
    """

    #: The fractional operator used for the derivative.
    d: FractionalOperator
    #: A callable used to predict the time step.
    predict_time_step: ScalarStateFunction

    #: Right-hand side source term.
    source: StateFunction
    #: The initial and final times of the simulation.
    tspan: tuple[float, float | None]
    #: Values used to reconstruct the required initial conditions.
    y0: tuple[Array, ...]

    @property
    def name(self) -> str:
        """An identifier for the method."""
        return type(self).__name__.replace("Method", "")

    @property
    @abstractmethod
    def order(self) -> float:
        """Expected order of convergence of the method."""


@singledispatch
def make_initial_condition(
    m: FractionalDifferentialEquationMethod,
    t: float,
    y0: tuple[Array, ...],
) -> Array:
    raise NotImplementedError(f"initial condition for '{type(m).__name__}'")


@singledispatch
def evolve(
    m: FractionalDifferentialEquationMethod,
    *,
    history: History | None = None,
    maxit: int | None = None,
    verbose: bool = True,
) -> Iterator[StepResult]:
    """Evolve the fractional-order ordinary differential equation in time.

    :arg m: method used to evolve the FODE.
    :arg f: right-hand side of the FODE.
    :arg y0: initial conditions for the FODE.
    :returns: a :class:`StepResult` (usually a :class:`StepCompleted`) containing
        the solution at a time :math:`t`.
    """

    if history is None:
        history = History()

    n = 0
    t, tfinal = m.tspan
    y = make_initial_condition(m, t, m.y0)

    history.dump(t, y)
    yield StepCompleted(t=t, iteration=n, dt=0.0, y=y)

    while True:
        if tfinal is not None and t >= tfinal:
            break

        if maxit is not None and n >= maxit:
            break

        try:
            dt = m.predict_time_step(t, y)
            if not np.isfinite(dt):
                raise ValueError(f"Invalid time step at iteration {n}: {dt!r}")

            if tfinal is not None:
                # NOTE: adding 1.0e-15 to ensure that t >= tfinal is true
                dt = min(dt, tfinal - t) + 1.0e-15

            y = advance(m, history, t, y, dt)
            n += 1
            t += dt

            history.dump(t, y)
            yield StepCompleted(t=t, iteration=n, dt=dt, y=y)
        except Exception as exc:
            logger.error("Step failed.", exc_info=exc)
            yield StepFailed(t=t, iteration=n)


@singledispatch
def advance(
    m: FractionalDifferentialEquationMethod,
    history: History,
    t: float,
    y: Array,
    dt: float,
) -> Array:
    """Advance the solution *y* by *dt* from the current time *t*."""
    raise NotImplementedError(f"'advance' functionality for '{type(m).__name__}'")


# }}}


# {{{ Caputo


@dataclass(frozen=True)
class CaputoDifferentialEquationMethod(FractionalDifferentialEquationMethod):
    r"""A generic method used to solve fractional ordinary differential
    equations (FODE) with the Caputo derivative.
    """


@make_initial_condition.register(CaputoDifferentialEquationMethod)
def _make_initial_condition_caputo(
    m: CaputoDifferentialEquationMethod,
    t: float,
    y0: tuple[Array, ...],
) -> Array:
    return y0[0]


# {{{ forward Euler


@dataclass(frozen=True)
class CaputoForwardEulerMethod(CaputoDifferentialEquationMethod):
    """The first-order forward Euler discretization of the Caputo derivative."""

    @property
    def order(self) -> float:
        return 1.0


@advance.register(CaputoForwardEulerMethod)
def _advance_caputo_forward_euler(
    m: CaputoForwardEulerMethod,
    history: History,
    t: float,
    y: Array,
    dt: float,
) -> Array:
    from math import gamma

    n = history.nhistory
    alpha = m.d.order
    t = t + dt

    # FIXME: this is intensely inefficient
    ynext = sum(
        [(t - m.tspan[0]) ** k / gamma(k + 1) * y0k for k, y0k in enumerate(m.y0)],
        np.zeros_like(y),
    )
    ts = [*history.thistory, t]

    for k in range(n):
        _, yk = history.load(k)
        omega = ((t - ts[k]) ** alpha - (t - ts[k + 1]) ** alpha) / gamma(1 + alpha)
        ynext += omega * m.source(ts[k], yk)

    return ynext


# }}}


# {{{ Crank-Nicolson


@dataclass(frozen=True)
class CaputoCrankNicolsonMethod(CaputoDifferentialEquationMethod):
    r"""The Crank-Nicolson discretization of the Caputo derivative.

    The Crank-Nicolson method is a convex combination of the forward Euler
    and the backward Euler method. This implementation uses a parameter
    :attr:`theta` to interpolate between the two.

    Note that for :math:`\theta = 0` we get the forward Euler method, which
    is first order, for :math:`\theta = 1` we get the backward Euler method,
    which is first order, and for :math:`\theta = 1/2` we get the Crank-Nicolson
    method, which is order :math:`1 + \alpha`. This method only becomes second
    order in the limit of :math:`\alpha \to 1`.
    """

    #: Parameter weight between the forward and backward Euler methods. The value
    #: of :math:`\theta = 1/2` gives the standard Crank-Nicolson method.
    theta: float

    source_jac: StateFunction | None

    if __debug__:

        def __post_init__(self) -> None:
            if not 0.0 <= self.theta <= 1.0:
                raise ValueError(
                    f"'theta' parameter must be in [0, 1]: got {self.theta}"
                )

    @property
    def order(self) -> float:
        return (1.0 + self.d.order) if self.theta == 0.5 else 1.0

    def solve(self, t: float, y0: Array, c: float, r: Array) -> Array:
        """Solves an implicit update formula.

        This function will solve an equation of the form

        .. math::

            y - c * f(t, y) = r

        for the solution :math:`y`. This is specific to first-order FODEs.

        :arg t: time at which the solution *y* is evaluated.
        :arg y: unknown solution at time *t*.
        :arg c: constant for the source term *f* that corresponds to
            :attr:`FractionalDifferentialEquationMethod.source`.
        :arg r: right-hand side term.

        :returns: solution :math:`y^*` of the above root finding problem.
        """

        def func(y: Array) -> Array:
            return np.array(y - c * self.source(t, y) - r)

        def jac(y: Array) -> Array:
            assert self.source_jac is not None
            return np.array(1 - c * self.source_jac(t, y))

        import scipy.optimize as so

        result = so.root(
            func,
            y0,
            jac=jac if self.source_jac is not None else None,
            method="lm",
            options={"ftol": 1.0e-10},
        )

        return np.array(result.x)


@advance.register(CaputoCrankNicolsonMethod)
def _advance_caputo_crank_nicolson(
    m: CaputoCrankNicolsonMethod,
    history: History,
    t: float,
    y: Array,
    dt: float,
) -> Array:
    from math import gamma

    n = history.nhistory
    alpha = m.d.order
    t = t + dt
    ts = [*history.thistory, t]

    # FIXME: this is intensely inefficient
    fnext = sum(
        [(t - ts[0]) ** k / gamma(k + 1) * y0k for k, y0k in enumerate(m.y0)],
        np.zeros_like(y),
    )

    for k in range(n - 1):
        omega = ((t - ts[k]) ** alpha - (t - ts[k + 1]) ** alpha) / gamma(1 + alpha)

        # add forward term
        _, yk = history.load(k)
        if m.theta != 0.0:
            fnext += omega * m.theta * m.source(ts[k], yk)

        # add backward term
        if m.theta != 1.0:
            _, yk = history.load(k + 1)
            fnext += omega * (1 - m.theta) * m.source(ts[k + 1], yk)

    k = n - 1
    omega = ((t - ts[k]) ** alpha - (t - ts[k + 1]) ** alpha) / gamma(1 + alpha)

    if m.theta != 0.0:
        # add last forward
        _, yk = history.load(k)
        fnext += omega * m.theta * m.source(ts[k], yk)

    if m.theta != 1.0:
        ynext = m.solve(ts[-1], y, omega * (1 - m.theta), fnext)
    else:
        ynext = fnext

    return ynext


# }}}

# }}}
