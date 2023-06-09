# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property, singledispatch
from typing import Iterator

from pycaputo.derivatives import FractionalOperator
from pycaputo.fode.history import History
from pycaputo.logging import get_logger
from pycaputo.utils import Array, CallbackFunction, ScalarStateFunction, StateFunction

logger = get_logger(__name__)


def make_predict_time_step_fixed(dt: float) -> ScalarStateFunction:
    """
    :returns: a callable returning a fixed time step *dt*.
    """

    if dt < 0:
        raise ValueError(f"Time step should be positive: {dt}")

    def predict_time_step(t: float, y: Array) -> float:
        return dt

    return predict_time_step


def make_predict_time_step_graded(
    tspan: tuple[float, float], maxit: int, r: int = 2
) -> ScalarStateFunction:
    r"""Construct a time step that is smaller around the initial time.

    This graded grid of time steps is described in [Garrappa2015b]_. It
    essentially takes the form

    .. math::

        \Delta t_n = \frac{T - t_0}{N^r} ((n + 1)^r - n^r),

    where the time interval is :math:`[t_0, T]` and :math:`N` time steps are
    taken. This graded grid can give full second-order convergence for certain
    methods such as the Predictor-Corrector method (e.g. implemented by
    :class:`~pycaputo.fode.CaputoPECEMethod`).

    :arg tspan: time interval :math:`[t_0, T]`.
    :arg maxit: maximum number of iterations to take in the interval.
    :arg r: a grading exponent that controls the clustering of points at :math:`t_0`.

    :returns: a callable returning a graded time step.
    """
    if maxit <= 0:
        raise ValueError(f"Negative number of iterations not allowed: {maxit}")

    if r < 1:
        raise ValueError(f"Gradient exponent must be >= 1: {r}")

    if tspan[0] > tspan[1]:
        raise ValueError(f"Invalid time interval: {tspan}")

    h = (tspan[1] - tspan[0]) / maxit**r

    def predict_time_step(t: float, y: Array) -> float:
        # FIXME: this works, but seems a bit overkill just to get the iteration
        n = round(((t - tspan[0]) / h) ** (1 / r))

        return float(h * ((n + 1) ** r - n**r))

    return predict_time_step


# {{{ events


@dataclass(frozen=True)
class Event:
    """Event after attempting to advance in a time step."""


@dataclass(frozen=True)
class StepFailed(Event):
    """Result of a failed update to time :attr:`t`."""

    #: Current time.
    t: float
    #: Current iteration.
    iteration: int


@dataclass(frozen=True)
class StepCompleted(Event):
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


# }}}


# {{{ interface


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

    #: The fractional derivative order used for the derivative.
    derivative_order: tuple[float, ...]
    #: A callable used to predict the time step.
    predict_time_step: float | ScalarStateFunction

    #: Right-hand side source term.
    source: StateFunction
    #: The initial and final times of the simulation. The final time can be
    #: *None* if evolving the equation to an unknown final time, e.g. by
    #: setting *maxit* in :func:`evolve`.
    tspan: tuple[float, float | None]
    #: Values used to reconstruct the required initial conditions.
    y0: tuple[Array, ...]

    if __debug__:

        def __post_init__(self) -> None:
            if not self.y0:
                raise ValueError("No initial conditions given")

            shape = self.y0[0]
            if not all(y0.shape == shape for y0 in self.y0[1:]):
                raise ValueError("Initial conditions have different shapes")

            from math import ceil

            m = ceil(self.largest_derivative_order)
            if m != len(self.y0):
                raise ValueError(
                    "Incorrect number of initial conditions: "
                    f"got {len(self.y0)}, but expected {m} arrays"
                )

    @cached_property
    def largest_derivative_order(self) -> float:
        return max(self.derivative_order)

    @property
    def is_constant_time_step(self) -> bool:
        """A flag for whether the method uses a constant time step."""
        return not callable(self.predict_time_step)

    @property
    def name(self) -> str:
        """An identifier for the method."""
        return type(self).__name__.replace("Method", "")

    @property
    @abstractmethod
    def order(self) -> float:
        """Expected order of convergence of the method."""

    @property
    @abstractmethod
    def d(self) -> tuple[FractionalOperator, ...]:
        """The fractional operators used by this method."""


@singledispatch
def evolve(
    m: FractionalDifferentialEquationMethod,
    *,
    callback: CallbackFunction | None = None,
    history: History | None = None,
    maxit: int | None = None,
    verbose: bool = True,
) -> Iterator[Event]:
    """Evolve the fractional-order ordinary differential equation in time.

    :arg m: method used to evolve the FODE.
    :arg f: right-hand side of the FODE.
    :arg y0: initial conditions for the FODE.
    :arg history: a :class:`History` instance that handles checkpointing the
        necessary state history for the method *m*.
    :arg maxit: the maximum number of iterations.
    :arg verbose: print additional iteration details.

    :returns: an :class:`Event` (usually a :class:`StepCompleted`) containing
        the solution at a time :math:`t`.
    """
    raise NotImplementedError(f"'evolve' functionality for '{type(m).__name__}'")


@singledispatch
def advance(
    m: FractionalDifferentialEquationMethod,
    history: History,
    t: float,
    y: Array,
) -> Array:
    """Advance the solution *y* by to the time *t*.

    This function takes ``(history[t_0, ... t_n], t_{n + 1}, y_n)`` and is
    expected to update the history with values at :math:`t_{n + 1}`. The time
    steps can be recalculated from the history if necessary.

    :returns: value of :math:`y_{n + 1}` at time :math:`t_{n + 1}`.
    """
    raise NotImplementedError(f"'advance' functionality for '{type(m).__name__}'")


@singledispatch
def make_initial_condition(m: FractionalDifferentialEquationMethod) -> Array:
    """Construct an initial condition for the method *m*."""
    raise NotImplementedError(type(m).__name__)


# }}}
