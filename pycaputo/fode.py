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
from pycaputo.utils import Array, CallbackFunction, ScalarStateFunction, StateFunction

logger = get_logger(__name__)


# {{{ interface


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


# {{{ history


@dataclass(frozen=True)
class StateHistory:
    """A class the holds the history of the state variables.

    For a simple method, this can be only :math:`(t_n, y_n)` at every time step.
    However, more complex methods can also checkpoint the right-hand side
    evaluations or other intermediary calculations.
    """


@dataclass(frozen=True)
class SourceHistory(StateHistory):
    """A state history that holds the right-hand side evaluations."""

    #: Time of the evaluation
    t: float
    #: Evaluation of the right-hand side.
    f: Array


@dataclass(frozen=True)
class History:
    """A class handling the history checkpointing of an evolution equation.

    This class essentially acts as a :class:`list` where items cannot be removed.
    """

    #: History of state variables, required to compute the memory term.
    history: list[StateHistory] = field(default_factory=list, repr=False)
    #: Time instances of each entry in the :attr:`history`.
    ts: list[float] = field(default_factory=list, repr=False)

    def __bool__(self) -> bool:
        """
        :returns: *False* if the history is empty and *True* otherwise.
        """
        return bool(self.history)

    def __len__(self) -> int:
        """
        :returns: the number of checkpointed solutions.
        """
        return len(self.history)

    def __iter__(self) -> Iterator[StateHistory]:
        return iter(self.history)

    def __getitem__(self, k: int) -> StateHistory:
        """
        :returns: a :class:`StateHistory` from the *k*-th checkpoint.
        """
        nhistory = len(self)
        if k == -1:
            k = nhistory - 1

        if not 0 <= k < nhistory:
            raise IndexError(f"history index out of range: 0 <= {k} < {nhistory}")

        return self.history[k]

    def append(self, value: StateHistory) -> None:
        """Add the *state* to the existing history."""
        self.history.append(value)


# }}}


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

    if history is None:
        history = History()

    n = 0
    t, tfinal = m.tspan
    y = make_initial_condition(m, t, m.y0)

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
            dt = m.predict_time_step(t, y)

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
            yield StepCompleted(t=t, iteration=n, dt=dt, y=y)
        except Exception as exc:
            if verbose:
                logger.error("Failed to advance time step.", exc_info=exc)

            yield StepFailed(t=t, iteration=n)


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


# }}}

# {{{ utils


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
    tspan: tuple[float, float], maxit: int, r: int
) -> ScalarStateFunction:
    r"""Construct a time step that is smaller around the initial time.

    This graded grid of time steps is described in [Garrappa2015b]_. It
    essentially takes the form

    .. math::

        \Delta t_n = \frac{T - t_0}{N^r} ((n + 1)^r - n^r),

    where the time interval is :math:`[t_0, T]` and :math:`N` time steps are
    taken. This graded grid can give full second-order convergence for certain
    methods such as the Predictor-Corrector method (e.g. implemented by
    :class:`CaputoPECEMethod`).

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
) -> Array:
    history.ts.append(t)
    if not history:
        history.append(SourceHistory(t=t, f=m.source(t, y)))
        return y

    from math import gamma

    n = len(history)
    alpha = m.d.order

    # add initial conditions
    ynext = np.zeros_like(y)
    for k, y0k in enumerate(m.y0):
        ynext += (t - m.tspan[0]) ** k / gamma(k + 1) * y0k

    # add history term
    ts = history.ts
    for k in range(n):
        yk = history[k]
        assert isinstance(yk, SourceHistory)

        omega = ((t - ts[k]) ** alpha - (t - ts[k + 1]) ** alpha) / gamma(1 + alpha)
        ynext += omega * yk.f

    history.append(SourceHistory(t=ts[-1], f=m.source(ts[-1], ynext)))
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
) -> Array:
    history.ts.append(t)
    if not history:
        history.append(SourceHistory(t=t, f=m.source(t, y)))
        return y

    from math import gamma

    n = len(history)
    alpha = m.d.order

    # add initial conditions
    fnext = np.zeros_like(y)
    for k, y0k in enumerate(m.y0):
        fnext += (t - m.tspan[0]) ** k / gamma(k + 1) * y0k

    # compute explicit memory term
    ts = history.ts
    for k in range(n - 1):
        omega = ((t - ts[k]) ** alpha - (t - ts[k + 1]) ** alpha) / gamma(1 + alpha)

        # add forward term
        if m.theta != 0.0:
            yk = history[k]
            assert isinstance(yk, SourceHistory)
            fnext += omega * m.theta * yk.f

        # add backward term
        if m.theta != 1.0:
            yk = history[k + 1]
            assert isinstance(yk, SourceHistory)
            fnext += omega * (1 - m.theta) * yk.f

    # add last forward
    k = n - 1
    omega = ((t - ts[k]) ** alpha - (t - ts[k + 1]) ** alpha) / gamma(1 + alpha)

    if m.theta != 0.0:
        yk = history[k]
        assert isinstance(yk, SourceHistory)
        fnext += omega * m.theta * yk.f

    # solve implicit equation
    if m.theta != 1.0:
        ynext = m.solve(ts[-1], y, omega * (1 - m.theta), fnext)
    else:
        ynext = fnext

    history.append(SourceHistory(t=ts[-1], f=m.source(ts[-1], ynext)))
    return ynext


# }}}


# {{{ Predictor-Corector (PEC and PECE)


@dataclass(frozen=True)
class CaputoPredictorCorrectorMethod(CaputoDifferentialEquationMethod):
    r"""The Predictor-Corrector discretization of the Caputo derivative.

    In their classic forms (see e.g. [Diethelm2002]_), these are methods of
    order :math:`1 + \alpha` with good stability properties.

    In general, the corrector step can be repeated multiple times to achieve
    convergence using :attr:`corrector_iterations`. In the limit of
    :math:`k \to \infty`, it is equivalent to a Adams-Moulton method solved by
    fixed point iteration.

    Note that using a high number of corrector iterations is not recommended, as
    the fixed point iteration is not guaranteed to converge, e.g. for very stiff
    problems. In that case it is better to use an implicit method and, e.g.,
    a Newton iteration to solve the root finding problem.
    """

    #: Number of repetitions of the corrector step.
    corrector_iterations: int

    if __debug__:

        def __post_init__(self) -> None:
            if self.corrector_iterations < 1:
                raise ValueError(
                    "More than one corrector iteration is required:"
                    f" {self.corrector_iterations}"
                )

    @property
    def order(self) -> float:
        return 1.0 + self.d.order


@dataclass(frozen=True)
class CaputoPECEMethod(CaputoPredictorCorrectorMethod):
    """The Predict-Evaluate-Correct-Evaluate (PECE) discretization of the
    Caputo Derivative.

    This method is described in [Diethelm2002]_ in its simplest case with a
    single corrector step, which effectively gives the so-called PECE scheme.
    The corrector step can be repeated any number of times to give the
    :math:`PE(CE)^k` methods (see
    :attr:`CaputoPredictorCorrectorMethod.corrector_iterations`).
    """


@dataclass(frozen=True)
class CaputoPECMethod(CaputoPredictorCorrectorMethod):
    """The Predict-Evaluate-Correct (PEC) discretization of the Caputo derivative.

    This is a predictor-corrector similar to :class:`CaputoPECEMethod`, where
    the previous evaluation of the predictor is used to avoid an additional
    right-hand side call. Like the PECE method, the corrector step can be
    repeated multiple times for improved error results.
    """


@advance.register(CaputoPredictorCorrectorMethod)
def _advance_caputo_predictor_corrector(
    m: CaputoPredictorCorrectorMethod,
    history: History,
    t: float,
    y: Array,
) -> Array:
    history.ts.append(t)
    if not history:
        history.append(SourceHistory(t=t, f=m.source(t, y)))
        return y

    from math import gamma

    n = len(history)
    alpha = m.d.order
    ts = history.ts
    gamma1 = gamma(1 + alpha)
    gamma2 = gamma(2 + alpha)

    # add initial conditions
    y0 = np.zeros_like(y)
    for k, y0k in enumerate(m.y0):
        y0 += (t - m.tspan[0]) ** k / gamma(k + 1) * y0k

    # predictor step (forward Euler)
    yp = np.copy(y0)
    omega_e = np.empty(n)
    for k in range(n):
        yk = history[k]
        assert isinstance(yk, SourceHistory)

        omega_e[k] = ((t - ts[k]) ** alpha - (t - ts[k + 1]) ** alpha) / gamma1
        yp += omega_e[k] * yk.f

    # corrector step (Adams-Bashforth 2)
    yexplicit = np.copy(y0)
    for k in range(n - 1):
        yk = history[k]
        assert isinstance(yk, SourceHistory)

        dt = ts[k + 1] - ts[k]
        omega = (
            (t - ts[k + 1]) ** (alpha + 1) / gamma2 / dt
            - (t - ts[k]) ** (alpha + 1) / gamma2 / dt
            + (t - ts[k]) ** alpha / gamma1
        )
        yexplicit += omega * yk.f

        yk = history[k + 1]
        assert isinstance(yk, SourceHistory)

        omega = (
            (t - ts[k]) ** (alpha + 1) / gamma2 / dt
            - (t - ts[k + 1]) ** (alpha + 1) / gamma2 / dt
            - (t - ts[k + 1]) ** alpha / gamma1
        )
        yexplicit += omega * yk.f

    k = n - 1
    yk = history[k]
    assert isinstance(yk, SourceHistory)

    dt = ts[k + 1] - ts[k]
    omega = (
        (t - ts[k + 1]) ** (alpha + 1) / gamma2 / dt
        - (t - ts[k]) ** (alpha + 1) / gamma2 / dt
        + (t - ts[k]) ** alpha / gamma1
    )
    yexplicit += omega * yk.f

    # corrector iterations
    omega = (
        (t - ts[k]) ** (alpha + 1) / gamma2 / dt
        - (t - ts[k + 1]) ** (alpha + 1) / gamma2 / dt
        - (t - ts[k + 1]) ** alpha / gamma1
    )
    for _ in range(m.corrector_iterations):
        fp = m.source(t, yp)
        yp = yexplicit + omega * fp

    ynext = yp
    f = fp if isinstance(m, CaputoPECMethod) else m.source(ts[-1], ynext)
    history.append(SourceHistory(t=ts[-1], f=f))

    return ynext


# }}}

# }}}
