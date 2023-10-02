# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property, partial, singledispatch
from typing import Iterator

import numpy as np

from pycaputo.derivatives import FractionalOperator
from pycaputo.fode.history import History
from pycaputo.logging import get_logger
from pycaputo.utils import Array, CallbackFunction, StateFunction

logger = get_logger(__name__)


# {{{ time step


class StepEstimateError(RuntimeError):
    """An exception raised when a time step estimate has failed."""


@dataclass(frozen=True)
class TimeSpan(ABC):
    """A description of a discrete time interval."""

    #: Start of the time interval.
    tstart: float
    #: End of the time interval (leave *None* for infinite time stepping).
    tfinal: float | None
    #: Number of time steps (leave *None* for infinite time stepping).
    nsteps: int | None

    if __debug__:

        def __post_init__(self) -> None:
            if self.tfinal is not None and self.tstart > self.tfinal:
                raise ValueError("Invalid time interval: 'tstart' > 'tfinal'")

            if self.nsteps is not None and self.nsteps <= 0:
                raise ValueError(
                    f"Number of iterations must be positive: {self.nsteps}"
                )

    def finished(self, n: int, t: float) -> bool:
        """Check if the evolution should finish at iteration *n* and time *t*."""
        if self.tfinal is not None and t >= self.tfinal:
            return True

        if self.nsteps is not None and n >= self.nsteps:
            return True

        return False

    @abstractmethod
    def get_next_time_step_raw(self, n: int, t: float, y: Array) -> float:
        """Get a raw estimate of the time step at the given state values.

        This function is meant to be implemented by subclasses, but
        :meth:`~pycaputo.fode.TimeSpan.get_next_time_step` should be used to
        ensure a consistent time stepping.
        """

    def get_next_time_step(self, n: int, t: float, y: Array) -> float:
        r"""Get an estimate of the time step at the given state values.

        :arg n: current iteration.
        :arg t: starting time, i.e. the time step is estimated for the interval
            :math:`[t, t + \Delta t]`.
        :arg y: state value at the time *t*.
        :returns: an estimate for the time step :math:`\Delta t`.
        :raises StepEstimateError: When the next time step cannot be estimated
            or is not finite.
        """
        try:
            dt = self.get_next_time_step_raw(n, t, y)
        except Exception as exc:
            raise StepEstimateError("Failed to get next time step") from exc

        if not np.isfinite(dt):
            raise StepEstimateError(f"Time step is not finite: {dt}")

        # add eps to ensure that t >= tfinal is true
        if self.tfinal is not None:
            eps = float(5.0 * np.finfo(y.dtype).eps)
            dt = min(dt, self.tfinal - t) + eps

        # TODO: Would be nice to have some smoothing over time steps so that
        # they don't vary too much here, but that may be unwanted?

        return dt


@dataclass(frozen=True)
class FixedTimeSpan(TimeSpan):
    """A :class:`TimeSpan` with a fixed time step."""

    #: The fixed time step that should be used.
    dt: float

    def get_next_time_step_raw(self, n: int, t: float, y: Array) -> float:
        """See :meth:`pycaputo.fode.TimeSpan.get_next_time_step_raw`."""
        return self.dt

    @classmethod
    def from_data(
        cls,
        dt: float,
        tstart: float = 0.0,
        tfinal: float | None = None,
        nsteps: int | None = None,
    ) -> FixedTimeSpan:
        """Create a consistent time span with a fixed time step.

        This ensures that the following relation holds:
        ``tfinal = tstart + nsteps * dt`` for all given values. This can be
        achieved by small modifications to either *nsteps* or *dt*.

        :arg dt: desired time step (chosen time step may be slightly smaller).
        :arg tstart: start of the time span.
        :arg tfinal: end of the time span, which is not required.
        :arg nsteps: number of time steps in the span, which is not required.
        """

        if tfinal is not None:
            nsteps = int((tfinal - tstart) / dt) + 1
            dt = (tfinal - tstart) / nsteps
        elif nsteps is not None:
            tfinal = tstart + nsteps * dt
        else:
            # nsteps and tfinal are None, so nothing we can do here
            pass

        return cls(tstart=tstart, tfinal=tfinal, nsteps=nsteps, dt=dt)


@dataclass(frozen=True)
class GradedTimeSpan(TimeSpan):
    r"""A :class:`TimeSpan` with a variable graded time step.

    This graded grid of time steps is described in [Garrappa2015b]_. It
    essentially takes the form

    .. math::

        \Delta t_n = \frac{t_f - t_s}{N^r} ((n + 1)^r - n^r),

    where the time interval is :math:`[t_s, t_f]` and :math:`N` time steps are
    taken. This graded grid can give full second-order convergence for certain
    methods such as the Predictor-Corrector method (e.g. implemented by
    :class:`~pycaputo.fode.CaputoPECEMethod`).
    """

    #: A grading exponent that controls the clustering of points at :math:`t_s`.
    r: int

    if __debug__:

        def __post_init__(self) -> None:
            if self.tfinal is None:
                raise ValueError("'tfinal' must be given for the graded estimate.")

            if self.nsteps is None:
                raise ValueError("'nsteps' must be given for the graded estimate")

            super().__post_init__()
            if self.r < 1:
                raise ValueError(f"Exponent must be >= 1: {self.r}")

    def get_next_time_step_raw(self, n: int, t: float, y: Array) -> float:
        """See :meth:`pycaputo.fode.TimeSpan.get_next_time_step_raw`."""
        assert self.tfinal is not None
        assert self.nsteps is not None

        h = (self.tfinal - self.tstart) / self.nsteps**self.r
        return float(h * ((n + 1) ** self.r - n**self.r))


@dataclass(frozen=True)
class FixedLipschitzTimeSpan(TimeSpan):
    r"""A :class:`TimeSpan` that uses the Lipschitz constant to estimate the time step.

    This method estimates the time step using (Theorem 2.5 from [Baleanu2012]_)

    .. math::

        \Delta t = \frac{1}{(\Gamma(2 - \alpha) L)^{\frac{1}{\alpha}}},

    where :math:`L` is the Lipschitz constant estimated by :attr:`lipschitz_constant`.
    Note that this is essentially a fixed time step estimate.
    """

    #: An estimate of the Lipschitz contants of the right-hand side :math:`f(t, y)`
    #: for all times :math:`t \in [t_s, t_f]`.
    lipschitz_constant: float
    #: Fractional order of the derivative.
    alpha: float

    def get_next_time_step_raw(self, n: int, t: float, y: Array) -> float:
        """See :meth:`pycaputo.fode.TimeSpan.get_next_time_step_raw`."""
        if y.shape != (1,):
            raise ValueError(f"Only scalar functions are supported: {y.shape}")

        from math import gamma

        L = self.lipschitz_constant
        return float((gamma(2 - self.alpha) * L) ** (-1.0 / self.alpha))


@dataclass(frozen=True)
class LipschitzTimeSpan(TimeSpan):
    r"""A :class:`TimeSpan` that uses the Lipschitz constant to estimate the time step.

    This uses the same logic as :class:`FixedLipschitzTimeSpan`, but computes the
    Lipschitz constant using the estimate from
    :func:`~pycaputo.lipschitz.estimate_lipschitz_constant` at each time :math:`t`.

    .. warning::

        This method requires many evaluations of the right-hand side function
        *f* and will not be efficient in practice. Ideally, the Lipschitz
        constant is approximated by some theoretical result.
    """

    #: The right-hand side function used in the differential equation.
    f: StateFunction
    #: Fractional order of the derivative.
    alpha: float
    #: An expected domain of the state variable. The Lipschitz constant is
    #: estimated by sampling in this domain.
    yspan: tuple[float, float]

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()

            if not 0.0 < self.alpha < 1.0:
                raise ValueError(f"Derivative order must be in (0, 1): {self.alpha}")

            if self.yspan[0] >= self.yspan[1]:
                raise ValueError("Invalid yspan interval: ystart > yfinal")

    def get_next_time_step_raw(self, n: int, t: float, y: Array) -> float:
        """See :meth:`pycaputo.fode.TimeSpan.get_next_time_step_raw`."""
        if y.shape != (1,):
            raise ValueError(f"Only scalar functions are supported: {y.shape}")

        from math import gamma

        from pycaputo.lipschitz import estimate_lipschitz_constant

        L = estimate_lipschitz_constant(
            partial(self.f, t), self.yspan[0], self.yspan[1]
        )
        return float((gamma(2 - self.alpha) * L) ** (-1.0 / self.alpha))


# }}}


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
    #: A reason on why the step failed (if available).
    reason: str


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
    #: An instance describing the discrete time span being simulated.
    tspan: TimeSpan

    #: Right-hand side source term.
    source: StateFunction
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
    verbose: bool = True,
) -> Iterator[Event]:
    """Evolve the fractional-order ordinary differential equation in time.

    :arg m: method used to evolve the FODE.
    :arg f: right-hand side of the FODE.
    :arg y0: initial conditions for the FODE.
    :arg history: a :class:`History` instance that handles checkpointing the
        necessary state history for the method *m*.
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

    This function takes ``(history[t_s, ... t_n], t_{n + 1}, y_n)`` and is
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
