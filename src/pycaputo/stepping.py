# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property, singledispatch
from typing import TYPE_CHECKING, Any, Generic, Iterable, Iterator

import numpy as np

from pycaputo.derivatives import FractionalOperator
from pycaputo.events import Event
from pycaputo.history import History
from pycaputo.utils import Array, StateFunctionT, cached_on_first_arg, gamma

if TYPE_CHECKING:
    # NOTE: avoid cyclic import
    from pycaputo.controller import Controller


@cached_on_first_arg
def gamma1p(m: FractionalDifferentialEquationMethod[StateFunctionT]) -> Array:
    r"""A cached vectorized value of :math:`\Gamma(1 + \alpha_i)`."""
    return gamma(1 + m.alpha)


@cached_on_first_arg
def gamma2p(m: FractionalDifferentialEquationMethod[StateFunctionT]) -> Array:
    r"""A cached vectorized value of :math:`\Gamma(2 + \alpha_i)`."""
    return gamma(2 + m.alpha)


@cached_on_first_arg
def gamma2m(m: FractionalDifferentialEquationMethod[StateFunctionT]) -> Array:
    r"""A cached vectorized value of :math:`\Gamma(2 - \alpha_i)`."""
    return gamma(2 - m.alpha)


@dataclass(frozen=True)
class FractionalDifferentialEquationMethod(ABC, Generic[StateFunctionT]):
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
    #: An instance describing the discrete time being simulated.
    control: Controller

    #: Right-hand side source term.
    source: StateFunctionT
    #: Values used to reconstruct the required initial conditions.
    y0: tuple[Array, ...]

    if __debug__:

        def __post_init__(self) -> None:
            if not self.y0:
                raise ValueError("No initial conditions given")

            shape = self.y0[0]
            if not all(y0.shape == shape for y0 in self.y0[1:]):
                raise ValueError("Initial conditions have different shapes")

            if self.y0[0].size != len(self.derivative_order):
                raise ValueError("Derivative orders must match state size")
            from math import ceil

            m = ceil(self.largest_derivative_order)
            if m != len(self.y0):
                raise ValueError(
                    "Incorrect number of initial conditions: "
                    f"got {len(self.y0)}, but expected {m} arrays"
                )

    @property
    def name(self) -> str:
        """An identifier for the method."""
        return type(self).__name__.replace("Method", "")

    @cached_property
    def largest_derivative_order(self) -> float:
        """Largest order in :attr:`derivative_order`."""
        return max(self.derivative_order)

    @cached_property
    def smallest_derivative_order(self) -> float:
        """Smallest order in :attr:`derivative_order`."""
        return min(self.derivative_order)

    @cached_property
    def alpha(self) -> Array:
        """A cached vectorized form of :attr:`derivative_order`."""
        return np.array(self.derivative_order)

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
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    *,
    history: History[Any] | None = None,
    dtinit: float | None = None,
) -> Iterator[Event]:
    """Evolve the fractional-order ordinary differential equation in time.

    :arg m: method used to evolve the FODE.
    :arg history: a :class:`~pycaputo.history.History` instance that handles
        checkpointing the necessary state history for the method *m*.
    :arg dtinit: an initial time step used to start the simulation. If none is
        provided the controller of the method will be used to estimate it
        (see :attr:`~pycaputo.stepping.FractionalDifferentialEquationMethod.control`).

    :returns: an :class:`~pycaputo.events.Event` (usually a
        :class:`~pycaputo.events.StepCompleted`) containing
        the solution at a time :math:`t`.
    """
    raise NotImplementedError(f"'evolve' functionality for '{type(m).__name__}'")


@singledispatch
def advance(
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    history: History[Any],
    y: Array,
    dt: float,
) -> Iterable[Array]:
    r"""Advance the solution with the *history* by a time step *dt*.

    This function takes ``history[t_s, ... t_n]`` with the history up to the
    time :math:`t_n` and the time step :math:`\Delta t` used to evolve to the
    next time:math:`t_{n + 1}`.

    :arg history: the history of all previous time steps.
    :arg y: state solution from the previous time step.
    :arg dt: the time step to use to evolve to the next step.
    :returns: an iterable where the first item is the solution at the next time
        step and the remaining items contain additional information about the
        evolution left to the individual methods.
    """
    raise NotImplementedError(f"'advance' functionality for '{type(m).__name__}'")


@singledispatch
def make_initial_condition(
    m: FractionalDifferentialEquationMethod[StateFunctionT],
) -> Array:
    """Construct an initial condition for the method *m*."""
    raise NotImplementedError(type(m).__name__)


# }}}
