# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property, singledispatch
from typing import TYPE_CHECKING, Any, Generic

from pycaputo.derivatives import FractionalOperatorT
from pycaputo.events import Event
from pycaputo.history import History
from pycaputo.typing import Array, StateFunctionT

if TYPE_CHECKING:
    # NOTE: avoid cyclic import
    from pycaputo.controller import Controller


@dataclass(frozen=True)
class FractionalDifferentialEquationMethod(
    ABC,
    Generic[FractionalOperatorT, StateFunctionT],
):
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

    ds: tuple[FractionalOperatorT, ...]
    """The fractional operators used for each equation."""
    control: Controller
    """An instance describing the discrete time being simulated."""

    source: StateFunctionT
    """Right-hand side source term."""
    y0: tuple[Array, ...]
    """Values used to reconstruct the required initial conditions."""

    if __debug__:

        def __post_init__(self) -> None:
            if not self.ds:
                raise ValueError("No fractional operators given")

            if not self.y0:
                raise ValueError("No initial conditions given")

            shape = self.y0[0]
            if not all(y0.shape == shape for y0 in self.y0[1:]):
                raise ValueError(
                    "Initial conditions have different shapes: "
                    f"{[y0.shape for y0 in self.y0]}"
                )

            if self.y0[0].size != len(self.ds):
                raise ValueError(
                    f"Fractional operator must match state size: got {len(self.ds)} "
                    f"operators for initial conditions of size {self.y0[0].size}"
                )

            y = self.source(0.0, self.y0[0])
            if y.shape != self.y0[0].shape:
                raise ValueError(
                    "Array returned by 'source' does not match y0: "
                    f"got shape {y.shape} for y0 of shape {shape}"
                )

    @property
    def name(self) -> str:
        """An identifier for the method."""
        return type(self).__name__.replace("Method", "")

    @property
    @abstractmethod
    def derivative_order(self) -> tuple[float, ...]:
        r"""A number that represents the *fractional* order of the operators in
        :attr:`~pycaputo.stepping.FractionalDifferentialEquationMethod.ds`. For
        example, in the case of the Caputo derivative, this is just the order
        :math:`\alpha`.
        """

    @cached_property
    def smallest_derivative_order(self) -> float:
        """Smallest value of the
        :attr:`~pycaputo.stepping.FractionalDifferentialEquationMethod.derivative_order`.
        """
        return min(self.derivative_order)

    @cached_property
    def largest_derivative_order(self) -> float:
        """Largest value of the
        :attr:`~pycaputo.stepping.FractionalDifferentialEquationMethod.derivative_order`.
        """
        return max(self.derivative_order)

    @property
    @abstractmethod
    def order(self) -> float:
        r"""Expected order of convergence of the method.

        In general, the order of convergence will depend on the parameters of
        the fractional operator (e.g. the order :math:`\alpha` of the Caputo
        derivative), but also on the smoothness of the solutions and the temporal
        mesh that is used. The meaning of the order is therefore not clear.
        """

    @abstractmethod
    def make_default_history(self) -> History[Any]:
        """Construct a default history for the method."""


@singledispatch
def evolve(
    m: FractionalDifferentialEquationMethod[FractionalOperatorT, StateFunctionT],
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
    m: FractionalDifferentialEquationMethod[FractionalOperatorT, StateFunctionT],
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
    m: FractionalDifferentialEquationMethod[FractionalOperatorT, StateFunctionT],
) -> Array:
    """Construct an initial condition for the method *m*."""
    raise NotImplementedError(type(m).__name__)


# }}}
