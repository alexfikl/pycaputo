# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pycaputo.logging import get_logger
from pycaputo.utils import Array

logger = get_logger(__name__)


# {{{ interface


@dataclass(frozen=True)
class State:
    """A class the holds the history of the state variables.

    For a simple method, this can be only :math:`(t_n, y_n)` at every time step.
    However, more complex methods can also checkpoint the right-hand side
    evaluations or other intermediary calculations.
    """


class History(ABC):
    """A class handling the history checkpointing of an evolution equation.

    It essentially acts as a queue from which the items cannot be removed. For
    inspiration check out :class:`collections.deque`.
    """

    @abstractmethod
    def __bool__(self) -> bool:
        """
        :returns: *False* if the history is empty and *True* otherwise.
        """

    @abstractmethod
    def __len__(self) -> int:
        """
        :returns: the number of checkpointed states.
        """

    @abstractmethod
    def __getitem__(self, k: int) -> State:
        """
        :returns: a compound :class:`State` from the *k*-th checkpoint.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all items from the history."""

    @abstractmethod
    def append(self, t: float, y: Array) -> None:
        """Add a state to the current history."""


# }}}


# {{{ VariableProductIntegrationHistory


@dataclass(frozen=True)
class ProductIntegrationState(State):
    """A state history that holds the right-hand side evaluations used by
    standard Product Integration methods.
    """

    #: Time of the evaluation.
    t: float
    #: Evaluation of the right-hand side.
    f: Array


@dataclass(frozen=True)
class VariableProductIntegrationHistory(History):
    """A history for Product Integration methods with variable time step."""

    #: History of state variables, required to compute the memory term.
    history: list[ProductIntegrationState] = field(default_factory=list, repr=False)
    #: Time instances of each entry in the :attr:`history`.
    ts: list[float] = field(default_factory=list, repr=False)

    def __bool__(self) -> bool:
        return bool(self.history)

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, k: int) -> ProductIntegrationState:
        nhistory = len(self.history)
        k = (k + nhistory) if k < 0 else k

        if not 0 <= k < nhistory:
            raise IndexError(f"History index out of range: 0 <= {k} < {nhistory}")

        return self.history[k]

    def clear(self) -> None:
        self.history.clear()
        self.ts.clear()

    def append(self, t: float, y: Array) -> None:
        self.history.append(ProductIntegrationState(t=t, f=y))


# }}}


# {{{ FixedSizeHistory


@dataclass(frozen=True)
class FixedState(State):
    """A view into the state values at a given time."""

    #: Time at which the state function is evaluated at.
    t: float
    #: State function stored by the :class:`FixedSizeHistory`.
    value: Array


@dataclass(frozen=True)
class FixedSizeHistory(History):
    """A history with a preallocated fixed set of arrays."""

    history: Array = field(repr=False)
    ts: Array = field(repr=False)
    filled: int = 0

    def __bool__(self) -> bool:
        return self.filled == 0

    def __len__(self) -> int:
        return self.filled

    def __getitem__(self, k: int) -> FixedState:
        if k == -1:
            k = self.filled - 1

        if not 0 <= k < self.filled:
            raise IndexError(f"history index out of range: 0 <= {k} < {self.filled}")

        return FixedState(t=self.ts[k], value=self.history[k])

    def clear(self) -> None:
        object.__setattr__(self, "filled", 0)

    def append(self, t: float, y: Array) -> None:
        object.__setattr__(self, "filled", self.filled + 1)

        k = self.filled
        self.history[k] = y
        self.ts[k] = t


# }}}
