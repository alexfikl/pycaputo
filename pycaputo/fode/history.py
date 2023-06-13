# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from pycaputo.logging import get_logger
from pycaputo.utils import Array

logger = get_logger(__name__)


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
