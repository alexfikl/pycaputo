# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Generic, Iterable, TypeVar

import numpy as np

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

    def __iter__(self) -> Iterable[Any]:
        for f in fields(self):
            yield getattr(self, f.name)


#: Invariant type variable bound to :class:`State`.
T = TypeVar("T", bound=State)


class History(ABC, Generic[T]):
    """A class handling the history checkpointing of an evolution equation.

    It essentially acts as a queue from which the items cannot be removed. For
    inspiration check out :class:`collections.deque`.

    The object itself is a :class:`~collections.abc.Sequence` and implements the
    required methods.

    .. automethod:: __bool__
    .. automethod:: __len__
    .. automethod:: __getitem__
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
    def __getitem__(self, k: int) -> T:
        """Load a checkpoint from the history.

        :returns: a compound :class:`State` from the *k*-th checkpoint.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all items from the history.

        Note that this will only reset relevant counters in the object. If the
        history is stored on disk, it will not delete the files.
        """

    @abstractmethod
    def append(self, t: float, y: Array) -> None:
        """Add a checkpoint to the current history."""


@dataclass(frozen=True)
class InMemoryHistory(History[T]):
    """A history with contiguous in-memory storage.

    This storage automatically grows when it exceeds its capacity and allows
    for easy retrieval of specific time steps. Note that, to resize the arrays
    no external references to them must exist. For example,
    ``ts = history.ts[:n]`` will create such a reference and a copy should be
    made instead if absolutely necessary.
    """

    #: An array of shape ``(n, ...)`` containing the stored values. Note that
    #: this is not necessarily the solution itself, but can be any data that is
    #: required for storage in the method.
    storage: Array = field(repr=False)
    #: An array of shape ``(n,)`` containing the fixed time steps.
    ts: Array = field(repr=False)

    filled: int = field(default=0, repr=False, init=False)

    @property
    def capacity(self) -> int:
        """The maximum size currently available for storage."""
        return self.ts.size

    @property
    def current_time(self) -> float:
        """The current time instance stored in the history."""
        return float(self.ts[self.filled - 1])

    def __bool__(self) -> bool:
        return self.filled > 0

    def __len__(self) -> int:
        return self.filled

    def clear(self) -> None:
        object.__setattr__(self, "filled", 0)

    def resize(self, new_size: int) -> None:
        """Forcefully resize the storage of the history.

        :arg new_size: the new desired size.
        """
        # NOTE: mostly following the growth pattern of python lists
        # https://github.com/python/cpython/blob/76bef3832bae64664882e27ecb6f89800a12cf43/Objects/listobject.c#L73

        new_size = new_size + (new_size >> 3) + (3 if new_size < 9 else 6)
        self.storage.resize((new_size, *self.storage.shape[1:]))
        self.ts.resize((new_size,))

    def append(self, t: float, y: Array) -> None:
        k = self.filled
        if k == self.capacity - 1:
            self.resize(k + 1)

        self.storage[k] = y
        self.ts[k] = t

        object.__setattr__(self, "filled", self.filled + 1)

    @classmethod
    def empty(
        cls,
        n: int = 512,
        shape: tuple[int, ...] = (1,),
        dtype: Any = None,
    ) -> InMemoryHistory[T]:
        """Construct a :class:`InMemoryHistory` for given sizes.

        :arg n: number of time steps that will be stored.
        :arg shape: shape of each state array that will be stored.
        """
        dtype = np.dtype(dtype)
        return cls(
            storage=np.empty((n, *shape), dtype=dtype),
            ts=np.empty(n, dtype=dtype),
        )

    @classmethod
    def empty_like(cls, y: Array, n: int = 512) -> InMemoryHistory[T]:
        """Construct a :class:`InMemoryHistory` for *y*."""
        return cls.empty(n=n, shape=y.shape, dtype=y.dtype)


# }}}


# {{{ ProductIntegrationHistory


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
class ProductIntegrationHistory(InMemoryHistory[ProductIntegrationState]):
    """A history for Product Integration methods with variable time step."""

    def __getitem__(self, k: int) -> ProductIntegrationState:
        if k == -1:
            k = self.filled - 1

        if not 0 <= k < self.filled:
            raise IndexError(f"History index out of range: 0 <= {k} < {self.filled}")

        return ProductIntegrationState(self.ts[k], self.storage[k])


# }}}
