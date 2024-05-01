# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import singledispatch

from pycaputo.derivatives import FractionalOperator
from pycaputo.grid import Points
from pycaputo.utils import Array, ArrayOrScalarFunction


@dataclass(frozen=True)
class DerivativeMethod(ABC):
    """A generic method used to evaluate a fractional derivative at a set of points."""

    @property
    def name(self) -> str:
        """An (non-unique) identifier for the differentiation method."""
        return type(self).__name__.replace("Method", "")

    @property
    @abstractmethod
    def d(self) -> FractionalOperator:
        """The fractional operator that is being discretized by the method."""


@singledispatch
def diff(m: DerivativeMethod, f: ArrayOrScalarFunction, p: Points) -> Array:
    """Evaluate the fractional derivative of *f* at *p* using method *m*.

    This is a low-level function that should be implemented by methods. For
    a higher-level function see :func:`pycaputo.diff`. Note that this function
    evaluates the derivative at all points in *p*, not just at *p[-1]*.

    :arg m: method used to evaluate the derivative.
    :arg f: a simple function for which to evaluate the derivative.
    :arg p: an array of points at which to evaluate the derivative.
    """
    raise NotImplementedError(
        f"Cannot evaluate fractional derivative with method '{type(m).__name__}'"
    )
