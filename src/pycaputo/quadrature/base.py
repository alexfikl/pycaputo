# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import singledispatch

from pycaputo.derivatives import FractionalOperator
from pycaputo.grid import Points
from pycaputo.typing import Array, ArrayOrScalarFunction


@dataclass(frozen=True)
class QuadratureMethod(ABC):
    """A generic method used to evaluate a fractional integral at a set of points."""

    @property
    def name(self) -> str:
        """An identifier for the quadrature method."""
        return type(self).__name__.replace("Method", "")

    @property
    @abstractmethod
    def d(self) -> FractionalOperator:
        """The fractional operator that is being discretized by the method."""


@singledispatch
def quad(m: QuadratureMethod, f: ArrayOrScalarFunction, p: Points) -> Array:
    """Evaluate the fractional integral of *f* at *p* using method *m*.

    :arg m: method used to evaluate the integral.
    :arg f: a simple function for which to evaluate the integral. If the
        method requires higher-order derivatives (e.g. for Hermite interpolation),
        this function can also be a
        :class:`~pycaputo.typing.DifferentiableScalarFunction`.
    :arg p: an array of points at which to evaluate the integral.
    """

    raise NotImplementedError(
        f"Cannot evaluate fractional integral with method '{type(m).__name__}'"
    )
