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
class QuadratureMethod(ABC):
    """A generic method used to evaluate a fractional integral."""

    #: Description of the integral that is approximated.
    d: FractionalOperator

    if __debug__:

        def __post_init__(self) -> None:
            if self.d.order >= 0:
                raise ValueError(
                    f"Integral requires a negative order: order is '{self.d.order}'"
                )

    @property
    @abstractmethod
    def name(self) -> str:
        """An identifier for the quadrature method."""
        return type(self).__name__.replace("Method", "")

    @property
    @abstractmethod
    def order(self) -> float:
        """Expected order of convergence of the method."""


@singledispatch
def quad(m: QuadratureMethod, f: ArrayOrScalarFunction, x: Points) -> Array:
    """Evaluate the fractional integral of *f* using the points *x*.

    :arg m: method used to evaluate the integral.
    :arg f: a simple function for which to evaluate the integral. If the
        method requires higher-order derivatives (e.g. for Hermite interpolation),
        this function can also be a
        :class:`~pycaputo.utils.DifferentiableScalarFunction`.
    :arg x: an array of points at which to evaluate the integral.
    """

    raise NotImplementedError(
        f"Cannot evaluate integral with method '{type(m).__name__}'"
    )
