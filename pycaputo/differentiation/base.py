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

    d: FractionalOperator

    if __debug__:

        def __post_init__(self) -> None:
            if not self.supports(self.d.order):
                raise ValueError(
                    f"Method '{self.name}' does not support derivative order "
                    f"'{self.d.order}'"
                )

    @property
    def name(self) -> str:
        """An identifier for the differentiation method."""
        return type(self).__name__.replace("Method", "")

    @property
    @abstractmethod
    def order(self) -> float:
        """Expected order of convergence of the method."""

    @abstractmethod
    def supports(self, alpha: float) -> bool:
        """
        :returns: *True* if the method supports computing the fractional
            order derivative of order *alpha* and *False* otherwise.
        """


@singledispatch
def diff(m: DerivativeMethod, f: ArrayOrScalarFunction, p: Points) -> Array:
    """Evaluate the fractional derivative of *f* at *x*.

    Note that not all numerical methods can evaluate the derivative at all
    points in *x*. For example, the :class:`CaputoL1Method` cannot evaluate the
    derivative at ``x[0]``.

    :arg m: method used to evaluate the derivative.
    :arg f: a simple function for which to evaluate the derivative.
    :arg p: an array of points at which to evaluate the derivative.
    """
    raise NotImplementedError(
        f"Cannot evaluate fractional derivative with method '{type(m).__name__}'"
    )
