# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from pycaputo.utils import Array


@dataclass(frozen=True)
class Points:
    """A collection of points in an interval :math:`[a, b]`."""

    #: The left boundary of the interval.
    a: float
    #: The right boundary of the interval.
    b: float
    #: The array of points.
    x: Array

    @property
    def n(self) -> int:
        """Number of points in the collection."""
        return self.x.size

    @cached_property
    def dx(self) -> Array:
        """Distance between points."""
        return np.diff(self.x)


def make_stretched_points(
    n: int, a: float = 0.0, b: float = 1.0, strength: float = 4.0
) -> Points:
    """Construct a :class:`Points` on :math:`[a, b]`.

    This uses a custom function that clusters points around the midpoint
    based on the strength *strength*.

    :arg n: number of points in :math:`[a, b]`.
    :arg strength: a positive number that constrols the clustering of points at
        the midpoint, i.e. a larger number denotes more points.
    """
    x = np.linspace(0, 1, n)
    x = x + strength * (0.5 - x) * (1 - x) * x**2

    return Points(a=a, b=b, x=a + (b - a) * x)


@dataclass(frozen=True)
class UniformPoints(Points):
    """A uniform set of points in :math:`[a, b]`."""


def make_uniform_points(n: int, a: float = 0.0, b: float = 1.0) -> UniformPoints:
    """Construct a :class:`UniformPoints` on :math:`[a, b]`.

    :arg n: number of points in :math:`[a, b]`.
    """
    return UniformPoints(a=a, b=b, x=np.linspace(a, b, n))


@dataclass(frozen=True)
class UniformMidpoints(Points):
    """A set of points on :math:`[a, b]` formed from midpoints of a uniform grid.

    This set of points consists of :math:`x_0 = a` and :math:`x_j` are the
    midpoints of the uniform grid :class:`UniformPoints`
    """


def make_uniform_midpoints(n: int, a: float = 0.0, b: float = 1.0) -> UniformMidpoints:
    """Construct a :class:`UniformMidpoints` on :math:`[a, b]`.

    :arg n: number of points in :math:`[a, b]`.
    """
    x = np.linspace(a, b, n)
    x[1:] = (x[1:] + x[:-1]) / 2

    return UniformMidpoints(a=a, b=b, x=x)
