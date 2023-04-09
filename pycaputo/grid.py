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


@dataclass(frozen=True)
class UniformPoints(Points):
    """A uniform set of points in :math:`[a, b]`."""


def make_uniform_points(n: int, a: float = 0.0, b: float = 1.0) -> UniformPoints:
    """Construct a uniform point distribution.

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
    x = np.linspace(a, b, n)
    x[1:] = np.diff(x)

    return UniformMidpoints(a=a, b=b, x=x[:-1])
