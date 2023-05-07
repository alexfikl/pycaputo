# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Dict

import numpy as np

from pycaputo.logging import get_logger
from pycaputo.utils import Array

logger = get_logger(__name__)

# {{{ non-uniform points


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
    r"""Construct a :class:`Points` on :math:`[a, b]`.

    This uses a custom function that clusters points around the midpoint
    based on the strength *strength*. The stretching is given by

    .. math::

        \phi(\xi) = \xi + A (x_c - x) (1 - \xi) \xi^2

    where :math:`A` is the *strength* and :math:`x_c = 1/2` for the midpoint.

    :arg n: number of points in :math:`[a, b]`.
    :arg strength: a positive number that constrols the clustering of points at
        the midpoint, i.e. a larger number denotes more points.
    """
    x = np.linspace(0, 1, n)
    x = x + strength * (0.5 - x) * (1 - x) * x**2

    return Points(a=a, b=b, x=a + (b - a) * x)


def make_stynes_points(
    n: int, a: float = 0.0, b: float = 1.0, gamma: float = 2.0
) -> Points:
    r"""Construct a graded set of points on :math:`[a, b]`.

    This builds the graded mesh from [Stynes2017]_. The stretching function
    is given by

    .. math::

        \phi(\xi) = \xi^\gamma,

    where the optimal grading :math:`\gamma` is :math:`(2 - \alpha) / \alpha`.
    Note that, according to [Stynes2017]_, this grading is optimal for the L1
    method.

    :arg n: number of points in :math:`[a, b]`.
    :arg gamma: mesh grading -- a larger :math:`\gamma` leads to more clustering
        of points near the origin :math:`a`.
    """

    x = np.linspace(0.0, 1.0, n)
    return Points(a=a, b=b, x=a + (b - a) * x**gamma)


# }}}


# {{{ uniform


@dataclass(frozen=True)
class UniformPoints(Points):
    """A uniform set of points in :math:`[a, b]`."""


def make_uniform_points(n: int, a: float = 0.0, b: float = 1.0) -> UniformPoints:
    """Construct a :class:`UniformPoints` on :math:`[a, b]`.

    :arg n: number of points in :math:`[a, b]`.
    """
    return UniformPoints(a=a, b=b, x=np.linspace(a, b, n))


# }}}

# {{{ midpoints


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


# }}}


# {{{ Jacobi-Gauss-Lobatto


@dataclass(frozen=True)
class JacobiGaussLobattoPoints(Points):
    """A set of Jacobi-Gauss-Lobatto points on :math:`[a, b]`.

    See :func:`scipy.special.roots_jacobi`. This can also be used for quadrature
    using :attr:`w` and is accurate for polynomials of degreen up to
    :math:`2 n - 3`.
    """

    #: Parameter of the Jacobi polynomials.
    alpha: float
    #: Parameter of the Jacobi poylnomials.
    beta: float
    #: Jacobi-Gauss-Lobatto quadrature weights on :math:`[a, b]`.
    w: Array


def make_jacobi_gauss_lobatto_points(
    n: int,
    a: float = 0.0,
    b: float = 1.0,
    *,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> JacobiGaussLobattoPoints:
    r"""Construct a set of Jacobi-Gauss-Lobatto points on :math:`[a, b]`.

    For the special cases of :math:`\alpha = \beta = 0` we get the
    Legendre-Gauss-Lobatto points and for :math:`\alpha = \beta = -1/2` we
    get the Chebyshev-Gauss-Lobatto points.

    :arg n: number of points in :math:`[a, b]`.
    :arg alpha: parameter for the Jacobi polynomial.
    :arg beta: parameter for the Jacobi polynomial.
    """
    if n < 3:
        raise ValueError("At least 3 points are required")

    if n > 30:
        from warnings import warn

        warn("Evaluating Jacobi nodes for large n > 30 might be numerically unstable")

    from scipy.special import roots_jacobi

    xi, w = np.empty(n), np.empty(n)

    xi[1:-1], w[1:-1] = roots_jacobi(n - 2, alpha, beta)

    # add Lobatto points
    xi[0], xi[-1] = -1.0, 1.0
    w[0] = w[-1] = 2 / (n * (n - 1))

    # translate affinely to [a, b] from [-1, 1]
    x = (b + a) / 2 + (b - a) / 2 * xi
    w = (b - a) / 2 * w

    return JacobiGaussLobattoPoints(a=a, b=b, x=x, alpha=alpha, beta=beta, w=w)


# }}}


# {{{ make

REGISTERED_POINTS: Dict[str, Callable[..., Points]] = {
    "stretch": make_stretched_points,
    "stynes": make_stynes_points,
    "uniform": make_uniform_points,
    "midpoints": make_uniform_midpoints,
}


def make_points_from_name(name: str, n: int, a: float = 0.0, b: float = 1.0) -> Points:
    """Construct a set of points by name.

    :arg name: the name of the point set.
    :arg n: number of points in :math:`[a, b]`.
    """
    if name not in REGISTERED_POINTS:
        raise ValueError(
            "Unknown point distribution '{}'. Known distributions are '{}'".format(
                name, "', '".join(REGISTERED_POINTS)
            )
        )

    return REGISTERED_POINTS[name](n, a=a, b=b)


# }}}
