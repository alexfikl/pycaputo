# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any

import numpy as np
from array_api_compat import array_namespace

from pycaputo.logging import get_logger
from pycaputo.typing import Array

log = get_logger(__name__)

# {{{ non-uniform points


@dataclass(frozen=True)
class Points:
    """A collection of points in an interval :math:`[a, b]`."""

    a: float
    """The left boundary of the interval."""
    b: float
    """The right boundary of the interval."""
    x: Array
    """The array of points."""

    def __len__(self) -> int:
        return len(self.x)

    @property
    def dtype(self) -> np.dtype[Any]:
        """The :class:`numpy.dtype` of the points in the set."""
        return np.dtype(self.x.dtype)

    @property
    def size(self) -> int:
        """The number of points in the set."""
        return self.x.size

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the points array in the set."""
        return self.x.shape  # type: ignore[no-any-return]

    @cached_property
    def dx(self) -> Array:
        """Distance between points."""
        xp = array_namespace(self.x)
        return xp.diff(self.x)  # type: ignore[no-any-return]

    @cached_property
    def xm(self) -> Array:
        """Array of midpoints."""
        return (self.x[1:] + self.x[:-1]) / 2

    @cached_property
    def dxm(self) -> Array:
        """Distance between midpoints."""
        xp = array_namespace(self.x)
        return xp.diff(self.xm)  # type: ignore[no-any-return]

    def translate(self, a: float, b: float) -> Points:
        """Linearly translate the set of points to the new interval :math:`[a, b]`."""
        # translate from [a', b'] to [0, 1] to [a, b]
        x = a + (b - a) / (self.b - self.a) * (self.x - self.a)
        return replace(self, a=self.a, b=self.b, x=x)


def make_stynes_points(
    n: int,
    a: float = 0.0,
    b: float = 1.0,
    *,
    r: float | None = 3.0,
    alpha: float | None = None,
    xp: Any = None,
) -> Points:
    r"""Construct a graded set of points on :math:`[a, b]`.

    This builds the graded mesh from [Stynes2017]_. The stretching function
    is given by

    .. math::

        \phi(\xi) = \xi^r,

    where the optimal grading :math:`r` is :math:`(2 - \alpha) / \alpha`.
    Note that, according to [Stynes2017]_, this grading is optimal for the L1
    method, but could be beneficial for other methods as well.

    :arg n: number of points in :math:`[a, b]`.
    :arg alpha: order of the fractional operator. The order is used to choose an
        optimal grading *r* according to [Stynes2017]_.
    :arg r: mesh grading -- a larger :math:`r` leads to more clustering
        of points near the origin :math:`a`. If set to *None*, a default optimal
        value is determined from *alpha*.
    """
    if xp is None:
        xp = np

    if r is None and alpha is None:
        raise ValueError("Must provide either 'alpha' or 'r' to define grading")

    if r is None:
        assert alpha is not None
        if 0.0 < alpha <= 1.0:
            r = (2 - alpha) / alpha
        else:
            raise ValueError("Grading estimate is only valid for 'alpha' in (0, 1)")

    x = xp.linspace(0.0, 1.0, n)
    return Points(a=a, b=b, x=a + (b - a) * x**r)


def make_stretched_points(
    n: int,
    a: float = 0.0,
    b: float = 1.0,
    *,
    strength: float = 4.0,
    xp: Any = None,
) -> Points:
    r"""Construct a :class:`Points` on :math:`[a, b]`.

    This uses a custom function that clusters points around the midpoint
    based on the strength *strength*. The stretching is given by

    .. math::

        \phi(\xi) = \xi + A (x_c - x) (1 - \xi) \xi^2

    where :math:`A` is the *strength* and :math:`x_c = 1/2` for the midpoint.

    **Note**: This function is mostly meant for testing.

    :arg n: number of points in :math:`[a, b]`.
    :arg strength: a positive number that controls the clustering of points at
        the midpoint, i.e. a larger number denotes more points.
    """
    if xp is None:
        xp = np

    x = xp.linspace(0, 1, n)
    x = x + strength * (0.5 - x) * (1 - x) * x**2

    return Points(a=a, b=b, x=a + (b - a) * x)


def make_sine_points(
    n: int,
    a: float = 0.0,
    b: float = 1.0,
    *,
    A: float = 0.5,
    omega: float = 1.0,
) -> Points:
    r"""Construct a set of points on :math:`[a, b]` with a sinusoidal spacing.

    The time step will have a magnitude proportional to

    .. math::

        \Delta x_k \propto 1 + A \sin \left(2 \pi \omega \frac{k}{n - 2}\right)

    The amplitude :math:`A` also controls the smallest time step allowed on this
    grid. In particular, when the sine takes its minimal value of :math:`-1`,
    an amplitude of :math:`A = ±1` can give almost zero spacing.

    **Note**: This function is mostly meant for testing.

    :arg A: amplitude of the sine wave in :math:`(-1, 1)`.
    :arg omega: wavenumber of the sine wave.
    """
    if not -1.0 < A < 1.0:
        raise ValueError(f"Amplitude 'A' not in (-1, 1): {A}")

    if omega <= 0:
        raise ValueError(f"Wavenumber 'omega' cannot be non-positive: '{omega}'")

    # create spacing in [0, 1]
    k = np.arange(n - 1) / (n - 2)
    dx = 1.0 + A * np.sin(2.0 * np.pi * omega * k)
    dx /= np.sum(dx)

    # cumsum everything up to get the actual grid points
    x = np.empty(n)
    x[0] = 0
    x[1:] = np.cumsum(dx)
    assert np.linalg.norm(np.diff(x) - dx) < 1.0e-15

    return Points(a=a, b=b, x=a + (b - a) * x)


# }}}


# {{{ uniform


@dataclass(frozen=True)
class UniformPoints(Points):
    """A uniform set of points in :math:`[a, b]`."""


def make_uniform_points(
    n: int,
    a: float = 0.0,
    b: float = 1.0,
    *,
    xp: Any = None,
) -> UniformPoints:
    """Construct a :class:`UniformPoints` on :math:`[a, b]`.

    :arg n: number of points in :math:`[a, b]`.
    """
    if xp is None:
        xp = np

    return UniformPoints(a=a, b=b, x=xp.linspace(a, b, n))


# }}}

# {{{ midpoints


@dataclass(frozen=True)
class MidPoints(Points):
    """A set of points on :math:`[a, b]` formed from midpoints of another grid.

    This set of points consists of :math:`x_0 = a` and :math:`x_j` are the
    midpoints of another grid.
    """


def make_uniform_midpoints(
    n: int,
    a: float = 0.0,
    b: float = 1.0,
    *,
    xp: Any = None,
) -> MidPoints:
    """Construct a :class:`MidPoints` on :math:`[a, b]`.

    :arg n: number of points in :math:`[a, b]`.
    """
    if xp is None:
        xp = np

    x = xp.linspace(a, b, n)
    x[1:] = (x[1:] + x[:-1]) / 2

    return MidPoints(a=a, b=b, x=x)


def make_midpoints_from(p: Points) -> MidPoints:
    """Construct a grid of midpoints from a given set of points *p*.

    The new grid will contain the point :math:`x_0` and the midpoints of *p*.
    """
    xp = array_namespace(p.x)

    x = xp.empty_like(p.x)
    x[0] = p.x[0]
    x[1:] = p.xm

    return MidPoints(a=p.a, b=p.b, x=x)


# }}}


# {{{ Jacobi-Gauss-Lobatto


@dataclass(frozen=True)
class JacobiGaussLobattoPoints(Points):
    """A set of Jacobi-Gauss-Lobatto points on :math:`[a, b]`.

    See :func:`scipy.special.roots_jacobi`. This can also be used for quadrature
    using :attr:`w` and is accurate for polynomials of degreen up to
    :math:`2 n - 3`.
    """

    alpha: float
    """Parameter of the Jacobi polynomials."""
    beta: float
    """Parameter of the Jacobi poylnomials."""
    w: Array
    """Jacobi-Gauss-Lobatto quadrature weights on :math:`[a, b]`."""

    def translate(self, a: float, b: float) -> Points:
        """Linearly translate the set of points to the new interval :math:`[a, b]`."""
        # translate from [a', b'] to [0, 1] to [a, b]
        r = (b - a) / (self.b - self.a)
        x = a - r * self.a + r * self.x
        w = r * self.w

        return replace(self, a=a, b=b, x=x, w=w)


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

    if n > 100:
        from warnings import warn

        warn(
            "Evaluating Jacobi nodes for large n > 100 might be numerically unstable",
            stacklevel=2,
        )

    from pycaputo.jacobi import jacobi_gauss_lobatto_nodes, jacobi_gauss_lobatto_weights

    xi = jacobi_gauss_lobatto_nodes(n, alpha, beta)
    wi = jacobi_gauss_lobatto_weights(xi, alpha, beta)

    # translate affinely to [a, b] from [-1, 1]
    x = (b + a) / 2 + (b - a) / 2 * xi
    w = (b - a) / 2 * wi

    return JacobiGaussLobattoPoints(a=a, b=b, x=x, alpha=alpha, beta=beta, w=w)


# }}}


# {{{ make

REGISTERED_POINTS: dict[str, Callable[..., Points]] = {
    "jacobi": make_jacobi_gauss_lobatto_points,
    "midpoints": make_uniform_midpoints,
    "sine": make_sine_points,
    "stretch": make_stretched_points,
    "stynes": make_stynes_points,
    "uniform": make_uniform_points,
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
