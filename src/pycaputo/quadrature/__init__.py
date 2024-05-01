# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pycaputo.derivatives import FractionalOperator, Side
from pycaputo.grid import Points
from pycaputo.quadrature.base import QuadratureMethod, quad


def guess_method_for_order(
    p: Points,
    d: float | FractionalOperator,
) -> QuadratureMethod:
    """Construct a :class:`QuadratureMethod` for the given points *p* and
    integral *d*.

    Note that in general not all methods support arbitrary sets of points or
    arbitrary orders, so specialized methods must be chosen. This function is
    mean to make a reasonable guess at a high-order method. If other properties
    are required (e.g. stability), then a manual choice is better.

    :arg p: a set of points on which to evaluate the fractional operator.
    :arg d: a fractional operator to discretize.
    """
    from pycaputo import grid
    from pycaputo.derivatives import RiemannLiouvilleDerivative
    from pycaputo.quadrature import riemann_liouville as rl

    if not isinstance(d, FractionalOperator):
        d = RiemannLiouvilleDerivative(alpha=d, side=Side.Left)

    m: QuadratureMethod | None = None

    if isinstance(d, RiemannLiouvilleDerivative):
        if isinstance(p, grid.JacobiGaussLobattoPoints):
            m = rl.SpectralJacobi(d.alpha)
        else:
            m = rl.Trapezoidal(d.alpha)

    if m is None:
        raise ValueError(
            "Cannot determine an adequate method for the operator "
            f"'{d!r}' and points of type '{type(p).__name__}'."
        )

    return m


__all__ = (
    "QuadratureMethod",
    "guess_method_for_order",
    "quad",
)
