# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pycaputo.derivatives import FractionalOperator, Side
from pycaputo.differentiation.base import DerivativeMethod, diff
from pycaputo.grid import Points


def guess_method_for_order(
    p: Points,
    d: float | FractionalOperator,
) -> DerivativeMethod:
    """Construct a :class:`DerivativeMethod` for the given points *p* and
    derivative *d*.

    Note that in general not all methods support arbitrary sets of points or
    arbitrary orders, so specialized methods must be chosen. This function is
    mean to make a reasonable guess at a high-order method. If other properties
    are required (e.g. stability), then a manual choice is better.

    :arg p: a set of points on which to evaluate the fractional operator.
    :arg d: a fractional operator to discretize. If only a float is given, the
        common Caputo derivative is used.
    """
    from pycaputo import grid
    from pycaputo.derivatives import CaputoDerivative, RiemannLiouvilleDerivative
    from pycaputo.differentiation import caputo
    from pycaputo.differentiation import riemann_liouville as rl

    m: DerivativeMethod | None = None
    if not isinstance(d, FractionalOperator):
        d = CaputoDerivative(order=d, side=Side.Left)

    if isinstance(d, CaputoDerivative):
        if isinstance(p, grid.JacobiGaussLobattoPoints):
            m = caputo.SpectralJacobi(d.order)
        elif 0 < d.order < 1:
            if isinstance(p, grid.UniformMidpoints):
                m = caputo.ModifiedL1(d.order)
            else:
                m = caputo.L1(d.order)
        elif 1 < d.order < 2 and isinstance(p, grid.UniformPoints):
            m = caputo.L2C(d.order)
    elif isinstance(d, RiemannLiouvilleDerivative):
        if 0 < d.order < 1:
            m = rl.L1(d.order)
        elif 1 < d.order < 2 and isinstance(p, grid.UniformPoints):
            m = rl.L2C(d.order)

    if m is None:
        raise ValueError(
            "Cannot determine an adequate method for the "
            f"'{type(d).__name__}' of order '{d.order}' and points of type "
            f"'{type(p).__name__}'."
        )

    return m


__all__ = (
    "DerivativeMethod",
    "diff",
    "guess_method_for_order",
)
