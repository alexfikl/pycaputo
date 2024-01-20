# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pycaputo.derivatives import FractionalOperator, RiemannLiouvilleDerivative, Side
from pycaputo.grid import Points
from pycaputo.quadrature.base import QuadratureMethod, quad
from pycaputo.quadrature.riemann_liouville import (
    RiemannLiouvilleConvolutionMethod,
    RiemannLiouvilleCubicHermiteMethod,
    RiemannLiouvilleMethod,
    RiemannLiouvilleRectangularMethod,
    RiemannLiouvilleSimpsonMethod,
    RiemannLiouvilleSpectralMethod,
    RiemannLiouvilleTrapezoidalMethod,
)


def make_method_from_name(
    name: str,
    d: float | FractionalOperator,
) -> QuadratureMethod:
    """Instantiate a :class:`QuadratureMethod` given the name *name*.

    :arg d: a fractional operator that should be discretized by the method. If
        the method does not support this operator, it can fail.
    """

    methods: dict[str, type[QuadratureMethod]] = {
        cls.__name__: cls for cls in quad.registry
    }
    if name not in methods:
        raise ValueError(
            "Unknown quadrature method '{}'. Known methods are '{}'".format(
                name, "', '".join(methods)
            )
        )

    if not isinstance(d, FractionalOperator):
        d = RiemannLiouvilleDerivative(order=d, side=Side.Left)

    return methods[name](d)


def guess_method_for_order(
    p: Points,
    d: float | FractionalOperator,
) -> QuadratureMethod:
    """Construct a :class:`QuadratureMethod` for the given points *p* and
    integral *d*.

    Note that in general not all methods support arbitrary sets of points or
    arbitrary orders, so specialized methods must be chosen. This function is
    mean to make a reasonable guess at a high-order method. If other properties
    are required (e.g. stability), then a manual choice is better, perhaps
    using :func:`make_method_from_name`.

    :arg p: a set of points on which to evaluate the fractional operator.
    :arg d: a fractional operator to discretize.
    """
    from pycaputo.grid import JacobiGaussLobattoPoints

    if not isinstance(d, FractionalOperator):
        d = RiemannLiouvilleDerivative(order=d, side=Side.Left)

    m: QuadratureMethod | None = None

    if isinstance(d, RiemannLiouvilleDerivative):
        if isinstance(p, JacobiGaussLobattoPoints):
            m = RiemannLiouvilleSpectralMethod(d)
        else:
            m = RiemannLiouvilleTrapezoidalMethod(d)

    if m is None:
        raise ValueError(
            "Cannot determine an adequate method for the "
            f"'{type(d).__name__}' of order '{d.order}' and points of type "
            f"'{type(p).__name__}'."
        )

    return m


__all__ = (
    "QuadratureMethod",
    "RiemannLiouvilleConvolutionMethod",
    "RiemannLiouvilleCubicHermiteMethod",
    "RiemannLiouvilleMethod",
    "RiemannLiouvilleRectangularMethod",
    "RiemannLiouvilleSimpsonMethod",
    "RiemannLiouvilleSpectralMethod",
    "RiemannLiouvilleTrapezoidalMethod",
    "guess_method_for_order",
    "make_method_from_name",
    "quad",
)
