# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pycaputo.derivatives import CaputoDerivative, FractionalOperator, Side
from pycaputo.differentiation.base import DerivativeMethod, diff
from pycaputo.differentiation.caputo import (
    CaputoDerivativeMethod,
    CaputoL1Method,
    CaputoL2CMethod,
    CaputoL2Method,
    CaputoModifiedL1Method,
    CaputoSpectralMethod,
)
from pycaputo.differentiation.riemann_liouville import (
    RiemannLiouvilleDerivativeMethod,
    RiemannLiouvilleFromCaputoDerivativeMethod,
    RiemannLiouvilleL1Method,
    RiemannLiouvilleL2CMethod,
    RiemannLiouvilleL2Method,
)
from pycaputo.grid import Points


def make_method_from_name(
    name: str,
    d: float | FractionalOperator,
) -> DerivativeMethod:
    """Instantiate a :class:`DerivativeMethod` given the name *name*.

    :arg d: a fractional operator that should be discretized by the method. If
        the method does not support this operator, it can fail.
    """

    methods: dict[str, type[DerivativeMethod]] = {
        cls.__name__: cls for cls in diff.registry
    }
    if name not in methods:
        raise ValueError(
            "Unknown differentiation method '{}'. Known methods are '{}'".format(
                name, "', '".join(methods)
            )
        )

    if not isinstance(d, FractionalOperator):
        d = CaputoDerivative(order=d, side=Side.Left)

    return methods[name](d)


def guess_method_for_order(
    p: Points,
    d: float | FractionalOperator,
) -> DerivativeMethod:
    """Construct a :class:`DerivativeMethod` for the given points *p* and
    derivative *d*.

    Note that in general not all methods support arbitrary sets of points or
    arbitrary orders, so specialized methods must be chosen. This function is
    mean to make a reasonable guess at a high-order method. If other properties
    are required (e.g. stability), then a manual choice is better, perhaps
    using :func:`make_method_from_name`.

    :arg p: a set of points on which to evaluate the fractional operator.
    :arg d: a fractional operator to discretize.
    """
    from pycaputo.grid import JacobiGaussLobattoPoints, UniformMidpoints, UniformPoints

    m: DerivativeMethod | None = None
    if not isinstance(d, FractionalOperator):
        d = CaputoDerivative(order=d, side=Side.Left)

    if isinstance(d, CaputoDerivative):
        if isinstance(p, JacobiGaussLobattoPoints):
            m = CaputoSpectralMethod(d)
        elif 0 < d.order < 1:
            if isinstance(p, UniformMidpoints):
                m = CaputoModifiedL1Method(d)
            else:
                m = CaputoL1Method(d)
        elif 1 < d.order < 2 and isinstance(p, UniformPoints):
            m = CaputoL2CMethod(d)

    if m is None:
        raise ValueError(
            "Cannot determine an adequate method for the "
            f"'{type(d).__name__}' of order '{d.order}' and points of type "
            f"'{type(p).__name__}'."
        )

    return m


__all__ = (
    "CaputoDerivativeMethod",
    "CaputoL1Method",
    "CaputoL2CMethod",
    "CaputoL2Method",
    "CaputoModifiedL1Method",
    "CaputoSpectralMethod",
    "DerivativeMethod",
    "RiemannLiouvilleDerivativeMethod",
    "RiemannLiouvilleFromCaputoDerivativeMethod",
    "RiemannLiouvilleL1Method",
    "RiemannLiouvilleL2CMethod",
    "RiemannLiouvilleL2Method",
    "diff",
    "guess_method_for_order",
    "make_method_from_name",
)
