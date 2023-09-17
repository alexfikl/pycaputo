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
from pycaputo.grid import Points

REGISTERED_METHODS: dict[str, type[DerivativeMethod]] = {
    "CaputoL1Method": CaputoL1Method,
    "CaputoL2CMethod": CaputoL2CMethod,
    "CaputoL2Method": CaputoL2Method,
    "CaputoModifiedL1Method": CaputoModifiedL1Method,
    "CaputoSpectralMethod": CaputoSpectralMethod,
}


def register_method(
    name: str,
    method: type[DerivativeMethod],
    *,
    force: bool = False,
) -> None:
    """Register a new derivative approximation method.

    :arg name: a canonical name for the method.
    :arg method: a class that will be used to construct the method.
    :arg force: if *True*, any existing methods will be overwritten.
    """

    if not force and name in REGISTERED_METHODS:
        raise ValueError(
            f"A method by the name '{name}' is already registered. Use 'force=True' to"
            " overwrite it."
        )

    REGISTERED_METHODS[name] = method


def make_method_from_name(
    name: str,
    d: float | FractionalOperator,
) -> DerivativeMethod:
    """Instantiate a :class:`DerivativeMethod` given the name *name*.

    :arg d: a fractional operator that should be discretized by the method. If
        the method does not support this operator, it can fail.
    """
    if name not in REGISTERED_METHODS:
        raise ValueError(
            "Unknown differentiation method '{}'. Known methods are '{}'".format(
                name, "', '".join(REGISTERED_METHODS)
            )
        )

    if not isinstance(d, FractionalOperator):
        d = CaputoDerivative(order=d, side=Side.Left)

    return REGISTERED_METHODS[name](d)


def guess_method_for_derivative(
    p: Points,
    d: float | FractionalOperator,
) -> DerivativeMethod:
    """Construct a :class:`DerivativeMethod` for the given points *p* and
    derivative *d*.

    Note that in general not all methods support arbitrary sets of points or
    arbitrary orders, so specialized methods must be chosen. This function is
    mean to make a reasonable guess at a high-order method. If other properties
    are required (e.g. stability), then a manual choice is better, perhaps
    using :func:`make_diff_from_name`.

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
    "diff",
    "guess_method_for_derivative",
    "make_method_from_name",
    "register_method",
)
