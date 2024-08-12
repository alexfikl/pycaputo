# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo.derivatives import FractionalOperator
from pycaputo.differentiation.base import (
    DerivativeMethod,
    FunctionCallableError,
    diff,
    differentiation_matrix,
    diffs,
    quadrature_weights,
)
from pycaputo.grid import Points
from pycaputo.typing import Array, ArrayOrScalarFunction, Scalar


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
        common (left) Caputo derivative is used.
    """
    from pycaputo import grid
    from pycaputo.derivatives import CaputoDerivative, RiemannLiouvilleDerivative
    from pycaputo.differentiation import caputo
    from pycaputo.differentiation import riemann_liouville as rl

    m: DerivativeMethod | None = None
    if not isinstance(d, FractionalOperator):
        d = CaputoDerivative(alpha=d)

    if isinstance(d, CaputoDerivative):
        if isinstance(p, grid.JacobiGaussLobattoPoints):
            m = caputo.Jacobi(d.alpha)
        elif 0 < d.alpha < 1:
            if isinstance(p, grid.MidPoints):
                m = caputo.ModifiedL1(d.alpha)
            else:
                m = caputo.L1(d.alpha)
        elif 1 < d.alpha < 2 and isinstance(p, grid.UniformPoints):
            m = caputo.L2C(d.alpha)
    elif isinstance(d, RiemannLiouvilleDerivative):
        if 0 < d.alpha < 1:
            m = rl.L1(d.alpha)
        elif 1 < d.alpha < 2 and isinstance(p, grid.UniformPoints):
            m = rl.L2C(d.alpha)

    if m is None:
        raise ValueError(
            "Cannot determine an adequate method for operator "
            f"'{d!r}' and points of type '{type(p).__name__}'."
        )

    return m


def quadrature_weights_fallback(m: DerivativeMethod, p: Points, n: int) -> Array:
    """Evaluate the quadrature weights for the method *m* at the points *p.x[n]*.

    This function attempts to construct the quadrature weights using
    :func:`~pycaputo.differentiation.quadrature_weights`. If not available, it
    falls back to constructing the full differentiation matrix using
    :func:`~pycaputo.differentiation.differentiation_matrix` and selecting
    the *nth* row.

    .. warning::

        Falling back to :func:`~pycaputo.differentiation.differentiation_matrix`
        will be significantly slower, as it required computing the full matrix.
        Use this function with care.
    """

    try:
        return quadrature_weights(m, p, n)
    except NotImplementedError:
        pass

    if not 0 <= n <= p.size:
        raise IndexError(f"Index 'n' out of range: 0 <= {n} < {p.size}")

    W = differentiation_matrix(m, p)
    return W[n, : n + 1]


def diffs_fallback(
    m: DerivativeMethod, f: ArrayOrScalarFunction, p: Points, n: int
) -> Scalar:
    """Evaluate the fractional derivative of *f* using method *m* at the point
    *p.x[n]* using the underlying grid.

    This function attempts to evaluate the fractional derivative using
    :func:`~pycaputo.differentiation.diffs`. If not available, it falls back
    to evaluating the derivative at all points using
    :func:`~pycaputo.differentiation.diff` and selecting the desired point.

    .. warning::

        Falling back to the :func:`~pycaputo.differentiation.diff` function
        will be significantly slower, since all the points on the grid *p* are
        evaluated. Use this function with care.
    """
    try:
        return diffs(m, f, p, n)
    except NotImplementedError:
        pass

    if not 0 <= n <= p.size:
        raise IndexError(f"Index 'n' out of range: 0 <= {n} < {p.size}")

    result = diff(m, f, p)
    return np.array(result[n])


__all__ = (
    "DerivativeMethod",
    "FunctionCallableError",
    "diff",
    "differentiation_matrix",
    "diffs",
    "diffs_fallback",
    "guess_method_for_order",
    "quadrature_weights",
    "quadrature_weights_fallback",
)
