# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo.derivatives import (
    CaputoDerivative,
    FractionalOperator,
    RiemannLiouvilleDerivative,
    Side,
)
from pycaputo.differentiation import DerivativeMethod
from pycaputo.grid import Points
from pycaputo.typing import Array, ArrayOrScalarFunction, ScalarFunction


def diff(
    f: ArrayOrScalarFunction,
    p: Points,
    d: FractionalOperator | float,
) -> Array:
    """Compute the fractional-order derivative of *f* at the points *p*.

    :arg f: an array or callable to compute the derivative of. If this is an
        array, it is assumed that it is evaluated at every point in *p*.
    :arg p: a set of points at which to compute the derivative.
    :arg d: a fractional operator. If this is just a number, the standard
        Caputo derivative will be used.
    """
    from pycaputo.differentiation import diff as _diff
    from pycaputo.differentiation import guess_method_for_order

    if d is None:
        raise ValueError("'d' is required if 'method is not given")

    if not isinstance(d, FractionalOperator):
        d = CaputoDerivative(d, side=Side.Left)

    m = guess_method_for_order(p, d)

    return _diff(m, f, p)


def quad(
    f: ArrayOrScalarFunction,
    p: Points,
    d: FractionalOperator | float,
) -> Array:
    """Compute the fractional-order integral of *f* at the points *p*.

    :arg f: an array or callable to compute the integral of. If this is an
        array, it is assumed that it is evaluated at every point in *p*.
    :arg p: a set of points at which to compute the integral.
    :arg d: a fractional operator. If this is just a number, the standard
        Riemann-Liouville integral will be used.
    """
    from pycaputo.quadrature import guess_method_for_order
    from pycaputo.quadrature import quad as _quad

    if d is None:
        raise ValueError("'d' is required if 'method' is not given")

    if not isinstance(d, FractionalOperator):
        d = RiemannLiouvilleDerivative(d, side=Side.Left)

    m = guess_method_for_order(p, d)

    return _quad(m, f, p)


def grad(
    m: DerivativeMethod,
    f: ScalarFunction,
    p: Points,
    x: Array,
    a: Array | None = None,
) -> Array:
    """Compute the fractional-order gradient of *f* at the points *p*.

    The gradient is computed component by component using :func:`diff`. The
    arguments also have the same meaning.

    :arg p: a set of :class:`~pycaputo.grid.Points` on :math:`[0, 1]` that will
        be linearly transformed to use as a grid for computing the gradient.
        Essentially, this will do :math:`p_i = a_i + (x_i - a_i) * p`.
    :arg x: a set of points at which to compute the gradient.
    :arg a: a set of starting points of the fractional operator, which will be
        computed on :math:`[a_i, x_i]`.
    """
    # {{{ normalize inputs

    if a is None:
        a = np.zeros_like(x)

    if x.shape != a.shape:
        raise ValueError(
            f"Inconsistent values for 'x' and 'a': got shape {x.shape} points but"
            f" shape {a.shape} starts"
        )

    if any(x[i] <= a[i] for i in np.ndindex(x.shape)):
        raise ValueError("Lower limits 'a' must be smaller than 'x'")

    # }}}

    def make_component_f(i: tuple[int, ...]) -> ScalarFunction:
        x_r = x[..., None]
        e_i = np.zeros_like(x_r)
        e_i[i] = 1.0

        def f_i(y: Array) -> Array:
            return f(x_r + (y - x[i]) * e_i)

        return f_i

    def make_component_p(i: tuple[int, ...]) -> Points:
        return p.translate(a[i], x[i])

    from pycaputo.differentiation import diff as _diff

    result = np.empty_like(x)
    for i in np.ndindex(x.shape):
        # FIXME: this should just compute the gradient at -1
        result[i] = _diff(m, make_component_f(i), make_component_p(i))[-1]

    return result


__all__ = ("diff", "grad", "quad")
