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
from pycaputo.quadrature import QuadratureMethod
from pycaputo.utils import Array, ArrayOrScalarFunction, ScalarFunction


def diff(
    f: ArrayOrScalarFunction,
    p: Points,
    d: FractionalOperator | float | None = None,
    *,
    method: DerivativeMethod | None = None,
) -> Array:
    """Compute the fractional-order derivative of *f* at the points *p*.

    :arg f: an array or callable to compute the derivative of. If this is an
        array, it is assumed that it is evaluated at every point in *p*.
    :arg p: a set of points at which to compute the derivative.
    :arg d: a fractional operator. If this is just a number, the standard
        Caputo derivative will be used.
    :arg method: the name of the method to use. If *None*, an appropriate method
        is select, otherwise the string should directly reference a class
        from :ref:`sec-differentiation`.
    """
    from pycaputo.differentiation import diff as _diff
    from pycaputo.differentiation import guess_method_for_order

    if method is None:
        if d is None:
            raise ValueError("'d' is required if 'method is not given")

        if not isinstance(d, FractionalOperator):
            d = CaputoDerivative(d, side=Side.Left)

        m = guess_method_for_order(p, d)
    elif isinstance(method, DerivativeMethod):
        if d is not None:
            raise ValueError("Cannot provide both order 'alpha' and 'method'")

        m = method
    else:
        raise TypeError(f"'method' has unsupported type: {type(method).__name__!r}")

    return _diff(m, f, p)


def quad(
    f: ArrayOrScalarFunction,
    p: Points,
    d: FractionalOperator | float | None = None,
    *,
    method: QuadratureMethod | None = None,
) -> Array:
    """Compute the fractional-order integral of *f* at the points *p*.

    :arg f: an array or callable to compute the integral of. If this is an
        array, it is assumed that it is evaluated at every point in *p*.
    :arg p: a set of points at which to compute the integral.
    :arg d: a fractional operator. If this is just a number, the standard
        Riemann-Liouville integral will be used.
    :arg method: the name of the method to use. If *None*, an appropriate method
        is select, otherwise the string should directly reference a class
        from :ref:`sec-quadrature`.
    """
    from pycaputo.quadrature import guess_method_for_order
    from pycaputo.quadrature import quad as _quad

    if method is None:
        if d is None:
            raise ValueError("'d' is required if 'method' is not given")

        if not isinstance(d, FractionalOperator):
            d = RiemannLiouvilleDerivative(d, side=Side.Left)

        m = guess_method_for_order(p, d)
    elif isinstance(method, QuadratureMethod):
        if d is not None:
            raise ValueError("Cannot provide both 'd' and 'method'")

        m = method
    else:
        raise TypeError(f"'method' has unsupported type: {type(method).__name__!r}")

    return _quad(m, f, p)


def grad(
    f: ScalarFunction,
    p: Points,
    x: Array,
    a: Array | None = None,
    d: FractionalOperator | float | None = None,
    *,
    method: DerivativeMethod | None = None,
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

    from pycaputo.differentiation import diff as _diff
    from pycaputo.differentiation import guess_method_for_order

    if method is None:
        if d is None:
            raise ValueError("'d' is required if 'method is not given")

        if not isinstance(d, FractionalOperator):
            d = CaputoDerivative(d, side=Side.Left)

        m = guess_method_for_order(p, d)
    elif isinstance(method, DerivativeMethod):
        if d is not None:
            raise ValueError("Cannot provide both 'd' and 'method'")

        m = method
    else:
        raise TypeError(f"'method' has unsupported type: {type(method).__name__!r}")

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

    result = np.empty_like(x)
    for i in np.ndindex(x.shape):
        # FIXME: this should just compute the gradient at -1
        result[i] = _diff(m, make_component_f(i), make_component_p(i))[-1]

    return result


__all__ = ("diff", "grad", "quad")
