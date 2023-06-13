# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from pycaputo.grid import Points
from pycaputo.utils import Array, ArrayOrScalarFunction


def diff(
    f: ArrayOrScalarFunction,
    p: Points,
    alpha: float,
    *,
    method: str | None = None,
) -> Array:
    """Compute the fractional-order derivative of *f* at the points *p*.

    :arg f: an array or callable to compute the derivative of. If this is an
        array, it is assumed that it is evaluated at every point in *p*.
    :arg p: a set of points at which to compute the derivative.
    :arg alpha: the order of the fractional derivative.
    :arg method: the name of the method to use. If *None*, an appropriate method
        is select, otherwise the string should directly reference a class
        from :ref:`sec-differentiation`.
    """
    import pycaputo.differentiation as pyd

    if method is None:
        m = pyd.guess_method_for_order(p, alpha)
    else:
        m = pyd.make_method_from_name(method, alpha)

    return pyd.diff(m, f, p)


def quad(
    f: ArrayOrScalarFunction,
    p: Points,
    alpha: float,
    *,
    method: str | None = None,
) -> Array:
    """Compute the fractional-order integral of *f* at the points *p*.

    :arg f: an array or callable to compute the integral of. If this is an
        array, it is assumed that it is evaluated at every point in *p*.
    :arg p: a set of points at which to compute the integral.
    :arg alpha: the order of the fractional integral.
    :arg method: the name of the method to use. If *None*, an appropriate method
        is select, otherwise the string should directly reference a class
        from :ref:`sec-quadrature`.
    """
    import pycaputo.quadrature as pyq

    if method is None:
        m = pyq.guess_method_for_order(p, alpha)
    else:
        m = pyq.make_method_from_name(method, alpha)

    return pyq.quad(m, f, p)


__all__ = ("diff", "quad")
