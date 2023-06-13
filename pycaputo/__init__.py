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

    :arg f: an
    """
    import pycaputo.differentiation as pyd

    if method is None:
        m = pyd.guess_method_for_order(p, alpha)
    else:
        m = pyd.make_method_from_name(method, alpha)

    return pyd.diff(m, f, p)


__all__ = ("diff", "quad")
