# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from functools import singledispatch

import numpy as np

from pycaputo.derivatives import CaputoDerivative, Side
from pycaputo.grid import Points
from pycaputo.utils import ScalarFunction

# {{{ interface


@dataclass(frozen=True)
class DerivativeMethod:
    """A generic method used to evaluate a fractional derivative at a point."""


@singledispatch
def evaluate(m: DerivativeMethod, f: ScalarFunction, x: Points) -> np.ndarray:
    """Evaluate the fractional derivative of *f*.

    :arg m: method used to evaluate the derivative.
    :arg f: a simple function for which to evaluate the derivative.
    :arg x: an array of points at which to evaluate the derivative.
    """
    raise NotImplementedError(
        f"Cannot evaluate function with method '{type(m).__name__}'"
    )


# }}}


# {{{ L1Method


@dataclass(frozen=True)
class CaputoL1Method(DerivativeMethod):
    r"""Implements the L1 method for the Caputo fractional derivative
    of order :math:`\alpha in (0, 1)`.

    This method is defined in Section 4.1 from [Li2020]_.
    """

    #: The type of the Caputo derivative.
    d: CaputoDerivative


def make_caputo_l1(order: float, side: Side = Side.Left) -> CaputoL1Method:
    """Construct a :class:`CaputoL1Method` to evaluate derivatives of
    order :math:`order`.

    :arg order: the order of the fractional derivative, which should be in
        :math:`(0, 1)`.
    :arg side: the side of the derivative.
    """
    if not (0 < order < 1):
        raise ValueError(
            "CaputoL1Method only supports order of (0, 1): order is '{order}'"
        )

    return CaputoL1Method(d=CaputoDerivative(order=order))


@evaluate.register(CaputoL1Method)
def _evaluate_l1method(m: CaputoL1Method, f: ScalarFunction, p: Points) -> np.ndarray:
    import math

    from pycaputo.grid import UniformPoints

    x = p.x
    fx = f(x)
    alpha = m.d.order

    df = np.zeros_like(x)
    if isinstance(p, UniformPoints):
        c = p.dx[0] ** alpha * math.gamma(2 - alpha)

        # NOTE: [Li2020] Equation 4.3
        for n in range(1, df.size):
            k = np.arange(n)
            omega = (k + 1) ** (1 - alpha) - k ** (1 - alpha)
            df[n] = np.sum(omega * np.diff(fx[: n + 1])[::-1]) / c
    else:
        c = math.gamma(2 - alpha)

        # NOTE: [Li2020] Equation 4.20
        for n in range(1, df.size - 1):
            omega = (
                (x[n] - x[:n]) ** (1 - alpha) - (x[n] - x[1 : n + 1]) ** (1 - alpha)
            ) / p.dx[:n]
            df[n] = np.sum(omega * np.diff(fx[: n + 1])) / c

    return df


# }}}
