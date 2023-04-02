# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from functools import singledispatch

import numpy as np

from pycaputo.derivatives import CaputoDerivative, Side
from pycaputo.utils import ScalarFunction

# {{{ interface


@dataclass(frozen=True)
class DerivativeMethod:
    """A generic method used to evaluate a fractional derivative at a point."""


@singledispatch
def evaluate(alg: DerivativeMethod, f: ScalarFunction, x: np.ndarray) -> np.ndarray:
    """Evaluate the fractional derivative of *f*.

    :arg alg: method used to evaluate the derivative.
    :arg f: a simple function for which to evaluate the derivative.
    :arg x: an array of points at which to evaluate the derivative.
    """
    raise NotImplementedError(
        f"Cannot evaluate function with method '{type(alg).__name__}'"
    )


# }}}


# {{{ L1Method


@dataclass(frozen=True)
class CaputoL1Method(DerivativeMethod):
    r"""Implements the L1 method for the Caputo fractional derivative
    of order :math:`\alpha in (0, 1)`.
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
def _evaluate_l1method(
    alg: CaputoL1Method, f: ScalarFunction, x: np.ndarray
) -> np.ndarray:
    import math

    fx = f(x)
    h = np.diff(x)

    alpha = alg.d.order
    g_alpha = math.gamma(2 - alpha)

    df = np.empty_like(x)
    for i in range(1, df.size - 1):
        b = (
            1
            / (g_alpha * h[:i])
            * ((x[i] - x[:i]) ** (1 - alpha) - (x[i] - x[1 : i + 1]) ** (1 - alpha))
        )
        df[i] = np.sum(b * np.diff(fx[: i + 1]))

    return df


# }}}
