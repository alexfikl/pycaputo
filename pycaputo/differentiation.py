# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from functools import singledispatch

import numpy as np

from pycaputo.derivatives import CaputoDerivative
from pycaputo.utils import ScalarFunction

# {{{ interface


@dataclass(frozen=True)
class DerivativeAlgorithm:
    pass


@singledispatch
def evaluate(alg: DerivativeAlgorithm, f: ScalarFunction, x: np.ndarray) -> np.ndarray:
    raise NotImplementedError(
        f"Cannot evaluate function with algorithm '{type(alg).__name__}'"
    )


# }}}


# {{{ L1Method


@dataclass(frozen=True)
class CaputoL1Algorithm(DerivativeAlgorithm):
    r"""Implements the L1 algorithm for the Caputo fractional derivative
    of order :math:`\alpha in (0, 1)`.

    .. attribute:: d

        A :class:`pycaputo.CaputoDerivative` of order :math:`\alpha`.
    """

    d: CaputoDerivative


def make_caputo_l1_algorithm(order: float) -> CaputoL1Algorithm:
    if not (0 < order < 1):
        raise ValueError(
            "CaputoL1Algorithm only supports order of (0, 1): order is '{order}'"
        )

    return CaputoL1Algorithm(d=CaputoDerivative(order=order))


@evaluate.register(CaputoL1Algorithm)
def _evaluate_l1method(
    alg: CaputoL1Algorithm, f: ScalarFunction, x: np.ndarray
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
