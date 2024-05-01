# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

from pycaputo.derivatives import RiemannLiouvilleDerivative, Side
from pycaputo.grid import Points
from pycaputo.logging import get_logger
from pycaputo.utils import Array, ArrayOrScalarFunction

from . import caputo
from .base import DerivativeMethod, diff

logger = get_logger(__name__)


@dataclass(frozen=True)
class RiemannLiouvilleDerivativeMethod(DerivativeMethod):
    """A method used to evaluate a
    :class:`~pycaputo.derivatives.RiemannLiouvilleDerivative`."""

    alpha: float
    """Order of the Riemann-Liouville derivative that is being discretized."""

    if __debug__:

        def __post_init__(self) -> None:
            if self.alpha < 0:
                raise ValueError(f"Negative orders are not supported: {self.alpha}")

    @property
    def d(self) -> RiemannLiouvilleDerivative:
        return RiemannLiouvilleDerivative(self.alpha, side=Side.Left)


# {{{ RiemannLiouvilleFromCaputoDerivativeMethod


@dataclass(frozen=True)
class RiemannLiouvilleFromCaputoDerivativeMethod(RiemannLiouvilleDerivativeMethod):
    r"""Implements approximations of the Riemann-Liouville derivative based on
    existing approximations of the Caputo derivative.

    For smooth functions, we have that

    .. math::

        D_{RL}^\alpha[f](x) = D_{C}^\alpha[f](x)
            + \sum_{k = 0}^{n - 1}
                \frac{(x - a)^{k - \alpha}}{\Gamma(k - \alpha + 1)}
                f^{(k)}(a).

    so any method approximating the Caputo derivative can be repurposed to
    also approximate the Riemann-Liouville derivative with the addition of
    initial terms.
    """

    @property
    def base(self) -> DerivativeMethod:
        raise NotImplementedError


@dataclass(frozen=True)
class L1(RiemannLiouvilleFromCaputoDerivativeMethod):
    r"""A discretization based on the :class:`~pycaputo.differentiation.caputo.L1`
    method.
    """

    @cached_property
    def base(self) -> DerivativeMethod:
        return caputo.L1(self.d.alpha)


@dataclass(frozen=True)
class L2(RiemannLiouvilleFromCaputoDerivativeMethod):
    r"""A discretization based on the :class:`~pycaputo.differentiation.caputo.L2`
    method.
    """

    @cached_property
    def base(self) -> DerivativeMethod:
        return caputo.L2(self.d.alpha)


@dataclass(frozen=True)
class L2C(RiemannLiouvilleFromCaputoDerivativeMethod):
    r"""A discretization based on the :class:`~pycaputo.differentiation.caputo.L2C`
    method.
    """

    @cached_property
    def base(self) -> DerivativeMethod:
        return caputo.L2C(self.alpha)


@diff.register(L1)
@diff.register(L2)
@diff.register(L2C)
def _diff_rl_from_caputo(
    m: RiemannLiouvilleFromCaputoDerivativeMethod,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    alpha = m.alpha
    x = p.x
    fx = f(x) if callable(f) else f

    df = diff(m.base, fx, p)
    df[1:] += fx[0] / (x[1:] - x[0]) ** alpha / math.gamma(1 - alpha)

    if 0.0 < alpha <= 1.0:
        # NOTE: handled above
        pass
    elif 1.0 < alpha <= 2.0:
        # NOTE: this is a 3rd order approximation of the derivative at f(a)
        # that works on non-uniform grids too (derived by Mathematica)
        dfx = (
            # fmt: off
            (fx[0] - fx[1]) / (x[0] - x[1])
            + (fx[0] - fx[2]) / (x[0] - x[2])
            + (fx[0] - fx[3]) / (x[0] - x[3])
            - (fx[1] - fx[2]) / (x[1] - x[2])
            - (fx[1] - fx[3]) * (x[0] - x[2]) / ((x[1] - x[2]) * (x[1] - x[3]))
            + (fx[2] - fx[3]) * (x[0] - x[1]) / ((x[1] - x[2]) * (x[2] - x[3]))
            # fmt: on
        )

        df[1:] += dfx / (x[1:] - x[0]) ** (alpha - 1) / math.gamma(2 - alpha)
    else:
        raise NotImplementedError(f"Unsupported derivative order: {alpha}")

    return df


# }}}
