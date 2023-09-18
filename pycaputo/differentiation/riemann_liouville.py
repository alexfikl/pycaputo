# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

from pycaputo.derivatives import CaputoDerivative, RiemannLiouvilleDerivative
from pycaputo.grid import Points
from pycaputo.logging import get_logger
from pycaputo.utils import Array, ArrayOrScalarFunction

from .base import DerivativeMethod, diff
from .caputo import CaputoDerivativeMethod

logger = get_logger(__name__)


@dataclass(frozen=True)
class RiemannLiouvilleDerivativeMethod(DerivativeMethod):
    """A method used to evaluate a
    :class:`~pycaputo.derivatives.RiemannLiouvilleDerivative`."""

    #: A Riemann-Liouville derivative to discretize.
    d: RiemannLiouvilleDerivative

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()
            if not isinstance(self.d, RiemannLiouvilleDerivative):
                raise TypeError(
                    "Expected a Riemann-Liouville derivative:"
                    f" '{type(self.d).__name__}'"
                )


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

    #: A Riemann-Liouville derivative to discretize
    d: RiemannLiouvilleDerivative

    @property
    def caputo(self) -> CaputoDerivativeMethod:
        raise AttributeError

    @property
    def order(self) -> float:
        return self.caputo.order

    def supports(self, alpha: float) -> bool:
        return self.caputo.supports(alpha)


@dataclass(frozen=True)
class RiemannLiouvilleL1Method(RiemannLiouvilleFromCaputoDerivativeMethod):
    r"""A discretization based on :class:`~pycaputo.differentiation.CaputoL1Method`.

    This method is of order :math:`\mathcal{O}(h^{2 - \alpha})` and supports
    arbitrary grids.
    """

    @cached_property
    def caputo(self) -> CaputoDerivativeMethod:
        from .caputo import CaputoL1Method

        d = CaputoDerivative(self.d.order, self.d.side)
        return CaputoL1Method(d)


@dataclass(frozen=True)
class RiemannLiouvilleL2Method(RiemannLiouvilleFromCaputoDerivativeMethod):
    r"""A discretization based on :class:`~pycaputo.differentiation.CaputoL2Method`.

    This method is of order :math:`\mathcal{O}(h^{3 - \alpha})` and supports
    uniform grids.
    """

    @cached_property
    def caputo(self) -> CaputoDerivativeMethod:
        from .caputo import CaputoL2Method

        d = CaputoDerivative(self.d.order, self.d.side)
        return CaputoL2Method(d)


@dataclass(frozen=True)
class RiemannLiouvilleL2CMethod(RiemannLiouvilleFromCaputoDerivativeMethod):
    r"""A discretization based on :class:`~pycaputo.differentiation.CaputoL2CMethod`.

    This method is of order :math:`\mathcal{O}(h^{3 - \alpha})` and supports
    uniform grids.
    """

    @cached_property
    def caputo(self) -> CaputoDerivativeMethod:
        from .caputo import CaputoL2CMethod

        d = CaputoDerivative(self.d.order, self.d.side)
        return CaputoL2CMethod(d)


@diff.register(RiemannLiouvilleL1Method)
@diff.register(RiemannLiouvilleL2Method)
@diff.register(RiemannLiouvilleL2CMethod)
def _diff_rl_from_caputo(
    m: RiemannLiouvilleFromCaputoDerivativeMethod,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    alpha = m.d.order
    x = p.x
    fx = f(x) if callable(f) else f

    df = diff(m.caputo, fx, p)
    df[1:] += fx[0] / (x[1:] - x[0]) ** alpha / math.gamma(1 - alpha)

    if 0.0 < alpha <= 1.0:
        # NOTE: handled above
        pass
    elif 1.0 < alpha <= 2.0:
        # NOTE: this is a 3rd order approximation of the derivative at f(a)
        # that works on non-uniform grids too (derived by Mathematica)
        # fmt: off
        dfx = (
            (fx[0] - fx[1]) / (x[0] - x[1])
            + (fx[0] - fx[2]) / (x[0] - x[2])
            + (fx[0] - fx[3]) / (x[0] - x[3])
            - (fx[1] - fx[2]) / (x[1] - x[2])
            - (fx[1] - fx[3]) * (x[0] - x[2]) / ((x[1] - x[2]) * (x[1] - x[3]))
            + (fx[2] - fx[3]) * (x[0] - x[1]) / ((x[1] - x[2]) * (x[2] - x[3]))
        )
        # fmt: on

        df[1:] += dfx / (x[1:] - x[0]) ** (alpha - 1) / math.gamma(2 - alpha)
    else:
        raise AssertionError

    return df


# }}}
