# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from pycaputo.derivatives import RiemannLiouvilleDerivative, Side
from pycaputo.grid import Points
from pycaputo.logging import get_logger
from pycaputo.typing import Array, ArrayOrScalarFunction, Scalar

from . import caputo
from .base import DerivativeMethod, diff, diffs, quadrature_weights

log = get_logger(__name__)


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


# {{{ RiemannLiouvilleFromCaputoMethod


@dataclass(frozen=True)
class RiemannLiouvilleFromCaputoMethod(RiemannLiouvilleDerivativeMethod):
    r"""Implements approximations of the Riemann-Liouville derivative based on
    existing approximations of the Caputo derivative. For smooth functions,
    we have that

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
class L1(RiemannLiouvilleFromCaputoMethod):
    r"""A discretization based on the :class:`~pycaputo.differentiation.caputo.L1`
    method.
    """

    @cached_property
    def base(self) -> DerivativeMethod:
        return caputo.L1(self.d.alpha)


@dataclass(frozen=True)
class L2(RiemannLiouvilleFromCaputoMethod):
    r"""A discretization based on the :class:`~pycaputo.differentiation.caputo.L2`
    method.
    """

    @cached_property
    def base(self) -> DerivativeMethod:
        return caputo.L2(self.d.alpha)


@dataclass(frozen=True)
class L2C(RiemannLiouvilleFromCaputoMethod):
    r"""A discretization based on the :class:`~pycaputo.differentiation.caputo.L2C`
    method.
    """

    @cached_property
    def base(self) -> DerivativeMethod:
        return caputo.L2C(self.alpha)


def _rl_d1_boundary_coefficients(x: Array) -> Array:
    # NOTE: this is a 3rd order approximation of the derivative at f(a)
    # that works on non-uniform grids too (derived by Mathematica)

    x0, x1, x2, x3 = x[:4]
    c0 = 1.0 / (x0 - x1) + 1.0 / (x0 - x2) + 1.0 / (x0 - x3)
    c1 = -(x0 - x2) * (x0 - x3) / ((x0 - x1) * (x1 - x2) * (x1 - x3))
    c2 = -(x0 - x1) * (x0 - x3) / ((x0 - x2) * (x2 - x1) * (x2 - x3))
    c3 = -(x0 - x1) * (x0 - x2) / ((x0 - x3) * (x3 - x1) * (x3 - x2))

    return np.array([c0, c1, c2, c3])


def _rl_correction_weights(
    m: RiemannLiouvilleFromCaputoMethod,
    x: Array,
    n: int,
) -> Array:
    xn = x[n]
    alpha = m.alpha
    w0 = 1.0 / (xn - x[0]) ** alpha / math.gamma(1 - alpha)

    if 0.0 < alpha < 1.0:
        w = np.array([w0], dtype=x.dtype)
    elif 1.0 < alpha < 2.0:
        w = (
            _rl_d1_boundary_coefficients(x)
            / (xn - x[0]) ** (alpha - 1)
            / math.gamma(2 - alpha)
        )
        w[0] += w0
    else:
        raise NotImplementedError

    return w


def _rl_correction(m: RiemannLiouvilleFromCaputoMethod, x: Array, fx: Array) -> Array:
    alpha = m.alpha
    df = fx[0] / (x[1:] - x[0]) ** alpha / math.gamma(1 - alpha)

    if 0.0 < alpha <= 1.0:
        # NOTE: handled above
        pass
    elif 1.0 < alpha <= 2.0:
        dfx = _rl_d1_boundary_coefficients(x) @ fx[:4]
        df += dfx / (x[1:] - x[0]) ** (alpha - 1) / math.gamma(2 - alpha)
    else:
        raise NotImplementedError(f"Unsupported derivative order: {alpha}")

    return np.array(df)


@quadrature_weights.register(L1)
@quadrature_weights.register(L2)
@quadrature_weights.register(L2C)
def _quadrature_weights_rl_from_caputo(
    m: RiemannLiouvilleFromCaputoMethod,
    p: Points,
    n: int,
) -> Array:
    w = quadrature_weights(m.base, p, n)

    wc = _rl_correction_weights(m, p.x, n)
    w[: wc.size] += wc

    return w


@diffs.register(L1)
@diffs.register(L2)
@diffs.register(L2C)
def _diffs_rl_from_caputo(
    m: RiemannLiouvilleFromCaputoMethod,
    f: ArrayOrScalarFunction,
    p: Points,
    n: int,
) -> Scalar:
    x = p.x
    fx: Array = f(x[: n + 1]) if callable(f) else f[: n + 1]

    df = diffs(m.base, fx, p, n)

    wc = _rl_correction_weights(m, p.x, n)
    return df + wc @ fx[: wc.size]


@diff.register(L1)
@diff.register(L2)
@diff.register(L2C)
def _diff_rl_from_caputo(
    m: RiemannLiouvilleFromCaputoMethod,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    x = p.x
    fx = f(x) if callable(f) else f

    df = diff(m.base, fx, p)
    df[1:] += _rl_correction(m, p.x, fx)

    return df


# }}}
