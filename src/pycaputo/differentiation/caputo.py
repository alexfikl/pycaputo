# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from pycaputo.derivatives import CaputoDerivative
from pycaputo.grid import Points, UniformMidpoints, UniformPoints
from pycaputo.logging import get_logger
from pycaputo.utils import Array, ArrayOrScalarFunction

from .base import DerivativeMethod, diff

logger = get_logger(__name__)


@dataclass(frozen=True)
class CaputoDerivativeMethod(DerivativeMethod):
    """A method used to evaluate a :class:`~pycaputo.derivatives.CaputoDerivative`."""

    #: A Caputo derivative to discretize.
    d: CaputoDerivative

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()
            if not isinstance(self.d, CaputoDerivative):
                raise TypeError(
                    f"Expected a Caputo derivative: '{type(self.d).__name__}'"
                )


# {{{ Caputo L1 Method


@dataclass(frozen=True)
class CaputoL1Method(CaputoDerivativeMethod):
    r"""Implements the L1 method for the Caputo fractional derivative
    of order :math:`\alpha \in (0, 1)`.

    This method is defined in Section 4.1.1 (II) from [Li2020]_ for general
    non-uniform grids. Note that it cannot compute the derivative at the
    starting point, i.e. :math:`D_C^\alpha[f](a)` is undefined.

    This method is of order :math:`\mathcal{O}(h^{2 - \alpha})` and supports
    arbitrary grids.
    """

    @property
    def order(self) -> float:
        return 2 - self.d.order

    def supports(self, alpha: float) -> bool:
        return 0 < alpha < 1


def _weights_l1(m: CaputoL1Method, p: Points) -> Iterator[Array]:
    x, dx = p.x, p.dx
    alpha = m.d.order
    w0 = 1 / math.gamma(2 - alpha)

    # NOTE: weights given in [Li2020] Equation 4.20
    if isinstance(p, UniformPoints):
        w0 = w0 / p.dx[0] ** alpha
        k = np.arange(x.size - 1)

        for n in range(1, x.size):
            w = (n - k[:n]) ** (1 - alpha) - (n - k[:n] - 1) ** (1 - alpha)
            yield w0 * w
    else:
        for n in range(1, x.size):
            w = (
                (x[n] - x[:n]) ** (1 - alpha) - (x[n] - x[1 : n + 1]) ** (1 - alpha)
            ) / dx[:n]

            yield w0 * w


@diff.register(CaputoL1Method)
def _diff_l1method(m: CaputoL1Method, f: ArrayOrScalarFunction, p: Points) -> Array:
    dfx = np.diff(f(p.x) if callable(f) else f)

    df = np.empty(p.x.shape, dtype=dfx.dtype)
    df[0] = np.nan

    for n, w in enumerate(_weights_l1(m, p)):
        df[n + 1] = np.sum(w * dfx[: n + 1])

    return df


@dataclass(frozen=True)
class CaputoModifiedL1Method(CaputoL1Method):
    r"""Implements the modified L1 method for the Caputo fractional derivative
    of order :math:`\alpha \in (0, 1)`.

    This method is defined in Section 4.1.1 (III) from [Li2020]_ for quasi-uniform
    grids. Note that it cannot compute the derivative at the starting point, i.e.
    :math:`D_C^\alpha[f](a)` is undefined.

    This method is of order :math:`\mathcal{O}(h^{2 - \alpha})` and requires
    a special grid constructed by :func:`~pycaputo.grid.make_uniform_midpoints`.
    """


def _weights_modified_l1(m: CaputoModifiedL1Method, p: Points) -> Iterator[Array]:
    if not isinstance(p, UniformMidpoints):
        raise NotImplementedError(
            f"'{type(m).__name__}' does not implement 'weights' for"
            f" '{type(p).__name__}' grids"
        )

    # NOTE: weights from [Li2020] Equation 4.51
    # FIXME: this does not use the formula from the book; any benefit to it?
    alpha = m.d.order
    wc = 1 / math.gamma(2 - alpha) / p.dx[-1] ** alpha
    k = np.arange(p.x.size)

    # NOTE: first interval has a size of h / 2 and is weighted differently
    w0 = 2 * ((k[:-1] + 0.5) ** (1 - alpha) - k[:-1] ** (1 - alpha))

    for n in range(1, p.x.size):
        w = (n - k[:n]) ** (1 - alpha) - (n - k[:n] - 1) ** (1 - alpha)
        w[0] = w0[n - 1]

        yield wc * w


@diff.register(CaputoModifiedL1Method)
def _diff_modified_l1method(
    m: CaputoModifiedL1Method, f: ArrayOrScalarFunction, p: Points
) -> Array:
    if not isinstance(p, UniformMidpoints):
        raise NotImplementedError(
            f"'{type(m).__name__}' does not implement 'diff' for '{type(p).__name__}'"
            " grids"
        )

    dfx = np.diff(f(p.x) if callable(f) else f)

    df = np.empty(p.x.size)
    df[0] = np.nan

    for n, w in enumerate(_weights_modified_l1(m, p)):
        df[n + 1] = np.sum(w * dfx[: n + 1])

    return df


# }}}


# {{{ Caputo L2 Method


@dataclass(frozen=True)
class CaputoL2Method(CaputoDerivativeMethod):
    r"""Implements the L2 method for the Caputo fractional derivative
    of order :math:`\alpha \in (1, 2)`.

    This method is defined in Section 4.1.2 from [Li2020]_. Note that
    it cannot compute the derivative at the starting point, i.e.
    :math:`D_C^\alpha[f](a)` is undefined.

    This method is of order :math:`\mathcal{O}(h^{3 - \alpha})` and supports
    only uniform grids.
    """

    @property
    def order(self) -> float:
        return 1

    def supports(self, alpha: float) -> bool:
        return 1 < alpha < 2


def _weights_l2(alpha: float, i: int | Array, k: int | Array) -> Array:
    return np.array((i - k) ** (2 - alpha) - (i - k - 1) ** (2 - alpha))


@diff.register(CaputoL2Method)
def _diff_l2method(m: CaputoL2Method, f: ArrayOrScalarFunction, p: Points) -> Array:
    # precompute variables
    x = p.x
    fx = f(x) if callable(f) else f

    alpha = m.d.order
    w0 = 1 / math.gamma(3 - alpha)

    # NOTE: [Li2020] Section 4.2
    # NOTE: the method is not written as in [Li2020] and has several tweaks:
    # * terms are written as `sum(w * f'')` instead of `sum(w * f)`, which
    #   makes it easier to express w
    # * boundary terms are approximated with a biased stencil.

    df = np.empty_like(x)
    df[0] = np.nan

    if isinstance(p, UniformPoints):
        w0 = w0 / p.dx[0] ** alpha
        k = np.arange(fx.size)

        ddf = np.zeros(fx.size - 1, dtype=fx.dtype)
        ddf[:-1] = fx[2:] - 2 * fx[1:-1] + fx[:-2]
        ddf[-1] = 2 * fx[-1] - 5 * fx[-2] + 4 * fx[-3] - fx[-4]

        for n in range(1, df.size):
            df[n] = w0 * np.sum(_weights_l2(alpha, n, k[:n]) * ddf[:n])
    else:
        raise NotImplementedError(
            f"'{type(m).__name__}' not implemented for '{type(p).__name__}' grids"
        )

    return df


# }}}


# {{{ CaputoL2CMethod


@dataclass(frozen=True)
class CaputoL2CMethod(CaputoL2Method):
    r"""Implements the L2C method for the Caputo fractional derivative
    of order :math:`\alpha \in (1, 2)`.

    This method is defined in Section 4.1.2 from [Li2020]_. Note that
    it cannot compute the derivative at the starting point, i.e.
    :math:`D_C^\alpha[f](a)` is undefined.

    This method is of order :math:`\mathcal{O}(h^{3 - \alpha})` and supports
    only uniform grids.
    """

    @property
    def order(self) -> float:
        return 3 - self.d.order


@diff.register(CaputoL2CMethod)
def _diff_uniform_l2cmethod(
    m: CaputoL2CMethod, f: ArrayOrScalarFunction, p: Points
) -> Array:
    # precompute variables
    x = p.x
    fx = f(x) if callable(f) else f

    alpha = m.d.order
    w0 = 1 / math.gamma(3 - alpha)

    # NOTE: [Li2020] Section 4.2
    df = np.empty_like(x)
    df[0] = np.nan

    if isinstance(p, UniformPoints):
        w0 = w0 / (2 * p.dx[0] ** alpha)
        k = np.arange(fx.size)

        ddf = np.zeros(fx.size - 1, dtype=fx.dtype)
        ddf[1:-1] = (fx[3:] - fx[2:-1]) - (fx[1:-2] - fx[:-3])
        ddf[0] = 3 * fx[0] - 7 * fx[1] + 5 * fx[2] - fx[3]
        ddf[-1] = 3 * fx[-1] - 7 * fx[-2] + 5 * fx[-3] - fx[-4]

        for n in range(1, df.size):
            df[n] = w0 * np.sum(_weights_l2(alpha, n, k[:n]) * ddf[:n])
    else:
        raise NotImplementedError(
            f"'{type(m).__name__}' not implemented for '{type(p).__name__}'"
        )

    return df


# }}}

# {{{ CaputoSpectralMethod


@dataclass(frozen=True)
class CaputoSpectralMethod(CaputoDerivativeMethod):
    r"""Caputo derivative approximation using spectral methods based
    on Jacobi polynomials.

    This method is described in more detail in Section 4.4 of [Li2020]_. It
    approximates the function by projecting it to the Jacobi polynomial basis
    and constructing a quadrature rule, i.e.

    .. math::

        D^\alpha[f](x_j) = D^\alpha[p_N](x_j)
                         = \sum_{k = m}^N w^\alpha_{jk} \hat{f}_k,

    where :math:`p_N` is a degree :math:`N` polynomial approximating :math:`f`.
    Then, :math:`w^\alpha_{jk}` are a set of weights and :math:`\hat{f}_k` are
    the modal coefficients. Here, we approximate the function by the Jacobi
    polynomials :math:`P^{(u, v)}`.

    This method is of the order of the Jacobi polynomials and requires
    a Gauss-Jacobi-Lobatto grid (for the projection :math:`\hat{f}_k`) as
    constructed by :func:`~pycaputo.grid.make_jacobi_gauss_lobatto_points`.
    """

    @property
    def name(self) -> str:
        return "CSpec"

    @property
    def order(self) -> float:
        return np.inf

    def supports(self, alpha: float) -> bool:
        return 0 < alpha < np.inf


@diff.register(CaputoSpectralMethod)
def _diff_jacobi(m: CaputoSpectralMethod, f: ArrayOrScalarFunction, p: Points) -> Array:
    from pycaputo.grid import JacobiGaussLobattoPoints

    if not isinstance(p, JacobiGaussLobattoPoints):
        raise TypeError(
            f"Only JacobiGaussLobattoPoints points are supported: '{type(p).__name__}'"
        )

    from pycaputo.jacobi import jacobi_caputo_derivative, jacobi_project

    # NOTE: Equation 3.63 [Li2020]
    fx = f(p.x) if callable(f) else f
    fhat = jacobi_project(fx, p)

    df = np.zeros_like(fhat)
    for n, Dhat in jacobi_caputo_derivative(p, m.d.order):
        df += fhat[n] * Dhat

    return df


# }}}
