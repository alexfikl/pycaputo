# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import singledispatch
from typing import Iterator

import numpy as np

from pycaputo.derivatives import CaputoDerivative, Side
from pycaputo.grid import Points, UniformMidpoints, UniformPoints
from pycaputo.logging import get_logger
from pycaputo.utils import Array, ArrayOrScalarFunction

logger = get_logger(__name__)

# {{{ interface


@dataclass(frozen=True)
class DerivativeMethod(ABC):
    """A generic method used to evaluate a fractional derivative at a set of points."""

    @property
    def name(self) -> str:
        """An identifier for the method."""
        return type(self).__name__.replace("Method", "")

    @property
    @abstractmethod
    def order(self) -> float:
        """Expected order of convergence of the method."""

    def supports(self, alpha: float) -> bool:
        """
        :returns: *True* if the method supports computing the fractional
            order derivative of order *alpha* and *False* otherwise.
        """
        return False


@singledispatch
def weights(m: DerivativeMethod, p: Points, n: int | None = None) -> Iterator[Array]:
    r"""Evaluates the weights used in the derivative approximation.

    In general, a fractional order derivative is a convolution with a given
    kernel (e.g. a :math:`\log` kernel for the
    :class:`~pycaputo.derivatives.HadamardDerivative`) and can be written as

    .. math::

        D^\alpha[f](x_n) = \sum_{k = 0}^n w_{nk} f(x_k),

    where the convolution weights are given by :math:`w_{nk}`. This is not the
    case for all methods, so this does not need to be implemented.

    :arg n: if given, gives the weights for only for :math:`w_{nk}`.
    :returns: weights for computing the derivative approximations at each point
        in *p* except at :math:`x_0`.
    """
    raise NotImplementedError(f"Weights for method '{type(m).__name__}'")


@singledispatch
def diff(m: DerivativeMethod, f: ArrayOrScalarFunction, p: Points) -> Array:
    """Evaluate the fractional derivative of *f* at *x*.

    Note that not all numerical methods can evaluate the derivative at all
    points in *x*. For example, the :class:`CaputoL1Method` cannot evaluate the
    derivative at ``x[0]``.

    :arg m: method used to evaluate the derivative.
    :arg f: a simple function for which to evaluate the derivative.
    :arg p: an array of points at which to evaluate the derivative.
    """
    raise NotImplementedError(
        f"Cannot evaluate derivative with method '{type(m).__name__}'"
    )


@dataclass(frozen=True)
class CaputoDerivativeMethod(DerivativeMethod):
    """A method used to evaluate a :class:`~pycaputo.derivatives.CaputoDerivative`."""

    #: The type of the Caputo derivative.
    d: CaputoDerivative


# }}}


# {{{ Caputo L1 Method


@dataclass(frozen=True)
class CaputoL1Method(CaputoDerivativeMethod):
    r"""Implements the L1 method for the Caputo fractional derivative
    of order :math:`\alpha \in (0, 1)`.

    This method is defined in Section 4.1.1 (II) from [Li2020]_ for general
    non-uniform grids. Note that it cannot compute the derivative at the
    starting point, i.e. :math:`D_C^\alpha[f](a)` is undefined.

    This method is of order :math:`\mathcal{O}(h^{2 - \alpha})`.
    """

    if __debug__:

        def __post_init__(self) -> None:
            if not self.supports(self.d.order):
                raise ValueError(
                    f"'{type(self).__name__}' supports orders in (0, 1): "
                    f"got order '{self.d.order}'"
                )

    @property
    def order(self) -> float:
        return 2 - self.d.order

    def supports(self, alpha: float) -> bool:
        return 0 < alpha < 1


@weights.register(CaputoL1Method)
def _weights_l1_method(m: CaputoL1Method, p: Points) -> Iterator[Array]:
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

    for n, w in enumerate(weights(m, p)):
        df[n + 1] = np.sum(w * dfx[: n + 1])

    return df


@dataclass(frozen=True)
class CaputoModifiedL1Method(CaputoL1Method):
    r"""Implements the modified L1 method for the Caputo fractional derivative
    of order :math:`\alpha \in (0, 1)`.

    This method is defined in Section 4.1.1 (III) from [Li2020]_ for quasi-uniform
    grids. Note that it cannot compute the derivative at the starting point, i.e.
    :math:`D_C^\alpha[f](a)` is undefined.

    This method is of order :math:`\mathcal{O}(h^{2 - \alpha})`.
    """


@weights.register(CaputoModifiedL1Method)
def _weights_modified_l1method(m: CaputoModifiedL1Method, p: Points) -> Iterator[Array]:
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

    for n, w in enumerate(weights(m, p)):
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

    This method is of order :math:`\mathcal{O}(h^{3 - \alpha})`.
    """

    if __debug__:

        def __post_init__(self) -> None:
            if not self.supports(self.d.order):
                raise ValueError(
                    f"'{type(self).__name__}' supports orders in (1, 2): "
                    f"got order '{self.d.order}'"
                )

    @property
    def order(self) -> float:
        return 1

    def supports(self, alpha: float) -> bool:
        return 1 < alpha < 2


def l2uweights(alpha: float, i: int | Array, k: int | Array) -> Array:
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
            df[n] = w0 * np.sum(l2uweights(alpha, n, k[:n]) * ddf[:n])
    else:
        raise NotImplementedError(
            f"'{type(m).__name__}' not implemented for '{type(p).__name__}' grids"
        )

    return df


@dataclass(frozen=True)
class CaputoL2CMethod(CaputoL2Method):
    r"""Implements the L2C method for the Caputo fractional derivative
    of order :math:`\alpha \in (1, 2)`.

    This method is defined in Section 4.1.2 from [Li2020]_. Note that
    it cannot compute the derivative at the starting point, i.e.
    :math:`D_C^\alpha[f](a)` is undefined.

    This method is of order :math:`\mathcal{O}(h^{3 - \alpha})`.
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
            df[n] = w0 * np.sum(l2uweights(alpha, n, k[:n]) * ddf[:n])
    else:
        raise NotImplementedError(
            f"'{type(m).__name__}' not implemented for '{type(p).__name__}'"
        )

    return df


# }}}


# {{{ Caputo Spectral Method


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
    """

    @property
    def name(self) -> str:
        return "CSpec"

    @property
    def order(self) -> float:
        return np.inf


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

# {{{ make


REGISTERED_METHODS: dict[str, type[DerivativeMethod]] = {
    "CaputoL1Method": CaputoL1Method,
    "CaputoL2CMethod": CaputoL2CMethod,
    "CaputoL2Method": CaputoL2Method,
    "CaputoModifiedL1Method": CaputoModifiedL1Method,
    "CaputoSpectralMethod": CaputoSpectralMethod,
}


def make_diff_from_name(
    name: str,
    order: float,
    *,
    side: Side = Side.Left,
) -> DerivativeMethod:
    if name not in REGISTERED_METHODS:
        raise ValueError(
            "Unknown differentiation method '{}'. Known methods are '{}'".format(
                name, "', '".join(REGISTERED_METHODS)
            )
        )

    d = CaputoDerivative(order=order, side=side)
    method = REGISTERED_METHODS[name](d)  # type: ignore[call-arg]

    if not method.supports(order):
        raise ValueError(f"Method '{name}' does not support derivative order '{order}'")

    return method


# }}}
