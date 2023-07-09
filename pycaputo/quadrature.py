# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import singledispatch

import numpy as np

from pycaputo.derivatives import RiemannLiouvilleDerivative, Side
from pycaputo.grid import Points
from pycaputo.utils import Array, ArrayOrScalarFunction

# {{{ interface


@dataclass(frozen=True)
class QuadratureMethod(ABC):
    """A generic method used to evaluate a fractional integral."""

    @property
    @abstractmethod
    def name(self) -> str:
        """An identifier for the quadrature method."""

    @property
    @abstractmethod
    def order(self) -> float:
        """Expected order of convergence of the method."""


@singledispatch
def quad(m: QuadratureMethod, f: ArrayOrScalarFunction, x: Points) -> Array:
    """Evaluate the fractional integral of *f* using the points *x*.

    :arg m: method used to evaluate the integral.
    :arg f: a simple function for which to evaluate the integral.
    :arg x: an array of points at which to evaluate the integral.
    """

    raise NotImplementedError(
        f"Cannot evaluate integral with method '{type(m).__name__}'"
    )


# }}}


# {{{ RiemannLiouvilleMethod


@dataclass(frozen=True)
class RiemannLiouvilleMethod(QuadratureMethod):
    """Quadrature method for the Riemann-Liouville integral."""

    #: Description of the integral that is integrated.
    d: RiemannLiouvilleDerivative

    if __debug__:

        def __post_init__(self) -> None:
            if self.d.order >= 0:
                raise ValueError(
                    f"Integral requires a negative order: order is '{self.d.order}'"
                )


@dataclass(frozen=True)
class RiemannLiouvilleRectangularMethod(RiemannLiouvilleMethod):
    r"""Riemann-Liouville integral approximation using the rectangular formula.

    The rectangular formula is derived in Section 3.1 I from [Li2020]_. It
    uses a piecewise constant approximation on each subinterval and cannot
    be used to evaluate the value at the starting point, i.e.
    :math:`I_{RL}^\alpha[f](a)` is not defined.

    This method is of order :math:`\mathcal{O}(h)` and supports arbitrary grids.
    """

    #: Weight used in the approximation :math:`\theta f_k + (1 - \theta) f_{k + 1}`.
    theta: float = 0.5

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()
            if not 0.0 <= self.theta <= 1.0:
                raise ValueError(
                    f"Weight is expected to be in [0, 1]: theta is '{self.theta}'"
                )

    @property
    def name(self) -> str:
        return "RLRect"

    @property
    def order(self) -> float:
        if self.theta == 0.5:
            return min(2.0, 1.0 - self.d.order)

        return 1.0


@quad.register(RiemannLiouvilleRectangularMethod)
def _quad_rl_rect(
    m: RiemannLiouvilleRectangularMethod,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    x = p.x
    fx = f(x) if callable(f) else f
    alpha = -m.d.order
    w0 = 1 / math.gamma(1 + alpha)

    fc = m.theta * fx[:-1] + (1 - m.theta) * fx[1:]

    # compute integral
    qf = np.empty_like(fx)
    qf[0] = np.nan

    for n in range(1, qf.size):
        w = (x[n] - x[:n]) ** alpha - (x[n] - x[1 : n + 1]) ** alpha
        qf[n] = w0 * np.sum(w * fc[:n])

    return qf


@dataclass(frozen=True)
class RiemannLiouvilleTrapezoidalMethod(RiemannLiouvilleMethod):
    r"""Riemann-Liouville integral approximation using the trapezoidal formula.

    The rectangular formula is derived in Section 3.1 II from [Li2020]_. It
    uses a linear approximation on each subinterval and cannot be used to
    evaluate the value at the starting point, i.e.
    :math:`I_{RL}^\alpha[f](a)` is not defined.

    This method is of order :math:`\mathcal{O}(h^2)` and supports arbitrary grids.
    """

    @property
    def name(self) -> str:
        return "RLTrap"

    @property
    def order(self) -> float:
        return 2.0


@quad.register(RiemannLiouvilleTrapezoidalMethod)
def _quad_rl_trap(
    m: RiemannLiouvilleTrapezoidalMethod,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    from pycaputo.grid import UniformPoints

    x = p.x
    fx = f(x) if callable(f) else f
    alpha = -m.d.order
    w0 = 1 / math.gamma(2 + alpha)

    # compute integral
    qf = np.empty_like(fx)
    qf[0] = np.nan

    if isinstance(p, UniformPoints):
        k = np.arange(qf.size)
        w0 = w0 * p.dx[0] ** alpha

        w = k[:-1] ** (1 + alpha) - (k[:-1] - alpha) * k[1:] ** alpha
        qf[1:] = w0 * (w * fx[0] + fx[1:])

        # NOTE: [Li2020] Equation 3.15
        for n in range(2, qf.size):
            w = (
                (n - k[1:n] + 1) ** (1 + alpha)
                - 2 * (n - k[1:n]) ** (1 + alpha)
                + (n - k[1:n] - 1) ** (1 + alpha)
            )

            qf[n] += w0 * np.sum(w * fx[1:n])
    else:
        for n in range(1, qf.size):
            dl, dr = x[n] - x[:n], x[n] - x[1 : n + 1]
            wl = dr ** (1 + alpha) + dl**alpha * (alpha * p.dx[:n] - dr)
            wr = dl ** (1 + alpha) - dr**alpha * (alpha * p.dx[:n] + dl)

            qf[n] = w0 * np.sum((wl * fx[:n] + wr * fx[1 : n + 1]) / p.dx[:n])

    return qf


@dataclass(frozen=True)
class RiemannLiouvilleSpectralMethod(RiemannLiouvilleMethod):
    r"""Riemann-Liouville integral approximation using spectral methods based
    on Jacobi polynomials.

    This method is described in more detail in Section 3.3 of [Li2020]_. It
    approximates the function by projecting it to the Jacobi polynomial basis
    and constructing a quadrature rule, i.e.

    .. math::

        I^\alpha[f](x_j) = I^\alpha[p_N](x_j)
                         = \sum_{k = 0}^N w^\alpha_{jk} \hat{f}_k,

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
        return "RLSpec"

    @property
    def order(self) -> float:
        return np.inf


@quad.register(RiemannLiouvilleSpectralMethod)
def _quad_rl_spec(
    m: RiemannLiouvilleSpectralMethod,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    from pycaputo.grid import JacobiGaussLobattoPoints

    if not isinstance(p, JacobiGaussLobattoPoints):
        raise TypeError(
            f"Only JacobiGaussLobattoPoints points are supported: '{type(p).__name__}'"
        )

    from pycaputo.jacobi import jacobi_project, jacobi_riemann_liouville_integral

    # NOTE: Equation 3.63 [Li2020]
    fx = f(p.x) if callable(f) else f
    fhat = jacobi_project(fx, p)

    df = np.zeros_like(fhat)
    for n, Phat in enumerate(jacobi_riemann_liouville_integral(p, -m.d.order)):
        df += fhat[n] * Phat

    return df


# }}}


# {{{


@dataclass(frozen=True)
class RiemannLiouvilleConvolutionMethod(RiemannLiouvilleMethod):
    r"""Riemann-Liouville integral approximation using the convolution quadratures
    of [Lubich1986]_.

    This method is described in detail in [Lubich1986]_ and allows approximations
    of the form

    .. math::

        I^\alpha[f](x_j) =
            \Delta x^\alpha \sum_{i = 0}^j w_{j - i} f_i
            + \Delta x^\alpha \sum_{i = 0}^s \omega_{ji} f_i

    where :math:`w_j` are referred to as the convolution weights and
    :math:`\omega_{ji}` are referred to as the starting weights. The starting
    weights are used to guaranteed high-order behavior around the origin.

    These quadrature methods are modelled on the Backward Differencing Formulas
    (BDF) and only support orders up to :math:`6`.

    This quadrature method only supports on uniform grids.
    """

    #: The order of the convolution quadrature method. Only orders up to 6 are
    #: currently supported, see
    #: :func:`~pycaputo.generating_functions.lubich_bdf_weights` for additional
    #: details.
    quad_order: int
    #: An exponent used in constructing the starting weights of the quadrature.
    #: Negative values will allow for certain singularities at the origin, while
    #: a default of :math:`\beta = 1` will benefit a smooth function. Setting
    #: this to ``float("inf")`` will disable the starting weights.
    beta: float

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()

            if not 1 <= self.quad_order <= 6:
                raise ValueError(
                    f"Only orders 1 <= q <= 6 are supported: {self.quad_order}"
                )

            if self.beta.is_integer() and self.beta <= 0:
                raise ValueError(
                    f"Values of beta in 0, -1, ... are not supported: {self.beta}"
                )

    @property
    def name(self) -> str:
        return "RLConv"

    @property
    def order(self) -> int:
        return self.quad_order


@quad.register(RiemannLiouvilleConvolutionMethod)
def _quad_rl_conv(
    m: RiemannLiouvilleConvolutionMethod,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    from pycaputo.grid import UniformPoints

    if not isinstance(p, UniformPoints):
        raise TypeError(f"Only uniforms points are supported: '{type(p).__name__}'")

    from pycaputo.generating_functions import (
        lubich_bdf_starting_weights,
        lubich_bdf_starting_weights_count,
        lubich_bdf_weights,
    )

    fx = f(p.x) if callable(f) else f
    alpha = -m.d.order
    dxa = p.dx[0] ** alpha

    qf = np.empty_like(fx)
    qf[0] = np.nan

    w = lubich_bdf_weights(alpha, m.quad_order, p.n)

    if np.isfinite(m.beta):
        s = lubich_bdf_starting_weights_count(m.quad_order, alpha)
        omegas = lubich_bdf_starting_weights(w, s, alpha, beta=m.beta)

        for n, omega in enumerate(omegas):
            qc = np.sum(w[: n + s][::-1] * fx[: n + s])
            qs = np.sum(omega * fx[: s + 1])

            qf[n + s] = dxa * (qc + qs)
    else:
        for n in range(1, qf.size):
            qf[n] = dxa * np.sum(w[: n + s][::-1] * fx[: n + s])

    return qf


# }}}


# {{{ make


REGISTERED_METHODS: dict[str, type[QuadratureMethod]] = {
    "RiemannLiouvilleRectangularMethod": RiemannLiouvilleRectangularMethod,
    "RiemannLiouvilleTrapezoidalMethod": RiemannLiouvilleTrapezoidalMethod,
    "RiemannLiouvilleSpectralMethod": RiemannLiouvilleSpectralMethod,
    "RiemannLiouvilleConvolutionMethod": RiemannLiouvilleConvolutionMethod,
}


def register_method(
    name: str,
    method: type[QuadratureMethod],
    *,
    force: bool = False,
) -> None:
    """Register a new integral approximation method.

    :arg name: a canonical name for the method.
    :arg method: a class that will be used to construct the method.
    :arg force: if *True*, any existing methods will be overwritten.
    """

    if not force and name in REGISTERED_METHODS:
        raise ValueError(
            f"A method by the name '{name}' is already registered. Use 'force=True' to"
            " overwrite it."
        )

    REGISTERED_METHODS[name] = method


def make_method_from_name(
    name: str,
    alpha: float,
    *,
    side: Side = Side.Left,
) -> QuadratureMethod:
    """Instantiate a :class:`QuadratureMethod` given the name *name*.

    :arg alpha: the order of the fractional integral. Not all methods support
        all orders, so this choice may be invalid.
    """
    if name not in REGISTERED_METHODS:
        raise ValueError(
            "Unknown differentiation method '{}'. Known methods are '{}'".format(
                name, "', '".join(REGISTERED_METHODS)
            )
        )

    d = RiemannLiouvilleDerivative(order=alpha, side=side)
    return REGISTERED_METHODS[name](d)  # type: ignore[call-arg]


def guess_method_for_order(
    p: Points,
    alpha: float,
    *,
    side: Side = Side.Left,
) -> QuadratureMethod:
    """Construct a :class:`QuadratureMethod` for the given order
    *alpha* and points *p*.

    Note that in general not all methods support arbitrary sets of points, so
    specialized methods must be chosen.

    :arg alpha: the order of the fractional integral.
    :arg p: a set of points on which to evaluate the fractional integral.
    """
    from pycaputo.grid import JacobiGaussLobattoPoints

    d = RiemannLiouvilleDerivative(order=alpha, side=side)
    m: QuadratureMethod | None = None

    if isinstance(p, JacobiGaussLobattoPoints):
        m = RiemannLiouvilleSpectralMethod(d)
    else:
        m = RiemannLiouvilleTrapezoidalMethod(d)

    if m is None:
        raise ValueError(
            "Cannot determine an adequate method for "
            f"alpha = {alpha} and '{type(p).__name__}'."
        )

    return m


# }}}
