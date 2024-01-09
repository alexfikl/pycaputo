# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pycaputo.derivatives import RiemannLiouvilleDerivative
from pycaputo.grid import Points
from pycaputo.utils import Array, ArrayOrScalarFunction, DifferentiableScalarFunction

from .base import QuadratureMethod, quad


@dataclass(frozen=True)
class RiemannLiouvilleMethod(QuadratureMethod):
    """Quadrature method for the Riemann-Liouville integral."""

    #: Description of the integral that is approximated.
    d: RiemannLiouvilleDerivative

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()
            if not isinstance(self.d, RiemannLiouvilleDerivative):
                raise TypeError(
                    f"Expected a Riemann-Liouville integral: '{type(self.d).__name__}'"
                )


# {{{ RiemannLiouvilleRectangularMethod


@dataclass(frozen=True)
class RiemannLiouvilleRectangularMethod(RiemannLiouvilleMethod):
    r"""Riemann-Liouville integral approximation using the rectangular formula.

    The rectangular formula is derived in Section 3.1.(I) from [Li2020]_. It
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


# }}}


# {{{ RiemannLiouvilleTrapezoidalMethod


@dataclass(frozen=True)
class RiemannLiouvilleTrapezoidalMethod(RiemannLiouvilleMethod):
    r"""Riemann-Liouville integral approximation using the trapezoidal formula.

    The trapezoidal formula is derived in Section 3.1.(II) from [Li2020]_. It
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
        # NOTE: this expressions match the Mathematica result
        for n in range(1, qf.size):
            dl, dr = x[n] - x[:n], x[n] - x[1 : n + 1]
            wl = dr ** (1 + alpha) + dl**alpha * (alpha * p.dx[:n] - dr)
            wr = dl ** (1 + alpha) - dr**alpha * (alpha * p.dx[:n] + dl)

            qf[n] = w0 * np.sum((wl * fx[:n] + wr * fx[1 : n + 1]) / p.dx[:n])

    return qf


# }}}


# {{{ RiemannLiouvilleSimpsonMethod


@dataclass(frozen=True)
class RiemannLiouvilleSimpsonMethod(RiemannLiouvilleMethod):
    r"""Riemann-Liouville integral approximation using Simpson's method.

    This method is described in more detail in Section 3.3.(III) of [Li2020]_. It
    uses a quadratic approximation on each subinterval and cannot be used to
    evaluate the value at the starting point, i.e.
    :math:`I_{RL}^\alpha[f](a)` is not defined.

    This method is of order :math:`\mathcal{O}(h^3)` and supports uniform grids.
    """

    @property
    def name(self) -> str:
        return "RLSimpson"

    @property
    def order(self) -> float:
        return 3.0


@quad.register(RiemannLiouvilleSimpsonMethod)
def _quad_rl_simpson(
    m: RiemannLiouvilleSimpsonMethod,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    from pycaputo.grid import UniformPoints

    if not callable(f):
        raise TypeError(f"Input 'f' needs to be a callable: {type(f).__name__}")

    if not isinstance(p, UniformPoints):
        raise TypeError(f"Only uniform points are supported: {type(p).__name__}")

    fx = f(p.x)
    fm = f(p.xm)

    alpha = -m.d.order
    w0 = p.dx[0] ** alpha / math.gamma(3 + alpha)
    indices = np.arange(fx.size)

    # compute integral
    qf = np.empty_like(fx)
    qf[0] = np.nan

    # NOTE: [Li2020] Equation 3.19 and 3.20
    n = indices[1:]
    w = (
        4.0 * (n ** (2 + alpha) - (n - 1) ** (2 + alpha))
        - (2 + alpha) * (3 * n ** (1 + alpha) + (n - 1) ** (1 + alpha))
        + (2 + alpha) * (1 + alpha) * n**alpha
    )
    what = 4.0 * (
        (2 + alpha) * (n ** (1 + alpha) + (n - 1) ** (1 + alpha))
        - 2 * (n ** (2 + alpha) - (n - 1) ** (2 + alpha))
    )
    # add k == 0 and k == n cases so we don't have to care about them
    qf[1:] = (w * fx[0] + what * fm[0]) + (2 - alpha) * fx[1:]

    # add 1 <= k <= n - 1 cases
    for n in range(2, qf.size):  # type: ignore[assignment]
        k = indices[1:n]
        # fmt: off
        w = (
            4 * (
                (n + 1 - k) ** (2 + alpha)
                - (n - 1 - k) ** (2 + alpha))
            - (2 + alpha) * (
                (n + 1 - k) ** (1 + alpha)
                + 6 * (n - k) ** (1 + alpha)
                + (n - 1 - k) ** (1 + alpha))
            )
        what = 4.0 * (
            (2 + alpha) * (
                (n - k) ** (1 + alpha)
                + (n - 1 - k) ** (1 + alpha))
            - 2 * (
                (n - k) ** (2 + alpha)
                - (n - 1 - k) ** (2 + alpha))
        )
        # fmt: on

        qf[n] += np.sum(w * fx[1:n]) + np.sum(what * fm[1:n])

    return np.array(w0 * qf)


# }}}


# {{{ RiemannLiouvilleCubicHermiteMethod


@dataclass(frozen=True)
class RiemannLiouvilleCubicHermiteMethod(RiemannLiouvilleMethod):
    r"""Riemann-Liouville integral approximation using a cubic Hermite interpolant.

    This method is described in more detail in Section 3.3.(B) of [Li2020]_. It
    uses a cubic approximation on each subinterval and cannot be used to
    evaluate the value at the starting point, i.e.
    :math:`I_{RL}^\alpha[f](a)` is not defined.

    Note that Hermite interpolants require derivative values at the grid points.

    This method is of order :math:`\mathcal{O}(h^4)` and supports uniform grids.
    """

    @property
    def name(self) -> str:
        return "RLCHermite"

    @property
    def order(self) -> float:
        return 4.0


@quad.register(RiemannLiouvilleCubicHermiteMethod)
def _quad_rl_cubic_hermite(
    m: RiemannLiouvilleCubicHermiteMethod,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    from pycaputo.grid import UniformPoints

    if not isinstance(f, DifferentiableScalarFunction):
        raise TypeError(f"Input 'f' needs to be a callable: {type(f).__name__}")

    if not isinstance(p, UniformPoints):
        raise TypeError(f"Only uniform points are supported: {type(p).__name__}")

    fx = f(p.x, d=0)
    fp = f(p.x, d=1)

    alpha = -m.d.order
    h = p.dx[0]
    w0 = h**alpha / math.gamma(4 + alpha)
    indices = np.arange(fx.size)

    # compute integral
    qf = np.empty_like(fx)
    qf[0] = np.nan

    # NOTE: [Li2020] Equation 3.29 and 3.30
    n = indices[1:]
    w = n**alpha * (
        12 * n**3 - 6 * (3 + alpha) * n**2 + (1 + alpha) * (2 + alpha) * (3 + alpha)
    ) - 6 * (n - 1) ** (2 + alpha) * (1 + 2 * n + alpha)
    what = n ** (1 + alpha) * (
        6 * n**2 - 4 * (3 + alpha) * n + (2 + alpha) * (3 + alpha)
    ) - 2 * (n - 1) ** (2 + alpha) * (3 * n + alpha)
    qf[1:] = (
        w * fx[0]
        + what * h * fp[0]
        + 6.0 * (1 + alpha) * fx[1:]
        - 2 * alpha * h * fp[1:]
    )

    for n in range(2, qf.size):  # type: ignore[assignment]
        k = indices[1:n]
        # fmt: off
        w = 6 * (
                4 * (n - k) ** (3 + alpha)
                + (n - k - 1) ** (2 + alpha) * (2 * k - 2 * n - 1 - alpha)
                + (n - k + 1) ** (2 + alpha) * (2 * k - 2 * n + 1 + alpha))
        what = (
            2 * (n - k - 1) ** (2 + alpha) * (3 * k - 3 * n - alpha)
            - (n - k) ** (2 + alpha) * (8 * alpha + 24)
            - 2 * (n - k + 1) ** (2 + alpha) * (3 * k - 3 * n + alpha))
        # fmt: on

        qf[n] += np.sum(w * fx[1:n]) + h * np.sum(what * fp[1:n])

    return np.array(w0 * qf)


# }}}


# {{{ RiemannLiouvilleSpectralMethod: Jacobi polynomials


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


# {{{ RiemannLiouvilleConvolutionMethod: Lubich


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

    w = lubich_bdf_weights(-alpha, m.quad_order, p.size)

    if np.isfinite(m.beta):
        s = lubich_bdf_starting_weights_count(m.quad_order, alpha)
        omegas = lubich_bdf_starting_weights(w, s, alpha, beta=m.beta)

        for n, omega in enumerate(omegas):
            qc = np.sum(w[: n + s][::-1] * fx[: n + s])
            qs = np.sum(omega * fx[: s + 1])

            qf[n + s] = dxa * (qc + qs)
    else:
        for n in range(1, qf.size):
            qf[n] = dxa * np.sum(w[:n][::-1] * fx[:n])

    return qf


# }}}
