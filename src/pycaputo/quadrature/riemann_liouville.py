# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from pycaputo.derivatives import RiemannLiouvilleDerivative, Side
from pycaputo.grid import Points
from pycaputo.utils import Array, ArrayOrScalarFunction, DifferentiableScalarFunction

from .base import QuadratureMethod, quad


@dataclass(frozen=True)
class RiemannLiouvilleMethod(QuadratureMethod):
    """Quadrature method for the Riemann-Liouville integral."""

    alpha: float
    """Order of the Riemann-Liouville integral that is being discretized."""

    if __debug__:

        def __post_init__(self) -> None:
            if self.alpha > 0:
                raise ValueError(f"Positive orders are not supported: {self.alpha}")

    @property
    def d(self) -> RiemannLiouvilleDerivative:
        return RiemannLiouvilleDerivative(self.alpha, side=Side.Left)


# {{{ Rectangular


@dataclass(frozen=True)
class Rectangular(RiemannLiouvilleMethod):
    r"""Riemann-Liouville integral approximation using the rectangular formula.

    The rectangular formula is derived in Section 3.1.(I) from [Li2020]_ for
    general non-uniform grids.
    """

    theta: float = 0.5
    r"""Weight used in the approximation :math:`\theta f_k + (1 - \theta) f_{k + 1}`."""

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


@quad.register(Rectangular)
def _quad_rl_rect(
    m: Rectangular,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    x = p.x
    fx = f(x) if callable(f) else f
    alpha = -m.alpha
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


# {{{ Trapezoidal


@dataclass(frozen=True)
class Trapezoidal(RiemannLiouvilleMethod):
    r"""Riemann-Liouville integral approximation using the trapezoidal formula.

    The trapezoidal formula is derived in Section 3.1.(II) from [Li2020]_ for
    general non-uniform grids.
    """

    @property
    def name(self) -> str:
        return "RLTrap"


@quad.register(Trapezoidal)
def _quad_rl_trap(
    m: Trapezoidal,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    from pycaputo.grid import UniformPoints

    x = p.x
    fx = f(x) if callable(f) else f
    alpha = -m.alpha
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


# {{{ Simpson


@dataclass(frozen=True)
class Simpson(RiemannLiouvilleMethod):
    r"""Riemann-Liouville integral approximation using Simpson's method.

    This method is described in more detail in Section 3.3.(III) of [Li2020]_
    for uniform grids.
    """

    @property
    def name(self) -> str:
        return "RLSimpson"


@quad.register(Simpson)
def _quad_rl_simpson(
    m: Simpson,
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

    alpha = -m.alpha
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


# {{{ CubicHermite


@dataclass(frozen=True)
class CubicHermite(RiemannLiouvilleMethod):
    r"""Riemann-Liouville integral approximation using a cubic Hermite interpolant.

    This method is described in more detail in Section 3.3.(B) of [Li2020]_
    for uniform grids.

    Note that Hermite interpolants require derivative values at the grid points.
    """

    @property
    def name(self) -> str:
        return "RLCubicHermite"


@quad.register(CubicHermite)
def _quad_rl_cubic_hermite(
    m: CubicHermite,
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

    alpha = -m.alpha
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


# {{{ SpectralJacobi


@dataclass(frozen=True)
class SpectralJacobi(RiemannLiouvilleMethod):
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
        return "RLJacobi"


@quad.register(SpectralJacobi)
def _quad_rl_spec(
    m: SpectralJacobi,
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
    for n, Phat in enumerate(jacobi_riemann_liouville_integral(p, -m.alpha)):
        df += fhat[n] * Phat

    return df


# }}}


# {{{ SplineLagrange


@dataclass(frozen=True)
class SplineLagrange(RiemannLiouvilleMethod):
    """Riemann-Lioville integral approximation using the piecewise Lagrange
    spline method from [Cardone2021]_.

    Note that, unlike other methods, this method requires evaluating the function
    :math:`f` at interior points on each interval (based on :attr:`xi`). The
    integral itself, however, is only evaluated at the given points :math:`x`.

    This method has an order that depends on the reference points :attr:`xi` and
    supports arbitrary grids.
    """

    npoints: int
    """Number of points used on each element."""

    if __debug__:

        def __post_init__(self) -> None:
            from warnings import warn

            super().__post_init__()

            if self.npoints > 16:
                warn(
                    "Evaluating Lagrange polynomials of order > 16 might be "
                    "numerically unstable",
                    stacklevel=2,
                )

            if not np.all([0 < xi < 1 for xi in self.xi]):
                raise ValueError(f"Reference nodes are not in [0, 1]: {self.xi}")

    @property
    def name(self) -> str:
        return "RLSpline"

    @cached_property
    def xi(self) -> Array:
        """Reference points used to construct the Lagrange polynomials on each
        element.

        By default, this function constructs the Gauss-Legendre nodes based on
        :attr:`npoints`. This function can be easily overwritten to make use of
        different points. However, they must be in :math:`[0, 1]`.
        """
        from numpy.polynomial.legendre import leggauss

        xi, _ = leggauss(self.npoints)  # type: ignore[no-untyped-call]
        return np.array(xi + 1.0) / 2.0


@quad.register(SplineLagrange)
def _quad_rl_spline(
    m: SplineLagrange,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    from pycaputo.lagrange import lagrange_riemann_liouville_integral

    xi = m.xi
    alpha = -m.alpha

    if not callable(f):
        raise TypeError(
            f"'{type(m).__name__}' only supports callable functions: 'f' is a "
            f"{type(f).__name__}"
        )

    x = p.x[:-1].reshape(-1, 1) + p.dx.reshape(-1, 1) * xi
    fx = f(x)

    qf = np.empty(p.size, dtype=fx.dtype)
    qf[0] = np.nan

    for n, w in enumerate(lagrange_riemann_liouville_integral(p, xi, alpha)):
        qf[n + 1] = np.sum(w * fx[: n + 1])

    return qf


# }}}


# {{{ Lubich


@dataclass(frozen=True)
class Lubich(RiemannLiouvilleMethod):
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

    quad_order: int
    """The order of the convolution quadrature method. Only orders up to 6 are
    currently supported, see :func:`~pycaputo.generating_functions.lubich_bdf_weights`
    for additional details.
    """

    beta: float
    r"""An exponent used in constructing the starting weights of the quadrature.
    Negative values will allow for certain singularities at the origin, while
    a default of :math:`\beta = 1` will benefit a smooth function. Setting
    this to ``float("inf")`` will disable the starting weights.
    """

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


@quad.register(Lubich)
def _quad_rl_conv(
    m: Lubich,
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
    alpha = -m.alpha
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
