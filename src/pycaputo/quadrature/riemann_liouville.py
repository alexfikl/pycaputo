# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from pycaputo.derivatives import RiemannLiouvilleDerivative, Side
from pycaputo.grid import Points
from pycaputo.typing import (
    Array,
    ArrayOrScalarFunction,
    DifferentiableScalarFunction,
    ScalarFunction,
)

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

    if not isinstance(p, UniformPoints):
        raise TypeError(f"Only uniform points are supported: {type(p).__name__}")

    # FIXME: isinstance(f, DifferentiableScalarFunction) does not work?
    assert isinstance(f, DifferentiableScalarFunction)

    try:
        fx = f(p.x, d=0)
        fp = f(p.x, d=1)
    except TypeError as exc:
        raise TypeError(
            f"{type(m).__name__!r} requires a 'DifferentiableScalarFunction': "
            f"f is a {type(f).__name__!r}"
        ) from exc

    alpha = -m.alpha
    h = p.dx[0]
    w0 = h**alpha / math.gamma(4 + alpha)
    indices = np.arange(fx.size)

    # compute integral
    qf = np.empty_like(fx)
    qf[0] = np.nan

    # NOTE: [Li2020] Equation 3.29 and 3.30
    n = indices[1:]
    w: Array = n**alpha * (
        12 * n**3 - 6 * (3 + alpha) * n**2 + (1 + alpha) * (2 + alpha) * (3 + alpha)
    ) - 6 * (n - 1) ** (2 + alpha) * (1 + 2 * n + alpha)
    what: Array = n ** (1 + alpha) * (
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

        xi, _ = leggauss(self.npoints)
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


# {{{


@dataclass(frozen=True)
class DiffusiveRiemannLiouvilleMethod(RiemannLiouvilleMethod):
    r"""Quadrature method for the Riemann-Liouville integral based on diffusive
    approximations.

    This class of methods is very different from the standard product integration
    rules (e.g. :class:`Rectangular`). For a description, see Section 3.4 in
    [Li2020]_. In general, we assume that the integral can be written as

    .. math::

        D_{RL}^\alpha[f](x) = \int_0^\infty \phi(x, \omega) \,\mathrm{d} \omega

    where the auxiliary function :math:`\phi` has some desired properties and
    can be computed by other methods. Then, a quadrature rule can be constructed
    to evaluate the integral above (see :meth:`nodes_and_weights`).
    """

    @abstractmethod
    def nodes_and_weights(self) -> tuple[Array, Array]:
        r"""Compute the nodes and weights for the quadrature used by the method.

        :returns: a tuple of ``(omega, w)`` of nodes and weights to be used by
            the method.
        """


# }}}


# {{{ YuanAgrawal


def _diffusive_gamma_max_timestep(omega: Array) -> float:
    # NOTE: the time step needs to satisfy `h L < 1`, where L is the Lipschitz
    # constant for `fun`. In this case, that's linear
    #   L = `max(omega^2) = omega[-1]^2`
    return float(1.0 / np.max(omega) ** 2)


def _diffusive_gamma_solve_ivp(
    m: DiffusiveRiemannLiouvilleMethod,
    f: ScalarFunction,
    p: Points,
    omega: Array,
    *,
    method: str = "Radau",
    qtol: float = 1.0,
) -> Array:
    from scipy.integrate import solve_ivp

    alpha = -m.alpha
    omega_a = omega ** (1 - 2 * alpha)
    omega_b = -(omega**2)
    omega_jac = np.diag(omega_b)

    def fun(t: Array, phi: Array) -> Array:
        return np.array(omega_a * f(t) + omega_b * phi)

    def fun_jac(t: float, phi: Array) -> Array:
        return omega_jac

    phi0 = np.zeros_like(omega)
    result = solve_ivp(
        fun,
        (p.a, p.b),
        phi0,
        method=method,
        t_eval=p.x[1:],
        jac=fun_jac,
        # NOTE: qtol is used to further decrease the error based on the expected
        # quadrature error in the method.
        rtol=1.0e-3 * qtol,
        atol=1.0e-6 * qtol,
    )

    return np.array(result.y)


@dataclass(frozen=True)
class YuanAgrawal(DiffusiveRiemannLiouvilleMethod):
    r"""Riemann-Liouville integral approximation using the diffusive approximation
    from [Yuan2002]_.

    For this method, the auxiliary variable :math:`\phi` satisfies the following
    ordinary differential equation

    .. math::

        \frac{\partial \phi}{\partial \xi}(\xi; \omega) =
            \omega^{1 - 2 \alpha} f(\xi)
            - \omega^2 \phi(\xi; \omega).

    This method is described in Section 3 of [Yuan2002]_. It is slightly
    modified here by using the generalized Gauss-Laguerre quadrature rule
    (see [Diethelm2008]_).

    This method is only valid for :math:`0 < \alpha < 1`.
    """

    method: str
    """Numerical method used to solve the initial value problems for the
    diffusive representation. This method is passed to
    :func:`scipy.integrate.solve_ivp`.
    """

    quad_order: int
    """Order of the quadrature method used in the approximation
    (see :meth:`nodes_and_weights`)."""

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()

            if self.alpha <= -1:
                raise ValueError(
                    f"The {type(self).__name__!r} method is only valid for "
                    f"-1 < alpha < 0: {self.alpha}"
                )

            from scipy.integrate._ivp.ivp import METHODS  # noqa: PLC2701

            if self.method not in METHODS:
                raise ValueError(
                    "Unsupported method: '{}'. Known methods are: '{}'".format(
                        self.method, "', '".join(METHODS)
                    )
                )

    @property
    def _qtol(self) -> float:
        # NOTE: Theorem 4 in [Diethelm2008] gives the estimate quadrature error
        alphabar = 1.0 + 2 * self.alpha
        return float(0.75 * self.quad_order ** (alphabar - 1))

    def nodes_and_weights(self) -> tuple[Array, Array]:
        from scipy.special import roots_genlaguerre

        # get Gauss-Laguerre quadrature
        alpha = -self.alpha
        beta = 1.0 - 2.0 * alpha

        omega, w = roots_genlaguerre(self.quad_order, beta)

        # transform for Yuan-Agrawal method
        fac = 2.0 * np.sin(alpha * np.pi) / np.pi
        w = fac * omega ** (-beta) * np.exp(omega) * w

        return omega, w


@quad.register(YuanAgrawal)
def _quad_rl_yuan_agrawal(
    m: YuanAgrawal,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    if not callable(f):
        raise TypeError(
            f"{type(m).__name__!r} requires a callable: f is a {type(f).__name__!r}"
        )

    x = p.x
    dtype = np.array(f(p.x[0])).dtype

    # solve ODE at quadrature nodes
    omega, w = m.nodes_and_weights()
    phi = _diffusive_gamma_solve_ivp(m, f, p, omega, method=m.method, qtol=m._qtol)

    # compute RL integral
    qf = np.empty_like(x, dtype=dtype)
    qf[0] = np.nan
    qf[1:] = np.einsum("i,ij->j", w, phi)

    return qf


# }}}


# {{{ BirkSong


@dataclass(frozen=True)
class Diethelm(YuanAgrawal):
    r"""Riemann-Liouville integral approximation using the diffusive approximation
    from [Diethelm2008]_.

    This method uses the weights

    .. math::

        (1 - \omega)^{1 - 2 \alpha} (1 + \omega)^{2 \alpha - 1}.

    This method is only valid for :math:`0 < \alpha < 1`.
    """

    @property
    def _qtol(self) -> float:
        # FIXME: in this case, the error should be spectral, so it's not clear what
        # to use here. This seems to work well for the test case
        alphabar = 1.0 + 2 * self.alpha
        return float(0.75 * self.quad_order ** (alphabar - 3))

    def nodes_and_weights(self) -> tuple[Array, Array]:
        from scipy.special import roots_jacobi

        # get Gauss-Jacobi quadrature rule
        alpha = -self.alpha
        alphabar = 1 - 2.0 * alpha
        beta = alphabar
        gamma = -alphabar

        omega, w = roots_jacobi(self.quad_order, beta, gamma)

        # transform for Diethelm method
        fac = 4.0 * np.sin(alpha * np.pi) / np.pi
        w = fac * w / (1 - omega) ** beta / (1 + omega) ** (gamma + 2)
        omega = (1 - omega) / (1 + omega)

        return omega, w


# }}}


# {{{ Birk-Song


@dataclass(frozen=True)
class BirkSong(YuanAgrawal):
    r"""Riemann-Liouville integral approximation using the diffusive approximation
    from [Birk2010]_.

    This method uses the weights

    .. math::

        (1 - \omega)^{\bar{\alpha}} (1 + \omega)^{-\bar{\alpha}},

    where :math:`\bar{\alpha} = 1 - 2 \alpha`.

    This method is only valid for :math:`0 < \alpha < 1`.
    """

    @property
    def _qtol(self) -> float:
        # FIXME: in this case, the error should be spectral, so it's not clear what
        # to use here. This seems to work well for the test case
        alphabar = 1.0 + 2 * self.alpha
        return float(0.75 * self.quad_order ** (alphabar - 3))

    def nodes_and_weights(self) -> tuple[Array, Array]:
        from scipy.special import roots_jacobi

        # get Gauss-Jacobi quadrature rule
        alpha = -self.alpha
        alphabar = 1 - 2.0 * alpha
        beta = 2 * alphabar + 1
        gamma = -(2 * alphabar - 1)

        omega, w = roots_jacobi(self.quad_order, beta, gamma)

        # transform for BirkSong method
        fac = 8.0 * np.sin(alpha * np.pi) / np.pi
        w = fac * w / (1 - omega) ** (beta - 1) / (1 + omega) ** (gamma + 3)
        omega = ((1 - omega) / (1 + omega)) ** 2

        return omega, w


# }}}
