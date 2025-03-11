# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from array_api_compat import array_namespace

from pycaputo.derivatives import CaputoDerivative, Side
from pycaputo.grid import MidPoints, Points, UniformPoints, make_midpoints_from
from pycaputo.logging import get_logger
from pycaputo.typing import (
    Array,
    ArrayOrScalarFunction,
    DifferentiableScalarFunction,
    Scalar,
)

from .base import (
    DerivativeMethod,
    FunctionCallableError,
    diff,
    diffs,
    quadrature_weights,
)

log = get_logger(__name__)


@dataclass(frozen=True)
class CaputoMethod(DerivativeMethod):
    """A method used to evaluate a :class:`~pycaputo.derivatives.CaputoDerivative`."""

    alpha: float
    """Order of the Caputo derivative that is being discretized."""

    if __debug__:

        def __post_init__(self) -> None:
            if self.alpha < 0:
                raise ValueError(f"Negative orders are not supported: {self.alpha}")

    @property
    def d(self) -> CaputoDerivative:
        return CaputoDerivative(self.alpha)


# {{{ L1


@dataclass(frozen=True)
class L1(CaputoMethod):
    r"""Implements the L1 method for the Caputo fractional derivative
    of order :math:`\alpha \in (0, 1)`.

    This method is defined in Section 4.1.1 (II) from [Li2020]_ for general
    non-uniform grids.
    """

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()
            if not 0 < self.alpha < 1:
                raise ValueError(
                    f"'{type(self).__name__}' only supports 0 < alpha < 1: {self.alpha}"
                )


def _caputo_piecewise_constant_integral(x: Array, alpha: float) -> Array:
    r"""Computes the integrals

    .. math::

        a_{nk} =
            \frac{1}{\Gamma(1 - \alpha)}
            \int_{x_k}^{x_{k + 1}} \frac{1}{(x_n - s)^\alpha} ds

    for :math:`0 \le k \le n - 1`.
    """

    xn = x[-1]
    gamma = math.gamma(2 - alpha)
    return ((xn - x[:-1]) ** (1 - alpha) - (xn - x[1:]) ** (1 - alpha)) / gamma  # type: ignore[no-any-return]


def _caputo_l1_weights(x: Array, dx: Array, n: int, alpha: float) -> Array:
    xp = array_namespace(x)
    if n == 0:
        return xp.array(np.nan)  # type: ignore[no-any-return]

    x = x[: n + 1]
    dx = dx[:n]
    a = _caputo_piecewise_constant_integral(x, alpha) / dx

    # NOTE: the first step of the discretization is just
    #   sum a_{ik} f'_k
    # and we need to re-arrange the sum with the approximation of the derivative
    return xp.concatenate([-a[:1], a[:-1] - a[1:], a[-1:]])  # type: ignore[no-any-return]


@quadrature_weights.register(L1)
def _quadrature_weights_caputo_l1(m: L1, p: Points, n: int) -> Array:
    if not 0 <= n < p.size:
        raise IndexError(f"Index 'n' out of range: 0 <= {n} < {p.size}")

    if n == 0:
        return np.array([], dtype=p.dtype)

    return _caputo_l1_weights(p.x, p.dx, n, m.alpha)


@diffs.register(L1)
def _diffs_caputo_l1(m: L1, f: ArrayOrScalarFunction, p: Points, n: int) -> Scalar:
    if not 0 <= n < p.size:
        raise IndexError(f"Index 'n' out of range: 0 <= {n} < {p.size}")

    if n == 0:
        return np.array([np.nan])

    w = _caputo_l1_weights(p.x, p.dx, n, m.alpha)
    fx = f(p.x[: w.size]) if callable(f) else f[: w.size]

    return np.sum(w * fx)  # type: ignore[no-any-return]


@diff.register(L1)
def _diff_caputo_l1(m: L1, f: ArrayOrScalarFunction, p: Points) -> Array:
    fx = f(p.x) if callable(f) else f
    if fx.shape[0] != p.size:
        raise ValueError(
            f"Shape of 'f' does match points: got {fx.shape} expected {p.shape}"
        )

    xp = array_namespace(fx)
    alpha = m.alpha

    # FIXME: in the uniform case, we can also do an FFT, but we need different
    # weights for that, so we leave it like this for now
    return xp.array(  # type: ignore[no-any-return]
        [
            xp.sum(_caputo_l1_weights(p.x, p.dx, n, alpha) * fx[: n + 1])
            for n in range(fx.size)
        ],
        dtype=fx.dtype,
    )


# }}}


# {{{ ModifiedL1


@dataclass(frozen=True)
class ModifiedL1(CaputoMethod):
    r"""Implements the modified L1 method for the Caputo fractional derivative
    of order :math:`\alpha \in (0, 1)`.

    This method is defined in Section 4.1.1 (III) from [Li2020]_. Note that this
    method evaluates the fractional derivative at the midpoints
    :math:`(x_i + x_{i - 1}) / 2` for any given input grid except
    :class:`~pycaputo.grid.MidPoints`.

    As noted in [Li2020]_, this method has the same order of convergence as the
    standard L1 method. However the weights can be used to construct a
    Crank-Nicolson type method.
    """

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()
            if not 0 < self.alpha < 1:
                raise ValueError(
                    f"'{type(self).__name__}' only supports 0 < alpha < 1: {self.alpha}"
                )


@quadrature_weights.register(ModifiedL1)
def _quadrature_weights_caputo_modified_l1(m: ModifiedL1, p: Points, n: int) -> Array:
    if isinstance(p, MidPoints):
        return quadrature_weights.dispatch(L1)(m, p, n)
    else:
        p = make_midpoints_from(p)
        w = quadrature_weights.dispatch(L1)(m, p, n)

        wp = np.empty((n + 1,), dtype=w.dtype)
        wp[0] = w[0] / 2.0
        wp[-1] = w[-1] / 2.0
        wp[1:-1] = (w[1:] + w[:-1]) / 2.0

        return wp


@diffs.register(ModifiedL1)
def _diffs_caputo_modified_l1(
    m: ModifiedL1, f: ArrayOrScalarFunction, p: Points, n: int
) -> Scalar:
    if not isinstance(p, MidPoints):
        p = make_midpoints_from(p)

        if not callable(f):
            f[1:] = (f[:-1] + f[1]) / 2.0

    return diffs.dispatch(L1)(m, f, p, n)


@diff.register(ModifiedL1)
def _diff_caputo_modified_l1(
    m: ModifiedL1, f: ArrayOrScalarFunction, p: Points
) -> Array:
    if not isinstance(p, MidPoints):
        p = make_midpoints_from(p)

        if not callable(f):
            f[1:] = (f[:-1] + f[1]) / 2.0

    return diff.dispatch(L1)(m, f, p)


# }}}


# {{{ L2


@dataclass(frozen=True)
class L2(CaputoMethod):
    r"""Implements the L2 method for the Caputo fractional derivative
    of order :math:`\alpha \in (1, 2)`.

    This method is defined in Section 4.1.2 from [Li2020]_ for uniform grids.
    The variant implemented here supports arbitrary non-uniform grids.

    .. note::

        Unlike the method from [Li2020]_, we do not assume knowledge of points
        outside of the domain. Instead a biased stencil is used at the boundary.
    """

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()
            if not 1 < self.alpha < 2:
                raise ValueError(
                    f"'{type(self).__name__}' only supports 1 < alpha < 2: {self.alpha}"
                )


def _caputo_d2_boundary_coefficients(x: Array, s: Scalar) -> Array:
    r"""Get coefficients for a second-order approximation of the second derivative
    at the boundary.

    .. math::

        \frac{\partial^2 f}{\partial x^2}(s) =
            c_0 f(x_0) + c_1 f(x_1) + c_2 f(x_2) + c_3 f(x_3)

    """
    x0, x1, x2, x3 = x[:4]
    c0 = 2.0 * (3.0 * s - x1 - x2 - x3) / ((x0 - x1) * (x0 - x2) * (x0 - x3))
    c1 = 2.0 * (x0 + x2 + x3 - 3.0 * s) / ((x0 - x1) * (x1 - x2) * (x1 - x3))
    c2 = 2.0 * (x0 + x1 + x3 - 3.0 * s) / ((x0 - x2) * (x2 - x1) * (x2 - x3))
    c3 = 2.0 * (x0 + x1 + x2 - 3.0 * s) / ((x0 - x3) * (x3 - x1) * (x3 - x2))

    return np.array([c0, c1, c2, c3])


def _caputo_l2_weights(p: Points, n: int, alpha: float) -> Array:
    x = p.x[: n + 1]
    dx = p.dx[:n]
    dxm = p.dxm[: n - 1]

    # weights have size at least 4 for the initial biased stencil
    w = np.zeros(max(4, n + 1), dtype=x.dtype)
    a = _caputo_piecewise_constant_integral(x, alpha - 1)

    # add boundary stencil
    c_l = _caputo_d2_boundary_coefficients(p.x[:4], x[0])
    w[:4] += a[0] * c_l

    # add interior stencils
    d_l = 1.0 / (dx[:-1] * dxm)
    d_r = 1.0 / (dx[1:] * dxm)
    d_m = -(d_l + d_r)

    if n > 1:
        w[0] += a[1] * d_l[0]
        w[1] += a[1] * d_m[0]
        w[n] += a[-1] * d_r[-1]

    if n > 2:
        w[1] += a[2] * d_l[1]
        w[n - 1] += a[-1] * d_m[-1] + a[-2] * d_r[-2]

    if n > 3:
        w[2 : n - 1] += a[1:-2] * d_r[:-2] + a[2:-1] * d_m[1:-1] + a[3:] * d_l[2:]

    return w


@quadrature_weights.register(L2)
def _quadrature_weights_caputo_l2(m: L2, p: Points, n: int) -> Array:
    if not 0 <= n <= p.size:
        raise IndexError(f"Index 'n' out of range: 0 <= {n} < {p.size}")

    if n == 0:
        return np.array([], dtype=p.dtype)

    return _caputo_l2_weights(p, n, m.alpha)


@diffs.register(L2)
def _diffs_caputo_l2(m: L2, f: ArrayOrScalarFunction, p: Points, n: int) -> Array:
    if not 0 <= n <= p.size:
        raise IndexError(f"Index 'n' out of range: 0 <= {n} < {p.size}")

    if n == 0:
        return np.array([np.nan])

    w = quadrature_weights(m, p, n)
    fx = f(p.x[: w.size]) if callable(f) else f[: w.size]

    return np.sum(w * fx)  # type: ignore[no-any-return]


@diff.register(L2)
def _diff_caputo_l2(m: L2, f: ArrayOrScalarFunction, p: Points) -> Array:
    fx = f(p.x) if callable(f) else f
    if fx.shape[0] != p.size:
        raise ValueError(
            f"Shape of 'f' does match points: got {fx.shape} expected {p.shape}"
        )

    # variables
    x = p.x
    dx = p.dx
    dxm = p.dxm
    alpha = m.alpha

    # estimate second-order derivative
    # NOTE: this is faster than using the quadrature weights
    ddfx = np.empty(p.size - 1, dtype=fx.dtype)

    # interior
    cm = 1.0 / (dx[:-1] * dxm)
    cp = 1.0 / (dx[1:] * dxm)
    ddfx[1:] = cm * fx[:-2] - (cm + cp) * fx[1:-1] + cp * fx[2:]

    # boundary
    c = _caputo_d2_boundary_coefficients(x[:4], x[0])
    ddfx[0] = c @ fx[: c.size]

    # FIXME: in the uniform case, we can also do an FFT, but we need different
    # weights for that, so we leave it like this for now
    df = np.empty_like(fx)
    df[0] = np.nan

    for n in range(1, df.size):
        a = _caputo_piecewise_constant_integral(x[: n + 1], alpha - 1)
        df[n] = np.sum(a * ddfx[: a.size])

    return df


# }}}


# {{{ L2C


@dataclass(frozen=True)
class L2C(CaputoMethod):
    r"""Implements the L2C method for the Caputo fractional derivative
    of order :math:`\alpha \in (1, 2)`.

    This method is defined in Section 4.1.2 from [Li2020]_ on uniform grids.
    """

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()
            if not 1 < self.alpha < 2:
                raise ValueError(
                    f"'{type(self).__name__}' only supports 0 < alpha < 1: {self.alpha}"
                )


def _weights_l2(alpha: float, i: int | Array, k: int | Array) -> Array:
    return np.array((i - k) ** (2 - alpha) - (i - k - 1) ** (2 - alpha))


@diff.register(L2C)
def _diff_l2c_method(m: L2C, f: ArrayOrScalarFunction, p: Points) -> Array:
    # precompute variables
    x = p.x
    fx: Array = f(x) if callable(f) else f

    alpha = m.alpha
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


# {{{ L2F


@dataclass(frozen=True)
class L2F(CaputoMethod):
    r"""Implements the L2 method for the Caputo fractional derivative
    of order :math:`\alpha \in (1, 2)`.

    This is similar to :class:`L2`, but it assumes that *f* is a callable that can
    be evaluated at :math:`f(x_{-1})`, i.e. outside of the domain :math:`[a, b]`.
    For symmetry, we set :math:`x_{-1} = a - \Delta x_0`. This allows using
    the same centered stencil for all the points in the domain.

    This method mainly exists for comparison with the literature, as [Li2020]_
    also uses the value outside of the interval. Note that this is not always
    possible, e.g. for :math:`\sqrt{x}` on :math:`[0, 1]`.
    """

    side: Side


def _caputo_l2f_weights(p: Points, n: int, alpha: float) -> Array:
    # weights have at least size 3 for the initial stencil
    w = np.zeros(max(3, n + 2), dtype=p.dtype)

    # variables
    x = p.x[: n + 1]
    dx = p.dx[:n]
    dxm = p.dxm[: n - 1]

    a = _caputo_piecewise_constant_integral(x, alpha - 1)

    # add boundary stencil
    dx2 = p.dx[0] ** 2
    c_l, c_m, c_r = 1.0 / dx2, -2.0 / dx2, 1.0 / dx2

    w[0] = a[0] * c_l
    w[1] = a[0] * c_m
    w[2] = a[0] * c_r

    # add interior stencils
    d_l = 1.0 / (dx[:-1] * dxm)
    d_r = 1.0 / (dx[1:] * dxm)
    d_m = -(d_l + d_r)

    if n > 1:
        w[1] += a[1] * d_l[0]
        w[2] += a[1] * d_m[0]
        w[n + 1] += a[-1] * d_r[-1]

    if n > 2:
        w[2] += a[2] * d_l[1]
        w[n] += a[-1] * d_m[-1] + a[-2] * d_r[-2]

    if n > 3:
        w[3:n] += a[1:-2] * d_r[:-2] + a[2:-1] * d_m[1:-1] + a[3:] * d_l[2:]

    return w


@quadrature_weights.register(L2F)
def _quadrature_weights_caputo_l2f(m: L2F, p: Points, n: int) -> Array:
    if m.side == Side.Right:
        raise NotImplementedError("Right boundary approximation not implemented")

    if not 0 <= n <= p.size:
        raise IndexError(f"Index 'n' out of range: 0 <= {n} < {p.size}")

    if n == 0:
        return np.array([], dtype=p.dtype)

    return _caputo_l2f_weights(p, n, m.alpha)


@diffs.register(L2F)
def _diffs_caputo_l2f(m: L2F, f: ArrayOrScalarFunction, p: Points, n: int) -> Array:
    if m.side == Side.Right:
        raise NotImplementedError("Right boundary approximation not implemented")

    if not callable(f):
        raise FunctionCallableError(
            f"The '{type(m).__name__}' method requires evaluating the function "
            f"values. 'f' is not callable: {type(f)}"
        )

    if not 0 <= n <= p.size:
        raise IndexError(f"Index 'n' out of range: 0 <= {n} < {p.size}")

    if n == 0:
        return np.array([np.nan])

    w = _caputo_l2f_weights(p, n, m.alpha)
    x = np.empty(w.size, dtype=p.dtype)
    x[0] = p.a - p.dx[0]
    x[1:] = p.x
    fx = f(x)

    return np.sum(w * fx)  # type: ignore[no-any-return]


@diff.register(L2F)
def _diff_caputo_l2f(m: L2F, f: ArrayOrScalarFunction, p: Points) -> Array:
    if not callable(f):
        raise FunctionCallableError(
            f"The '{type(m).__name__}' method requires evaluating the function "
            f"values. 'f' is not callable: {type(f)}"
        )

    # variables
    x = p.x
    dx = p.dx
    dxm = p.dxm
    alpha = m.alpha
    fx = f(p.x)

    # estimate second-order derivative
    # NOTE: this is faster than using the quadrature weights
    ddfx = np.empty(p.size - 1, dtype=fx.dtype)

    cm = 1.0 / dx[:-1]
    cp = 1.0 / dx[1:]

    if m.side == Side.Left:
        fa = f(p.a - dx[0])
        ddfx[1:] = (cm * fx[:-2] - (cm + cp) * fx[1:-1] + cp * fx[2:]) / dxm
        ddfx[0] = (fx[1] - 2.0 * fx[0] + fa) / dx[0] ** 2
    else:
        fb = f(p.b + dx[-1])
        ddfx[:-1] = (cm * fx[:-2] - (cm + cp) * fx[1:-1] + cp * fx[2:]) / dxm
        ddfx[-1] = (fx[-2] - 2.0 * fx[-1] + fb) / dx[-1] ** 2

    # FIXME: in the uniform case, we can also do an FFT, but we need different
    # weights for that, so we leave it like this for now
    df = np.empty_like(fx)
    df[0] = np.nan

    for n in range(1, df.size):
        a = _caputo_piecewise_constant_integral(x[: n + 1], alpha - 1)
        df[n] = np.sum(a * ddfx[: a.size])

    return df


# }}}


# {{{ LXD


@dataclass(frozen=True)
class LXD(CaputoMethod):
    r"""Implements the LX method for the Caputo fractional derivative
    of order :math:`\alpha \in (0, \infty)`.

    This method is equivalent to :class:`L1` (or :class:`L2` method), but
    evaluation requires explicit knowledge of the required derivative, i.e. the
    *f* function must be a callable satisfying
    :class:`~pycaputo.typing.DifferentiableScalarFunction`. With explicit
    knowledge of the derivative, we can evaluate any fractional order.

    The derivatives are always evaluated at the midpoint of each interval.
    """

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()


@quadrature_weights.register(LXD)
def _quadrature_weights_caputo_lxd(m: LXD, p: Points, n: int) -> Array:
    if not 0 <= n <= p.size:
        raise IndexError(f"Index 'n' out of range: 0 <= {n} < {p.size}")

    if n == 0:
        return np.array([])

    d = m.d
    x = p.x[: n + 1]
    return _caputo_piecewise_constant_integral(x, d.alpha - d.n + 1)


@diffs.register(LXD)
def _diffs_caputo_lxd(m: LXD, f: ArrayOrScalarFunction, p: Points, n: int) -> Scalar:
    if not 0 <= n <= p.size:
        raise IndexError(f"Index 'n' out of range: 0 <= {n} < {p.size}")

    if n == 0:
        return np.array([np.nan])

    if not callable(f):
        raise FunctionCallableError(
            f"The '{type(m).__name__}' method requires evaluating the first "
            f"derivative. 'f' is not callable: {type(f)}"
        )

    # FIXME: isinstance(f, DifferentiableScalarFunction) does not work?
    assert isinstance(f, DifferentiableScalarFunction)
    d = m.d

    try:
        w = quadrature_weights(m, p, n)
        dfx = f((p.x[1 : n + 1] + p.x[:n]) / 2, d=d.n)
    except TypeError as exc:
        raise TypeError(
            f"{type(m).__name__!r} requires a 'DifferentiableScalarFunction': "
            f"f is a {type(f).__name__!r}"
        ) from exc

    return np.sum(w * dfx)  # type: ignore[no-any-return]


@diff.register(LXD)
def _diff_caputo_lxd(m: LXD, f: ArrayOrScalarFunction, p: Points) -> Array:
    if not callable(f):
        raise FunctionCallableError(
            f"The '{type(m).__name__}' method requires evaluating the first "
            f"derivative. 'f' is not callable: {type(f)}"
        )

    # FIXME: isinstance(f, DifferentiableScalarFunction) does not work?
    assert isinstance(f, DifferentiableScalarFunction)
    alpha_hat = m.alpha - m.d.n + 1
    x = p.x
    xm = p.xm

    try:
        dfx = f(xm, d=m.d.n)
    except TypeError as exc:
        raise TypeError(
            f"{type(m).__name__!r} requires a 'DifferentiableScalarFunction': "
            f"f is a {type(f).__name__!r}"
        ) from exc

    df = np.empty((p.size, *dfx.shape[1:]), dtype=dfx.dtype)
    df[0] = np.nan

    for n in range(1, df.size):
        w = _caputo_piecewise_constant_integral(x[: n + 1], alpha_hat)
        df[n] = np.sum(w * dfx[:n])

    return df


# }}}


# {{{ Jacobi


@dataclass(frozen=True)
class Jacobi(CaputoMethod):
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


@diff.register(Jacobi)
def _diff_jacobi(m: Jacobi, f: ArrayOrScalarFunction, p: Points) -> Array:
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
    for n, Dhat in jacobi_caputo_derivative(p, m.alpha):
        df += fhat[n] * Dhat

    return df


# }}}


# {{{ DiffusiveCaputoMethod


@dataclass(frozen=True)
class DiffusiveCaputoMethod(CaputoMethod):
    r"""Quadrature method for the Caputo derivative based on diffusive approximations.

    See :class:`~pycaputo.quadrature.riemann_liouville.DiffusiveRiemannLiouvilleMethod`
    for details on the method itself. Approximations for the Caputo derivative
    generally follow the same construction.
    """

    @abstractmethod
    def nodes_and_weights(self) -> tuple[Array, Array]:
        r"""Compute the nodes and weights for the quadrature used by the method.

        :returns: a tuple of ``(omega, w)`` of nodes and weights to be used by
            the method.
        """


# }}}


# {{{ YuanAgrawal


def _diffusive_gamma_solve_ivp(
    m: DiffusiveCaputoMethod,
    f: DifferentiableScalarFunction,
    p: Points,
    omega: Array,
    *,
    method: str = "Radau",
    qtol: float = 1.0,
) -> Array:
    from scipy.integrate import solve_ivp

    # construct coefficients
    alpha = m.alpha
    n = m.d.n
    omega_a = omega ** (2 * alpha - 2 * n + 1)
    omega_b = -(omega**2)
    omega_jac = np.diag(omega_b)

    def fun(t: Array, phi: Array) -> Array:
        return omega_a * f(t, d=n) + omega_b * phi  # type: ignore[no-any-return]

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
class YuanAgrawal(DiffusiveCaputoMethod):
    r"""Caputo derivative approximation using the diffusive approximation
    from [Yuan2002]_.

    See the approximation for the Riemann-Liouville fractional integral from
    :class:`~pycaputo.quadrature.riemann_liouville.YuanAgrawal` for details
    on the method. The main difference is that the ODE for :math:`\phi` is

    .. math::

        \frac{\partial \phi}{\partial \xi}(\xi; \omega) =
            \omega^{2 \alpha - 2 m + 1} f^{(m)}(\xi)
            - \omega^2 \phi(\xi; \omega),

    where :math:`m` is the integer part of :math:`\alpha`. As such, this problem
    has the added difficulty of computing :math:`f^{(m)}(\xi)`. The current
    implementation requires an analytical expression for the derivative
    (see :class:`~pycaputo.typing.DifferentiableScalarFunction`).
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
        alphabar = 2 * self.alpha - 2 * self.d.n + 1
        return float(0.75 * self.quad_order ** (alphabar - 1))

    def nodes_and_weights(self) -> tuple[Array, Array]:
        from scipy.special import roots_genlaguerre

        # get Gauss-Laguerre quadrature
        alpha = self.alpha
        n = self.d.n
        beta = 2.0 * alpha - 2 * n + 1

        omega, w = roots_genlaguerre(self.quad_order, beta)

        # transform for Yuan-Agrawal method
        fac = 2.0 * (-1) ** (n - 1) * np.sin(alpha * np.pi) / np.pi
        w = fac * omega ** (-beta) * np.exp(omega) * w

        return omega, w


@diff.register(YuanAgrawal)
def _diff_caputo_yuan_agrawal(
    m: YuanAgrawal,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    # FIXME: isinstance(f, DifferentiableScalarFunction) does not work?
    assert isinstance(f, DifferentiableScalarFunction)

    try:
        dtype = np.array(f(p.x[0], d=0)).dtype
    except TypeError:
        raise TypeError(
            f"{type(m).__name__!r} requires a 'DifferentiableScalarFunction': "
            f"f is a {type(f).__name__!r}"
        ) from None

    x = p.x

    # solve ODE at quadrature nodes
    omega, w = m.nodes_and_weights()
    phi = _diffusive_gamma_solve_ivp(m, f, p, omega, method=m.method, qtol=m._qtol)

    # compute RL integral
    qf = np.empty_like(x, dtype=dtype)
    qf[0] = np.nan
    qf[1:] = np.einsum("i,ij->j", w, phi)

    return qf


# }}}


# {{{ Diethelm


@dataclass(frozen=True)
class Diethelm(YuanAgrawal):
    r"""Caputo derivative approximation using the diffusive approximation
    from [Diethelm2008]_.

    This method uses the weights

    .. math::

        (1 - \omega)^{\bar{\alpha}} (1 + \omega)^{-\bar{\alpha}},

    where :math:`\bar{\alpha} = 2 \alpha - 2 m + 1`.
    """

    @property
    def _qtol(self) -> float:
        # FIXME: in this case, the error should be spectral, so it's not clear what
        # to use here. This seems to work well for the test case
        alphabar = 2 * self.alpha - 2 * self.d.n + 1
        return float(0.75 * self.quad_order ** (alphabar - 3))

    def nodes_and_weights(self) -> tuple[Array, Array]:
        from scipy.special import roots_jacobi

        # get Gauss-Jacobi quadrature rule
        alpha = self.alpha
        n = self.d.n
        alphabar = 2.0 * alpha - 2 * n + 1
        beta = alphabar
        gamma = -alphabar

        omega, w = roots_jacobi(self.quad_order, beta, gamma)

        # transform for Diethelm method
        fac = 4.0 * np.sin(alpha * np.pi) / np.pi
        w = fac * w / (1 - omega) ** beta / (1 + omega) ** (gamma + 2)
        omega = (1 - omega) / (1 + omega)

        return omega, w


# }}}


# {{{ BirkSong


@dataclass(frozen=True)
class BirkSong(YuanAgrawal):
    r"""Caputo derivative approximation using the diffusive approximation
    from [Birk2010]_.

    This method uses the weights

    .. math::

        (1 - \omega)^{2 \bar{\alpha} + 1} (1 + \omega)^{-(2 \bar{\alpha} - 1)},

    where :math:`\bar{\alpha} = 2 \alpha - 2 m + 1`.
    """

    @property
    def _qtol(self) -> float:
        # FIXME: in this case, the error should be spectral, so it's not clear what
        # to use here. This seems to work well for the test case
        alphabar = 2 * self.alpha - 2 * self.d.n + 1
        return float(0.75 * self.quad_order ** (alphabar - 3))

    def nodes_and_weights(self) -> tuple[Array, Array]:
        from scipy.special import roots_jacobi

        # get Gauss-Jacobi quadrature rule
        alpha = self.alpha
        n = self.d.n
        alphabar = 2.0 * alpha - 2 * n + 1
        beta = 2 * alphabar + 1
        gamma = -(2 * alphabar - 1)

        omega, w = roots_jacobi(self.quad_order, beta, gamma)

        # transform for BirkSong method
        fac = 8.0 * np.sin(alpha * np.pi) / np.pi
        w = fac * w / (1 - omega) ** (beta - 1) / (1 + omega) ** (gamma + 3)
        omega = ((1 - omega) / (1 + omega)) ** 2

        return omega, w


# }}}
