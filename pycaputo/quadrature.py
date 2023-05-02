# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import singledispatch
from typing import Dict, Type

import numpy as np

from pycaputo.derivatives import RiemannLiouvilleDerivative, Side
from pycaputo.grid import Points
from pycaputo.utils import Array, ScalarFunction

# {{{ interface


@dataclass(frozen=True)
class QuadratureMethod(ABC):
    """A generic method used to evalute a fractional integral."""

    @property
    @abstractmethod
    def name(self) -> str:
        """An identifier for the quadrature method."""

    @property
    @abstractmethod
    def order(self) -> float:
        """Expected order of convergence of the method."""


@singledispatch
def quad(m: QuadratureMethod, f: ScalarFunction, x: Points) -> Array:
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

    This method is of order :math:`\mathcal{O}(h)`.
    """

    #: Weight used in the approximation :math:`w f_k + (1 - w) f_{k + 1}`.
    weight: float = 0.5

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()
            if not 0.0 <= self.weight <= 1.0:
                raise ValueError(
                    f"Weight is expected to be in [0, 1]: weight is '{self.weight}'"
                )

    @property
    def name(self) -> str:
        return "RLRect"

    @property
    def order(self) -> float:
        if self.weight == 0.5:
            return min(2.0, 1.0 - self.d.order)

        return 1.0


@quad.register(RiemannLiouvilleRectangularMethod)
def _quad_rl_rect(
    m: RiemannLiouvilleRectangularMethod,
    f: ScalarFunction,
    p: Points,
) -> Array:
    x = p.x
    fx = f(x)
    alpha = -m.d.order
    w0 = 1 / math.gamma(1 + alpha)

    fc = m.weight * fx[:-1] + (1 - m.weight) * fx[1:]

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

    This method is of order :math:`\mathcal{O}(h^2)`.
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
    f: ScalarFunction,
    p: Points,
) -> Array:
    from pycaputo.grid import UniformPoints

    x = p.x
    fx = f(x)
    alpha = -m.d.order
    w0 = 1 / math.gamma(2 + alpha)

    # compute integral
    qf = np.empty_like(fx)
    qf[0] = np.nan

    if isinstance(p, UniformPoints):
        k = np.arange(qf.size)
        w0 = w0 * p.dx[0] ** alpha

        w = k[:-1] ** (1 + alpha) - (k[:-1] - alpha) * k[1:] ** alpha
        qf[1:] = fx[0] + w * fx[1:]

        # NOTE: [Li2020] Equation 3.15
        for n in range(1, qf.size):
            w = (
                (n - k[1 : n - 1] + 1) ** (1 + alpha)
                - 2 * (n - k[1 : n - 1]) ** (1 + alpha)
                + (n - k[1 : n - 1] - 1) ** (1 + alpha)
            )

            qf[n] += w0 * np.sum(w * fx[1 : n - 1])
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
    approximates the polynomial by projecting the function on each interval
    to the Jacobi basis and constructing a quadrature rule, i.e.

    .. math::

        I^\alpha[f](x_j) = I^\alpha[p_N](x_j) = \sum_{k = 0}^N w_{jk} f(x_k),

    where :math:`p_N` is a degree :math:`N` polynomial approximating :math:`f`
    and :math:`w_{jk}` are a set of weights. Here, we approximate the function
    by the Jacobi polynomials :math:`P^{(\alpha, \beta)}`.
    """

    #: Maximum total degree of the Jacobi polynomials.
    degree: int = 16
    #: First Jacobi polynomial coefficient.
    j_alpha: float = 0
    #: Second Jacobi polynomial coefficient.
    j_beta: float = 0

    @property
    def name(self) -> str:
        return "RLSpec"

    @property
    def order(self) -> float:
        return self.degree


def _quad_rl_spec_uniform_legendre(
    m: RiemannLiouvilleSpectralMethod, f: ScalarFunction, p: Points
) -> Array:
    raise NotImplementedError


def _quad_rl_spec_uniform_chebyshev(
    m: RiemannLiouvilleSpectralMethod, f: ScalarFunction, p: Points
) -> Array:
    raise NotImplementedError


def _quad_rl_spec_uniform_jacobi(
    m: RiemannLiouvilleSpectralMethod, f: ScalarFunction, p: Points
) -> Array:
    raise NotImplementedError


@quad.register(RiemannLiouvilleSpectralMethod)
def _quad_rl_spec(
    m: RiemannLiouvilleSpectralMethod,
    f: ScalarFunction,
    p: Points,
) -> Array:
    from pycaputo.grid import UniformPoints

    if not isinstance(p, UniformPoints):
        raise TypeError(f"Only uniform points are supported: '{type(p).__name__}'")

    if m.j_alpha == 0 and m.j_beta == 0:
        df = _quad_rl_spec_uniform_legendre(m, f, p)
    elif m.j_alpha == -0.5 and m.j_beta == -0.5:
        df = _quad_rl_spec_uniform_chebyshev(m, f, p)
    else:
        df = _quad_rl_spec_uniform_jacobi(m, f, p)

    return df


# }}}


# {{{ make


REGISTERED_METHODS: Dict[str, Type[QuadratureMethod]] = {
    "RiemannLiouvilleRectangularMethod": RiemannLiouvilleRectangularMethod,
    "RiemannLiouvilleTrapezoidalMethod": RiemannLiouvilleTrapezoidalMethod,
}


def make_quad_from_name(
    name: str,
    order: float,
    *,
    side: Side = Side.Left,
) -> QuadratureMethod:
    if name not in REGISTERED_METHODS:
        raise ValueError(
            "Unknown differentiation method '{}'. Known methods are '{}'".format(
                name, "', '".join(REGISTERED_METHODS)
            )
        )

    d = RiemannLiouvilleDerivative(order=order, side=side)
    return REGISTERED_METHODS[name](d)  # type: ignore[call-arg]


# }}}
