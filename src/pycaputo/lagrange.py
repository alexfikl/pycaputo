# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from pycaputo.grid import Points
from pycaputo.logging import get_logger
from pycaputo.utils import Array, ScalarFunction

logger = get_logger(__name__)


# {{{


@dataclass(frozen=True)
class _LagrangePoly:
    n: int
    """Order of the Lagrange polynomial."""
    p: Array
    """Nodes used to construct the polynomials."""

    def __call__(self, x: Array) -> Array:
        n, p = self.n, self.p
        return np.prod(
            [(x - p[i]) / (p[n] - p[i]) for i in range(p.size) if n != i], axis=0
        )


def lagrange_polynomials(p: Array) -> Iterator[ScalarFunction]:
    """Construct functions to evaluate the Lagrange polynomials."""
    for n in range(p.size):
        yield _LagrangePoly(n, p)


def vandermonde(x: Array) -> Array:
    """Compute the Vandermonde matrix for the monomial polynomials."""
    assert x.ndim == 1

    n = np.arange(x.size)
    x = x.reshape(-1, 1)

    return x**n


def monomial_coefficients(x: Array) -> Array:
    vdm = vandermonde(x)
    return np.linalg.pinv(vdm)


# }}}


# {{{ Lagrange Riemann-Liouville integral


def lagrange_riemann_liouville_integral(
    p: Points,
    alpha: float,
    n: int,
    *,
    q: int = 0,
) -> Array:
    r"""Compute the Riemann-Liouville integral of Lagrange polynomials.

    .. math::

        I^{q, \alpha}_{nk} \triangleq
            \frac{1}{\Gamma(\alpha)}
            int_{x_k}^{x_{k + 1}} (x_n - s)^{\alpha - 1}
            \prod_{i = 0}^p \frac{x - x_i}{x_j - x_i}
            \,\mathrm{d} s

    for :math:`k \in \{0, \dots, n\}`.

    :arg n: target point for which to compute the integrals.
    :arg q: order of the Lagrange polynomials to compute the integral of.
    :returns: the integrals :math:`L^{q, \alpha}_{nk}` for every subinterval
        :math:`[x_k, x_{k + 1}]`
    """
    raise NotImplementedError


# }}}


# {{{ Lagrange Caputo derivative


def lagrange_caputo_derivative(
    p: Points,
    alpha: float,
    n: int,
    *,
    q: int = 0,
) -> Array:
    r"""Compute the Caputo derivative of Lagrange polynomials.

    .. math::

        D^{q, \alpha}_{nk} \triangleq
            \frac{1}{\Gamma(m - \alpha)}
            int_{x_k}^{x_{k + 1}} (x_n - s)^{m - \alpha - 1}
            \frac{\mathrm{d}^m}{\mathrm{d} s^m}
            \left(\prod_{i = 0}^p \frac{x - x_i}{x_j - x_i}\right)
            \,\mathrm{d} s

    for :math:`k \in \{0, \dots, n\}` and :math:`m - 1 < \alpha \le m`.

    :arg n: target point for which to compute the derivatives.
    :arg q: order of the Lagrange polynomials to compute the derivative of.
    :returns: the derivatives :math:`L^{q, \alpha}_{nk}` for every subinterval
        :math:`[x_k, x_{k + 1}]`
    """
    raise NotImplementedError


# }}}
