# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from pycaputo.grid import Points
from pycaputo.logging import get_logger
from pycaputo.typing import Array, ScalarFunction

log = get_logger(__name__)


# {{{ Vandermonde matrix


@dataclass(frozen=True)
class _LagrangePoly:
    n: int
    """Order of the Lagrange polynomial."""
    p: Array
    """Nodes used to construct the polynomials."""

    def __call__(self, x: Array) -> Array:
        n, p = self.n, self.p
        result = np.prod(
            [(x - p[i]) / (p[n] - p[i]) for i in range(p.size) if n != i], axis=0
        )

        return np.array(result)


def lagrange_polynomials(p: Array) -> Iterator[ScalarFunction]:
    """Construct functions to evaluate the Lagrange polynomials."""
    for n in range(p.size):
        yield _LagrangePoly(n, p)


def vandermonde(x: Array) -> Array:
    """Compute the Vandermonde matrix for the monomial polynomials."""
    assert x.ndim == 1

    n = np.arange(x.size)
    x = x.reshape((-1, 1))

    return np.array(x**n)


def vandermonde_inverse(x: Array) -> Array:
    """Compute the inverse of the Vandermonde matrix from :func:`vandermonde`."""
    vdm = vandermonde(x)
    return np.linalg.pinv(vdm)


# }}}


# {{{ Lagrange Riemann-Liouville integral


def lagrange_riemann_liouville_integral(
    p: Points,
    xi: Array,
    alpha: float,
) -> Iterator[Array]:
    r"""Compute the Riemann-Liouville integral of Lagrange polynomials.

    Let :math:`\{\phi_{kj}(x)\}` be the :math:`q` Lagrange polynomials constructed
    from *xi*, defined on each interval :math:`[x_k, x_{k + 1}]` from the point
    grid *p*. We extend these polynomials to the full domain :math:`[a, b]` with
    0. Then, we have that their Riemann-Lioville integral is given by

    .. math::

        I^\alpha[\phi_{kj}](x_n) =
            \frac{1}{\Gamma(\alpha)} \int_{0}^{x_n}
            (x_n - s)^{\alpha - 1} \phi_{kj}(s) \,\mathrm{d}s.

    As the polynomials are zero except in the domain of definition, this simplifies
    to

    .. math::

        I^\alpha[\phi_{kj}](x_n) = L^\alpha_{nkj} =
            \frac{1}{\Gamma(\alpha)}
            \int_{x_k}^{x_{k + 1}} (x_n - s)^{\alpha - 1} \phi_{kj}(s)
            \,\mathrm{d} s

    for :math:`k \in \{0, \dots, n - 1\}`. For every :math:`n`, this function
    computes the ``(n, q)`` array :math:`(L^\alpha_n)_{k, j}`.

    :arg p: domain grid composed of elements :math:`[x_k, x_{k + 1}]`.
    :arg xi: reference nodes on the unit interval :math:`[0, 1]` in each element,
        that are used to construct the Lagrange polynomials.

    :returns: the integrals :math:`L^{\alpha}_{n, \cdot}` for each :math:`x_n`.
    """
    from scipy.special import beta, betainc, gamma

    x = p.x
    dx = p.dx

    dxa = (dx**alpha / gamma(alpha)).reshape(-1, 1)
    A = np.linalg.pinv(vandermonde(xi))

    for n in range(1, p.size):
        j = np.arange(xi.size)
        zn = ((x[n] - x[:n]) / dx[:n]).reshape(-1, 1)
        B = betainc(1 + j, alpha, 1 / zn) * beta(1 + j, alpha)

        # NOTE: in [Cardone2021] this is written as
        #   dx^alpha / gamma(alpha) * sum(A_{jk} z^{j + alpha} B(1/z, 1 + j, alpha))
        yield (dxa[:n] * (zn ** (j + alpha)) * B) @ A


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

        L^{q, \alpha}_{nk} \triangleq
            \frac{1}{\Gamma(m - \alpha)}
            \int_{x_k}^{x_{k + 1}} (x_n - s)^{m - \alpha - 1}
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
