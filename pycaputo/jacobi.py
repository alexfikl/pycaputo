# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from typing import Iterator, Tuple

import numpy as np

from pycaputo.grid import JacobiGaussLobattoPoints, Points
from pycaputo.utils import Array

# {{{ coefficients


def jacobi_gamma(n: int, alpha: float, beta: float) -> float:
    r"""Compute coefficient used in coefficient calculation of a function
    projected to the Jacobi basis (see Equation 3.54 in [Li2020]_).

    :returns: the coefficient :math:`\gamma^{\alpha, \beta}_n`.
    """
    from math import gamma

    if alpha == beta == -0.5 and n == 0:
        # NOTE: gamma(n + alpha + beta + 1) is not defined in this case
        return np.pi

    return (
        2 ** (alpha + beta + 1)
        * gamma(n + alpha + 1)
        * gamma(n + beta + 1)
        / ((2 * n + alpha + beta + 1) * gamma(n + 1) * gamma(n + alpha + beta + 1))
    )


def jacobi_rec_coefficients(
    n: int, alpha: float, beta: float
) -> Tuple[float, float, float]:
    r"""Computes coefficients for the Jacobi recursion relation
    (see Equation 3.51 [Li2020]_).

    :returns: the coefficients
        :math:`A^{\alpha, \beta}_n, B^{\alpha, \beta}_n` and
        :math:`C^{\alpha, \beta}_n`.
    """
    # NOTE: Equation 3.52 in [Li2020]
    a = 2 * n + alpha + beta
    b = (n + 1) * (n + alpha + beta + 1)

    A = (a + 1) * (a + 2) / (2 * b)
    B = (beta**2 - alpha**2) * (a + 1) / (2 * a * b)
    C = (n + alpha) * (n + beta) * (a + 2) / (a * b)

    return A, B, C


def jacobi_diff_rec_coefficients(
    n: int, alpha: float, beta: float
) -> Tuple[float, float, float]:
    r"""Computes coefficients for a derivative-based Jacobi recursion relation
    (see Equation 3.58 [Li2020]_).

    :returns: the coefficients
        :math:`\hat{A}^{\alpha, \beta}_n, \hat{B}^{\alpha, \beta}_n` and
        :math:`\hat{C}^{\alpha, \beta}_n`.
    """
    # NOTE: Equation 3.59 [Li2020]
    a = 2 * n + alpha + beta
    Ahat = -2 * (n + alpha) * (n + beta) / (a * (a + 1) * (n + alpha + beta))
    Bhat = 2 * (alpha - beta) / (a * (a + 1))
    Chat = 2 * (n + alpha + beta + 1) / ((a + 1) * (a + 2))

    return Ahat, Bhat, Chat


# }}}


# {{{ jacobi_polynomial


def jacobi_polynomial(
    p: Points,
    npoly: int,
    *,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> Iterator[Array]:
    r"""Evaluate the Jacobi polynomials :math:`P^{\alpha, \beta}_n` at the points *p*.

    Note that the Jacobi polynomials are only define on :math:`[-1, 1]`, so
    other intervals are simply translated to this in an affine manner.

    :arg p: a set of points at which to evaluate the polynomials.
    :arg alpha: parameter of the Jacobi polynomial.
    :arg beta: parameter of the Jacobi polynomial.
    :arg npoly: number (maximum order) of Jacobi polynomials to evaluate.

    :returns: the Jacobi polynomials evaluated at *p*.
    """
    xm = (p.b + p.a) / 2
    dx = (p.b - p.a) / 2
    x = (p.x - xm) / dx

    P0 = np.ones_like(x)
    yield P0

    P1 = (alpha + beta + 2) / 2 * x + (alpha - beta) / 2
    yield P1

    for n in range(2, npoly):
        # NOTE: Equation 3.51 in [Li2020]
        A, B, C = jacobi_rec_coefficients(n - 1, alpha, beta)
        Pn = (A * x - B) * P1 - C * P0
        yield Pn

        P1, P0 = Pn, P1


# }}}


# {{{ jacobi_riemann_liouville_integral


def jacobi_riemann_liouville_integral(
    p: JacobiGaussLobattoPoints, alpha: float
) -> Iterator[Array]:
    r"""Computes an integral of the Jacobi polynomials used in the definition
    of the Riemann-Liouville integral (see Section 3.3 (I) in [Li2020]_).

    This effectively computes

    .. math::

        \hat{P}^{u, v}_n(x) =
            \int_{-1}^x (x - s)^{\alpha - 1} P^{u, v}_n(s) \,\mathrm{d} s,

    where :math:`P^{u, v}_n` is the usual Jacobi polynomial.

    :returns: the Riemann-Liouville integral of the Jacobi polynomials of
        every order.
    """
    from math import gamma

    # translate x back to [-1, 1]
    xm = (p.b + p.a) / 2
    dx = (p.b - p.a) / 2
    x = (p.x - xm) / dx

    # NOTE: Equation 3.64 [Li2020]
    Phat0 = (x + 1) ** alpha / gamma(alpha + 1)
    yield dx**alpha * Phat0

    # fmt: off
    Phat1 = (
        x * (x + 1) ** alpha / gamma(alpha + 1)
        - alpha * (x + 1) ** (alpha + 1) / gamma(alpha + 2)
    )
    # fmt: on
    Phat1 = (p.alpha + p.beta + 2) / 2 * Phat1 + (p.alpha - p.beta) / 2 * Phat0
    yield dx**alpha * Phat1

    # NOTE: this holds the Jacobi polynomials at x = -1 in use in the recursion
    # FIXME: these have an exact formula:
    #       P^{alpha, beta}_n(-1) = (-1)^n binomial(n + beta, b)
    P0 = 1.0
    P1 = -(p.alpha + p.beta + 2) / 2 + (p.alpha - p.beta) / 2

    for n in range(2, x.size):
        A, B, C = jacobi_rec_coefficients(n - 1, p.alpha, p.beta)
        Ahat, Bhat, Chat = jacobi_diff_rec_coefficients(n - 1, p.alpha, p.beta)

        D = 1 + alpha * A * Chat
        P2 = -(A + B) * P1 - C * P0

        C0 = alpha * A * (Ahat * P0 + Bhat * P1 + Chat * P2) / (D * gamma(alpha + 1))
        C1 = (A * x - B - alpha * A * Bhat) / D
        C2 = (C + alpha * A * Ahat) / D
        Phatn = C0 * (x + 1) ** alpha + C1 * Phat1 - C2 * Phat0
        yield dx**alpha * Phatn

        P1, P0 = P2, P1
        Phat1, Phat0 = Phatn, Phat1


# }}}

# {{{ jacobi_project


def jacobi_project(f: Array, p: JacobiGaussLobattoPoints) -> Array:
    r"""Project a function to the basis of Jacobi polynomials
    (see Equation 4.61 [Li2020]_).

    Each coefficient is given by

    .. math::

        \hat{f}^{\alpha, \beta}_n = \frac{1}{\delta^{\alpha, \beta}_n}
            \sum_{k = 0}^{N - 1} f(x_k) P^{\alpha, \beta}_n(x_k) w_k,

    where :math:`(x_k, w_k)` are the Jacobi-Gauss-Lobatto quadrature nodes and
    weights.

    :arg f: an array of function values evaluated at the points of *p*.
    :arg p: a set of Jacobi-Gauss-Lobatto points.
    :returns: the spectral coefficients used to represent the function *f* in
        the Jacobi basis.
    """
    w = p.w
    alpha = p.alpha
    beta = p.beta

    fhat = np.empty(f.size)
    for n, Pn in enumerate(jacobi_polynomial(p, w.size, alpha=alpha, beta=beta)):
        # NOTE: Equation 3.61 in [Li2020]
        fhat[n] = np.sum(f * Pn * w) / jacobi_gamma(n, alpha, beta)

    fhat[-1] = fhat[-1] / (2 + (alpha + beta + 1) / (w.size - 2))

    return fhat


# }}}
