# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Iterator

import numpy as np

from pycaputo.grid import JacobiGaussLobattoPoints, Points
from pycaputo.utils import Array

# {{{ Jacobi polynomial related coefficients


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


def jacobi_diff_coefficient(n: int, k: int, alpha: float, beta: float) -> float:
    r"""Computes the coefficient for the Jacobi polynomial derivative
    (see Equation 3.102 [Shen2011]_).

    The same equation is given by Equation 3.57 [Li2020]_. However, it has an
    incorrect denominator, which should be :math:`\Gamma(n + \alpha + \beta + 1)`.

    :arg k: order of the derivative.
    :returns: the coefficient :math:`d^{\alpha, \beta}_{n, k}`.
    """
    from math import gamma

    return gamma(n + k + alpha + beta + 1) / (2.0**k * gamma(n + alpha + beta + 1))


def jacobi_rec_coefficients(
    n: int, alpha: float, beta: float
) -> tuple[float, float, float]:
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
) -> tuple[float, float, float]:
    r"""Computes coefficients for a derivative-based Jacobi recursion relation
    (see Equation 3.58 [Li2020]_).

    :returns: the coefficients
        :math:`\hat{A}^{\alpha, \beta}_n, \hat{B}^{\alpha, \beta}_n` and
        :math:`\hat{C}^{\alpha, \beta}_n`.
    """
    # NOTE: Equation 3.59 [Li2020]
    a = 2 * n + alpha + beta
    if n == 1:
        Ahat = 0.0
    else:
        Ahat = -2 * (n + alpha) * (n + beta) / (a * (a + 1) * (n + alpha + beta))
    Bhat = 2 * (alpha - beta) / (a * (a + 1))
    Chat = 2 * (n + alpha + beta + 1) / ((a + 1) * (a + 2))

    return Ahat, Bhat, Chat


def jacobi_weights_coefficient(n: int, alpha: float, beta: float) -> float:
    r"""Computes the coefficient used in the weights of the Jacobi-Gauss-Lobatto
    quadrature rule from Theorem 3.27 [Shen2011]_.

    :returns: the coefficient :math:`\tilde{G}^{\alpha, \beta}_n`.
    """
    from math import gamma

    return (
        2 ** (alpha + beta + 1)
        * gamma(n + alpha + 2)
        * gamma(n + beta + 2)
        / (gamma(n + 2) * gamma(n + alpha + beta + 2))
    )


# }}}

# {{{ Jacobi polynomial evaluation


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


def jacobi_derivative(n: int, k: int, alpha: float, beta: float, x: Array) -> Array:
    r"""Compute the :math:`k`-th derivative of a Jacobi polynomial.

    .. math::

        \frac{\mathrm{d}^k}{\mathrm{d} x^k} P^{\alpha, \beta}_n(x) =
        \frac{\Gamma(n + k + \alpha + \beta + 1)}{2^k \Gamma(n + \alpha + \beta + 1)}
        P^{\alpha + k, \beta + k}_{n - k}(x)

    :arg n: index of the Jacobi polynomial.
    :arg k: order of the derivative.
    """
    from math import gamma

    from scipy.special import eval_jacobi

    d = gamma(n + k + alpha + beta + 1) / gamma(n + alpha + beta + 1) / 2**k
    P = eval_jacobi(n - k, alpha + k, beta + k, x)

    return np.array(d * P)


# }}}

# {{{ Jacobi quadrature


def jacobi_gauss_lobatto_nodes(n: int, alpha: float, beta: float) -> Array:
    r"""Construct the Jacobi-Gauss-Lobatto nodes.

    The Jacobi-Gauss-Lobatto nodes :math:`\{x\}_{k = 1}^{n - 2}` are the
    zeros of the first derivative of the Jacobi polynomials
    :math:`P^{\alpha, \beta}_n(x)`, while :math:`x_0 = -1` and
    :math:`x_{n - 1} = 1`.

    :arg n: number of nodes to construct.
    :arg alpha: parameter of the Jacobi polynomial.
    :arg beta: parameter of the Jacobi polynomial.
    """
    from scipy.special import roots_jacobi

    x = np.empty(n)
    x[0] = -1.0
    x[-1] = 1.0
    x[1:-1], _, _ = roots_jacobi(n - 2, alpha + 1, beta + 1, mu=True)

    return x


def jacobi_gauss_lobatto_weights(x: Array, alpha: float, beta: float) -> Array:
    """Construct the Jacobi-Gauss-Lobatto quadrature weights.

    The weights are described in Theorem 3.27 [Shen2011]_ for the general case
    and in the following sections for special cases (e.g. Legendre-Gauss-Lobatto).
    In general, these weights have explicit, but complex expressions.

    Note that the quadrature rule with :math:`n` terms is accurate for
    polynomials up to :math:`2n - 1`.

    :arg x: Jacobi-Gauss-Lobatto nodes.
    :arg alpha: parameter of the Jacobi polynomial.
    :arg beta: parameter of the Jacobi polynomial.
    """
    from math import gamma

    # NOTE: Theorem 3.27 [Shen2011]
    n = x.size - 1
    w = np.empty_like(x)

    # fmt: off
    Gab = 2 ** (alpha + beta + 1) * gamma(n) / gamma(n + alpha + beta + 2)
    w[0] = (
        Gab * (beta + 1) * gamma(beta + 1) ** 2 * gamma(n + alpha + 1)
        / (gamma(n + beta + 1))
    )
    w[-1] = (
        Gab * (alpha + 1) * gamma(alpha + 1) ** 2 * gamma(n + beta + 1)
        / (gamma(n + alpha + 1))
    )

    Gab = jacobi_weights_coefficient(n - 2, alpha + 1, beta + 1)
    Jab = jacobi_derivative(n - 1, 1, alpha + 1, beta + 1, x[1:-1])
    w[1:-1] = Gab / (1 - x[1:-1]**2) ** 2 / Jab ** 2
    # fmt: on

    if __debug__:
        sum_w = np.sum(w)
        sum_w_ref = jacobi_gamma(0, alpha, beta)
        error = abs(sum_w - sum_w_ref) / abs(sum_w_ref)
        assert error < 100 * np.finfo(w.dtype).eps

    return w


# }}}

# {{{ Jacobi Riemann-Liouville integral


def jacobi_riemann_liouville_integral(
    p: JacobiGaussLobattoPoints, alpha: float, *, weighted: bool = True
) -> Iterator[Array]:
    r"""Computes an integral of the Jacobi polynomials used in the definition
    of the Riemann-Liouville integral (see Section 3.3 (I) in [Li2020]_).

    This effectively computes

    .. math::

        \hat{P}^{u, v}_n(x) =
            \int_{-1}^x (x - s)^{\alpha - 1} P^{u, v}_n(s) \,\mathrm{d} s,

    where :math:`P^{u, v}_n` is the usual Jacobi polynomial.

    :arg weighted: if *True*, the integral is taken in the interval :math:`[a, b]`,
        which gives an extra weight of :math:`h^{\alpha - 1}`.
    :returns: the Riemann-Liouville integral of the Jacobi polynomials of
        every order.
    """
    from math import gamma

    # translate x back to [-1, 1]
    xm = (p.b + p.a) / 2
    dx = (p.b - p.a) / 2
    x = (p.x - xm) / dx

    w = dx ** (alpha - 1) if weighted else 1.0

    # NOTE: Equation 3.64 [Li2020]
    Phat0 = (x + 1) ** alpha / gamma(alpha + 1)
    yield w * Phat0

    # fmt: off
    Phat1 = (
        x * (x + 1) ** alpha / gamma(alpha + 1)
        - alpha * (x + 1) ** (alpha + 1) / gamma(alpha + 2)
    )
    # fm    t: on
    Phat1 = (p.alpha + p.beta + 2) / 2 * Phat1 + (p.alpha - p.beta) / 2 * Phat0
    yield w * Phat1

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
        yield w * Phatn

        P1, P0 = P2, P1
        Phat1, Phat0 = Phatn, Phat1


# }}}

# {{{ Jacobi Caputo derivative


def jacobi_caputo_derivative(
    p: JacobiGaussLobattoPoints,
    alpha: float,
    *,
    weighted: bool = True,
) -> Iterator[tuple[int, Array]]:
    r"""Computes an integral of the Jacobi polynomials used in the definition
    of the Caputo derivative (see Section 4.4 in [Li2020]_).

    This effectively computes

    .. math::

        \hat{D}^{u, v, \alpha, m}_j(x) = \frac{1}{m - \alpha}
            \int_{-1}^x (x - s)^{m - \alpha - 1}
            \frac{\mathrm{d}^m}{\mathrm{d} s^m} P^{u, v}_j(s) \,\mathrm{d} s,

    where :math:`P^{u, v}_j` is the usual Jacobi polynomial and
    :math:`m - 1 < \alpha \le m`. Computing this integral is largely based on
    :func:`jacobi_riemann_liouville_integral` and Equation 3.50 [Li2020]_.
    Note that, by definition, :math:`\hat{D}^{u, v, \alpha, m}_j = 0` for
    :math:`0 \le j \le m - 1`.

    :arg weighted: if *True*, the integral is taken in the interval :math:`[a, b]`,
        which gives an extra weight of :math:`h^{-\alpha - 1}`.
    :returns: the Caputo derivative of the Jacobi polynomials of every order.
    """

    import math
    from dataclasses import replace

    m = int(math.ceil(alpha))
    pm = replace(p, alpha=p.alpha + m, beta=p.beta + m)
    # FIXME: the weight is h^-alpha in [Li2020], but that seems incorrect?
    w = ((p.b - p.a) / 2) ** (-alpha - 1) if weighted else 1

    for n, Phat in enumerate(
        jacobi_riemann_liouville_integral(pm, m - alpha, weighted=False)
    ):
        if n + m >= p.size:
            break

        # NOTE: compute D^{u, v, alpha, m}_{n + m} Equation 4.313 [Li2020]
        d = jacobi_diff_coefficient(n + m, m, p.alpha, p.beta)
        Dhat = d * Phat

        yield n + m, w * Dhat


# }}}

# {{{ Jacobi project


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
