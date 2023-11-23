# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from pycaputo.typing import Array

# {{{ FLMM: Starting Weights


def lmm_starting_weights(
    w: Array, sigma: Array, alpha: float, *, atol: float | None = None
) -> Iterator[tuple[int, Array]]:
    r"""Constructs starting weights for a given set of weights *w* of a
    fractional-order linear multistep method (FLMM).

    The starting weights are introduced to handle a lack of smoothness of the
    function :math:`f(x)` being integrated. They are constructed in such a way
    that they are exact for a series of monomials, which results in an accurate
    quadrature for functions of the form :math:`f(x) = f_i x^{\sigma_i} +
    x^{\beta} g(x)`, where :math:`g` is smooth and :math:`\beta < p`
    (see Remark 3.6 in [Li2020]_). This ensures that the Taylor expansion near
    the origin is sufficiently accurate.

    Therefore, they are obtained by solving

    .. math::

        \sum_{k = 0}^s w_{mk} k^{\sigma_q} =
        \frac{\Gamma(\sigma_q +1)}{\Gamma(\sigma_q + \alpha + 1)}
        m^{\sigma_q + \alpha} -
        \sum_{k = 0}^s w_{n - k} k^{\sigma_q}

    for a set of :math:`\sigma_q` powers. In the simplest case, we can just set
    :math:`\sigma \in \{0, 1, \dots, p - 1\}` and obtain integer powers. Other
    values can be chosen depending on the behaviour of :math:`f(x)` near the origin.

    Note that the starting weights are only computed for :math:`m \ge s`.
    The initial :math:`s` steps are expected to be computed in some other
    fashion.

    :arg w: convolution weights of an FLMM.
    :arg sigma: an array of monomial powers for which the starting weights are exact.
    :arg alpha: order of the fractional derivative to approximate.
    :arg atol: absolute tolerance used for the GMRES solver. If *None*, an
        exact LU-based solver is used instead (see Section 3.2 in [Diethelm2006]_
        for a discussion of these methods).

    :returns: the index and starting weights for every point :math:`m \ge s`.
    """

    from scipy.linalg import lu_factor, lu_solve
    from scipy.sparse.linalg import gmres
    from scipy.special import gamma

    s = sigma.size
    j = np.arange(1, s + 1).reshape(-1, 1)

    A = j**sigma
    assert A.shape == (s, s)
    assert np.all(np.isfinite(A))

    if atol is None:
        lu_p = lu_factor(A)

    for k in range(s, w.size):
        b = (
            gamma(sigma + 1) / gamma(sigma + alpha + 1) * k ** (sigma + alpha)
            - A @ w[k - s : k][::-1]
        )

        if atol is None:
            omega = lu_solve(lu_p, b)
        else:
            omega, _ = gmres(A, b, atol=atol)

        assert np.all(np.isfinite(omega))
        yield k, omega


def diethelm_starting_powers(order: int, alpha: float) -> Array:
    r"""Generate monomial powers for the starting weights from [Diethelm2006]_.

    The proposed starting weights are given in Section 3.1 from [Diethelm2006]_ as

    .. math::

        \sigma \in \{i + \alpha j \mid i, j \ge 0, i + \alpha j \le p - 1 \},

    where :math:`p` is the desired order. For certain values of :math:`\alpha`,
    these monomials can repeat, e.g. for :math:`\alpha = 0.1` we get the same
    value for :math:`(i, j) = (1, 0)` and :math:`(i, j) = (0, 10)`. This function
    returns only unique powers.

    :arg order: order of the LMM method.
    :arg alpha: order of the fractional operator.
    """
    from itertools import product

    result = np.array([
        gamma
        for i, j in product(range(order), repeat=2)
        if (gamma := i + abs(alpha) * j) <= order - 1
    ])

    return np.array(np.unique(result))


def lubich_starting_powers(order: int, alpha: float, *, beta: float = 1.0) -> Array:
    r"""Generate monomial powers for the starting weights from [Lubich1986]_.

    The proposed starting weights are given in Section 4.2 of [Lubich1986]_ as

    .. math::

        \sigma \in \{i + \beta - 1 \mid q \le p - 1\},

    where :math:`\beta` is chosen such that :math:`q + \beta - 1 \le p - 1`
    and :math:`q + \alpha + \beta - 1 < p`, according to Theorem 2.4 from
    [Lubich1986]_. In general multiple such :math:`\beta` exist and choosing
    more can prove beneficial.

    :arg order: order of the LMM method.
    :arg alpha: order of the fractional operator.
    """
    from math import floor

    # NOTE: trying to satisfy
    # 0 <= q <= p - \beta and 0 <= q < p + 1 - alpha - beta
    qmax = floor(max(0, min(order - beta, order - alpha - beta + 1))) + 1

    return np.array([q + beta - 1 for q in range(qmax)])


def garrappa_starting_powers(order: int, alpha: float) -> Array:
    r"""Generate monomial powers for the starting weights from
    `FLMM2 <https://www.mathworks.com/matlabcentral/fileexchange/47081-flmm2>`__.

    :arg order: order of the LMM method.
    :arg alpha: order of the fractional operator.
    """
    from math import ceil, floor

    if order == 2 and 0 < alpha < 1:
        # NOTE: this is the estimate from the FLMM2 MATLAB code
        k = floor(1 / abs(alpha))
    else:
        # NOTE: this should be vaguely the cardinality of `diethelm_bdf_starting_powers`
        k = ceil(order * (order - 1 + 2 * abs(alpha)) / (2 * abs(alpha)))

    return np.arange(k) * abs(alpha)


# }}}


# {{{ backward differentiation formulas


def lubich_bdf_weights(alpha: float, order: int, n: int) -> Array:
    r"""This function generates the weights for the BDF methods of [Lubich1986]_.

    Table 1 from [Lubich1986]_ gives the generating functions. The weights
    constructed here are the coefficients of the Taylor expansion of the
    generating functions. In particular, the generating functions are given
    by

    .. math::

        w^\alpha_p(\zeta) \triangleq
            \left(\sum_{k = 1}^p \frac{1}{k} (1 - \zeta)^k\right)^\alpha.

    The corresponding weights also have an explicit formulation based on the
    recursion formula (see Theorem 4 in [Garrappa2015b]_)

    .. math::

        w^\alpha_k \triangleq \frac{1}{k c_0} \sum_{j = 0}^{k - 1}
            (\alpha (k - j) - j) c_{k - j} w^\alpha_j

    where :math:`c_k` represent the coefficients of :math:`\zeta^k` in the
    generating function. While this formulae are valid for larger order :math:`p`,
    we restrict here to a maximum order of 6, as the classical BDF methods of
    larger orders are not stable.

    :arg alpha: power of the generating function, where a positive value is
        used to approximate the integral and a negative value is used to
        approximate the derivative.
    :arg order: order of the method, only :math:`1 \le p \le 6` is supported.
    :arg n: number of truncated terms in the power series of :math:`w^\alpha_k(\zeta)`.
    """
    if order <= 0:
        raise ValueError(f"Negative orders are not supported: {order}")

    if n <= order:
        raise ValueError(
            f"Number of points '{n}' cannot be smaller than the order '{order}'"
        )

    if order == 1:
        c = np.array([1.0, -1.0])
    elif order == 2:
        c = np.array([3.0 / 2.0, -2.0, 1.0 / 2.0])
    elif order == 3:
        c = np.array([11.0 / 6.0, -3.0, 3.0 / 2.0, -1.0 / 3.0])
    elif order == 4:
        c = np.array([25.0 / 12.0, -4.0, 3.0, -4.0 / 3.0, 1.0 / 4.0])
    elif order == 5:
        c = np.array([137.0 / 60.0, -5.0, 5.0, -10.0 / 3.0, 5.0 / 4.0, -1.0 / 5.0])
    elif order == 6:
        c = np.array([
            49.0 / 20.0,
            -6.0,
            15.0 / 2.0,
            -20.0 / 3.0,
            15.0 / 4.0,
            -6.0 / 5.0,
            1.0 / 6.0,
        ])
    else:
        raise ValueError(f"Unsupported order '{order}'")

    import scipy.special as ss

    w = np.empty(n + 1)
    indices = np.arange(n + 1)

    if order == 1:
        return np.array(ss.poch(indices, alpha - 1) / ss.gamma(alpha))
    else:
        w[0] = c[0] ** -alpha
        for k in range(1, n + 1):
            min_j = max(0, k - c.size + 1)
            max_j = k

            j = indices[min_j:max_j]
            omega = -(alpha * (k - j) + j) * c[1 : k + 1][::-1]

            w[k] = np.sum(omega * w[min_j:max_j]) / (k * c[0])

    return w


# }}}


# {{{ trapezoidal formulas


def trapezoidal_weights(alpha: float, n: int) -> Array:
    r"""This function constructions the weights for the trapezoidal method
    from Section 4.1 in [Garrappa2015b]_.

    The trapezoidal method has the generating function

    .. math::

        w^\alpha(\zeta) \triangleq \left(
            \frac{1}{2} \frac{1 + \zeta}{1 - \zeta}
        \right)^\alpha,

    which expands into an infinite series. This function truncates that series
    to the first *n* terms, which give the desired weights.

    :arg alpha: power of the generating function, where a positive value is
        used to approximate the integral and a negative value is used to
        approximate the derivative.
    :arg n: number of truncated terms in the power series of :math:`w^\alpha_k(\zeta)`.
    """

    try:
        from scipy.fft import fft, ifft
    except ImportError:
        from numpy.fft import fft, ifft

    omega_1 = np.empty(n + 1)
    omega_2 = np.empty(n + 1)

    omega_1[0] = omega_2[0] = 1.0
    for k in range(1, n + 1):
        omega_1[k] = ((alpha + 1) / k - 1) * omega_1[k - 1]
        omega_2[k] = (1 + (alpha - 1) / k) * omega_2[k - 1]

    # FIXME: this looks like it can be done faster by using rfft?
    omega_1_hat = fft(omega_1, 2 * omega_1.size)
    omega_2_hat = fft(omega_2, 2 * omega_2.size)
    omega = ifft(omega_1_hat * omega_2_hat)[: omega_1.size].real

    return np.array(omega / 2**alpha)


# }}}


# {{{ Newton-Gregory formulas


def newton_gregory_weights(alpha: float, k: int, n: int) -> Array:
    r"""This function constructions the weights for the Newton-Gregory method
    from Section 4.2 in [Garrappa2015b]_.

    The Newton-Gregory family of methods have the generating functions

    .. math::

        w^\alpha_k(\zeta) \triangleq
            \frac{G^\alpha_k(\zeta)}{(1 - \zeta)^\alpha},

    where :math:`G^\alpha_k(\zeta)` is the :math:`k`-term truncated power
    series expansion of

    .. math::

        G^\alpha(\zeta) = \left(\frac{1 - \zeta}{\log \zeta}\right)^\alpha.

    Note that this is not a classic expansion in the sense of [Lubich1986]_
    because the :math:`G^\alpha_k(\zeta)` function is first raised to the
    :math:`\alpha` power and then truncated to :math:`k` terms. In a standard
    scheme, the generating function :math:`w^\alpha_k(\zeta)` is the one that
    is truncated.

    :arg alpha: power of the generating function, where a positive value is
        used to approximate the integral and a negative value is used to
        approximate the derivative.
    :arg k: number of truncated terms in the power series of :math:`G^\alpha_k(\zeta)`.
    :arg n: number of truncated terms in the power series of :math:`w^\alpha_k(\zeta)`.
    """
    if k != 1:
        raise ValueError(f"Only 2-term truncations are supported: '{k + 1}'")

    omega = np.empty(n + 1)
    omega[0] = 1.0
    for i in range(1, n + 1):
        omega[i] = (1 + (alpha - 1) / i) * omega[i - 1]

    omega[1:] = (1 - alpha / 2) * omega[1:] + alpha / 2 * omega[:-1]
    omega[0] = 1 - alpha / 2.0

    return omega


# }}}
