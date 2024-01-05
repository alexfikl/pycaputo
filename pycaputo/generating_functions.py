# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Iterator

import numpy as np

from pycaputo.utils import Array

# {{{ Lubich1986 weights


def lubich_bdf_starting_weights_count(order: int, alpha: float) -> int:
    """An estimate for the number of starting weights from [Lubich1986]_.

    :arg order: order of the BDF method.
    :arg alpha: order of the fractional derivative.
    """
    from math import floor

    return floor(1 / abs(alpha)) - 1


def lubich_bdf_weights(alpha: float, order: int, n: int) -> Array:
    r"""This function generates the weights for the p-BDF methods of [Lubich1986]_.

    Table 1 from [Lubich1986]_ gives the generating functions. The weights
    constructed here are the coefficients of the Taylor expansion of the
    generating functions. In particular, the generating functions are given
    by

    .. math::

        w^\alpha_p(\zeta) \triangleq
            \left(\sum_{k = 1}^p \frac{1}{k} (1 - \zeta)^k\right)^\alpha.

    The corresponding weights also have an explicit formulation based on the
    recursion formula

    .. math::

        w^\alpha_k \triangleq \frac{1}{k c_0} \sum_{j = 0}^{k - 1}
            (\alpha (k - j) - j) c_{k - j} w^\alpha_j

    where :math:`c_k` represent the coefficients of :math:`\zeta^k` in the
    generating function. While this formulae are valid for larger order :math:`p`,
    we restrict here to a maximum order of 6, as the classical BDF methods of
    larger orders are not stable.

    :arg alpha: power of the generating function.
    :arg order: order of the method, only :math:`1 \le p \le 6` is supported.
    :arg n: number of weights.
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

    w = np.empty(n)
    indices = np.arange(n)

    w[0] = c[0] ** alpha
    for k in range(1, n):
        min_j = max(0, k - c.size + 1)
        max_j = k

        j = indices[min_j:max_j]
        omega = (alpha * (k - j) - j) * c[1 : k + 1][::-1]

        w[k] = np.sum(omega * w[min_j:max_j]) / (k * c[0])

    return w


def lubich_bdf_starting_weights(
    w: Array, s: int, alpha: float, *, beta: float = 1.0, atol: float | None = None
) -> Iterator[Array]:
    r"""Constructs starting weights for a given set of weights *w* from [Lubich1986]_.

    The starting weights are introduced to handle a lack of smoothness of the
    functions being integrated. They are constructed in such a way that they
    are exact for a series of monomials, which results in an accurate
    quadrature for functions of the form :math:`f(x) = x^{\beta - 1} g(x)`,
    where :math:`g` is smooth.

    Therefore, they are obtained by solving

    .. math::

        \sum_{k = 0}^s w_{mk} k^{q + \beta - 1} =
        \frac{\Gamma(q + \beta)}{\Gamma(q + \beta + \alpha)}
        m^{q + \beta + \alpha - 1} -
        \sum_{k = 0}^s w_{n - k} j^{q + \beta - 1}

    where :math:`q \in \{0, 1, \dots, s - 1\}` and :math:`\beta \ne \{0, -1, \dots\}`.
    In the simplest case, we can take :math:`\beta = 1` and obtain integer
    powers. Other values can be chosen depending on the behaviour of :math:`f(x)`
    near the origin.

    Note that the starting weights are only computed for :math:`m \ge s`.
    The initial :math:`s` steps are expected to be computed in some other
    fashion.

    :arg w: convolution weights defined by [Lubich1986]_.
    :arg s: number of starting weights.
    :arg alpha: order of the fractional derivative to approximate.
    :arg beta: order of the singularity at the origin.
    :arg atol: absolute tolerance used for the GMRES solver. If *None*, an
        exact LU-based solver is used instead.

    :returns: the starting weights for every point :math:`x_m` starting with
        :math:`m \ge s`.
    """

    if s <= 0:
        raise ValueError(f"Negative s is not allowed: {s}")

    if beta.is_integer() and beta <= 0:
        raise ValueError(f"Values of beta in 0, -1, ... are not supported: {beta}")

    from scipy.linalg import lu_factor, lu_solve
    from scipy.sparse.linalg import gmres
    from scipy.special import gamma

    q = np.arange(s) + beta - 1
    j = np.arange(1, s + 1).reshape(-1, 1)

    A = j**q
    assert A.shape == (s, s)

    if atol is None:
        lu, p = lu_factor(A)

    for k in range(s, w.size):
        b = (
            gamma(q + 1) / gamma(q + alpha + 1) * k ** (q + alpha)
            - A @ w[k - s : k][::-1]
        )

        if atol is None:
            omega = lu_solve((lu, p), b)
        else:
            omega, _ = gmres(A, b, atol=atol)

        yield omega


# }}}
