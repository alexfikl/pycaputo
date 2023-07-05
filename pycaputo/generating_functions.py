# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo.utils import Array

# {{{ Lubich1986 weights


def lubich_bdf(alpha: float, order: int, n: int) -> Array:
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
    if alpha <= 0:
        raise ValueError(f"Negative values of alpha are not supported: {alpha}")

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
        c = np.array(
            [
                49.0 / 20.0,
                -6.0,
                15.0 / 2.0,
                -20.0 / 3.0,
                15.0 / 4.0,
                -6.0 / 5.0,
                1.0 / 6.0,
            ]
        )
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

        print(k, j, omega, w[min_j:max_j])
        w[k] = np.sum(omega * w[min_j:max_j]) / (k * c[0])

    return w


# }}}
