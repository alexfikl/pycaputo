# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import enum
import math
from typing import Union

import numpy as np

from pycaputo.logging import get_logger
from pycaputo.utils import Array

logger = get_logger(__name__)


@enum.unique
class Algorithm(enum.Enum):
    Series = enum.auto()


def _mittag_leffler_series(
    z: complex,
    *,
    alpha: float,
    beta: float,
) -> complex:
    eps = 2 * np.finfo(np.array(z).dtype).eps
    kmax = 2048

    result, term = 0.0, 1.0
    k = 0
    while abs(term) > eps and k <= kmax:
        term = z**k / math.gamma(alpha * k + beta)
        result += term
        k += 1

    if abs(term) > eps:
        logger.error("Series did not converge for E[%g, %g](%g)", alpha, beta, z)

    return result


def mittag_leffler(
    z: Union[float, complex, Array],
    alpha: float = 0.0,
    beta: float = 1.0,
    alg: Algorithm = Algorithm.Series,
) -> Array:
    r"""Evaluate the Mittag-Leffler function :math:`E_{\alpha, \beta}(z)`.

    Several special cases are handled explicitly and otherwise, the
    approximation algorithm can be chosen by *alg*.

    :arg z: values at which to compute the Mittag-Leffler function.
    :arg alpha: parameter of the function.
    :arg beta: parameter of the function.
    :arg alg: the algorithm used to compute the function.
    """
    if alpha < 0 or beta < 0:
        raise NotImplementedError(
            "Negative parameters are not implemented: "
            + f"alpha '{alpha}' and beta '{beta}'"
        )

    # NOTE: special cases taken from:
    #       https://arxiv.org/abs/0909.0230
    if beta == 1:
        if alpha == 0:
            return 1 / (1 - z)
        if alpha == 1:
            return np.exp(z)
        if alpha == 2:
            return np.cosh(np.sqrt(z))
        if alpha == 3:
            z = np.cbrt(z)
            return (np.exp(z) + 2 * np.exp(-z / 2) * np.cos(np.sqrt(3) * z / 2)) / 3
        if alpha == 4:
            z = np.sqrt(np.sqrt(z))
            return (np.cos(z) + np.cosh(z)) / 2
        if alpha == 0.5:
            try:
                from scipy.special import erfc

                return np.exp(z**2) * erfc(-z)
            except ImportError:
                pass

    if beta == 2:
        if alpha == 1:
            return (np.exp(z) - 1) / z
        if alpha == 2:
            z = np.sqrt(z)
            return np.sinh(z) / z

    if alpha == 0 and np.all(abs(z) < 1):
        return 1 / (1 - z) / math.gamma(beta)

    if alg == Algorithm.Series:
        return np.vectorize(_mittag_leffler_series)(z, alpha=alpha, beta=beta)

    raise ValueError(f"Unknown algorithm: '{alg}'")
