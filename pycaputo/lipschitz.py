# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import scipy.stats

from pycaputo.logging import get_logger
from pycaputo.utils import Array, ScalarFunction

logger = get_logger(__name__)


# {{{ estimate_lipschitz_constant


def uniform_diagonal_sample(
    a: float,
    b: float,
    n: int,
    *,
    delta: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[Array, Array]:
    r"""Sample points uniformly around the diagonal of :math:`[a, b] \times [a, b]`.

    This function samples point in

    .. math::

        \{(x, y) \in [a, b] \times [a, b] \mid |x - y| < \delta\}.

    :arg n: number of points to sample.
    :arg rng: random number generator to use, which defaults to
        :func:`numpy.random.default_rng`.
    :returns: an array of shape ``(2, n)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    # TODO: is this actually uniformly distributed on the domain?
    # FIXME: any nice vectorized way to do this?
    x, y = np.empty((2, n))
    for i in range(n):
        x[i] = rng.uniform(a, b)
        y[i] = rng.uniform(max(a, x[i] - delta), min(x[i] + delta, b))

    return x, y


def uniform_sample_maximum_slopes(
    f: ScalarFunction,
    a: float,
    b: float,
    *,
    delta: float | None = None,
    nslopes: int = 8,
    nbatches: int = 64,
    rng: np.random.Generator | None = None,
) -> Array:
    r"""Uniformly sample slopes of *f* in a neighborhood the diagonal of
    :math:`[a, b] \times [a, b]`.

    This function uses :func:`uniform_diagonal_sample` and takes the same
    arguments as :func:`estimate_lipschitz_constant`.

    :returns: an array of shape ``(nbatches,)`` of maximum slopes for the function
        *f* on the interval :math:`[a, b]`.
    """
    if a > b:
        raise ValueError(f"Invalid interval: [{a}, {b}]")

    d = np.sqrt(2) * (b - a)
    if delta is None:
        # NOTE: in the numerical tests from [Wood1996] they take `delta = 0.05`
        # regardless of the interval, but here we take a scaled version by default
        # that's about `0.044` on `[-1, 1]` and grows on bigger intervals
        delta = 1 / 64 * d

    if delta <= 0:
        raise ValueError(f"delta should be positive: {delta}")

    if rng is None:
        rng = np.random.default_rng()

    smax = np.empty(nbatches)
    for m in range(nbatches):
        x, y = uniform_diagonal_sample(a, b, nslopes, delta=delta, rng=rng)
        s = np.abs(f(x) - f(y)) / (np.abs(x - y) + 1.0e-15)

        smax[m] = np.max(s)

    return smax


def fit_reverse_weibull(x: Array) -> scipy.stats.rv_continuous:
    """Fits a reverse Weibull distribution to the CDF of *x*.

    See :data:`scipy.stats.weibull_max` for details on the distribution.

    :returns: the distribution with the optimal parameters.
    """
    # NOTE: MLE seems to work better than MM
    c, loc, scale = scipy.stats.weibull_max.fit(x, method="MLE")
    return scipy.stats.weibull_max(c=c, loc=loc, scale=scale)


def estimate_lipschitz_constant(
    f: ScalarFunction,
    a: float,
    b: float,
    *,
    delta: float | None = None,
    nslopes: int = 8,
    nbatches: int = 64,
    rng: np.random.Generator | None = None,
) -> float:
    r"""Estimate the Lipschitz constant of *f* based on [Wood1996]_.

    The Lipschitz constant is defined, for a function
    :math:`f: [a, b] \to \mathbb{R}`, by

    .. math::

        |f(x) - f(y)| \le L |x - y|, \qquad \forall x, y \in [a, b]

    .. warning::

        The current implementation of this function seems to underpredict the
        Lipschitz constant, unlike the results from [Wood1996]_. This is likely
        a consequence of the method used to fit the data to the Weibull
        distribution.

    :arg a: left-hand side of the domain.
    :arg b: right-hand side of the domain.
    :arg delta: a width of a strip around the diagonal of the
        :math:`[a, b] \times [a, b]` domain for :math:`(x, y)` sampling.
    :arg nslopes: number of slopes to sample.
    :arg nbatches: number of batches of slopes to sample.

    :returns: an estimate of the Lipschitz constant.
    """

    smax = uniform_sample_maximum_slopes(
        f,
        a,
        b,
        delta=delta,
        nslopes=nslopes,
        nbatches=nbatches,
        rng=rng,
    )
    cdf = fit_reverse_weibull(smax)

    # NOTE: the estimate for the Lipschitz constant is the location parameter
    return float(cdf.kwds["loc"])


# }}}
