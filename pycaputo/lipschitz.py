# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo.logging import get_logger
from pycaputo.utils import Array, ScalarFunction

logger = get_logger(__name__)


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

    :arg a: left-hand side of the domain.
    :arg b: right-hand side of the domain.
    :arg delta: a width of a strip around the diagonal of the
        :math:`[a, b] \times [a, b]` domain for :math:`(x, y)` sampling.
    :arg nslopes: number of slopes to sample.
    :arg nbatches: number of batches of slopes to sample.

    :returns: an estimate of the Lipschitz constant.
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

    # generate slope maxima
    maxs = np.empty(nbatches)
    for m in range(nbatches):
        x, y = uniform_diagonal_sample(a, b, nslopes, delta=delta, rng=rng)
        s = np.abs(f(x) - f(y)) / (np.abs(x - y) + 1.0e-7)

        maxs[m] = np.max(s)

    # fit the slopes to the three-parameter inverse Weibull distribution
    import scipy.optimize
    import scipy.stats

    maxs = np.sort(maxs)
    F = scipy.stats.ecdf(maxs)
    Fx = F.cdf.probabilities

    def cdf_inverse_weibull(x: Array, p: Array) -> Array:
        u, v, w = p
        return np.where(x < u, np.exp(-((u - x) ** w) / v), 1)

    def cdf_inverse_weibull_jac(x: Array, p: Array) -> Array:
        u, v, w = p
        cdf = cdf_inverse_weibull(x, p)
        return np.stack(
            [
                np.where(x < u, -w * (u - x) ** (w - 1) / v * cdf, 0.0),
                np.where(x < u, (u - x) ** w / v**2 * cdf, 0.0),
                np.where(x < u, -((u - x) ** w) * np.log(u - x) / v * cdf, 0.0),
            ]
        )

    def f_opt(p: Array) -> float:
        r = 0.5 * np.sum((Fx - cdf_inverse_weibull(maxs, p)) ** 2)
        assert np.all(np.isfinite(r))
        return float(r)

    def f_opt_jac(p: Array) -> Array:
        a = Fx - cdf_inverse_weibull(maxs, p)
        dW_du, dW_dv, dW_dw = cdf_inverse_weibull_jac(maxs, p)

        r = np.array([a @ dW_du, a @ dW_dv, a @ dW_dw])
        assert np.all(np.isfinite(r))
        return r

    # NOTE: 0: location parameter u, 1: scale parameter v, 2: shape parameter w
    x0 = np.array([maxs[nbatches // 2], 1.0, 1.0])
    result = scipy.optimize.minimize(
        f_opt,
        x0,
        jac=f_opt_jac,
        # bounds=[(0.0, np.inf), (0, 1.0), (1.0, 5.0)],
        method="SLSQP",
        options={"disp": True},
    )
    logger.info("Convergence: x0 %s maxs %s\n%s", x0, np.max(maxs), result)
    print(f(x0), f(result.x), f(np.array([1.0, 0.004, 1.0])))
    if not result.success:
        logger.error("Unable to determine Lipschitz constant")

    # NOTE: the estimate for the Lipschitz constant is the location parameter
    return float(result.x[0])
