# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycaputo.logging import get_logger
from pycaputo.utils import Array, ScalarFunction

logger = get_logger(__name__)


# {{{ estimate_lipschitz_constant


@dataclass
class InverseWeibullCumulativeDistributionFunction:
    r"""Evaluates a three-parameter family of inverse Weibull distributions.

    .. math::

        F(x) =
        \begin{cases}
        \displaystyle
        \exp \left(-\frac{(u - x)^w}{v}\right), & \quad x < u, \\
        1, & x > u.
        \end{cases}
    """

    #: Location parameter.
    u: float
    #: Scale parameter.
    v: float
    #: Shape parameter.
    w: float

    def __call__(self, x: Array) -> Array:
        return self.evaluate(x)

    def evaluate(self, x: Array) -> Array:
        """Evaluate the CDF at the given points."""
        return np.where(x < self.u, np.exp(-((self.u - x) ** self.w) / self.v), 1)

    def jac(self, x: Array) -> Array:
        """Evaluate the Jacobian of the CDF with respect to the parameters.

        :arg x: points at which to evaluate the CDF.
        :returns: an array of shape ``(3, x.size)``.
        """

        u, v, w = self.u, self.v, self.w
        cdf = self.evaluate(x)
        return np.stack(
            [
                np.where(x < u, -w * (u - x) ** (w - 1) / v * cdf, 0.0),
                np.where(x < u, (u - x) ** w / v**2 * cdf, 0.0),
                np.where(x < u, -((u - x) ** w) * np.log(u - x) / v * cdf, 0.0),
            ]
        )


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
        s = np.abs(f(x) - f(y)) / (np.abs(x - y) + 1.0e-7)

        smax[m] = np.max(s)

    return smax


def fit_inverse_weibull(
    x: Array, *, verbose: bool = False
) -> InverseWeibullCumulativeDistributionFunction:
    """Fits an inverse Weibull distribution to the CDF of *x*.

    See :class:`InverseWeibullCumulativeDistributionFunction` for details on the
    functions involved.

    :returns: the distribution with the optimal parameters.
    """

    import scipy.optimize
    import scipy.stats

    x = np.sort(x)
    empirical_cdf = scipy.stats.ecdf(x).cdf.probabilities

    def f_opt(p: Array) -> Array:
        F = InverseWeibullCumulativeDistributionFunction(*p)
        return np.array(empirical_cdf - F.evaluate(x))

    def f_opt_jac(p: Array) -> Array:
        F = InverseWeibullCumulativeDistributionFunction(*p)
        return F.jac(x)

    # NOTE: 0: location parameter u, 1: scale parameter v, 2: shape parameter w
    x0 = np.array([np.median(x), 0.1, 1.0])
    result = scipy.optimize.least_squares(
        f_opt,
        x0,
        # jac=f_opt_jac,
        method="lm",
        verbose=verbose,
    )

    if verbose:
        logger.info("Convergence:%s\n%s", result.x, result)

    if not result.success:
        raise RuntimeError("Unable to determine constant")

    return InverseWeibullCumulativeDistributionFunction(*result.x)


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

    smax = uniform_sample_maximum_slopes(
        f,
        a,
        b,
        delta=delta,
        nslopes=nslopes,
        nbatches=nbatches,
        rng=rng,
    )
    try:
        cdf = fit_inverse_weibull(smax)

        # NOTE: the estimate for the Lipschitz constant is the location parameter
        return cdf.u
    except RuntimeError:
        logger.error("Unable to determine constant.")
        return float(np.max(smax))


# }}}
