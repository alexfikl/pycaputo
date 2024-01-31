# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import enum
import math
from typing import Any

import numpy as np

from pycaputo.logging import get_logger
from pycaputo.utils import Array

logger = get_logger(__name__)


@enum.unique
class Algorithm(enum.Enum):
    """Algorithm used to compute the Mittag-Leffler function."""

    #: The standard series definition is used to compute the function. This
    #: choice can be very slow to converge for certain arguments and parameters.
    #: It is not recommended for production use.
    Series = enum.auto()

    #: Algorithm by [Diethelm2005]_.
    Diethelm = enum.auto()

    #: Algorithm by [Garrappa2015]_.
    Garrappa = enum.auto()

    #: Algorithm by [Ortigueira2019]_.
    Ortigueira = enum.auto()


# {{{ series


def mittag_leffler_series(
    z: complex,
    *,
    alpha: float,
    beta: float,
    eps: float | None = None,
    kmax: int | None = None,
) -> complex:
    if eps is None:
        eps = 2 * float(np.finfo(np.array(z).dtype).eps)

    if kmax is None:
        kmax = 2048

    result, term = 0j, 1j
    k = 0
    while abs(term) > eps and k <= kmax:
        term = z**k / math.gamma(alpha * k + beta)
        result += term
        k += 1

    if abs(term) > eps:
        logger.error("Series did not converge for E[%g, %g](%g)", alpha, beta, z)

    return result


# }}}


# {{{ Diethelm


def _ml_quad_k(
    a: float,
    b: float,
    alpha: float,
    beta: float,
    z: complex,
    *,
    eps: float,
    delta: float | None = None,
) -> complex:
    if delta is None:
        delta = 1.0e-2 * math.sqrt(eps)

    sin_pb = math.sin(np.pi * (1 - beta))
    sin_ab = math.sin(np.pi * (1 - beta + alpha))
    cos_pa = math.cos(np.pi * alpha)
    inv_aa = 1 / alpha
    inv_ab = (1 - beta) / alpha

    def K(chi: float) -> complex:  # noqa: N802
        r = (
            (chi * sin_pb - z * sin_ab)
            / (chi**2 - 2 * chi * z * cos_pa + z**2)
            / (alpha * np.pi)
        )

        return complex(chi**inv_ab * math.exp(-(chi**inv_aa)) * r)

    from functools import partial

    from scipy.integrate import quad

    quad = partial(quad, epsabs=eps, epsrel=eps, limit=1000, complex_func=True)
    if a < abs(z) < b:
        rl, _ = quad(K, a, abs(z) - delta)
        rr, _ = quad(K, abs(z) + delta, b)

        r = rl + rr
    else:
        r, _ = quad(K, a, b)

    return complex(r)


def _ml_quad_p(
    a: float, b: float, alpha: float, beta: float, rho: float, z: complex, *, eps: float
) -> complex:
    inv_ab = 1 + (1 - beta) / alpha
    rho_inva = rho ** (1 / alpha)
    rho_invb = rho**inv_ab

    def P(phi: float) -> complex:  # noqa: N802:
        omega = phi * inv_ab + rho_inva * math.sin(phi / alpha)
        r = (
            (math.cos(omega) + 1j * math.sin(omega))
            / (rho * (math.cos(phi) + 1j * math.sin(phi)) - z)
            / (2.0 * alpha * np.pi)
        )
        return complex(rho_invb * math.exp(rho_inva * math.cos(phi / alpha)) * r)

    from scipy.integrate import quad

    r, _ = quad(
        P,
        a,
        b,
        epsabs=eps,
        epsrel=eps,
        limit=1000,
        complex_func=True,
    )

    return complex(r)


def mittag_leffler_diethelm(
    z: complex,
    *,
    alpha: float,
    beta: float,
    eps: float | None = None,
    zeta: float | None = None,
) -> complex:
    if eps is None:
        eps = 2 * float(np.finfo(np.array(z).dtype).eps)

    if zeta is None:
        zeta = 0.9

    if alpha == 0:
        return mittag_leffler_series(z, alpha=alpha, beta=beta, eps=eps)

    assert eps is not None
    assert zeta is not None

    # NOTE: implements Algorithm 4 from [Diethelm2005]
    if abs(z) == 0:
        return 1 / math.gamma(beta)

    if alpha > 1:
        k0 = math.floor(alpha) + 1
        z = z ** (1.0 / k0)
        alpha = alpha / k0

        def rec_ml(k: int) -> complex:
            return mittag_leffler_diethelm(
                z * np.exp(2j * np.pi * k / k0),
                alpha=alpha,
                beta=beta,
                eps=eps,
                zeta=zeta,
            )

        return sum((rec_ml(k) for k in range(k0)), 0.0) / k0

    zabs = abs(z)
    if zabs < zeta:
        # FIXME: Gamma(alpha * k + beta) really overflows for large k0; there's
        # probably some way to figure out when it gets too large and truncate
        # k0 further

        k0 = max(
            math.ceil((1 - beta) / alpha),
            math.ceil(math.log(eps * (1 - zabs)) / math.log(zabs)),
        )

        from scipy.special import gamma

        return sum((z**k / gamma(alpha * k + beta) for k in range(k0 + 1)), 0.0)

    import cmath

    zarg = abs(cmath.phase(z))
    if zabs < math.floor(10 + 5 * alpha):
        if beta >= 0:
            chi0 = max(1, 2 * zabs, (-math.log(eps * np.pi / 6)) ** alpha)
        else:
            babs = abs(beta)
            chi0 = (
                -2 * math.log(eps * np.pi / 6 / (babs + 2) / (2 * babs) ** babs)
            ) ** alpha
            chi0 = max((1 + babs) ** alpha, 2 * zabs, chi0)

        api = alpha * np.pi
        if zarg > api and abs(zarg - api) > eps:
            if beta < 1 + alpha:
                K = _ml_quad_k(0, chi0, alpha, beta, z, eps=eps)
                P = 0j
            else:
                K = _ml_quad_k(1, chi0, alpha, beta, z, eps=eps)
                P = _ml_quad_p(-api, api, alpha, beta, 1.0, z, eps=eps)

            return K + P

        if zarg < alpha * np.pi and abs(zarg - alpha * np.pi) > eps:
            if beta < 1 + alpha:
                K = _ml_quad_k(0, chi0, alpha, beta, z, eps=eps)
                P = 0j
            else:
                K = _ml_quad_k(zabs / 2, chi0, alpha, beta, z, eps=eps)
                P = _ml_quad_p(-api, api, alpha, beta, zabs / 2, z, eps=eps)

            R = z ** ((1 - beta) / alpha) * cmath.exp(z ** (1 / alpha)) / alpha
            return K + P + R

        # NOTE: modified based on
        #   https://sistemas.fc.unesp.br/ojs/index.php/revistacqd/article/view/306
        K = _ml_quad_k(zabs + 1, chi0, alpha, beta, z, eps=eps)
        P = _ml_quad_p(-api, api, alpha, beta, zabs + 1, z, eps=eps)

        return K + P

    k0 = math.floor(-math.log(eps) / math.log(zabs))
    result = -sum((z**-k / math.gamma(beta - alpha * k) for k in range(k0)), 0.0)

    if zarg < 0.75 * alpha * np.pi:
        result += z ** ((1 - beta) / alpha) * cmath.exp(z ** (1 / alpha)) / alpha

    return result


# }}}


# {{{ Ortigueira


def mittag_leffler_ortigueira(
    z: complex,
    *,
    alpha: float,
    beta: float,
    eps: float | None = None,
    kmax: int | None = None,
) -> complex:
    if eps is None:
        eps = 2 * float(np.finfo(np.array(z).dtype).eps)

    if kmax is None:
        kmax = 2048

    if abs(z) == 0:
        return 1 / math.gamma(beta)

    raise NotImplementedError


# }}}


# {{{ Garrappa


def mittag_leffler_garrapa(
    z: complex,
    *,
    alpha: float,
    beta: float,
) -> complex:
    if abs(z) == 0:
        return 1 / math.gamma(beta)

    raise NotImplementedError


# }}}


def mittag_leffler_special(
    z: float | complex | Array,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Array | None:
    r"""Compute special cases of the Mittag-Leffler function.

    For special values of the :math:`(\alpha, \beta)` parameters, the Mitta-Leffler
    function reduces to more well-known special functions. For example,
    :math:`E_{1, 1}(x) = \exp(x)`.

    :return: the value of the Mittag-Leffler function or *None* if no special
        case is known for the given pair :math:`(\alpha, \beta)`.
    """
    # NOTE: special cases taken from:
    #       https://arxiv.org/abs/0909.0230

    z = np.array(z)
    if beta == 1:
        if alpha == 0:
            return np.array(1 / (1 - z))
        if alpha == 1:
            return np.array(np.exp(z))
        if alpha == 2:
            return np.cosh(np.sqrt(z))  # type: ignore[no-any-return]
        if alpha == 3:
            z = np.cbrt(z)
            return (  # type: ignore[no-any-return]
                np.exp(z) + 2 * np.exp(-z / 2) * np.cos(np.sqrt(3) * z / 2)
            ) / 3
        if alpha == 4:
            z = np.sqrt(np.sqrt(z))
            return (np.cos(z) + np.cosh(z)) / 2  # type: ignore[no-any-return]
        if alpha == 0.5:
            from scipy.special import erfc

            return np.exp(z**2) * erfc(-z)  # type: ignore[no-any-return]

    if beta == 2:
        if alpha == 1:
            return np.array((np.exp(z) - 1) / z)
        if alpha == 2:
            z = np.sqrt(z)
            return np.sinh(z) / z

    if alpha == 0 and np.all(np.abs(z) < 1):
        return 1 / (1 - z) / math.gamma(beta)

    return None


def mittag_leffler(
    z: float | complex | Array,
    alpha: float = 1.0,
    beta: float = 1.0,
    *,
    alg: Algorithm | None = None,
    use_explicit: bool = True,
) -> Array:
    r"""Evaluate the Mittag-Leffler function :math:`E_{\alpha, \beta}(z)`.

    Several special cases are handled explicitly and otherwise, the
    approximation algorithm can be chosen by *alg*.

    :arg z: values at which to compute the Mittag-Leffler function.
    :arg alpha: parameter of the function.
    :arg beta: parameter of the function.
    :arg alg: the algorithm used to compute the function.
    :arg use_explicit: if *True*, explicit formulae are used for some known
        sets of parameters. These can be significantly faster.
    """
    if alpha < 0 or beta < 0:
        raise NotImplementedError(
            "Negative parameters are not implemented: "
            f"alpha '{alpha}' and beta '{beta}'"
        )

    if alg is None:
        # NOTE: for now this algorithm should be faster
        alg = Algorithm.Diethelm

    if use_explicit:
        result = mittag_leffler_special(z, alpha, beta)
        if result is not None:
            return result

    func: Any
    if alg == Algorithm.Series:
        func = mittag_leffler_series
    elif alg == Algorithm.Diethelm:
        func = mittag_leffler_diethelm
    elif alg == Algorithm.Garrappa:
        func = mittag_leffler_garrapa
    elif alg == Algorithm.Ortigueira:
        func = mittag_leffler_ortigueira
    else:
        raise ValueError(f"Unknown algorithm: '{alg}'")

    ml = np.vectorize(lambda zi: 0j + func(zi, alpha=alpha, beta=beta))
    return np.array(ml(z))
