# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import enum
import math
from functools import partial
from typing import Any

import numpy as np

from pycaputo.logging import get_logger
from pycaputo.utils import Array

logger = get_logger(__name__)


@enum.unique
class Algorithm(enum.Enum):
    """Algorithm used to compute the Mittag-Leffler function."""

    Series = enum.auto()
    """The standard series definition is used to compute the function. This
    choice can be very slow to converge for certain arguments and parameters.
    It is not recommended for production use.
    """

    Diethelm = enum.auto()
    """Algorithm by [Diethelm2005]_."""

    Garrappa = enum.auto()
    """Algorithm by [Garrappa2015]_."""

    Ortigueira = enum.auto()
    """Algorithm by [Ortigueira2019]_."""


# {{{ series


def _mittag_leffler_series(
    z: complex,
    *,
    alpha: float,
    beta: float,
    eps: float | None = None,
    kmax: int | None = None,
) -> complex:
    z = complex(z)
    if eps is None:
        eps = 5 * float(np.finfo(np.array(z).dtype).eps)

    if kmax is None:
        kmax = 2048

    if abs(z) >= 1.0:
        from warnings import warn

        warn(
            "The series expansion of the Mittag-Leffler function "
            "converges very slowly for z > 1. Use a more appropriate method.",
            stacklevel=2,
        )

    result, term = 0j, 1j
    k = 0
    while abs(term) > eps and k <= kmax:
        term = z**k / math.gamma(alpha * k + beta)
        result += term
        k += 1

    if abs(term) > eps:
        logger.error(
            "Series did not converge for E[%g, %g](%g+%gj)", alpha, beta, z.real, z.imag
        )

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


def _mittag_leffler_diethelm(
    z: complex,
    *,
    alpha: float,
    beta: float,
    eps: float | None = None,
    zeta: float | None = None,
) -> complex:
    if eps is None:
        eps = 5 * float(np.finfo(np.array(z).dtype).eps)

    if zeta is None:
        zeta = 0.9

    if alpha == 0:
        return _mittag_leffler_series(z, alpha=alpha, beta=beta, eps=eps)

    assert eps is not None
    assert zeta is not None

    # NOTE: implements Algorithm 4 from [Diethelm2005]
    if abs(z) == 0:
        return 1 / math.gamma(beta)

    # NOTE: ensure z is complex (some of the operations below are not well defined
    # otherwise e.g. z ** 1/k0 for negative z)
    z = 0.0j + z

    if alpha > 1:
        k0 = math.floor(alpha) + 1
        z = z ** (1.0 / k0)
        alpha = alpha / k0

        def rec_ml(k: int) -> complex:
            return _mittag_leffler_diethelm(
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
    result = -sum((z**-k / math.gamma(beta - alpha * k + eps) for k in range(k0)), 0.0)

    if zarg < 0.75 * alpha * np.pi:
        result += z ** ((1 - beta) / alpha) * cmath.exp(z ** (1 / alpha)) / alpha

    return result


# }}}


# {{{ Garrappa


def _find_optimal_bounded_param(
    t: float,
    phi_star0: float,
    phi_star1: float,
    p: float,
    q: float,
    *,
    log_eps: float,
    log_machine_eps: float,
    fac: float = 1.01,
    p_eps: float = 1.0e-14,
    q_eps: float = 1.0e-14,
    conservative_error_analysis: bool = False,
) -> tuple[float, float, float]:
    # set maximum value of fbar (the ratio of the tolerance to the machine epsilon)
    f_max = math.exp(log_eps - log_machine_eps)
    threshold = 2 * np.sqrt((log_eps - log_machine_eps) / t)

    # starting values
    phi_star0_sq = np.sqrt(phi_star0)
    phi_star1_sq = min(np.sqrt(phi_star1), threshold - phi_star0_sq)

    # determine phibar and admissible region
    if p < p_eps:
        if q < q_eps:
            phibar_star0_sq = phi_star0_sq
            phibar_star1_sq = phi_star1_sq
            adm_region = True
        else:
            phibar_star0_sq = phi_star0_sq
            if phi_star0_sq > 0:
                f_min = fac * (phi_star0_sq / (phi_star1_sq - phi_star0_sq)) ** q
            else:
                f_min = fac

            if f_min < f_max:
                f_bar = f_min + f_min / f_max * (f_max - f_min)
                fq = f_bar ** (-1 / q)
                phibar_star1_sq = (2 * phi_star1_sq - fq * phi_star0_sq) / (2 + fq)
                adm_region = True
            else:
                adm_region = False
    else:  # noqa: PLR5501
        if q < q_eps:
            phibar_star1_sq = phi_star1_sq
            f_min = fac * (phi_star1_sq / (phi_star1_sq - phi_star0_sq)) ** p
            if f_min < f_max:
                f_bar = f_min + f_min / f_max * (f_max - f_min)
                fp = f_bar ** (-1 / p)
                phibar_star0_sq = (2 * phi_star0_sq - fp * phi_star1_sq) / (2 - fp)
                adm_region = True
            else:
                adm_region = False
        else:
            f_min = (
                fac
                * (phi_star1_sq + phi_star0_sq)
                / (phi_star1_sq - phi_star0_sq) ** max(p, q)
            )
            if f_min < f_max:
                f_min = max(f_min, 1.5)
                f_bar = f_min + f_min / f_max * (f_max - f_min)
                fp = f_bar ** (-1 / p)
                fq = f_bar ** (-1 / q)

                if not conservative_error_analysis:
                    w = -phi_star1 * t / log_eps
                else:
                    w = -2 * phi_star1 * t / (log_eps - phi_star1 * t)

                den = 2 + w - (1 + w) * fp + fq
                phibar_star0_sq = (
                    (2 + w + fq) * phi_star0_sq + fp * phi_star1_sq
                ) / den
                phibar_star1_sq = (
                    -(1 + w) * fq * phi_star0_sq + (2 + w - (1 + w) * fp) * phi_star1_sq
                ) / den
                adm_region = True
            else:
                adm_region = False

    if adm_region:
        log_eps = log_eps - math.log(f_bar)
        if not conservative_error_analysis:
            w = -(phibar_star1_sq**2) * t / log_eps
        else:
            w = -2 * phibar_star1_sq**2 * t / (log_eps - phibar_star1_sq**2 * t)

        mu = (((1 + w) * phibar_star0_sq + phibar_star1_sq) / (2 + w)) ** 2
        h = (
            -2
            * np.pi
            / log_eps
            * (phibar_star1_sq - phibar_star0_sq)
            / ((1 + w) * phibar_star0_sq + phibar_star1_sq)
        )
        N = math.ceil(np.sqrt(1 - log_eps / t / mu) / h)
    else:
        mu = 0
        h = 0
        N = np.inf

    return mu, N, h


def _find_optimal_unbounded_param(
    t: float,
    phi_star: float,
    p: float,
    *,
    log_eps: float,
    log_machine_eps: float,
    fac: float = 1.01,
    p_eps: float = 1.0e-14,
) -> tuple[float, float, float]:
    phi_star_sq = np.sqrt(phi_star)
    phibar_star = (fac * phi_star) if phi_star > 0 else 0.01
    phibar_star_sq = np.sqrt(phibar_star)

    # search for fbar in [f_min, f_max]
    found = False
    f_min = 1
    f_max = 10
    f_tar = 5

    while not found:
        phi = phibar_star * t
        log_eps_t = log_eps / phi

        N: float = math.ceil(
            phi / np.pi * (1 - 3 * log_eps_t / 2 + math.sqrt(1 - 2 * log_eps_t))
        )
        A = np.pi * N / phi

        mu = phibar_star_sq * abs(4 - A) / abs(7 - math.sqrt(1 + 12 * A))
        fbar = ((phibar_star_sq - phi_star_sq) / mu) ** (-p)

        found = p < p_eps or f_min < fbar < f_max
        if not found:
            phibar_star_sq = f_tar ** (-1 / p) * mu + phi_star_sq
            phibar_star = phibar_star_sq**2

    mu = mu**2
    h = (-3 * A - 2 + 2 * math.sqrt(1 + 12 * A)) / (4 - A) / N

    # adjust integration parameters
    threshold = (log_eps - log_machine_eps) / t
    if mu > threshold:
        Q = 0.0 if abs(p) < p_eps else (f_tar ** (-1 / p) * math.sqrt(mu))
        phibar_star = (Q + math.sqrt(phi_star)) ** 2

        if phibar_star < threshold:
            w = math.sqrt(log_machine_eps / (log_machine_eps - log_eps))
            u = math.sqrt(-phibar_star * t / log_machine_eps)

            mu = threshold
            N = math.ceil(w * log_eps / (2 * np.pi * (u * w - 1)))
            h = w / N
        else:
            N = np.inf
            h = 0

    return mu, N, h


def _laplace_transform_inversion(
    t: float,
    z: complex,
    *,
    alpha: float,
    beta: float,
    eps: float,
    fac: float = 1.01,
) -> complex:
    if abs(z) < eps:
        return 1.0 / math.gamma(beta)

    # get machine precision and epsilon differences
    machine_eps = np.finfo(np.array(z).dtype).eps

    log_machine_eps = math.log(machine_eps)
    log_eps = math.log(eps)
    log_10 = math.log(10)
    d_log_eps = log_eps - log_machine_eps

    import cmath

    # evaluate relevant poles
    theta = cmath.phase(z)
    kmin = math.ceil(-alpha / 2 - theta / (2 * math.pi))
    kmax = math.floor(+alpha / 2 - theta / (2 * math.pi))
    k = np.arange(kmin, kmax + 1)
    s_star = abs(z) ** (1 / alpha) * np.exp(1j * (theta + 2 * np.pi * k) / alpha)

    # sort poles
    phi_star = (s_star.real + abs(s_star)) / 2
    s_star_index = np.argsort(phi_star)
    phi_star = phi_star[s_star_index]
    s_star = s_star[s_star_index]

    # filter out zero poles
    s_star_mask = phi_star > eps
    s_star = s_star[s_star_mask]
    phi_star = phi_star[s_star_mask]

    # add back the origin as a pole
    s_star = np.insert(s_star, 0, 0.0)
    phi_star = np.insert(phi_star, 0, 0.0)

    # strength of the singularities
    p = np.ones(s_star.shape, dtype=s_star.real.dtype)
    p[0] = max(0, -2 * (alpha - beta + 1))
    q = np.ones(s_star.shape, dtype=s_star.real.dtype)
    q[-1] = np.inf
    phi_star = np.insert(phi_star, phi_star.size, np.inf)

    # find admissible regions
    (region_index,) = np.nonzero(
        np.logical_and(
            phi_star[:-1] < d_log_eps / t,
            phi_star[:-1] < phi_star[1:],
        )
    )

    # evaluate parameters for LT inversion in each admissible region
    nregion = region_index[-1] + 1
    mu = np.full(nregion, np.inf, dtype=phi_star.dtype)
    N = np.full(nregion, np.inf, dtype=phi_star.dtype)
    h = np.full(nregion, np.inf, dtype=phi_star.dtype)

    found_region = False
    while not found_region:
        for j in region_index:
            if j < s_star.size - 1:
                mu[j], N[j], h[j] = _find_optimal_bounded_param(
                    t,
                    phi_star[j],
                    phi_star[j + 1],
                    p[j],
                    q[j],
                    log_eps=log_eps,
                    log_machine_eps=log_machine_eps,
                    fac=fac,
                )
            else:
                mu[j], N[j], h[j] = _find_optimal_unbounded_param(
                    t,
                    phi_star[j],
                    p[j],
                    log_eps=log_eps,
                    log_machine_eps=log_machine_eps,
                    fac=fac,
                )

        if np.min(N) > 200:
            log_eps += log_10
        else:
            found_region = True

        if log_eps >= 0.0:
            raise ValueError("Failed to find admissible region")

    # select region that contains the minimum number of nodes
    jmin = np.argmin(N)
    N_min = N[jmin]
    mu_min = mu[jmin]
    h_min = h[jmin]

    # evaluate inverse Laplace transform
    k = np.arange(-N_min, N_min + 1)
    hk = h_min * k
    zk = mu_min * (1j * hk + 1) ** 2
    zd = -2.0 * mu_min * hk + 2j * mu_min
    zexp = np.exp(zk * t)
    F = zk ** (alpha - beta) / (zk**alpha - z) * zd
    S = F * zexp

    integral = h_min * np.sum(S) / (2j * math.pi)

    # evaluate residues
    s_star_min = s_star[jmin + 1 :]
    residues = np.sum(1 / alpha * s_star_min ** (1 - beta) * np.exp(t * s_star_min))

    # sum up the results
    result = residues + integral

    return complex(result)


def _mittag_leffler_garrapa(
    z: complex, *, alpha: float, beta: float, eps: float | None = None
) -> complex:
    if eps is None:
        eps = 5 * float(np.finfo(np.array(z).dtype).eps)
        eps = 1.0e-15

    if abs(z) == 0:
        return 1 / math.gamma(beta)

    zinv = _laplace_transform_inversion(1.0, z, alpha=alpha, beta=beta, eps=eps)
    return zinv


# }}}


# {{{ Ortigueira


def _mittag_leffler_ortigueira(
    z: complex,
    *,
    alpha: float,
    beta: float,
    eps: float | None = None,
    kmax: int | None = None,
) -> complex:
    if eps is None:
        eps = 10 * float(np.finfo(np.array(z).dtype).eps)

    if kmax is None:
        kmax = 2048

    if abs(z) == 0:
        return 1 / math.gamma(beta)

    raise NotImplementedError


# }}}


# {{{ Mittag-Leffler


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
    if alpha == 0 and np.all(np.abs(z) < 1):
        return 1 / (1 - z) / math.gamma(beta)

    if beta == 1:
        if alpha == 1:
            return np.exp(z)  # type: ignore[no-any-return]
        if alpha == 2:
            return np.cosh(z ** (1 / 2))  # type: ignore[no-any-return]
        if alpha == 3:
            z3 = z ** (1 / 3)
            return (  # type: ignore[no-any-return]
                np.exp(z3) + 2 * np.exp(-z3 / 2) * np.cos(np.sqrt(3) * z3 / 2)
            ) / 3
        if alpha == 4:
            z = z ** (1 / 4)
            return (np.cos(z) + np.cosh(z)) / 2  # type: ignore[no-any-return]
        if alpha == 0.5:
            from scipy.special import erfc

            return np.exp(z**2) * erfc(-z)  # type: ignore[no-any-return]

    if beta == 2:
        if alpha == 1:
            return (np.exp(z) - 1) / z  # type: ignore[no-any-return]
        if alpha == 2:
            z = z ** (1 / 2)
            return np.sinh(z) / z  # type: ignore[no-any-return]

    return None


def mittag_leffler(
    z: float | complex | Array,
    alpha: float = 1.0,
    beta: float = 1.0,
    *,
    alg: Algorithm | str | None = None,
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
        # NOTE: for now this algorithm should be faster / better
        alg = Algorithm.Garrappa

    if isinstance(alg, str):
        try:
            alg = Algorithm[alg.capitalize()]
        except KeyError:
            raise ValueError(f"Unknown algorithm: '{alg}'") from None

    if use_explicit:
        result = mittag_leffler_special(z, alpha, beta)
        if result is not None:
            return result

    func: Any
    if alg == Algorithm.Series:
        func = _mittag_leffler_series
    elif alg == Algorithm.Diethelm:
        func = _mittag_leffler_diethelm
    elif alg == Algorithm.Garrappa:
        func = _mittag_leffler_garrapa
    elif alg == Algorithm.Ortigueira:
        func = _mittag_leffler_ortigueira
    else:
        raise ValueError(f"Unknown algorithm: '{alg}'")

    ml = np.vectorize(lambda zi: 0j + func(0j + zi, alpha=alpha, beta=beta))
    return np.array(ml(z))


# }}}


# {{{ sine / cosine


def caputo_derivative_sine(
    t: float | Array, alpha: float, alg: Algorithm | None = None
) -> Array:
    r"""Compute the :class:`~pycaputo.derivatives.CaputoDerivative` of the
    sine function.

    .. math::

        D^\alpha_C[\sin](t) =
            \cos \left(\frac{n \pi}{2}\right) t^{1 + n - \alpha}
                E_{2, 2 + n - \alpha}(-t^2)
            + \sin \left(\frac{n \pi}{2}\right) t^{n - \alpha}
                E_{2, 1 + n - \alpha}(-t^2),

    where :math:`E_{\alpha, \beta}` is the Mittag-Leffler function.
    """
    n = int(np.ceil(alpha))

    if n % 2 == 0:
        # sin(m pi / 2) == 0
        Eab = mittag_leffler(-(t**2), 2, 2 + n - alpha, alg=alg)
        result = np.cos(n * np.pi / 2) * t ** (1 + n - alpha) * Eab
    else:
        # cos(m pi / 2) == 0
        Eab = mittag_leffler(-(t**2), 2, 1 + n - alpha, alg=alg)
        result = np.sin(n * np.pi / 2) * t ** (n - alpha) * Eab

    result = np.array(result)
    # assert np.linalg.norm(result.imag) < 5 * np.finfo(result.dtype).eps

    return np.array(result.real)


# }}}
