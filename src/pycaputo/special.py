# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo import derivatives as ds
from pycaputo.logging import get_logger
from pycaputo.utils import Array

logger = get_logger(__name__)


# {{{ power


def _pow_derivative_caputo(
    d: ds.CaputoDerivative, t: float | Array, omega: float
) -> Array:
    from math import gamma

    alpha = d.alpha
    zeros = np.arange(d.n)

    if np.isin(omega, zeros):
        result = np.zeros_like(np.asarray(t))
    else:
        result = gamma(omega + 1) / gamma(omega - alpha + 1) * t ** (omega - d.alpha)

    return np.array(result)


def _pow_derivative_riemann_liouville(
    d: ds.RiemannLiouvilleDerivative, t: float | Array, omega: float
) -> Array:
    from math import gamma

    alpha = d.alpha
    zeros = alpha - np.arange(1, d.n + 1)

    if np.isin(omega, zeros):
        result = np.zeros_like(np.asarray(t))
    else:
        result = gamma(omega + 1) / gamma(omega - alpha + 1) * t ** (omega - d.alpha)

    return np.array(result)


def pow_derivative(
    d: ds.FractionalOperator, t: float | Array, *, t0: float = 0.0, omega: float = 1.0
) -> Array:
    r"""Evaluates the application of the fractional operator *d* to the power
    function.

    The function evaluates the derivative of :math:`(t - t_0)^\omega`.
    This assumes that the integral in the fractional operator is evaluated on
    :math:`[t_0, t]`.

    The following implementations are available:

    * For the (left) Riemann-Liouville (and Grünwald-Letnikov) derivatives
      see Equation 23 in [Garrappa2019]_.
    * For the (left) Caputo derivative see Equation 24 in [Garrappa2019]_.
    """

    if isinstance(d, ds.CaputoDerivative):
        return _pow_derivative_caputo(d, t - t0, omega)
    elif isinstance(d, ds.RiemannLiouvilleDerivative):
        return _pow_derivative_riemann_liouville(d, t - t0, omega)
    elif isinstance(d, ds.GrunwaldLetnikovDerivative):
        d = ds.RiemannLiouvilleDerivative(alpha=d.alpha, side=d.side)
        return _pow_derivative_riemann_liouville(d, t - t0, omega)
    else:
        raise NotImplementedError(
            f"Power function derivative not implemented for {type(d).__name__}"
        )


# }}}


# {{{ exponential


def _exp_derivative_caputo(
    d: ds.CaputoDerivative, t: float | Array, omega: float
) -> Array:
    from pycaputo.mittagleffler import Algorithm, mittag_leffler

    alpha = d.alpha
    n = d.n
    alg = Algorithm.Garrappa

    Eab = mittag_leffler(omega * t, 1, 1 + n - alpha, alg=alg)
    result = omega**n * t ** (n - alpha) * Eab

    return np.array(result.real)


def _exp_derivative_riemann_liouville(
    d: ds.RiemannLiouvilleDerivative, t: float | Array, omega: float
) -> Array:
    from pycaputo.mittagleffler import Algorithm, mittag_leffler

    alpha = d.alpha
    alg = Algorithm.Garrappa

    Eab = mittag_leffler(omega * t, 1, 1 - alpha, alg=alg)
    result = Eab / t**alpha

    return np.array(result.real)


def exp_derivative(
    d: ds.FractionalOperator, t: float | Array, *, t0: float = 0.0, omega: float = 1.0
) -> Array:
    r"""Evaluates the application of the fractional operator *d* to the exponential
    function.

    The function evaluates the derivative of :math:`\exp \omega (t - t_0)`.
    This assumes that the integral in the fractional operator is evaluated on
    :math:`[t_0, t]`.

    The following implementations are available:

    * For the (left) Riemann-Liouville (and Grünwald-Letnikov) derivatives
      see Proposition 9 in [Garrappa2019]_.
    * For the (left) Caputo derivative see Proposition 9 in [Garrappa2019]_.
    """

    if isinstance(d, ds.CaputoDerivative):
        return _exp_derivative_caputo(d, t - t0, omega)
    elif isinstance(d, ds.RiemannLiouvilleDerivative):
        return _exp_derivative_riemann_liouville(d, t - t0, omega)
    elif isinstance(d, ds.GrunwaldLetnikovDerivative):
        d = ds.RiemannLiouvilleDerivative(alpha=d.alpha, side=d.side)
        return _exp_derivative_riemann_liouville(d, t - t0, omega)
    else:
        raise NotImplementedError(
            f"Exponential derivative not implemented for {type(d).__name__}"
        )


# }}}


# {{{ sine


def _sin_derivative_caputo(
    d: ds.CaputoDerivative, t: float | Array, omega: float, *, alg: object = None
) -> Array:
    from pycaputo.mittagleffler import Algorithm, mittag_leffler

    if alg is None:
        alg = Algorithm.Garrappa

    assert isinstance(alg, Algorithm)

    alpha = d.alpha
    n = d.n

    tomega = omega * t

    if n % 2 == 0:
        Eab = mittag_leffler(-(tomega**2), 2, 2 + n - alpha, alg=alg)
        result = (-1) ** (n // 2) * tomega ** (n + 1) / t**alpha * Eab
    else:
        Eab = mittag_leffler(-(tomega**2), 2, 1 + n - alpha, alg=alg)
        result = (-1) ** ((n - 1) // 2) * tomega**n / t**alpha * Eab

    return np.array(result.real)


def _sin_derivative_riemann_liouville(
    d: ds.RiemannLiouvilleDerivative, t: float | Array, omega: float
) -> Array:
    from pycaputo.mittagleffler import Algorithm, mittag_leffler

    alpha = d.alpha
    alg = Algorithm.Garrappa

    tomega = omega * t
    Eab = mittag_leffler(-(tomega**2), 2, 2 - alpha, alg=alg)
    result = tomega / t**alpha * Eab

    return np.array(result.real)


def sin_derivative(
    d: ds.FractionalOperator, t: float | Array, *, t0: float = 0.0, omega: float = 1.0
) -> Array:
    r"""Evaluates the application of the fractional operator *d* to the sine function.

    The function evaluates the derivative of :math:`\sin \omega (t - t_0)`.
    This assumes that the integral in the fractional operator is evaluated on
    :math:`[t_0, t]`.

    The following implementations are available:

    * For the (left) Riemann-Liouville (and Grünwald-Letnikov) derivatives
      see Proposition 14 in [Garrappa2019]_.
    * For the (left) Caputo derivative see Proposition 14 in [Garrappa2019]_.
    """

    if isinstance(d, ds.CaputoDerivative):
        return _sin_derivative_caputo(d, t - t0, omega)
    elif isinstance(d, ds.RiemannLiouvilleDerivative):
        return _sin_derivative_riemann_liouville(d, t - t0, omega)
    elif isinstance(d, ds.GrunwaldLetnikovDerivative):
        d = ds.RiemannLiouvilleDerivative(alpha=d.alpha, side=d.side)
        return _sin_derivative_riemann_liouville(d, t - t0, omega)
    else:
        raise NotImplementedError(
            f"Sine derivative not implemented for {type(d).__name__}"
        )


# }}}


# {{{ cosine


def _cos_derivative_caputo(
    d: ds.CaputoDerivative, t: float | Array, omega: float
) -> Array:
    from pycaputo.mittagleffler import Algorithm, mittag_leffler

    alpha = d.alpha
    n = d.n
    alg = Algorithm.Garrappa

    tomega = omega * t

    if n % 2 == 0:
        Eab = mittag_leffler(-(tomega**2), 2, 1 + n - alpha, alg=alg)
        result = (-1) ** (n // 2) * tomega**n / t**alpha * Eab
    else:
        Eab = mittag_leffler(-(tomega**2), 2, 2 + n - alpha, alg=alg)
        result = (-1) ** ((n + 1) // 2) * tomega ** (n + 1) / t**alpha * Eab

    return np.array(result.real)


def _cos_derivative_riemann_liouville(
    d: ds.RiemannLiouvilleDerivative, t: float | Array, omega: float
) -> Array:
    from pycaputo.mittagleffler import Algorithm, mittag_leffler

    alpha = d.alpha
    alg = Algorithm.Garrappa

    Eab = mittag_leffler(-((omega * t) ** 2), 2, 1 - alpha, alg=alg)
    result = Eab / t**alpha

    return np.array(result.real)


def cos_derivative(
    d: ds.FractionalOperator, t: float | Array, *, t0: float = 0.0, omega: float = 1.0
) -> Array:
    r"""Evaluates the application of the fractional operator *d* to the cosine function.

    The function evaluates the derivative of :math:`\cos \omega (t - t_0)`.
    This assumes that the integral in the fractional operator is evaluated on
    :math:`[t_0, t]`.

    The following implementations are available:

    * For the (left) Riemann-Liouville (and Grünwald-Letnikov) derivatives
      see Proposition 15 in [Garrappa2019]_.
    * For the (left) Caputo derivative see Proposition 15 in [Garrappa2019]_.
    """

    if isinstance(d, ds.CaputoDerivative):
        return _cos_derivative_caputo(d, t - t0, omega)
    elif isinstance(d, ds.RiemannLiouvilleDerivative):
        return _cos_derivative_riemann_liouville(d, t - t0, omega)
    elif isinstance(d, ds.GrunwaldLetnikovDerivative):
        d = ds.RiemannLiouvilleDerivative(alpha=d.alpha, side=d.side)
        return _cos_derivative_riemann_liouville(d, t - t0, omega)
    else:
        raise NotImplementedError(
            f"Cosine derivative not implemented for {type(d).__name__}"
        )


# }}}
