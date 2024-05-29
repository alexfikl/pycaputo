# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

r"""
This reproduces Figure 4 from [Garrappa2015]_. It evaluates the Mittag-Leffler
function

.. math::

    E_{\frac{1}{2}, 1}(z) = \exp(z^2) \erfc(-z)

In this regime, the series expansion will fail to converge for :math:`|z| > 5`
and the algorithm from Diethelm2005 also fails.

.. [Garrappa2015] R. Garrappa,
    *Numerical Evaluation of Two and Three Parameter Mittag-Leffler Functions*,
    SIAM Journal on Numerical Analysis, Vol. 53, pp. 1350--1369, 2015,
    `DOI <https://doi.org/10.1137/140971191>`__.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

from pycaputo.logging import get_logger
from pycaputo.mittagleffler import mittag_leffler
from pycaputo.utils import Array

logger = get_logger("ml")


def pointwise_error(ref: Array, a: Array) -> Array:
    # NOTE: this is the error used in Equation 4.1 from [Garrappa2015]_.
    return np.abs(a - ref) / (1 + np.abs(ref))  # type: ignore[no-any-return]


# {{{ evaluate

alpha = 0.5
beta = 1

r = np.linspace(0.0, 12.0, 256)
arg = np.pi / 2
z = r * np.exp(1j * arg)

# 0. Reference result
result_ref = np.exp(z**2) * erfc(-z)
# result_ref = (np.cos(z**0.25) + np.cosh(z**0.25)) / 2

# 1. Series expansion algorithm
mask1 = np.abs(z) < 3.0
result_series = mittag_leffler(z[mask1], alpha, beta, alg="Series", use_explicit=False)
error_series = pointwise_error(result_ref[mask1], result_series)

# 2. Diethelm algorithm
mask2 = np.abs(z) < 3.0
result_diethelm = mittag_leffler(
    z[mask2], alpha, beta, alg="Diethelm", use_explicit=False
)
error_diethelm = pointwise_error(result_ref[mask2], result_diethelm)

# 3. Garrappa algorithm
result_garrappa = mittag_leffler(z, alpha, beta, alg="Garrappa", use_explicit=False)
error_garrappa = pointwise_error(result_ref, result_garrappa)

logger.info(
    "Error: Series %.8e Diethelm %.8e Garrappa %.8e",
    np.max(error_series),
    np.max(error_diethelm),
    np.max(error_garrappa),
)

# }}}

# {{{ plot

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise SystemExit(0) from exc

from pycaputo.utils import figure, set_recommended_matplotlib

set_recommended_matplotlib()

with figure("mittag-leffler-accuracy", figsize=(16, 8)) as fig:
    ax = fig.gca()

    ax.semilogy(r[mask1], error_series, label="Series")
    ax.semilogy(r[mask2], error_diethelm, label="Diethelm (2005)")
    ax.semilogy(r, error_garrappa, label="Garrappa (2015)")

    ax.set_xlabel("$|z|$")
    ax.set_ylabel("Error")
    ax.legend()

# }}}
