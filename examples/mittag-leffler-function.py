# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

r"""
This reproduces Figure 10 and Figure 11 from [Diethelm2005]_. It plots
:math:`E_{\alpha, \beta}(-x^\alpha)` for various values of :math:`\alpha` and
:math:`\beta`.

.. [Diethelm2005] K. Diethelm, N. J. Ford, A. D. Freed, Y. Luchko,
    *Algorithms for the Fractional Calculus: A Selection of Numerical Methods*,
    Computer Methods in Applied Mechanics and Engineering, Vol. 194, pp. 743--773, 2005,
    `DOI <https://doi.org/10.1016/j.cma.2004.06.006>`__.
"""

from __future__ import annotations

import numpy as np

from pycaputo.logging import get_logger
from pycaputo.mittagleffler import mittag_leffler

logger = get_logger("ml")

# {{{ evaluate

x = np.linspace(0.0, 5.0, 256)

E_alpha = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]
E_alpha_value = []

for alpha in E_alpha:
    z = -(x**alpha)
    value = mittag_leffler(z, alpha, 1.0)
    E_alpha_value.append(value.real)

E_beta = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
E_beta_value = []
for beta in E_beta:
    z = -x
    value = mittag_leffler(z, 1.0, beta)
    E_beta_value.append(value.real)

# }}}

# {{{ plot

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise SystemExit(0) from exc

from pycaputo.utils import figure, set_recommended_matplotlib

set_recommended_matplotlib()

with figure("mittag-leffler-figure10") as fig:
    ax = fig.gca()

    for alpha, value in zip(E_alpha, E_alpha_value):
        ax.plot(x, value, label=rf"$\alpha = {alpha:.2f}$")

    ax.set_xlim([0.0, 5.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$E_{\alpha, 1}(-x^\alpha)$")
    ax.legend()

with figure("mittag-leffler-figure11") as fig:
    ax = fig.gca()

    for beta, value in zip(E_beta, E_beta_value):
        ax.plot(x, value, label=rf"$\beta = {beta:.2f}$")

    ax.set_xlim([0.0, 5.0])
    ax.set_ylim([-0.5, 1.5])
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$E_{1, \beta}(-x)$")
    ax.legend()

# }}}
