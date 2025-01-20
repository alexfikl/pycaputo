# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

r"""Computes the matrix operator for the Caputo derivative and look at its
spectrum for various values of :math:`\alpha`.

The spectrum of the matrix is quite boring, since the matrix itself is lower
triangular, so all the eigenvalues are on the diagonal. From the method, we know
that

.. math::

    W_{n, n} = \frac{1}{\Gamma(2 - \alpha)} \frac{1}{\Delta x^\alpha}.

i.e. the diagonal is constant. This mostly serves as an example of how the
matrices can be constructed.
"""

from __future__ import annotations

import numpy as np

from pycaputo.differentiation import quadrature_weights
from pycaputo.differentiation.caputo import L1
from pycaputo.grid import make_uniform_points
from pycaputo.logging import get_logger
from pycaputo.typing import Array

log = get_logger("caputo_derivative_l1_spectrum")


# {{{ matrix


def get_l1_matrix(alpha: float, *, n: int, a: float, b: float) -> Array:
    W = np.zeros((n, n))

    meth = L1(alpha=alpha)
    points = make_uniform_points(n, a=0, b=1)

    for i in range(1, n):
        W[i, : i + 1] = quadrature_weights(meth, points, i + 1)
    W[0, 0] = W[1, 1]

    return W


# }}}


# {{{ conditioning

n = 256
alphas = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

for alpha in alphas:
    W = get_l1_matrix(alpha, n=n, a=0.0, b=1.0)
    kappa = np.linalg.cond(W)
    log.info("alpha = %.2f kappa = %.8e", alpha, kappa)

# }}}

# {{{ plot

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise SystemExit(0) from exc

from pycaputo.utils import figure, set_recommended_matplotlib

set_recommended_matplotlib()

with figure("caputo-derivative-l1-spectrum") as fig:
    ax = fig.gca()

    for alpha in alphas:
        W = get_l1_matrix(alpha, n=n, a=0.0, b=1.0)
        sigma = np.linalg.eigvals(W)
        ax.plot(sigma.real, sigma.imag, "o")

    ax.set_xlabel(r"$\Re$")
    ax.set_ylabel(r"$\Im$")

# }}}
