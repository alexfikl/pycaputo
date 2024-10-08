# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

r"""Compute the fractional derivative of a quadratic function for multiple
values of :math:`\alpha` to showcase how the fractional derivative smoothly
interpolates between the integer orders.

This uses a spectral method with Jacobi polynomials for the added accuracy.
"""

from __future__ import annotations

import numpy as np

from pycaputo.differentiation import diff
from pycaputo.differentiation.caputo import Jacobi
from pycaputo.grid import make_jacobi_gauss_lobatto_points
from pycaputo.typing import Array

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise SystemExit(0) from exc
else:
    from pycaputo.utils import figure, set_recommended_matplotlib

    set_recommended_matplotlib()


def f(x: Array) -> Array:
    return x**2


with figure("caputo-derivative-quadratic") as fig:
    ax = fig.gca()

    alphas = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
    ]

    p = make_jacobi_gauss_lobatto_points(32, a=0.0, b=2.0)

    for alpha in alphas:
        method = Jacobi(alpha=alpha)

        df_num = diff(method, f, p)
        ax.plot(p.x, df_num, color="k", alpha=0.2)

    ax.plot(p.x, f(p.x), label=r"$\alpha = 0$")
    ax.plot(p.x, 2.0 * p.x, label=r"$\alpha = 1$")
    ax.plot(p.x, np.full(p.x.shape, 2.0), label=r"$\alpha = 2$")

    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$D^\alpha_C[f](x)$")
    ax.legend()
