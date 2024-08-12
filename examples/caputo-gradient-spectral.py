# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

r"""This example computes the gradient of a simple function

.. math::

    f(\mathbf{x}) = \sum_{k = 0}^{n - 1} \left(\frac{1}{2} - x_k\right)^4

The gradient is defined in a standard fashion from the literature, see for
example [Wei2020].

It uses a spectral method based on Jacobi polynomials to compute the gradient.

.. [Wei2020] Y. Wei, Y. Kang, W. Yin, Y. Wang,
    *Generalization of the Gradient Method With Fractional Order Gradient Direction*,
    Journal of the Franklin Institute, Vol. 357, pp. 2514--2532, 2020,
    `DOI <https://doi.org/10.1016/j.jfranklin.2020.01.008>`__.
"""

from __future__ import annotations

import math

import numpy as np

from pycaputo.typing import Array

# {{{ evaluate


def f(x: Array) -> Array:
    return np.array(np.sum((0.5 - x) ** 4, axis=0))


def df_i(x: Array, alpha: float) -> Array:
    return np.array(
        -0.5 * x ** (1 - alpha) / math.gamma(2 - alpha)
        + 3 * x ** (2 - alpha) / math.gamma(3 - alpha)
        - 12 * x ** (3 - alpha) / math.gamma(4 - alpha)
        + 24 * x ** (4 - alpha) / math.gamma(5 - alpha)
    )


def df(x: Array, alpha: float) -> Array:
    return np.array(
        [df_i(x[i], alpha) for i in np.ndindex(x.shape)],
        dtype=x.dtype,
    )


# instantiate the desired method
from pycaputo.differentiation import caputo

alpha = 0.9
method = caputo.Jacobi(alpha=0.9)

# make a set of points at which to evaluate the gradient
rng = np.random.default_rng(42)
x = rng.uniform(3.0, 7.0, size=12)

# evaluate the gradient
from pycaputo import grad
from pycaputo.grid import make_jacobi_gauss_lobatto_points

p = make_jacobi_gauss_lobatto_points(32, a=0.0, b=1.0)
df_num = grad(method, f, p, x)

df_ref = df(x, alpha)
print(
    "Relative error: ",
    np.linalg.norm(df_num - df_ref) / np.linalg.norm(df_ref),
)


# }}}

# {{{ plot

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise SystemExit(0) from exc

from pycaputo.utils import figure, set_recommended_matplotlib

set_recommended_matplotlib()

with figure("caputo-gradient-spectral") as fig:
    ax = fig.gca()

    ax.plot(df_num, lw=5, label="Spectral")
    ax.plot(df_ref, "k--", label="Exact")

    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$D^\alpha_C[f](x)$")
    ax.legend()

# }}}
