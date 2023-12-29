# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import numpy as np

from pycaputo.utils import Array

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


from pycaputo.derivatives import CaputoDerivative, Side

d = CaputoDerivative(order=0.9, side=Side.Left)

from pycaputo import grad
from pycaputo.differentiation import CaputoSpectralMethod

method = CaputoSpectralMethod(d)

from pycaputo.grid import make_jacobi_gauss_lobatto_points

# make a set of points at which to evaluate the gradient
rng = np.random.default_rng(42)
x = rng.uniform(3.0, 7.0, size=12)

# evaluate the gradient
p = make_jacobi_gauss_lobatto_points(32, a=0.0, b=1.0)
df_num = grad(f, p, x, method=method)

df_ref = df(x, d.order)
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
