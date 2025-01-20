# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""Compute the fractional Caputo derivative of a function using the L1
method. This is a simple example showcasing how the API can be used.

Other methods are available at :mod:`pycaputo.differentiation`.
"""

from __future__ import annotations

import math

import numpy as np

from pycaputo.logging import get_logger
from pycaputo.typing import Array

log = get_logger("tutorial")

# {{{ evaluate


# [tutorial-func-start]
def f(x: Array) -> Array:
    return (0.5 - x) ** 4


def df(x: Array, alpha: float) -> Array:
    return np.array(
        -0.5 * x ** (1 - alpha) / math.gamma(2 - alpha)
        + 3 * x ** (2 - alpha) / math.gamma(3 - alpha)
        - 12 * x ** (3 - alpha) / math.gamma(4 - alpha)
        + 24 * x ** (4 - alpha) / math.gamma(5 - alpha)
    )
    # [tutorial-func-end]


# [tutorial-method-start]
from pycaputo.differentiation.caputo import L1

alpha = 0.9
method = L1(alpha=alpha)
# [tutorial-method-end]

# [tutorial-evaluate-start]
from pycaputo.grid import make_uniform_points

p = make_uniform_points(256, a=0.0, b=1.0)

from pycaputo.differentiation import diff

df_num = diff(method, f, p)
# [tutorial-evaluate-end]

df_ref = df(p.x, alpha)
log.info(
    "Relative error: %.12e",
    np.linalg.norm(df_num[1:] - df_ref[1:]) / np.linalg.norm(df_ref[1:]),
)


# }}}

# {{{ plot

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    log.warning("'matplotlib' is not available.")
    raise SystemExit(0) from exc

from pycaputo import _get_default_dark  # noqa: PLC2701
from pycaputo.utils import figure, set_recommended_matplotlib

for dark, suffix in _get_default_dark():
    set_recommended_matplotlib(dark=dark)

    with figure(f"tutorial-caputo-l1{suffix}") as fig:
        ax = fig.gca()

        ax.plot(p.x, df_num, lw=5, label="L1 Method")
        ax.plot(p.x[1:], df_ref[1:], "--", color="w" if dark else "k", label="Exact")

        ax.set_xlabel("$x$")
        ax.set_ylabel(r"$D^\alpha_C[f](x)$")
        ax.legend()

# }}}
