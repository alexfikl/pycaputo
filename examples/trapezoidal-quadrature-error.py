# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

r"""This example reproduces the error curves for trapezoidal quadrature from
Theorem 2.6 in [Diethelm2004]. In general, for a monomial

.. math::

    f(x) = (x - x_a)^\beta

we should expect the error to scale as

.. math::

    E \le
    \begin{cases}
    C (x_n - x_0)^{\alpha + \beta - 2} h^2,     & \quad \beta > 1, \\
    C (x_n - x_0)^{\alpha - 1} h^{1 + \beta}    & \quad 0 < \beta < 1.
    \end{cases}

.. [Diethelm2004] K. Diethelm, N. J. Ford, A. D. Freed,
    *Detailed Error Analysis for a Fractional Adams Method*,
    Numerical Algorithms, Vol. 36, pp. 31--52, 2004,
    `DOI <https://doi.org/10.1023/b:numa.0000027736.85078.be>`__.
"""

from __future__ import annotations

import numpy as np

from pycaputo.logging import get_logger
from pycaputo.typing import Array

log = get_logger("trapezoidal")


# {{{ function


def func_f(x: Array, *, beta: float = 2.0, x0: float = 0.0) -> Array:
    return (x - x0) ** beta


def func_df(x: Array, *, alpha: float, beta: float = 2.0, x0: float = 0.0) -> Array:
    from math import gamma

    return gamma(1 + beta) / gamma(1 + beta + alpha) * (x - x0) ** (beta + alpha)


# }}}


# {{{ quadrature

from pycaputo.quadrature import quad, riemann_liouville

alpha = 0.2
beta = 0.85
m = riemann_liouville.Trapezoidal(-alpha)

from pycaputo.grid import make_uniform_points

n = 128
xa, xb = 1.5, 3 * np.pi
p = make_uniform_points(n, xa, xb)

f_ref = func_f(p.x, beta=beta, x0=xa)
df_ref = func_df(p.x, alpha=alpha, beta=beta, x0=xa)
df_num = quad(m, f_ref, p)
df_num[0] = df_ref[0]

error = np.linalg.norm(df_ref - df_num)
log.info("Error: %.12e", error)

# }}}

# {{{ convergence

from pycaputo.utils import EOCRecorder

eoc = EOCRecorder(order=min(2.0, 1 + beta))

for n in [32, 64, 128, 256, 384, 512]:
    p = make_uniform_points(n, xa, xb)

    f_ref = func_f(p.x, beta=beta, x0=xa)
    df_ref = func_df(p.x, alpha=alpha, beta=beta, x0=xa)
    df_num = quad(m, f_ref, p)
    df_num[0] = df_ref[0]

    h_max = np.max(p.dx)
    error = np.linalg.norm(df_ref - df_num) / np.linalg.norm(df_ref)
    eoc.add_data_point(h_max, error)

log.info("Error:\n%s", eoc)

# }}}


# {{{ plot

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise SystemExit(0) from exc

from pycaputo.utils import figure, set_recommended_matplotlib

set_recommended_matplotlib()

e = np.abs(df_ref - df_num)
hmax = np.max(p.dx)

if beta < 1.0:
    e_ref = (p.x - xa) ** (alpha - 1.0) * hmax ** (1 + beta)
    e_label = r"$(t - t_0)^{\alpha - 1} \Delta t_{\text{max}}^{1 + \beta}$"
else:
    e_ref = (p.x - xa) ** (alpha + beta - 2) * hmax**2
    e_label = r"$(t - t_0)^{\alpha + \beta - 2} \Delta t_{\text{max}}^2$"

# NOTE: ensure the constant is the same
e_ref[0] = 0.0
c_ref = e[-1] / e_ref[-1]
e_ref = c_ref * e_ref

with figure(f"trapezoidal-quadrature-{100 * beta:03.0f}-df") as fig:
    ax = fig.gca()

    ax.plot(p.x, df_num, label="Approx")
    ax.plot(p.x, df_ref, "k--", label="Exact")

    ax.set_title(rf"$\alpha = {alpha:.2f} ~/~ \beta = {beta:.2f} $")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"${{}}^C D_{x_0}^{\alpha}[f](x)$")
    ax.legend()

with figure(f"trapezoidal-quadrature-{100 * beta:03.0f}-t") as fig:
    ax = fig.gca()

    ax.semilogy(p.x, e, label="Error")
    ax.semilogy(p.x, e_ref, "k--", label=e_label)

    ax.set_title(rf"$\alpha = {alpha:.2f} ~/~ \beta = {beta:.2f} $")
    ax.set_xlabel("$t$")
    ax.set_ylabel("Error")
    ax.legend()

# }}}
