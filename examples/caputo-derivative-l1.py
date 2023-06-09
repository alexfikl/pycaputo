# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import math

import matplotlib.pyplot as mp
import numpy as np

from pycaputo.utils import Array

# {{{ evaluate


def f(x: Array) -> Array:
    return (0.5 - x) ** 4


def df(x: Array, alpha: float) -> Array:
    return np.array(
        -0.5 * x ** (1 - alpha) / math.gamma(2 - alpha)
        + 3 * x ** (2 - alpha) / math.gamma(3 - alpha)
        - 12 * x ** (3 - alpha) / math.gamma(4 - alpha)
        + 24 * x ** (4 - alpha) / math.gamma(5 - alpha)
    )


from pycaputo import CaputoDerivative, CaputoL1Method, Side

d = CaputoDerivative(order=0.9, side=Side.Left)
method = CaputoL1Method(d)

from pycaputo import diff
from pycaputo.grid import make_uniform_points

p = make_uniform_points(256, a=0.0, b=1.0)
df_num = diff(method, f, p)

df_ref = df(p.x, d.order)
print(
    "Relative error: ",
    np.linalg.norm(df_num[1:] - df_ref[1:]) / np.linalg.norm(df_ref[1:]),
)


# }}}

# {{{ plot


from pycaputo.utils import set_recommended_matplotlib

set_recommended_matplotlib()

fig = mp.figure()
ax = fig.gca()

ax.plot(p.x, df_num, lw=5, label="$L1~ Method$")
ax.plot(p.x[1:], df_ref[1:], "k--", label="$Exact$")

ax.set_xlabel("$x$")
ax.set_ylabel(r"$D^\alpha_C[f](x)$")
ax.legend()

fig.tight_layout()
fig.savefig("caputo-derivative-l1.png")

# }}}
