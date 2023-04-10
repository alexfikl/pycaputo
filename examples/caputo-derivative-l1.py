import math

import matplotlib.pyplot as mp
import numpy as np

from pycaputo.utils import Array

# {{{ evaluate


def f(x: Array) -> Array:
    return (1 + x) ** 3


def df(x: Array, alpha: float) -> Array:
    return np.array(
        3 * x ** (1 - alpha) / math.gamma(2 - alpha)
        + 6 * x ** (2 - alpha) / math.gamma(3 - alpha)
        + 6 * x ** (3 - alpha) / math.gamma(4 - alpha)
    )


from pycaputo import CaputoDerivative, CaputoUniformL1Method, Side

d = CaputoDerivative(order=0.9, side=Side.Left)
method = CaputoUniformL1Method(d, modified=False)

from pycaputo import evaluate
from pycaputo.grid import make_uniform_points

p = make_uniform_points(256, a=0, b=1)
df_num = evaluate(method, f, p)


# }}}

# {{{ plot


from pycaputo.utils import set_recommended_matplotlib

set_recommended_matplotlib()

fig = mp.figure()
ax = fig.gca()

ax.plot(p.x, df_num, lw=5, label="Approximation")
ax.plot(p.x[1:], df(p.x[1:], d.order), "k--", label="Exact")

# NOTE: this is the color of the 'sphinx_rtd_theme' background
fig.patch.set_facecolor("#FCFCFC")
ax.set_facecolor("#FCFCFC")

ax.set_xlabel("$x$")
ax.set_ylabel(r"$D^\alpha_C[f](x)$")
ax.legend()

fig.savefig("caputo-derivative-l1.png")

# }}}
