# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import numpy as np
import scipy.stats as ss

from pycaputo.lipschitz import uniform_diagonal_sample
from pycaputo.utils import Array, figure, set_recommended_matplotlib


def f(x: Array) -> Array:
    return np.array(x - x**3 / 2)


def cdf_inverse_weibull(x: Array, *, u: float, v: float, w: float) -> Array:
    return np.where(x < u, np.exp(-((u - x) ** w) / v), 1.0)


# parameters
a, b = -1.0, 1.0
nslopes = 32
nbatches = 256
delta = 1 / 64 * np.sqrt(2) * (b - a)

# compute maxima

rng = np.random.default_rng(seed=42)
maxs = np.empty(nbatches)
for i in range(nbatches):
    x, y = uniform_diagonal_sample(a, b, nslopes, delta=delta, rng=rng)
    s = np.abs(f(x) - f(y)) / (np.abs(x - y) + 1.0e-7)

    maxs[i] = np.max(s)

maxs = np.sort(maxs)
F = ss.ecdf(maxs)
iw_cdf = cdf_inverse_weibull(maxs, u=1.0, v=0.004, w=1.0)
# iw_cdf = cdf_inverse_weibull(maxs, u=1.71, v=0.632, w=2.5)
print(f"Error: {np.linalg.norm(F.cdf.probabilities - iw_cdf)}")

# plot
set_recommended_matplotlib()

with figure("example-lipschitz-weibull") as fig:
    ax = fig.gca()

    ax.step(maxs, F.cdf.probabilities)
    ax.step(maxs, iw_cdf, "k--")
    ax.set_xlabel("$l$")
    ax.set_ylabel("$CDF$")

from pycaputo.lipschitz import estimate_lipschitz_constant

L = estimate_lipschitz_constant(
    f,
    a,
    b,
    delta=delta,
    nslopes=nslopes,
    nbatches=nbatches,
)
print(f"Lipschitz constant estimate for `x - x^3/2`: {L}")
