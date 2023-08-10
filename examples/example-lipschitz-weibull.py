# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import numpy as np
import scipy.stats as ss

from pycaputo.lipschitz import fit_reverse_weibull, uniform_sample_maximum_slopes
from pycaputo.utils import Array, figure, set_recommended_matplotlib


def f(x: Array) -> Array:
    return np.array(x - x**3 / 3)


# parameters

a, b = -1.0, 1.0
c, loc, scale = 1.0, 1.0, 0.002

nslopes = 11
nbatches = 100
delta = 0.05

# fit distribution

rng = np.random.default_rng(seed=42)
smax = uniform_sample_maximum_slopes(
    f,
    a,
    b,
    delta=delta,
    nslopes=nslopes,
    nbatches=nbatches,
    rng=rng,
)
smax = np.sort(smax)

cdf_opt = fit_reverse_weibull(smax)
cdf_ref = ss.weibull_max(c=c, loc=loc, scale=scale)
cdf_empirical = ss.ecdf(smax).cdf

print(
    "Optimal parameters: c {c:.12e} loc {loc:.12e} scale {scale:.12e}".format(
        **cdf_opt.kwds
    )
)

# plot

set_recommended_matplotlib()

with figure("example-lipschitz-weibull") as fig:
    ax = fig.gca()

    ax.step(smax, cdf_opt.cdf(smax), "k--", label="$Optimal$")
    ax.step(smax, cdf_ref.cdf(smax), "k:", label="$Example$")
    ax.step(smax, cdf_empirical.probabilities, label="$Empirical$")
    ax.set_xlabel("$s_{max}$")

    ax.legend()
