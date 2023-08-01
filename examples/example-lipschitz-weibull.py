# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import numpy as np
import scipy.stats as ss

from pycaputo.lipschitz import (
    InverseWeibullCumulativeDistributionFunction,
    fit_inverse_weibull,
    uniform_sample_maximum_slopes,
)
from pycaputo.utils import Array, figure, set_recommended_matplotlib


def f(x: Array) -> Array:
    return np.array(x - x**3 / 2)


# parameters

a, b = -1.0, 1.0
nslopes = 32
nbatches = 128
delta = 1 / 64 * np.sqrt(2) * (b - a)

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

cdf_opt = fit_inverse_weibull(smax)
cdf_ref = InverseWeibullCumulativeDistributionFunction(u=1.0, v=0.004, w=1.0)
cdf_empirical = ss.ecdf(smax).cdf

# plot
set_recommended_matplotlib()

with figure("example-lipschitz-weibull") as fig:
    ax = fig.gca()

    ax.step(smax, cdf_opt(smax), "k--", label="$Optimal$")
    ax.step(smax, cdf_ref(smax), "k:", label="$Example$")
    ax.step(smax, cdf_empirical.probabilities, label="$Empirical$")
    ax.set_xlabel("$s_{max}$")

    ax.legend()
