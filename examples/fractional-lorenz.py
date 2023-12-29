# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from functools import partial

import numpy as np

from pycaputo.logging import get_logger
from pycaputo.utils import Array

logger = get_logger("lorenz")

# {{{ Lorenz


def lorenz(t: float, y: Array, *, sigma: float, rho: float, beta: float) -> Array:
    return np.array(
        [
            sigma * (y[1] - y[0]),
            y[0] * (rho - y[2]) - y[1],
            y[0] * y[1] - beta * y[2],
        ]
    )


def lorenz_jac(t: float, y: Array, *, sigma: float, rho: float, beta: float) -> Array:
    # J_{ij} = d f_i / d y_j
    return np.array(
        [
            [-sigma, sigma, 0],
            [rho - y[2], -1, -y[0]],
            [y[1], y[0], -beta],
        ]
    )


# }}}


# {{{ solve

from pycaputo.controller import make_fixed_controller
from pycaputo.fode import CaputoWeightedEulerMethod

# NOTE: order example taken from https://doi.org/10.1016/j.chaos.2009.03.016
alpha = (0.985, 0.99, 0.99)
# NOTE: the (sigma, rho, beta) parameters are the classic Lorenz attractor
# parameters and we take alpha ~ 1 to obtain something similar here
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
y0 = np.array([-2.0, 1.0, -1.0])

stepper = CaputoWeightedEulerMethod(
    derivative_order=alpha,
    control=make_fixed_controller(1.0e-2, tstart=0.0, tfinal=75.0),
    source=partial(lorenz, sigma=sigma, rho=rho, beta=beta),
    source_jac=partial(lorenz_jac, sigma=sigma, rho=rho, beta=beta),
    y0=(y0,),
    theta=1.0,
)

from pycaputo.fode import StepCompleted, evolve

ts = []
ys = []

for event in evolve(stepper):
    assert isinstance(event, StepCompleted)
    ts.append(event.t)
    ys.append(event.y)

    logger.info(
        "[%06d] t = %.5e dt = %.5e energy %.5e",
        event.iteration,
        event.t,
        event.dt,
        np.linalg.norm(event.y),
    )


# }}}


# {{{ plot

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise SystemExit(0) from exc

from pycaputo.utils import figure, set_recommended_matplotlib

set_recommended_matplotlib()
t = np.array(ts)
y = np.array(ys).T

with figure("lorenz-cycle-xy") as fig:
    ax = fig.gca()

    ax.plot(y[0], y[1], ls="--")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

with figure("lorenz-cycle-xz") as fig:
    ax = fig.gca()

    ax.plot(y[0], y[2], ls="--")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")

with figure("lorenz-cycle-yz") as fig:
    ax = fig.gca()

    ax.plot(y[1], y[2], ls="--")
    ax.set_xlabel("$y$")
    ax.set_ylabel("$z$")

# }}}
