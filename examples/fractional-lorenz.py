# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from functools import partial

import numpy as np

from pycaputo.utils import Array

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
    return np.array(
        [
            [-sigma, sigma, 0],
            [rho - y[2], -1, -y[0]],
            [y[1], y[0], -beta],
        ]
    )


# }}}


# {{{ solve

from pycaputo.fode import CaputoWeightedEulerMethod

# NOTE: the (sigma, rho, beta) parameters are the classic Lorenz attractor
# parameters and we take alpha ~ 1 to obtain something similar here
alpha = 0.995
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
y0 = np.array([-2.0, 1.0, -1.0])

stepper = CaputoWeightedEulerMethod(
    derivative_order=alpha,
    predict_time_step=1.0e-2,
    source=partial(lorenz, sigma=sigma, rho=rho, beta=beta),
    source_jac=partial(lorenz_jac, sigma=sigma, rho=rho, beta=beta),
    tspan=(0, 75),
    y0=(y0,),
    theta=0.5,
)

from pycaputo.fode import StepCompleted, evolve

ts = []
ys = []

for event in evolve(stepper):
    assert isinstance(event, StepCompleted)
    ts.append(event.t)
    ys.append(event.y)

    print(
        f"[{event.iteration:6d}] "
        f"t = {event.t:8.5f} energy {np.linalg.norm(event.y):.5e}"
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

with figure("lorenz-cycle-xy.svg") as fig:
    ax = fig.gca()

    ax.plot(y[0], y[1], ls="--")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

with figure("lorenz-cycle-xz.svg") as fig:
    ax = fig.gca()

    ax.plot(y[0], y[2], ls="--")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")

with figure("lorenz-cycle-yz.svg") as fig:
    ax = fig.gca()

    ax.plot(y[1], y[2], ls="--")
    ax.set_xlabel("$y$")
    ax.set_ylabel("$z$")

# }}}
