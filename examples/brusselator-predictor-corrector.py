# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from functools import partial

import numpy as np

from pycaputo.utils import Array

# {{{ Brusselator


def brusselator(t: float, y: Array, *, a: float, mu: float) -> Array:
    return np.array(
        [
            a - (mu + 1) * y[0] + y[0] ** 2 * y[1],
            mu * y[0] - y[0] ** 2 * y[1],
        ]
    )


# }}}


# {{{ solve

from pycaputo.derivatives import CaputoDerivative, Side
from pycaputo.fode import CaputoPECEMethod, make_predict_time_step_fixed

alpha = 0.8
a = 1.0
mu = 4.0
y0 = np.array([1.0, 2.0])

stepper = CaputoPECEMethod(
    d=CaputoDerivative(order=alpha, side=Side.Left),
    predict_time_step=make_predict_time_step_fixed(1.0e-2),
    source=partial(brusselator, a=a, mu=mu),
    tspan=(0, 50),
    y0=(y0,),
    corrector_iterations=1,
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

from pycaputo.utils import figure, set_recommended_matplotlib

set_recommended_matplotlib()
t = np.array(ts)
y = np.array(ys).T

with figure("brusselator-predictor-corrector.svg") as fig:
    ax = fig.gca()

    ax.plot(t, y[1], "--", lw=3, label="$y$")
    ax.plot(t, y[0], lw=3, label="$x$")

    ax.set_xlabel("$t$")
    ax.legend(loc="lower left", bbox_to_anchor=(0.5, 0.97), ncol=2, mode="expand")

with figure("brusselator-predictor-corrector-cycle.svg") as fig:
    ax = fig.gca()

    ax.plot(y[0], y[1], ls="--")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

# }}}
