# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

r"""Solve the fractional Brusselator model for a standard choice of parameters
where it exhibits a limit cycle.

The example uses the PECE method with a fixed step size.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from pycaputo.logging import get_logger
from pycaputo.typing import Array

log = get_logger("tutorial")

# {{{ Brusselator


# [tutorial-func-start]
def brusselator(t: float, y: Array, *, a: float, mu: float) -> Array:
    return np.array([
        a - (mu + 1) * y[0] + y[0] ** 2 * y[1],
        mu * y[0] - y[0] ** 2 * y[1],
    ])
    # [tutorial-func-end]


# }}}


# {{{ solve

# [tutorial-method-start]
from pycaputo.controller import make_fixed_controller
from pycaputo.derivatives import CaputoDerivative as D
from pycaputo.fode import caputo

alpha = 0.8
a = 1.0
mu = 4.0
y0 = np.array([1.0, 2.0])

stepper = caputo.PECE(
    ds=(D(alpha), D(alpha)),
    control=make_fixed_controller(1.0e-2, tstart=0.0, tfinal=50.0),
    source=partial(brusselator, a=a, mu=mu),
    y0=(y0,),
    corrector_iterations=1,
)
# [tutorial-method-end]

# [tutorial-evolve-start]
from pycaputo.events import StepCompleted
from pycaputo.stepping import evolve

ts = []
ys = []

for event in evolve(stepper):
    assert isinstance(event, StepCompleted)
    ts.append(event.t)
    ys.append(event.y)

    log.info(
        "[%06d] t = %.5e dt = %.5e energy %.5e",
        event.iteration,
        event.t,
        event.dt,
        np.linalg.norm(event.y),
    )
# [tutorial-evolve-end]

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

    t = np.array(ts)
    y = np.array(ys).T

    with figure(f"tutorial-brusselator{suffix}") as fig:
        ax = fig.gca()

        ax.plot(t, y[1], "--", lw=3, label="$y$")
        ax.plot(t, y[0], lw=3, label="$x$")

        ax.set_xlabel("$t$")
        ax.legend(loc="lower left", bbox_to_anchor=(0.5, 1.0), ncol=2, mode="expand")

    with figure(f"tutorial-brusselator-cycle{suffix}") as fig:
        ax = fig.gca()

        ax.plot(y[0], y[1], ls="--")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

# }}}
