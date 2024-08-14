# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo.fode.gallery import Lu
from pycaputo.logging import get_logger
from pycaputo.utils import TicTocTimer

logger = get_logger("gallery")
time = TicTocTimer()

# {{{ solve

# setup up system (parameters from Figure 5.35 from [Petras2011])
alpha = (0.985, 0.99, 0.98)
y0 = np.array([0.2, 0.5, 0.3])
func = Lu(a=36.0, b=3.0, c=20.0)

logger.info("alpha %s y0 %s parameters %s", alpha, y0, func)

# setup up stepper
from pycaputo.controller import make_fixed_controller
from pycaputo.fode import caputo

dt = 5.0e-3
stepper = caputo.PECE(
    derivative_order=alpha,
    control=make_fixed_controller(dt, tstart=0.0, tfinal=90.0),
    source=func.source,
    y0=(y0,),
    corrector_iterations=1,
)

nsteps = stepper.control.nsteps
assert nsteps is not None

logger.info("%s", stepper.control)

# evolve system
from pycaputo.events import StepCompleted
from pycaputo.stepping import evolve

ys = []

time.tic()
for event in evolve(stepper, dtinit=dt):
    assert isinstance(event, StepCompleted)
    if event.iteration % 50 == 0:
        time.toc()
        logger.info("%s norm %.12e (%s)", event, np.linalg.norm(event.y), time.short())
        time.tic()

    ys.append(event.y)

y = np.array(ys).T

# }}}

# {{{ plot

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    logger.warning("'matplotlib' is not available.")
    raise SystemExit(0) from exc

from pycaputo.utils import figure, get_default_dark, set_recommended_matplotlib

for dark, suffix in get_default_dark():
    set_recommended_matplotlib(dark=dark, overrides={"lines": {"linewidth": 1}})

    with figure(f"gallery-lu{suffix}", projection="3d") as fig:
        ax = fig.gca()
        ax.view_init(elev=15, azim=-55, roll=0)

        ax.plot(y[0], y[1], y[2])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

# }}}
