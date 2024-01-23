# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

# This model is taken from [Naud2008] with the same parameters
#
# .. [Naud2008] R. Naud, N. Marcille, C. Clopath, W. Gerstner,
#       *Firing Patterns in the Adaptive Exponential Integrate-and-Fire Model*,
#       Biological Cybernetics, Vol. 99, pp. 335--347, 2008,
#       `DOI <https://doi.org/10.1007/s00422-008-0264-7>`__.

from __future__ import annotations

import os

import numpy as np

from pycaputo.integrate_fire import ad_ex
from pycaputo.logging import get_logger

logger = get_logger("integrate-and-fire")

# {{{ model

adaptive = "PYCAPUTO_AD_EX_NO_ADAPTIVE" not in os.environ

# time interval
tstart, tfinal = 0.0, 32.0
# fractional order
alpha = 0.91, 0.99

param = ad_ex.get_ad_ex_parameters("Naud4d")
model = ad_ex.AdExModel(param.nondim(alpha))

logger.info("Parameters:\n%s", model.param)

# }}}

# {{{ setup

from pycaputo.controller import (
    Controller,
    make_fixed_controller,
    make_jannelli_controller,
)

# initial condition
rng = np.random.default_rng(seed=42)
y0 = np.array([
    rng.uniform(model.param.v_reset - 10, model.param.v_reset),
    rng.uniform(),
])

if adaptive:
    dtinit = 1.0e-1
    dtmin = 1.0e-5
    c: Controller = make_jannelli_controller(
        tstart,
        tfinal,
        dtmin=dtmin,
        chimin=0.001,
        chimax=0.01,
        abstol=1.0e-4,
    )
else:
    dtinit = dtmin = 1.0e-4
    c = make_fixed_controller(dtinit, tstart=tstart, tfinal=tfinal)

stepper = ad_ex.CaputoAdExIntegrateFireL1Model(
    derivative_order=alpha,
    control=c,
    y0=(y0,),
    source=model,
)

# }}}

# {{{ evolution

from pycaputo.integrate_fire import StepAccepted, StepRejected
from pycaputo.stepping import evolve

ts = []
ys = []
spikes = []
eests = []

for event in evolve(stepper, dtinit=dtinit):
    if isinstance(event, StepAccepted):
        ts.append(event.t)
        ys.append(event.y)
        eests.append(event.eest)

        if event.spiked:
            spikes.append(event.iteration)
    elif isinstance(event, StepRejected):
        pass
    else:
        raise RuntimeError(event)

    logger.info("%s energy %.5e eest %+.5e", event, np.linalg.norm(event.y), event.eest)

# }}}

# {{{ plotting

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise SystemExit(0) from exc

from pycaputo.utils import figure, set_recommended_matplotlib

set_recommended_matplotlib()

# vectorize variables
s = np.array(spikes)
t = np.array(ts)
y = np.array(ys).T
eest = np.array(eests)

basename = f"integrate-fire-adex-{100 * alpha[0]:.0f}-{100 * alpha[1]:.0f}"
with figure(f"{basename}-v") as fig:
    ax = fig.gca()

    ax.plot(t, y[0], lw=3)
    ax.axhline(model.param.v_peak, color="k", ls="-")
    ax.axhline(model.param.v_reset, color="k", ls="--")
    ax.plot(t[s], y[0][s], "rv")
    ax.plot(t[s], np.full_like(t[s], model.param.v_reset), "r^")

    ax.set_xlabel("$t$")
    ax.set_ylabel("$V$")

with figure(f"{basename}-w") as fig:
    ax = fig.gca()

    ax.plot(t, y[1], lw=3)
    ax.plot(t[s], y[1][s], "r^")
    ax.plot(t[s], y[1][s] + model.param.b, "rv")

    ax.set_xlabel("$t$")
    ax.set_ylabel("$w$")

with figure(f"{basename}-dt") as fig:
    ax = fig.gca()

    ax.semilogy(t[:-1], np.diff(t))
    ax.axhline(dtinit, color="k", ls="--")
    ax.axhline(dtmin, color="k", ls="--")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\Delta t$")

with figure(f"{basename}-eest") as fig:
    ax = fig.gca()

    ax.plot(t, eest)
    ax.axhline(1.0, color="k", ls="--")
    ax.axhline(0.0, color="k", ls="--")
    ax.plot(t[s], eest[s], "ro")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$E_{est}$")

# }}}
