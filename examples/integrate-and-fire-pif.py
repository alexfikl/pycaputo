# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo.integrate_fire import pif
from pycaputo.logging import get_logger

logger = get_logger("integrate-and-fire")

# {{{ model

# time interval
tstart, tfinal = 0.0, 32.0
# fractional order
alpha = 0.8

param = pif.PIFDim(current=160, C=100, v_reset=-48, v_peak=0.0)
model = pif.PIFModel(param.nondim(alpha, v_ref=1.0, I_ref=20.0))

logger.info("Parameters:\n%s", model.param)

# }}}

# {{{ setup

from pycaputo.controller import make_jannelli_controller

# initial condition
rng = np.random.default_rng()
y0 = np.array([rng.uniform(model.param.v_reset, model.param.v_peak)])

tspike = model.param.first_spike_time(y0[0])
logger.info("tspike %.8e tstart %.8e, tfinal %.8e", tspike, tstart, tfinal)

dtinit = 1.0e-1
c = make_jannelli_controller(
    tstart,
    tfinal,
    dtmin=1.0e-5,
    chimin=0.01,
    chimax=0.1,
    abstol=1.0e-4,
)

stepper = pif.CaputoPerfectIntegrateFireL1Method(
    derivative_order=(alpha,),
    control=c,
    y0=(y0,),
    source=model.source,
    model=model,
)

# }}}

# {{{ evolution

from pycaputo.integrate_fire import StepAccepted, StepRejected, evolve

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
        raise RuntimeError("Evolution failed!")

    logger.info(
        "[%06d] t = %.5e dt = %.5e energy %.5e eest %.5e",
        event.iteration,
        event.t,
        event.dt,
        event.eest,
        np.linalg.norm(event.y),
    )

# }}}

# {{{ plotting

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise SystemExit(0) from exc

from pycaputo.utils import figure, set_recommended_matplotlib

set_recommended_matplotlib()
t = np.array(ts)
s = np.array(spikes)

y = np.array(ys).squeeze()
y_ref = (y0 + model.param.current * t**alpha / stepper.gamma1p).squeeze()

eest = np.array(eests)

with figure("integrate-fire-pif") as fig:
    ax = fig.gca()

    ax.plot(t, y, lw=3)
    ax.plot(t, y_ref, "k--")
    ax.axhline(model.param.v_peak, color="k", ls="-")
    ax.axhline(model.param.v_reset, color="k", ls="--")
    ax.plot(t[s], y[s], "ro")
    ax.axvline(tspike, ls="--")

    ax.set_xlabel("$t$")
    ax.set_ylabel("$V$")

with figure("integrate-fire-pif-dt") as fig:
    ax = fig.gca()

    ax.semilogy(t[:-1], np.diff(t))
    ax.axhline(dtinit, color="k", ls="--")
    ax.axhline(c.dtmin, color="k", ls="--")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\Delta t$")

with figure("integrate-fire-pif-eest") as fig:
    ax = fig.gca()

    ax.plot(t, eest)
    ax.axhline(1.0, color="k", ls="--")
    ax.axhline(0.0, color="k", ls="--")
    ax.plot(t[s], eest[s], "ro")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$E_{est}$")

# }}}
