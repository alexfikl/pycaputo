# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import numpy as np

from pycaputo.derivatives import CaputoDerivative as D
from pycaputo.integrate_fire import pif
from pycaputo.logging import get_logger

log = get_logger("integrate-and-fire")

# {{{ model

# time interval
tstart, tfinal = 0.0, 32.0
# fractional order
alpha = 0.8

param = pif.PIFDim(current=160, C=100, v_reset=-48, v_peak=0.0)
model = pif.PIFModel(param.nondim(alpha, V_ref=1.0, I_ref=20.0))

log.info("Parameters:\n%s", model.param)

# }}}

# {{{ setup

from pycaputo.controller import make_jannelli_controller

# initial condition
rng = np.random.default_rng()
y0 = np.array([rng.uniform(model.param.v_reset, model.param.v_peak)])

tspikes = model.param.constant_spike_times(tfinal, V0=y0[0])
log.info("tspike %.8e tstart %.8e, tfinal %.8e", tspikes[0], tstart, tfinal)
if tspikes.size < 2:
    raise ValueError(
        "This example expects at least two spikes. Try increasing 'alpha' or 'tfinal'."
    )

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
    ds=(D(alpha),),
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

    log.info("%s energy %.5e eest %+.5e", event, np.linalg.norm(event.y), event.eest)

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
y = np.array(ys)
eest = np.array(eests)

# get exact solution up to the second spike
t_ref = t[: s[1]]
y_ref = y0 + model.param.current * t_ref**alpha / math.gamma(1 + alpha)

# make variables dimensional for plotting
dim = model.param.ref
t = dim.time(t)
y = dim.var(y)
t_ref = dim.time(t_ref)
y_ref = dim.var(y_ref)
tspikes = dim.time(tspikes)

basename = f"integrate-fire-pif-{100 * alpha:.0f}"
with figure(basename) as fig:
    ax = fig.gca()

    ax.plot(t, y, lw=3)
    ax.plot(t_ref, y_ref, "k--")
    ax.axhline(param.v_peak, color="k", ls="-")
    ax.axhline(param.v_reset, color="k", ls="--")
    ax.plot(t[s], y[s], "ro")
    ax.plot(tspikes, np.full_like(tspikes, param.v_peak), "kx")

    ax.set_xlabel("$t$ (ms)")
    ax.set_ylabel("$V$ (mV)")

with figure(f"{basename}-dt") as fig:
    ax = fig.gca()

    ax.semilogy(t[:-1], np.diff(t))
    ax.axhline(dim.time(dtinit), color="k", ls="--")
    ax.axhline(dim.time(c.dtmin), color="k", ls="--")
    ax.set_xlabel("$t$ (ms)")
    ax.set_ylabel(r"$\Delta t$ (ms)")

with figure(f"{basename}-eest") as fig:
    ax = fig.gca()

    ax.plot(t, eest)
    ax.axhline(1.0, color="k", ls="--")
    ax.axhline(0.0, color="k", ls="--")
    ax.plot(t[s], eest[s], "ro")
    ax.set_xlabel("$t$ (ms)")
    ax.set_ylabel("$E_{est}$")

# }}}
