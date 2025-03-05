# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo.derivatives import CaputoDerivative as D
from pycaputo.integrate_fire import eif
from pycaputo.logging import get_logger

log = get_logger("integrate-and-fire")

# {{{ model

# time interval
tstart, tfinal = 0.0, 50.0
# fractional order
alpha = 0.95

param = eif.EIFDim(
    current=160,
    C=100,
    gl=12.0,
    e_leak=-60.0,
    delta_t=2.0,
    vt=-50.0,
    v_reset=-48,
    v_peak=0.0,
)
model = eif.EIFModel(param.nondim(alpha))

log.info("Parameters:\n%s", param)
log.info("Parameters:\n%s", model.param)

# }}}

# {{{ setup

from pycaputo.controller import make_jannelli_controller

# initial condition
rng = np.random.default_rng()
y0 = np.array([rng.uniform(model.param.v_reset, model.param.v_peak)])

dtinit = 1.0e-3
c = make_jannelli_controller(
    tstart,
    tfinal,
    dtmin=1.0e-5,
    chimin=0.01,
    chimax=0.1,
    abstol=1.0e-4,
)

stepper = eif.CaputoExponentialIntegrateFireL1Method(
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
t = np.array(ts)
s = np.array(spikes)
y = np.array(ys)
eest = np.array(eests)

# dimensionalize variables
dim = model.param.ref
t = dim.time(t)
y = dim.var(y)

basename = f"integrate-fire-eif-{100 * alpha:.0f}"
with figure(basename) as fig:
    ax = fig.gca()

    ax.plot(t, y, lw=3)
    ax.axhline(param.v_peak, color="k", ls="-")
    ax.axhline(param.v_reset, color="k", ls="--")
    ax.axhline(param.vt, color="k", ls=":")
    ax.plot(t[s], y[s], "ro")

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
