# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

# NOTE: the parameters and general setup for this are taken from Jannelli2020
# NOTE: the figures should be compared to Figure 6 and surrounding text

from __future__ import annotations

import numpy as np

from pycaputo.controller import JannelliIntegralController, estimate_initial_time_step
from pycaputo.fode import CaputoPECEMethod
from pycaputo.logging import get_logger
from pycaputo.utils import Array

logger = get_logger("van-der-pol-adaptive")

# {{{ van der Pol oscillator


def van_der_pol(t: float, y: Array, *, mu: float = 4.0) -> Array:
    return np.array([y[1], mu * (1.0 - y[0] ** 2) * y[1] - y[0]])


def van_der_pol_jac(t: float, y: Array, *, mu: float = 4.0) -> Array:
    return np.array(
        [[0.0, 1.0], [-mu * 2.0 * y[0] * y[1] - 1.0, mu * (1.0 - y[0] ** 2)]]
    )


# }}}


# {{{ set up problem

tstart, tfinal = 0.0, 4.0
c = JannelliIntegralController(
    tstart=tstart,
    tfinal=tfinal,
    nsteps=None,
    dtmin=1.0e-3,
    qmin=0.5,  # corresponds to sigma in Jannelli2020
    qmax=1.5,  # corresponds to rho in Jannelli2020
    chimin=0.03,
    chimax=0.3,
    abstol=1.0e-12,
    reltol=1.0e-1,  # not used by this method
)

alpha = 0.8
y0 = np.array([1.0, 0.0])

m = CaputoPECEMethod(
    derivative_order=(alpha, alpha),
    control=c,
    source=van_der_pol,
    y0=(y0,),
    corrector_iterations=2,
)

dt0 = 1.0e-1
dtest = estimate_initial_time_step(
    tstart,
    y0,
    m.source,
    m.smallest_derivative_order,
    trunc=m.order,
)
logger.info("Initial time step %.8e estimate %.8e", dt0, dtest)

# }}}


# {{{ evolve


from pycaputo.fode import StepAccepted, StepRejected, evolve

ts = []
ys = []

truncs = []
eests = []
qs = []

for event in evolve(m, dt=dt0):
    if isinstance(event, StepAccepted):
        ts.append(event.t)
        ys.append(event.y)

        truncs.append(event.trunc)
        eests.append(event.eest)
        qs.append(event.q)
    elif isinstance(event, StepRejected):
        pass
    else:
        raise ValueError("Evolution failed!")

    logger.info(
        "%s[%06d] t = %.5e dt = %.5e (eest %+.5e q %.5e) energy %.5e",
        "[green][A][/]" if event.eest <= 1.0 else "[red][R][/]",
        event.iteration,
        event.t,
        event.dt,
        event.eest,
        event.q,
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
trunc = np.array(truncs).T

with figure("van-der-pol-adaptive-pece-phase") as fig:
    ax = fig.gca()

    ax.plot(y[0], y[1], "o", ms=3, fillstyle="none")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    # ax.set_aspect("equal")

with figure("van-der-pol-adaptive-pece-solution", nrows=2, figsize=(8, 8)) as fig:
    ax = fig.axes

    ax[0].plot(t, y[0], "o-", ms=3, fillstyle="none", label="$x$")
    ax[0].plot(t, y[1], "o-", ms=3, fillstyle="none", label="$y$")
    ax[0].legend(loc="lower left", bbox_to_anchor=(0.5, 0.97), ncol=2, mode="expand")

    ax[1].semilogy(t[:-1], np.diff(t), "o-", fillstyle="none", ms=3)
    ax[1].set_xlabel("$t$")
    ax[1].set_ylabel(r"$\Delta t$")
    ax[1].set_ylim([c.dtmin, dt0])

with figure("van-der-pol-adaptive-pece-eest", nrows=2, figsize=(8, 8)) as fig:
    ax = fig.axes

    ax[0].plot(t, trunc[0], "o-", ms=3, fillstyle="none")
    ax[0].plot(t, trunc[1], "o-", ms=3, fillstyle="none")
    ax[0].set_ylabel(r"$\tau$")

    ax[1].plot(t, eests, "-")
    ax[1].axhline(1.0, ls="--", color="k")
    ax[1].axhline(0.0, ls="--", color="k")
    ax[1].set_xlabel("$t$")
    ax[1].set_ylabel(r"$E_{\text{est}}$")

# }}}
