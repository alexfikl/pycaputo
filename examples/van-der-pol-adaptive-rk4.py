# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

# NOTE: the parameters and general setup for this are taken from Hairer2010
# NOTE: the figures should be compared to Figure 8.1 and surrounding text

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycaputo.controller import (
    IntegralController,
    ProportionalIntegralController,
    estimate_initial_time_step,
    evaluate_error_estimate,
    evaluate_timestep_accept,
    evaluate_timestep_factor,
    evaluate_timestep_reject,
)
from pycaputo.derivatives import CaputoDerivative, FractionalOperator, Side
from pycaputo.fode import FractionalDifferentialEquationMethod
from pycaputo.logging import get_logger
from pycaputo.utils import Array

logger = get_logger("van-der-pol-adaptive")


# {{{ van der Pol oscillator


@dataclass(frozen=True)
class RungeKutta4Dummy(FractionalDifferentialEquationMethod):
    @property
    def name(self) -> str:
        return "RK4"

    @property
    def order(self) -> float:
        return 4.0

    @property
    def d(self) -> tuple[FractionalOperator, ...]:
        return tuple(
            [CaputoDerivative(d, side=Side.Left) for d in self.derivative_order]
        )


def van_der_pol(t: float, y: Array, *, eps: float = 1.0e-3) -> Array:
    return np.array([y[1], ((1.0 - y[0] ** 2) * y[1] - y[0]) / eps])


def van_der_pol_jac(t: float, y: Array, *, mu: float) -> Array:
    return np.array([[0.0, 1.0], [-2.0 * y[0] * y[1] - 1.0, mu * (1.0 - y[0] ** 2)]])


# }}}


# {{{ set up problem

name = "pi"
cls = IntegralController if name == "i" else ProportionalIntegralController

tstart, tfinal = 0.0, 2.0
c = cls(
    tstart=tstart,
    tfinal=tfinal,
    nsteps=None,
    dtmin=1.0e-8,
    qmin=0.5,
    qmax=10.0,
    abstol=1.0e-4,
    reltol=1.0e-4,
)

y0 = np.array([2.0, -0.6])
yp0 = van_der_pol(tstart, y0)

m = RungeKutta4Dummy(
    derivative_order=(1.0,),
    control=c,
    source=van_der_pol,
    y0=(y0,),
)

dt = 1.0e-4
logger.info(
    "Initial time step %.8e estimate %.8e",
    dt,
    estimate_initial_time_step(
        tstart,
        y0,
        m.source,
        1.0,
        trunc=m.order,
        abstol=c.abstol,
        reltol=c.reltol,
    ),
)

# }}}


# {{{ evolve


def rk4step(t: float, y: Array, dt: float) -> tuple[Array, Array]:
    k1 = dt * van_der_pol(t, y)
    k2 = dt * van_der_pol(t + 0.5 * dt, y + 0.5 * k1)
    k3 = dt * van_der_pol(t + 0.5 * dt, y + 0.5 * k2)
    k4 = dt * van_der_pol(t + dt, y + k3)
    yp = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return yp, dt * np.abs(yp - y)


order = 4

n = 0
y = yprev = y0
t = tstart
eps = float(5.0 * np.finfo(y.dtype).eps)

nsteps = 2**14
ts = np.empty(nsteps)
ys = np.empty((nsteps, y0.size))
qs = np.empty(nsteps)
eests = np.empty(nsteps)
dts = np.empty(nsteps)
rejects = np.empty(nsteps, dtype=bool)

nrejects = 0
ys[0] = y0
while not c.finished(n, t):
    # advance
    y, trunc = rk4step(t, yprev, dt)
    ymid, _ = rk4step(t, yprev, dt / 2)
    yhat, _ = rk4step(t + dt / 2, ymid, dt / 2)
    trunc = np.abs(y - yhat)

    # determine scaled error estimate
    eest = evaluate_error_estimate(c, m, trunc, y, yprev)
    # determine time step factor
    q = evaluate_timestep_factor(c, m, eest)

    # determine if the time step is accepted
    state = {"n": n, "t": t, "y": y, "eest": eest}
    if eest <= 1.0 or nrejects > 64:
        dtnext = evaluate_timestep_accept(c, m, q, dt, state)

        # advance values
        n += 1
        t += dt
        yprev = y

        # save solutions for plotting
        ts[n], ys[n] = t, y
        qs[n], eests[n], dts[n] = q, eest, dt
        rejects[n] = nrejects == 0

        nrejects = 0
    else:
        nrejects += 1
        dtnext = evaluate_timestep_reject(c, m, q, dt, state)

    logger.info(
        "[%s] [%.4d] t = %.8e dt %.8e (%.8e) trunc (%.8e, %.8e)",
        "[green]A[/green]" if eest <= 1.0 else "[red]R[/red]",
        n + 1,
        t,
        dt,
        dtnext,
        *trunc,
    )

    dt = float(min(dtnext, tfinal - t) + 5.0 * np.finfo(y.dtype).eps)

# }}}


# {{{ plot

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise SystemExit(0) from exc

from pycaputo.utils import figure, set_recommended_matplotlib

set_recommended_matplotlib()

with figure(f"example-adaptive-rk4-{name}-phase") as fig:
    ax = fig.gca()

    ax.plot(ys[:n, 0], ys[:n, 1], "o", ms=3, fillstyle="none")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")

with figure(f"example-adaptive-rk4-{name}-xy") as fig:
    ax = fig.gca()

    ax.plot(ts[:n], ys[:n, 0], "o", ms=3, fillstyle="none", label="$x$")
    # ax.plot(ts[:n], ys[:n, 1], "o", ms=3, fillstyle="none", label="$y$")
    ax.set_aspect(0.5)
    ax.set_xlabel("$t$")

with figure(f"example-adaptive-rk4-{name}-qs") as fig:
    ax = fig.gca()

    ax.plot(ts[:n], qs[1 : n + 1], "o-", ms=3, label="$q$")
    ax.plot(ts[:n], eests[1 : n + 1], "o--", ms=3, label=r"$E_{\text{est}}$")
    ax.axhline(c.qmin, ls="--", color="k")
    # ax.axhline(c.qmax, ls="--", color="k")
    ax.axhline(1.0, ls="-", color="k")
    ax.set_xlabel("$t$")
    ax.legend()

with figure(f"example-adaptive-rk4-{name}-dt") as fig:
    ax = fig.gca()

    ax.semilogy(ts[:n], dts[1 : n + 1], "k-")
    ax.semilogy(ts[:n][rejects[:n]], dts[1 : n + 1][rejects[:n]], "o", ms=1)
    ax.semilogy(ts[:n][~rejects[:n]], dts[1 : n + 1][~rejects[:n]], "rx", ms=4)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\Delta t$")
    ax.set_aspect(0.5)

# }}}
