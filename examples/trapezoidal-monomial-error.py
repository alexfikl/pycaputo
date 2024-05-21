# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import numpy as np
import numpy.linalg as la

from pycaputo.fode import caputo, special
from pycaputo.history import ProductIntegrationHistory
from pycaputo.logging import get_logger
from pycaputo.utils import Array, BlockTimer, gamma

logger = get_logger("trap")
rng = np.random.default_rng(seed=42)

# {{{ solution

# fractional order
alpha = 0.8
m = int(np.ceil(alpha))
# right-hand side power
beta = 2
# time interval
tstart, tfinal = 0.0, 5.0

# construct an exact solution of the form
#   y(t) = sum Yv[i] * (t - t_0) ** nu[i]
nns = int(np.ceil(3 / alpha))
nu = np.array([i + alpha * j for i in range(nns) for j in range(nns)])
nu = np.sort(nu[nu <= 2.0 + alpha])
Yv = rng.uniform(0.0, 3.0, size=nu.size)

# get term with nu == 2 alpha
Yvstar_ref = Yv[3]
nustar = nu[3]

# exact solution: y(t)
func_ref = special.CaputoMonomial(
    Yv=Yv, nu=nu, t0=tstart, alpha=alpha, beta=beta, c=1.0
)

# exact solution: (t - t_0) ** nustar
func_star = special.CaputoMonomial(
    Yv=np.array([1.0]),
    nu=np.array([nustar]),
    t0=tstart,
    alpha=alpha,
    beta=1.0,
    c=0.0,
)

# }}}

# {{{ evolve

from pycaputo.controller import (
    Controller,
    RandomController,
    make_fixed_controller,
    make_graded_controller,
    make_random_controller,
)

grid_type = "uniform"
if grid_type == "uniform":
    c: Controller = make_fixed_controller(5.0e-3, tstart=tstart, tfinal=tfinal)
elif grid_type == "graded":
    c = make_graded_controller(5.0e-3, tstart=tstart, tfinal=tfinal, alpha=alpha)
elif grid_type == "random":
    c = make_random_controller(tstart=tstart, tfinal=tfinal, rng=rng)
else:
    raise ValueError(f"Unknown grid type: '{grid_type}'")

im_stepper = caputo.Trapezoidal(
    derivative_order=(alpha,),
    control=c,
    source=func_ref.source,
    source_jac=func_ref.source_jac,
    y0=(func_ref(tstart),),
)

# Implicit stepper for `(t - t0)^{nu^star}` where `nu^star = 2 alpha`
en_stepper = caputo.Trapezoidal(
    derivative_order=(alpha,),
    control=c,
    source=func_star.source,
    source_jac=func_star.source_jac,
    y0=(func_star(tstart),),
)

from pycaputo.events import StepCompleted
from pycaputo.stepping import evolve

t_l: list[float] = []
y_im_l: list[Array] = []
y_ref_l: list[Array] = []
y_en_l: list[Array] = []
error_l: list[Array] = []
error_ref_l: list[Array] = []

en_l: list[Array] = []
en_ref_l: list[Array] = []
yv_l: list[Array] = []
yv_ref_l: list[Array] = []
rn_l: list[Array] = []
rn_ref_l: list[Array] = []

from pycaputo.fode.caputo import (
    _error_explicit_step,  # noqa: PLC2701
    _weights_quadrature_trapezoidal,  # noqa: PLC2701
)

with BlockTimer("evolve") as bt:
    dtmax = 0.0
    dtinit = getattr(c, "dtinit", None)
    history = ProductIntegrationHistory.empty_like(im_stepper.y0[0])

    for im_event, en_event in zip(
        evolve(im_stepper, history=history, dtinit=dtinit),
        evolve(en_stepper, dtinit=dtinit),
    ):
        assert isinstance(im_event, StepCompleted)
        assert isinstance(en_event, StepCompleted)
        assert abs(en_event.t - im_event.t) < 1.0e-14

        if not np.any(np.isfinite(im_event.y)):
            logger.error("%s | Implicit solution diverged: %r", im_event, im_event.y)
            break

        # compute exact solution
        n = im_event.iteration
        t_n = im_event.t
        y_ref = func_ref(t_n)

        # estimate reference remainder
        dtmax = max(im_event.dt, dtmax)
        En_ref = (
            0.0
            if en_event.iteration == 0
            else ((t_n - tstart) ** (alpha - 1.0) * dtmax ** (nustar - alpha + 1))
        )
        Rn_ref = Yvstar_ref * gamma(1 + nustar) / gamma(1 + nustar - alpha) * En_ref

        # get weights
        omegal, omegar = _weights_quadrature_trapezoidal(
            im_stepper, history.ts[: n + 1], n, n
        )

        # estimate remainder R_n
        if n >= 2:
            # estimate E_{n, \nu^\star}
            En = np.abs(en_event.y - func_star(en_event.t)).item()

            # estimate \Delta E = E_{n, \nu^{\star}}^{IM} - E_{n, \nu^\star}^{EX}
            ts = history.ts[n - 2 : n + 1]
            fs = (ts - tstart).reshape(-1, 1) ** (nustar - en_stepper.alpha)
            delta_e = _error_explicit_step(en_stepper, ts, fs)

            # estimate \Delta y = y_n^{IM} - y_n^{EX}
            fs = history.storage[n - 2 : n + 1]
            delta_y = _error_explicit_step(im_stepper, ts, fs)

            # estimate Y_{\nu^\star}
            Yvstar = gamma(1 + nustar - alpha) / gamma(1 + nustar) * (delta_y / delta_e)

            # estimate remainder
            Rn = 0.5 * Yvstar * gamma(1 + nustar) / gamma(1 + nustar - alpha) * En

            del fs
            del ts
        else:
            delta_e = delta_y = np.array([1.0e-15])
            En = np.abs(en_event.y - func_star(en_event.t)).item()
            Yvstar = np.array([Yvstar_ref])
            Rn = np.array([Rn_ref])

        # compute global error
        if n > 0:
            # multiply error with jacobian
            error = np.array([
                func_ref.source_jac(t_i, y_i) * e_i
                for t_i, y_i, e_i in zip(t_l, y_im_l, error_l)
            ]).reshape(-1, 1)
            error_ref = np.array([
                func_ref.source_jac(t_i, y_i) * e_i
                for t_i, y_i, e_i in zip(t_l, y_im_l, error_ref_l)
            ]).reshape(-1, 1)

            # add up the entire error
            # fmt: off
            error_history = (
                np.einsum("ij,ij->j", omegal, error)
                + np.einsum("ij,ij->j", omegar[:-1], error[1:]))
            error_ref_history = (
                np.einsum("ij,ij->j", omegal, error_ref)
                + np.einsum("ij,ij->j", omegar[:-1], error_ref[1:]))
            # fmt: on

            error_jac = 1.0 - omegar[-1] * func_ref.source_jac(t_n, im_event.y)
            error = (np.abs(Rn) + error_history) / error_jac
            error_ref = (np.abs(Rn_ref) + error_ref_history) / error_jac

            error = np.abs(error)
            error_ref = np.abs(error_ref)
        else:
            error = error_ref = np.array([0.0])

        # print iteration
        im_error = la.norm(y_ref - im_event.y, np.inf)
        logger.info("%s | error im %.8e model %.8e", im_event, im_error, error)
        logger.info(
            "   En %.8e En_ref %.8e Yv %.8e Yv_ref %.8e", En, En_ref, Yvstar, Yvstar_ref
        )
        logger.info(
            "   Delta pow %.8e y %.8e ratio %.8e", delta_e, delta_y, delta_y / delta_e
        )

        # append solutions
        t_l.append(t_n)
        y_im_l.append(im_event.y)
        y_en_l.append(en_event.y)
        y_ref_l.append(y_ref)

        error_l.append(error)
        error_ref_l.append(error_ref)

        rn_l.append(Rn)
        rn_ref_l.append(Rn_ref)
        en_l.append(En)
        en_ref_l.append(En_ref)
        yv_l.append(Yvstar)
        yv_ref_l.append(Yvstar_ref)

logger.info("%s", bt.pretty())

# }}}

# {{{ plot

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise SystemExit(0) from exc

from pycaputo.utils import figure, set_recommended_matplotlib

set_recommended_matplotlib()
basename = f"trapezoidal-monomial-{grid_type}-{beta * 100:03.0f}"

logger.info("Yvs: %s", Yv)
logger.info("nu:  %s / %s", nu, nu - alpha)

t = np.array(t_l)
y_im = np.array(y_im_l).T
y_en = np.array(y_en_l).T
y_ref = np.array(y_ref_l).T

error = np.array(error_l).T
error_ref = np.array(error_ref_l).T

rn = np.array(rn_l).T
rn_ref = np.array(rn_ref_l).T
en = np.array(en_l).T
en_ref = np.array(en_ref_l).T
yv = np.array(yv_l).T
yv_ref = np.array(yv_ref_l).T

mask = np.s_[1:]

y_en_ref = np.array([func_ref(t_j) for t_j in t])

if isinstance(c, RandomController):
    with figure(f"{basename}-timestep") as fig:
        ax = fig.gca()

        ax.semilogy(np.diff(t))
        ax.axhline(c.dtmin, color="k", linestyle="--")
        ax.axhline(c.dtmax, color="k", linestyle="--")

        ax.set_xlabel("$n$")
        ax.set_ylabel(r"$\Delta t_n$")

with figure(f"{basename}-solution") as fig:
    ax = fig.gca()

    ax.plot(t[mask], y_im[0, mask], "-", label="Approx")
    ax.plot(t[mask], y_ref[0, mask], "k--", label="Exact")

    ax.set_xlabel("$t$")
    ax.set_ylabel("$y$")
    ax.legend()

with figure(f"{basename}-power-error") as fig:
    ax = fig.gca()

    ax.semilogy(t[mask], en[mask], "-", label="Error")
    ax.semilogy(t[mask], en_ref[mask], "k--", label="Model")
    ax.set_ylim([1.0e-10, 1.0e-1])

    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$E_{n, \nu^\star - \alpha}$")
    ax.legend()

with figure(f"{basename}-ynustar") as fig:
    ax = fig.gca()

    ax.plot(t[mask], yv[0, mask], "-", label="Approx")
    ax.plot(t[mask], yv_ref[mask], "k--", label="Exact")

    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$Y_{\nu^\star}$")
    ax.legend()

with figure(f"{basename}-error", figsize=(12, 8)) as fig:
    ax = fig.gca()

    e_im = np.abs(y_ref[0, mask] - y_im[0, mask])

    ax.plot(t[mask], e_im, "-", label="Implicit")
    ax.semilogy(t[mask], rn[0, mask], "k--", label="Remainder")
    ax.semilogy(t[mask], error[0, mask], "r--", label="Estimate")
    ax.semilogy(t[mask], rn_ref[mask], "k:", label="Remainder (Ref)")
    ax.semilogy(t[mask], error_ref[0, mask], "r:", label="Estimate (Ref)")
    ax.set_ylim([1.0e-10, 1.0e-1])

    ax.set_xlim([tstart, tfinal])
    ax.set_xlabel("$t$")
    ax.set_ylabel("Error")
    ax.legend(bbox_to_anchor=(1.3, 1.0), prop={"size": 16})

# }}}
