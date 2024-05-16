# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from functools import partial

import numpy as np
import numpy.linalg as la

from pycaputo.fode import caputo
from pycaputo.logging import get_logger
from pycaputo.utils import Array, BlockTimer, gamma

logger = get_logger("trapezoidal")
rng = np.random.default_rng(seed=42)


@dataclass(frozen=True)
class Trapezoidal(caputo.Trapezoidal):
    def _get_kwargs(self, *, scalar: bool = True) -> dict[str, object]:
        return {"method": "newton"}


# {{{ function


def func_y(t: float, *, t0: float, alpha: float, Yv: Array, nu: Array) -> Array:
    result = np.sum(Yv * (t - t0) ** nu)
    return np.array([result])


def func_dy(t: float, *, t0: float, alpha: float, Yv: Array, nu: Array) -> Array:
    gYv = gamma(1 + nu) / gamma(1 + nu - alpha) * Yv
    result = np.sum(gYv[1:] * (t - t0) ** (nu[1:] - alpha))

    return np.array([result])


def func_f(
    t: float, y: Array, *, t0: float, alpha: float, Yv: Array, nu: Array, beta: float
) -> Array:
    y_ref = func_y(t, t0=t0, alpha=alpha, Yv=Yv, nu=nu)
    dy_ref = func_dy(t, t0=t0, alpha=alpha, Yv=Yv, nu=nu)

    result = dy_ref + y_ref**beta - y**beta
    return np.array(result)


def func_f_jac(
    t: float, y: Array, *, t0: float, alpha: float, Yv: Array, nu: Array, beta: float
) -> Array:
    return beta * y ** (beta - 1.0)


# }}}

# {{{ solution

# fractional order
alpha = 0.8
m = int(np.ceil(alpha))
# right-hand side power
beta = 1.5
# time interval
tstart, tfinal = 0.0, 2.0

# construct an exact solution of the form
#   sum Yv[i] * (t - t_0) ** nu[i]
nns = int(np.ceil(3 / alpha))
nu = np.array([i + alpha * j for i in range(nns) for j in range(nns)])
nu = np.sort(nu[nu <= 2.0 + alpha])
Yv = rng.uniform(-5.0, 5.0, size=nu.size)

# construct initial problem
kwargs = {"t0": tstart, "alpha": alpha, "Yv": Yv, "nu": nu}
solution = partial(func_y, **kwargs)
y0 = solution(tstart)

# }}}

# {{{ evolve

from pycaputo.controller import (
    RandomController,
    make_fixed_controller,
    make_random_controller,
)

if True:
    c = make_fixed_controller(1.0e-3, tstart=tstart, tfinal=tfinal)
else:
    c = make_random_controller(tstart=tstart, tfinal=tfinal, rng=rng)

ex_stepper = caputo.ExplicitTrapezoidal(
    derivative_order=(alpha,),
    control=c,
    source=partial(func_f, **kwargs, beta=beta),
    y0=(y0,),
)

im_stepper = Trapezoidal(
    derivative_order=(alpha,),
    control=c,
    source=partial(func_f, **kwargs, beta=beta),
    source_jac=partial(func_f_jac, **kwargs, beta=beta),
    y0=(y0,),
)

from pycaputo.events import StepCompleted
from pycaputo.stepping import evolve

t_l = []
y_ex_l = []
y_im_l = []
y_ref_l = []
rn_l = []
dtmax = 0.0

with BlockTimer("evolve") as bt:
    for ex_event, im_event in zip(
        evolve(ex_stepper, dtinit=c.dtinit),
        evolve(im_stepper, dtinit=c.dtinit),
    ):
        assert isinstance(im_event, StepCompleted)
        assert isinstance(ex_event, StepCompleted)
        assert abs(ex_event.t - im_event.t) < 1.0e-14

        if not np.any(np.isfinite(im_event.y)):
            logger.error(
                "%s | Implicit solution is not finite: %r", im_event, im_event.y
            )
            break

        if not np.any(np.isfinite(ex_event.y)):
            logger.error(
                "%s | Explicit solution is not finite: %r", ex_event, ex_event.y
            )
            break

        # compute exact solution
        t_n = im_event.t
        y_ref = solution(t_n)

        # compute error model
        dtmax = max(dtmax, im_event.dt)
        mask = np.s_[1:]
        if im_event.iteration == 0:
            En = np.zeros_like(Yv[mask])
        else:
            En = np.where(
                nu[mask] - alpha <= 1.0,
                (t_n - tstart) ** (alpha - 1.0) * dtmax ** (nu[mask] - alpha + 1),
                (t_n - tstart) ** (nu[mask] - 2.0) * dtmax**2,
            )
        Rn = np.sum(Yv[mask] * gamma(1 + nu[mask]) / gamma(1 + nu[mask] - alpha) * En)

        # print iteration
        ex_error = la.norm(y_ref - ex_event.y) / la.norm(y_ref)
        im_error = la.norm(y_ref - im_event.y) / la.norm(y_ref)
        logger.info("%s | error im %.8e ex %.8e", im_event, im_error, ex_error)

        # append solutions
        t_l.append(t_n)
        y_ex_l.append(ex_event.y)
        y_im_l.append(im_event.y)
        y_ref_l.append(y_ref)
        rn_l.append(Rn)

logger.info("%s", bt.pretty())

# }}}

# {{{ plot

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise SystemExit(0) from exc

from pycaputo.utils import figure, set_recommended_matplotlib

set_recommended_matplotlib()

logger.info("Yvs: %s", Yv)
logger.info("nu:  %s / %s", nu, nu - alpha)

t = np.array(t_l)
y_ex = np.array(y_ex_l).T
y_im = np.array(y_im_l).T
y_ref = np.array(y_ref_l).T
rn = np.array(rn_l).T
mask = np.s_[1:]

if isinstance(c, RandomController):
    with figure("trapezoidal-monomial-timestep") as fig:
        ax = fig.gca()

        ax.semilogy(np.diff(t))
        ax.axhline(c.dtmin, color="k", linestyle="--")
        ax.axhline(c.dtmax, color="k", linestyle="--")

        ax.set_xlabel("$n$")
        ax.set_ylabel(r"$\Delta t_n$")

with figure("trapezoidal-monomial-solution") as fig:
    ax = fig.gca()

    ax.plot(t[mask], y_im[0, mask], "-", label="Explicit")
    ax.plot(t[mask], y_ex[0, mask], "--", label="Implicit")
    ax.plot(t[mask], y_ref[0, mask], "k--", label="Exact")

    ax.set_xlabel("$t$")
    ax.legend()

with figure("trapezoidal-monomial-remainder") as fig:
    ax = fig.gca()

    e_ex = np.abs(y_ref[0, mask] - y_ex[0, mask]) / np.abs(y_ref[0, mask])
    e_im = np.abs(y_ref[0, mask] - y_im[0, mask]) / np.abs(y_ref[0, mask])
    c_i = e_im[-1] / rn[-1]
    c_i = 1.0

    ax.plot(t[mask], e_im, "-", label="Implicit")
    ax.semilogy(t[mask], e_ex, "--", label="Explicit")
    ax.semilogy(t[mask], c_i * rn[mask], "k--", label="Remainder")

    ax.set_xlim([tstart, tfinal])
    ax.set_xlabel("$t$")
    ax.legend()

# }}}
