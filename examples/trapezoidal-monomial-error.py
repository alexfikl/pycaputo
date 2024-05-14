# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from functools import partial

import numpy as np
import numpy.linalg as la

from pycaputo.logging import get_logger
from pycaputo.utils import Array, gamma

logger = get_logger("trapezoidal")

# {{{ function


def func_y(t: float, *, t0: float, alpha: float, Yv: Array, nu: Array) -> Array:
    return np.array(Yv * (t - t0) ** nu)


def func_f(
    t: float, y: Array, *, t0: float, alpha: float, Yv: Array, nu: Array, beta: float
) -> Array:
    y_ref = func_y(t, t0=t0, alpha=alpha, Yv=Yv, nu=nu)
    gYv = gamma(1 + nu) / gamma(1 + nu - alpha) * Yv
    return gYv * (t - t0) ** (nu - alpha) + (y**beta - y_ref**beta)


def func_f_jac(
    t: float, y: Array, *, t0: float, alpha: float, Yv: Array, nu: Array, beta: float
) -> Array:
    result = beta * y ** (beta - 1.0)
    return result if y.size == 1 else np.diag(result)


# }}}

# {{{ evolve

from pycaputo.controller import make_fixed_controller
from pycaputo.fode import caputo

alpha = 0.9
beta = np.e - 1.0
tstart, tfinal = 1.0, 1.0 + np.pi / 2

# construct an exact solution
nys = 3
Yv = 4.0 * np.exp(-np.linspace(0.1, 5, nys))
nns = nys + 5
nu = np.setdiff1d(
    np.sort([i + alpha * j for i in range(nns) for j in range(nns)]),
    [*range(nns), alpha, 1 + alpha, 2 + alpha, 3 + alpha],
)[:nys]

solution = partial(func_y, t0=tstart, alpha=alpha, Yv=Yv, nu=nu)
y0 = solution(tstart)

c = make_fixed_controller(5.0e-3, tstart=tstart, tfinal=tfinal)
ex_stepper = caputo.ExplicitTrapezoidal(
    derivative_order=(alpha,) * nys,
    control=c,
    source=partial(func_f, t0=tstart, alpha=alpha, Yv=Yv, nu=nu, beta=beta),
    y0=(y0,),
)

im_stepper = caputo.Trapezoidal(
    derivative_order=(alpha,) * nys,
    control=c,
    source=partial(func_f, t0=tstart, alpha=alpha, Yv=Yv, nu=nu, beta=beta),
    source_jac=partial(func_f_jac, t0=tstart, alpha=alpha, Yv=Yv, nu=nu, beta=beta),
    y0=(y0,),
)

from pycaputo.events import StepCompleted
from pycaputo.stepping import evolve

t_l = []
y_ex_l = []
y_im_l = []
y_ref_l = []
rn_l = []

for ex_event, im_event in zip(
    evolve(ex_stepper, dtinit=c.dt), evolve(im_stepper, dtinit=c.dt)
):
    assert isinstance(im_event, StepCompleted)
    assert isinstance(ex_event, StepCompleted)
    assert abs(ex_event.t - im_event.t) < 1.0e-14

    t_n = ex_event.t
    y_ref = solution(t_n)
    if ex_event.iteration == 0:
        En = np.zeros_like(Yv)
    else:
        En = np.where(
            nu - alpha <= 1.0,
            (t_n - tstart) ** (alpha - 1.0) * c.dt ** (1 + nu - alpha),
            (t_n - tstart) ** (nu - 2.0) * c.dt**2,
        )
    Rn = Yv * gamma(1 + nu) / gamma(1 + nu - alpha) * En

    t_l.append(t_n)
    y_ex_l.append(ex_event.y)
    y_im_l.append(im_event.y)
    y_ref_l.append(y_ref)
    rn_l.append(Rn)

    ex_error = la.norm(y_ref - ex_event.y) / la.norm(y_ref)
    im_error = la.norm(y_ref - im_event.y) / la.norm(y_ref)
    logger.info("%s error ex %.8e im %.8e", im_event, ex_error, im_error)

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

with figure("trapezoidal-monomial-solution") as fig:
    ax = fig.gca()

    for i in range(nys):
        (line,) = ax.plot(
            t[mask], y_im[i, mask], "-", label="Explicit" if i == 0 else None
        )
        ax.plot(
            t[mask],
            y_ex[i, mask],
            "--",
            label="Implicit" if i == 0 else None,
            color=line.get_color(),
        )
        ax.plot(t[mask], y_ref[i, mask], "k--", label="Exact" if i == 0 else None)

    ax.set_xlabel("$t$")
    ax.legend()

with figure("trapezoidal-monomial-remainder") as fig:
    ax = fig.gca()

    for idx, i in enumerate(range(nys)):
        # for idx, i in enumerate([0]):
        e_ex = np.abs(y_ref[i, mask] - y_ex[i, mask])
        e_im = np.abs(y_ref[i, mask] - y_im[i, mask])
        c_i = e_im[-1] / rn[i, -1]

        (line,) = ax.semilogy(t[mask], e_im, "-", label="Error" if idx == 0 else None)
        ax.semilogy(
            t[mask],
            e_ex,
            "--",
            label="Error" if idx == 0 else None,
            color=line.get_color(),
        )
        ax.semilogy(
            t[mask], c_i * rn[i, mask], "k--", label="Remainder" if idx == 0 else None
        )

    ax.set_xlim([tstart, tfinal])
    ax.set_ylim([1.0e-9, 1.0e-1])
    ax.plot("$t$")
    ax.legend()

# }}}
