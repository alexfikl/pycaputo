# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from functools import partial

import numpy as np
import numpy.linalg as la

from pycaputo.fode import caputo
from pycaputo.logging import get_logger
from pycaputo.utils import Array, BlockTimer, StateFunctionT, gamma

logger = get_logger("trapezoidal")
rng = np.random.default_rng()


@dataclass(frozen=True)
class Trapezoidal(caputo.Trapezoidal[StateFunctionT]):
    def _get_kwargs(self, *, scalar: bool = True) -> dict[str, object]:
        return {"method": "newton", "xtol": 1.0e-12, "rtol": 1.0e-12}


# {{{ function


def func_y(t: float, *, t0: float, alpha: float, Yv: Array, nu: Array) -> Array:
    result = np.sum(Yv * (t - t0) ** nu)
    return np.array([result])


def func_dy(t: float, *, t0: float, alpha: float, Yv: Array, nu: Array) -> Array:
    mask = nu > 0.0
    gYv = gamma(1 + nu) / gamma(1 + nu - alpha) * Yv
    result = np.sum(gYv[mask] * (t - t0) ** (nu[mask] - alpha))

    return np.array([result])


def func_f(
    t: float,
    y: Array,
    *,
    t0: float,
    alpha: float,
    Yv: Array,
    nu: Array,
    beta: float,
    c: float = 1.0,
) -> Array:
    y_ref = func_y(t, t0=t0, alpha=alpha, Yv=Yv, nu=nu)
    dy_ref = func_dy(t, t0=t0, alpha=alpha, Yv=Yv, nu=nu)

    result = dy_ref + c * (y_ref**beta - y**beta)
    return np.array(result)


def func_f_jac(
    t: float,
    y: Array,
    *,
    t0: float,
    alpha: float,
    Yv: Array,
    nu: Array,
    beta: float,
    c: float = 1.0,
) -> Array:
    if c == 0:
        return np.zeros_like(y)

    result = -c * beta * y ** (beta - 1.0)
    return result


# }}}

# {{{ solution

# fractional order
alpha = 0.8
m = int(np.ceil(alpha))
# right-hand side power
beta = 1.5
# time interval
tstart, tfinal = 0.0, 8.0

# construct an exact solution of the form
#   sum Yv[i] * (t - t_0) ** nu[i]
nns = int(np.ceil(3 / alpha))
nu = np.array([i + alpha * j for i in range(nns) for j in range(nns)])
nu = np.sort(nu[nu <= 2.0 + alpha])
Yv = rng.uniform(0.0, 1.0, size=nu.size)

# construct initial problem
kwargs = {"t0": tstart, "alpha": alpha, "Yv": Yv, "nu": nu}
solution = partial(func_y, **kwargs)
y0 = solution(tstart)

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

im_stepper = Trapezoidal(
    derivative_order=(alpha,),
    control=c,
    source=partial(func_f, **kwargs, beta=beta),
    source_jac=partial(func_f_jac, **kwargs, beta=beta),
    y0=(y0,),
)

# Implicit stepper for `(t - t0)^{nu^star}` where `nu^star = 2 alpha`
nustar = 3
kwargs = {**kwargs, "Yv": np.array([Yv[nustar]]), "nu": np.array([nu[nustar]])}
en_solution = partial(func_y, **kwargs)

en_stepper = Trapezoidal(
    derivative_order=(alpha,),
    control=c,
    source=partial(func_f, **kwargs, beta=beta, c=0.0),
    source_jac=partial(func_f_jac, **kwargs, beta=beta, c=0.0),
    y0=(en_solution(tstart),),
)

from pycaputo.events import StepCompleted
from pycaputo.stepping import evolve

t_l: list[float] = []
y_im_l: list[Array] = []
y_ref_l: list[Array] = []
y_en_l: list[Array] = []
en_l: list[Array] = []
en_ref_l: list[Array] = []
rn_l: list[Array] = []
error_l: list[Array] = []
dtmax = 0.0

with BlockTimer("evolve") as bt:
    dtinit = getattr(c, "dtinit", None)
    for im_event, en_event in zip(
        evolve(im_stepper, dtinit=dtinit),
        evolve(en_stepper, dtinit=dtinit),
    ):
        assert isinstance(im_event, StepCompleted)
        assert isinstance(en_event, StepCompleted)
        assert abs(en_event.t - im_event.t) < 1.0e-14

        if not np.any(np.isfinite(im_event.y)):
            logger.error("%s | Implicit solution diverged: %r", im_event, im_event.y)
            break

        # compute exact solution
        t_n = im_event.t
        y_ref = solution(t_n)

        # compute error model
        dtmax = max(dtmax, im_event.dt)

        En = np.abs(en_event.y - en_solution(en_event.t)).item()
        En_ref = (
            0.0
            if en_event.iteration == 0
            else ((t_n - tstart) ** (alpha - 1.0) * dtmax ** (nu[nustar] - alpha + 1))
        )
        Rn = Yv[nustar] * gamma(1 + nu[nustar]) / gamma(1 + nu[nustar] - alpha) * En

        # compute global error
        from pycaputo.fode.caputo import (
            _weights_quadrature_trapezoidal,  # noqa: PLC2701
        )

        n = im_event.iteration
        if n > 0:
            source_jac = im_stepper.source_jac
            assert source_jac is not None

            # get weights
            omegal, omegar = _weights_quadrature_trapezoidal(
                im_stepper, np.hstack([t_l, [t_n]]), n, n
            )
            # multiply error with jacobian
            error = np.array([
                source_jac(t_i, y_i) * e_i
                for t_i, y_i, e_i in zip(t_l, y_im_l, error_l)
            ]).reshape(-1, 1)
            # add up the entire error
            error = (
                Rn
                + np.einsum("ij,ij->j", omegal, error)
                + np.einsum("ij,ij->j", omegar[:-1], error[1:])
            ) / (1 - omegar[-1] * source_jac(t_n, im_event.y))

            error = np.abs(error)
        else:
            error = np.array([0.0])

        # print iteration
        im_error = la.norm(y_ref - im_event.y, np.inf) / la.norm(y_ref, np.inf)
        logger.info(
            "%s | error im %.8e model %.8e",
            im_event,
            im_error,
            error,
        )

        # append solutions
        t_l.append(t_n)
        y_im_l.append(im_event.y)
        y_en_l.append(en_event.y)
        y_ref_l.append(y_ref)
        rn_l.append(Rn)
        en_l.append(En)
        en_ref_l.append(En_ref)
        error_l.append(error)

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
rn = np.array(rn_l).T
en = np.array(en_l).T
en_ref = np.array(en_ref_l).T
error = np.array(error_l).T
mask = np.s_[1:]

y_en_ref = np.array([en_solution(t_j) for t_j in t])

if isinstance(c, RandomController):
    with figure(f"{basename}-timestep") as fig:
        ax = fig.gca()

        ax.semilogy(np.diff(t))
        ax.axhline(c.dtmin, color="k", linestyle="--")
        ax.axhline(c.dtmax, color="k", linestyle="--")

        ax.set_xlabel("$n$")
        ax.set_ylabel(r"$\Delta t_n$")

with figure(f"{basename}-enstar") as fig:
    ax = fig.gca()

    ax.plot(t[mask], y_en[0, mask], "-", label="Implicit")
    ax.plot(t[mask], y_en_ref[mask], "k--", label="Exact")

    ax.set_xlabel("$t$")
    ax.set_ylabel("$y$")
    ax.legend()

with figure(f"{basename}-enstar-error") as fig:
    ax = fig.gca()

    ax.semilogy(t[mask], en[mask], "-", label="Error")
    ax.semilogy(t[mask], en_ref[mask], "k--", label="Model")
    ax.set_ylim([1.0e-10, 1.0e-1])

    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$E_{n, \nu^\star - \alpha}$")
    ax.legend()


with figure(f"{basename}-solution") as fig:
    ax = fig.gca()

    ax.plot(t[mask], y_im[0, mask], "-", label="Explicit")
    ax.plot(t[mask], y_ref[0, mask], "k--", label="Exact")

    ax.set_xlabel("$t$")
    ax.set_ylabel("$y$")
    ax.legend()

with figure(f"{basename}-remainder") as fig:
    ax = fig.gca()

    e_im = np.abs(y_ref[0, mask] - y_im[0, mask])

    ax.plot(t[mask], e_im, "-", label="Implicit")
    ax.semilogy(t[mask], rn[mask], "k--", label="Remainder")
    ax.semilogy(t[mask], error[0, mask], "r--", label="Global Error")
    ax.set_ylim([1.0e-10, 1.0e-1])

    ax.set_xlim([tstart, tfinal])
    ax.set_xlabel("$t$")
    ax.set_ylabel("Error")
    ax.legend()

# }}}
