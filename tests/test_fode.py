# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib
from functools import partial
from typing import Callable

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.derivatives import CaputoDerivative, Side
from pycaputo.fode import (
    FractionalDifferentialEquationMethod,
    make_predict_time_step_fixed,
)
from pycaputo.logging import get_logger
from pycaputo.utils import Array, set_recommended_matplotlib

logger = get_logger("pycaputo.test_caputo")
set_recommended_matplotlib()


# {{{ test_predict_time_step_graded


def test_predict_time_step_graded() -> None:
    from pycaputo.fode import make_predict_time_step_graded

    maxit = 100
    tspan = (-1.5, 3.0)
    r = 3
    predict_time_step = make_predict_time_step_graded(tspan, maxit, r)

    n = np.arange(maxit)
    t_ref = tspan[0] + (n / maxit) ** r * (tspan[1] - tspan[0])

    t = np.empty_like(t_ref)
    dummy = np.empty(3)

    t[0] = tspan[0]
    for i in range(1, maxit):
        dt = predict_time_step(t[i - 1], dummy)
        t[i] = t[i - 1] + dt

    error = la.norm(t - t_ref) / la.norm(t_ref)
    logger.info("error: %.12e", error)
    assert error < 1.0e-15


# }}}


# {{{ test_caputo_fode


# {{{ solution


def fode_source(t: float, y: Array, *, alpha: float) -> Array:
    from math import gamma

    return np.array(t**2 - y + 2 * t ** (2 - alpha) / gamma(3 - alpha))


def fode_source_jac(t: float, y: Array, *, alpha: float) -> Array:
    return -np.ones_like(y)


# }}}


def fode_solution(t: float) -> Array:
    return np.array([t**2])


def forward_euler_factory(alpha: float, n: int) -> FractionalDifferentialEquationMethod:
    y0 = fode_solution(0.0)
    tspan = (0.0, 1.0)
    dt = (tspan[1] - tspan[0]) / n

    from pycaputo.fode import CaputoForwardEulerMethod

    return CaputoForwardEulerMethod(
        d=CaputoDerivative(order=alpha, side=Side.Left),
        predict_time_step=make_predict_time_step_fixed(dt),
        source=partial(fode_source, alpha=alpha),
        tspan=tspan,
        y0=(y0,),
    )


def backward_euler_factory(
    alpha: float, n: int
) -> FractionalDifferentialEquationMethod:
    y0 = fode_solution(0.0)
    tspan = (0.0, 1.0)
    dt = (tspan[1] - tspan[0]) / n

    from pycaputo.fode import CaputoCrankNicolsonMethod

    return CaputoCrankNicolsonMethod(
        d=CaputoDerivative(order=alpha, side=Side.Left),
        predict_time_step=make_predict_time_step_fixed(dt),
        source=partial(fode_source, alpha=alpha),
        tspan=tspan,
        y0=(y0,),
        # cr
        source_jac=partial(fode_source_jac, alpha=alpha),
        theta=0.0,
    )


def crank_nicolson_factory(
    alpha: float, n: int
) -> FractionalDifferentialEquationMethod:
    y0 = fode_solution(0.0)
    tspan = (0.0, 1.0)
    dt = (tspan[1] - tspan[0]) / n

    from pycaputo.fode import CaputoCrankNicolsonMethod

    return CaputoCrankNicolsonMethod(
        d=CaputoDerivative(order=alpha, side=Side.Left),
        predict_time_step=make_predict_time_step_fixed(dt),
        source=partial(fode_source, alpha=alpha),
        tspan=tspan,
        y0=(y0,),
        # cr
        source_jac=partial(fode_source_jac, alpha=alpha),
        theta=0.5,
    )


def pece_factory(alpha: float, n: int) -> FractionalDifferentialEquationMethod:
    y0 = fode_solution(0.0)
    tspan = (0.0, 1.0)
    dt = (tspan[1] - tspan[0]) / n

    from pycaputo.fode import CaputoPECEMethod

    return CaputoPECEMethod(
        d=CaputoDerivative(order=alpha, side=Side.Left),
        predict_time_step=make_predict_time_step_fixed(dt),
        source=partial(fode_source, alpha=alpha),
        tspan=tspan,
        y0=(y0,),
        # pece
        corrector_iterations=1,
    )


def pec_factory(alpha: float, n: int) -> FractionalDifferentialEquationMethod:
    y0 = fode_solution(0.0)
    tspan = (0.0, 1.0)
    dt = (tspan[1] - tspan[0]) / n

    from pycaputo.fode import CaputoPECMethod

    return CaputoPECMethod(
        d=CaputoDerivative(order=alpha, side=Side.Left),
        predict_time_step=make_predict_time_step_fixed(dt),
        source=partial(fode_source, alpha=alpha),
        tspan=tspan,
        y0=(y0,),
        # pec
        corrector_iterations=1,
    )


@pytest.mark.parametrize(
    "factory",
    [
        forward_euler_factory,
        backward_euler_factory,
        crank_nicolson_factory,
        pece_factory,
        # pec_factory,
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_caputo_fode(
    factory: Callable[[float, int], FractionalDifferentialEquationMethod],
    alpha: float,
    *,
    visualize: bool = True,
) -> None:
    from pycaputo.fode import StepCompleted, StepFailed, evolve
    from pycaputo.utils import EOCRecorder

    eoc = EOCRecorder()

    for n in [32, 64, 128, 256, 512]:
        m = factory(alpha, n)

        ts = []
        ys = []
        for event in evolve(m, verbose=True):
            if isinstance(event, StepFailed):
                raise ValueError("Step update failed")
            elif isinstance(event, StepCompleted):
                ts.append(event.t)
                ys.append(event.y)

        dt = m.predict_time_step(ts[-1], ys[-1])

        y_ref = fode_solution(ts[-1])
        error = la.norm(ys[-1] - y_ref) / la.norm(y_ref)
        logger.info("dt %.5f error %.12e", dt, error)

        eoc.add_data_point(dt, error)

    from dataclasses import replace

    eoc = replace(eoc, order=m.order)
    logger.info("\n%s", eoc)

    if visualize:
        t = np.array(ts)
        y = np.array(ys).squeeze()
        y_ref = np.array([fode_solution(ti) for ti in t]).squeeze()

        from pycaputo.utils import figure

        dirname = pathlib.Path(__file__).parent
        filename = f"test_caputo_fode_{m.name}_{alpha}".replace(".", "_").lower()
        with figure(dirname / filename) as fig:
            ax = fig.gca()

            ax.plot(t, y)
            ax.plot(t, y_ref, "k--")
            ax.set_xlabel("$t$")
            ax.set_ylabel("$y$")

    assert eoc.estimated_order > m.order - 0.25


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
