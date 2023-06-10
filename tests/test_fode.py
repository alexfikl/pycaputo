# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib
from functools import partial
from typing import Callable

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.derivatives import CaputoDerivative, Side
from pycaputo.fode import FractionalDifferentialEquationMethod
from pycaputo.logging import get_logger
from pycaputo.utils import Array, set_recommended_matplotlib

logger = get_logger("pycaputo.test_caputo")
set_recommended_matplotlib()


# {{{ test_caputo_fode


def fode_time_step(t: float, y: Array, *, dt: float) -> float:
    return dt


def fode_source(t: float, y: Array, *, alpha: float) -> Array:
    from math import gamma

    return np.array(t**2 - y + 2 * t ** (2 - alpha) / gamma(3 - alpha))


def fode_source_jac(t: float, y: Array, *, alpha: float) -> Array:
    return -np.ones_like(y)


def fode_solution(t: float) -> Array:
    return np.array([t**2])


def forward_euler_factory(alpha: float, n: int) -> FractionalDifferentialEquationMethod:
    y0 = fode_solution(0.0)
    tspan = (0.0, 1.0)
    dt = (tspan[1] - tspan[0]) / n

    from pycaputo.fode import CaputoForwardEulerMethod

    return CaputoForwardEulerMethod(
        d=CaputoDerivative(order=alpha, side=Side.Left),
        predict_time_step=partial(fode_time_step, dt=dt),
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
        predict_time_step=partial(fode_time_step, dt=dt),
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
        predict_time_step=partial(fode_time_step, dt=dt),
        source=partial(fode_source, alpha=alpha),
        tspan=tspan,
        y0=(y0,),
        # cr
        source_jac=partial(fode_source_jac, alpha=alpha),
        theta=0.5,
    )


@pytest.mark.parametrize(
    "factory",
    [
        forward_euler_factory,
        backward_euler_factory,
        crank_nicolson_factory,
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

    if not visualize:
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
