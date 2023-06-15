# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from functools import partial
from typing import Any, Callable

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo import fode
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


# {{{ solution: Section 3.3.1, Example 2 [Li2015]


def li2015_solution(t: float) -> Array:
    return np.array([t**5 - 3 * t**4 + 2 * t**3])


def li2015_source(t: float, y: Array, *, alpha: float) -> Array:
    from math import gamma

    f = (
        120 / gamma(6 - alpha) * t ** (5 - alpha)
        - 72 / gamma(5 - alpha) * t ** (4 - alpha)
        + 12 / gamma(4 - alpha) * t ** (3 - alpha)
        + (t**5 - 3 * t**4 + 2 * t**3) ** 2
    )

    return np.array([-y[0] ** 2 + f], dtype=y.dtype)


def li2015_source_jac(t: float, y: Array, *, alpha: float) -> Array:
    return np.array([[-2 * y[0]]])


# }}}


# {{{ solution: Equation 27 from [Garrappa2009]


def garrapa2009_solution(t: float) -> Array:
    return np.array([t**2])


def garrappa2009_source(t: float, y: Array, *, alpha: float) -> Array:
    from math import gamma

    return np.array(t**2 - y + 2 * t ** (2 - alpha) / gamma(3 - alpha))


def garrappa2009_source_jac(t: float, y: Array, *, alpha: float) -> Array:
    return -np.ones_like(y)


# }}}


def fode_factory(
    cls: type[fode.FractionalDifferentialEquationMethod], **kwargs: Any
) -> Any:
    y0 = garrapa2009_solution(0.0)
    tspan = (0.0, 1.0)

    def wrapper(alpha: float, n: int) -> fode.FractionalDifferentialEquationMethod:
        dt = (tspan[1] - tspan[0]) / n

        if "source_jac" in kwargs:
            kwargs["source_jac"] = partial(kwargs["source_jac"], alpha=alpha)

        return cls(
            derivative_order=alpha,
            predict_time_step=fode.make_predict_time_step_fixed(dt),
            source=partial(garrappa2009_source, alpha=alpha),
            tspan=tspan,
            y0=(y0,),
            **kwargs,
        )

    return pytest.param(wrapper, id=cls.__name__)


@pytest.mark.parametrize(
    "factory",
    [
        fode_factory(fode.CaputoForwardEulerMethod),
        fode_factory(
            fode.CaputoCrankNicolsonMethod,
            theta=0.0,
            source_jac=garrappa2009_source_jac,
        ),
        fode_factory(
            fode.CaputoCrankNicolsonMethod,
            theta=0.5,
            source_jac=garrappa2009_source_jac,
        ),
        fode_factory(fode.CaputoPECEMethod, corrector_iterations=1),
        # FIXME: this does not converge to the correct order with one iteration
        fode_factory(fode.CaputoPECMethod, corrector_iterations=2),
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_caputo_fode(
    factory: Callable[[float, int], fode.FractionalDifferentialEquationMethod],
    alpha: float,
    *,
    visualize: bool = False,
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

        dt = np.max(np.diff(np.array(ts)))

        y_ref = garrapa2009_solution(ts[-1])
        error = la.norm(ys[-1] - y_ref) / la.norm(y_ref)
        logger.info("dt %.5f error %.12e", dt, error)

        eoc.add_data_point(dt, error)

    from dataclasses import replace

    eoc = replace(eoc, order=m.order)
    logger.info("\n%s", eoc)

    if visualize:
        t = np.array(ts)
        y = np.array(ys).squeeze()
        y_ref = np.array([garrapa2009_solution(ti) for ti in t]).squeeze()

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
