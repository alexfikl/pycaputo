# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from functools import partial
from typing import Any, Callable

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.fode import caputo
from pycaputo.logging import get_logger
from pycaputo.stepping import FractionalDifferentialEquationMethod, evolve
from pycaputo.utils import Array, StateFunction, set_recommended_matplotlib

logger = get_logger("pycaputo.test_fode_caputo")
set_recommended_matplotlib()

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

    return np.array([-(y[0] ** 2) + f], dtype=y.dtype)


def li2015_source_jac(t: float, y: Array, *, alpha: float) -> Array:
    return np.array([[-2 * y[0]]])


# }}}


# {{{ solution: Equation 27 from [Garrappa2009]


def garrappa2009_solution(t: float) -> Array:
    return np.array([t**2])


def garrappa2009_source(t: float, y: Array, *, alpha: float) -> Array:
    from math import gamma

    return np.array(t**2 - y + 2 * t ** (2 - alpha) / gamma(3 - alpha))


def garrappa2009_source_jac(t: float, y: Array, *, alpha: float) -> Array:
    return -np.ones_like(y)


# }}}


# {{{ test_caputo_fode


def fode_factory(
    cls: type[FractionalDifferentialEquationMethod[StateFunction]],
    *,
    wrap: bool = True,
    **kwargs: Any,
) -> Any:
    nterms = kwargs.pop("nterms", 1)
    y0 = np.concatenate([garrappa2009_solution(0.0) for _ in range(nterms)])
    tspan = (0.0, 1.0)

    from dataclasses import fields

    has_source_jac = "source_jac" in {f.name for f in fields(cls)}

    def source(t: float, y: Array, *, alpha: tuple[float, ...]) -> Array:
        r = np.stack([
            garrappa2009_source(t, y[i], alpha=alpha[i]) for i in range(nterms)
        ])
        return r.squeeze()

    def source_jac(t: float, y: Array, *, alpha: tuple[float, ...]) -> Array:
        r = np.stack([
            garrappa2009_source_jac(t, y[i], alpha=alpha[i]) for i in range(nterms)
        ])
        return r.squeeze()

    def wrapper(
        alpha: float | tuple[float, ...], n: int
    ) -> FractionalDifferentialEquationMethod[StateFunction]:
        if not isinstance(alpha, tuple):
            alpha = (alpha,)

        assert len(alpha) == nterms
        dt = (tspan[1] - tspan[0]) / n

        if has_source_jac:
            kwargs["source_jac"] = partial(source_jac, alpha=alpha)

        from pycaputo.controller import make_fixed_controller

        return cls(
            derivative_order=alpha,
            control=make_fixed_controller(dt, tstart=tspan[0], tfinal=tspan[1]),
            source=partial(source, alpha=alpha),
            y0=(y0,),
            **kwargs,
        )

    if wrap:
        return pytest.param(wrapper, id=cls.__name__)
    else:
        return wrapper


@pytest.mark.parametrize(
    "factory",
    [
        fode_factory(caputo.ForwardEuler),
        fode_factory(caputo.WeightedEuler, theta=0.0),
        fode_factory(caputo.WeightedEuler, theta=0.5),
        fode_factory(caputo.PECE, corrector_iterations=1),
        # FIXME: this does not converge to the correct order with one iteration
        fode_factory(caputo.PEC, corrector_iterations=2),
        fode_factory(caputo.ModifiedPECE, corrector_iterations=1),
        fode_factory(caputo.L1),
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_caputo_fode(
    factory: Callable[
        [float, int], FractionalDifferentialEquationMethod[StateFunction]
    ],
    alpha: float,
    *,
    visualize: bool = False,
) -> None:
    from pycaputo.events import StepCompleted, StepFailed
    from pycaputo.utils import BlockTimer, EOCRecorder

    eoc = EOCRecorder()
    if not callable(factory):
        # NOTE: this is a pytest.param and we take out the callable
        factory = factory.values[0]

    for n in [32, 64, 128, 256, 512]:
        m = factory(alpha, n)

        with BlockTimer(name=m.name) as bt:
            ts = []
            ys = []
            for event in evolve(m):
                if isinstance(event, StepFailed):
                    raise ValueError("Step update failed")
                elif isinstance(event, StepCompleted):
                    ts.append(event.t)
                    ys.append(event.y)

        dt = np.max(np.diff(np.array(ts)))

        y_ref = garrappa2009_solution(ts[-1])
        error = la.norm(ys[-1] - y_ref) / la.norm(y_ref)
        logger.info("dt %.5f error %.12e (%s)", dt, error, bt)

        eoc.add_data_point(dt, error)

    from dataclasses import replace

    eoc = replace(eoc, order=m.order)
    logger.info("\n%s", eoc)

    if visualize:
        t = np.array(ts)
        y = np.array(ys).squeeze()
        y_ref = np.array([garrappa2009_solution(ti) for ti in t]).squeeze()

        from pycaputo.utils import figure

        dirname = pathlib.Path(__file__).parent
        filename = f"test_caputo_fode_{m.name}_{alpha}".replace(".", "_").lower()
        with figure(dirname / filename) as fig:
            ax = fig.gca()

            ax.plot(t, y, label=f"{m.name}")
            ax.plot(t, y_ref, "k--", label="Reference")
            ax.set_xlabel("$t$")
            ax.set_ylabel("$y$")
            ax.legend()

    assert eoc.estimated_order > m.order - 0.25


# }}}


# {{{ test_caputo_fode_system


@pytest.mark.parametrize(
    "factory",
    [
        fode_factory(caputo.ForwardEuler, nterms=3),
        fode_factory(caputo.WeightedEuler, theta=0.5, nterms=3),
    ],
)
def test_caputo_fode_system(
    factory: Callable[
        [tuple[float, ...], int], FractionalDifferentialEquationMethod[StateFunction]
    ],
    *,
    visualize: bool = False,
) -> None:
    from pycaputo.events import StepCompleted, StepFailed
    from pycaputo.utils import BlockTimer, EOCRecorder

    eoc = EOCRecorder()
    if not callable(factory):
        # NOTE: this is a pytest.param and we take out the callable
        factory = factory.values[0]

    alpha = (0.8, 0.7, 0.9)
    for n in [32, 64, 128, 256, 512]:
        m = factory(alpha, n)

        with BlockTimer(name=m.name) as bt:
            ts = []
            ys = []
            for event in evolve(m):
                if isinstance(event, StepFailed):
                    raise ValueError("Step update failed")
                elif isinstance(event, StepCompleted):
                    ts.append(event.t)
                    ys.append(event.y)

        dt = np.max(np.diff(np.array(ts)))

        y_ref = garrappa2009_solution(ts[-1])
        error = la.norm(ys[-1] - y_ref) / la.norm(y_ref)
        logger.info("dt %.5f error %.12e (%s)", dt, error, bt)

        eoc.add_data_point(dt, error)

    from dataclasses import replace

    eoc = replace(eoc, order=m.order)
    logger.info("\n%s", eoc)

    assert eoc.estimated_order > m.order - 0.25


# }}}

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
