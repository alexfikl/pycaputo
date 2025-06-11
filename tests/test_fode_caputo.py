# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.derivatives import CaputoDerivative, FractionalOperatorT
from pycaputo.fode import caputo
from pycaputo.logging import get_logger
from pycaputo.stepping import FractionalDifferentialEquationMethod, evolve
from pycaputo.typing import Array, StateFunction
from pycaputo.utils import get_environ_bool, set_recommended_matplotlib

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent
ENABLE_VISUAL = get_environ_bool("ENABLE_VISUAL")

log = get_logger(f"pycaputo.{TEST_FILENAME.stem}")
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


# {{{ test_fode_caputo


def fode_factory(
    cls: type[FractionalDifferentialEquationMethod[FractionalOperatorT, StateFunction]],
    *,
    wrap: bool = True,
    graded: bool = False,
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
        return r if r.size == 1 else r.squeeze()

    def source_jac(t: float, y: Array, *, alpha: tuple[float, ...]) -> Array:
        r = np.stack([
            garrappa2009_source_jac(t, y[i], alpha=alpha[i]) for i in range(nterms)
        ])
        return r if r.size == 1 else r.squeeze()

    def wrapper(
        alpha: float | tuple[float, ...], n: int
    ) -> FractionalDifferentialEquationMethod[FractionalOperatorT, StateFunction]:
        if not isinstance(alpha, tuple):
            alpha = (alpha,)

        assert len(alpha) == nterms
        dt = (tspan[1] - tspan[0]) / n

        if has_source_jac:
            kwargs["source_jac"] = partial(source_jac, alpha=alpha)

        from pycaputo.controller import (
            Controller,
            make_fixed_controller,
            make_graded_controller,
        )

        control: Controller
        if graded:
            control = make_graded_controller(
                tspan[0], tspan[1], dt=dt, alpha=max(alpha)
            )
        else:
            control = make_fixed_controller(dt, tstart=tspan[0], tfinal=tspan[1])

        return cls(
            ds=tuple(CaputoDerivative(alpha_i) for alpha_i in alpha),  # type: ignore[misc]
            control=control,
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
        fode_factory(caputo.BackwardEuler),
        fode_factory(caputo.WeightedEuler, theta=0.0),
        fode_factory(caputo.WeightedEuler, theta=0.5),
        fode_factory(caputo.Trapezoidal),
        fode_factory(caputo.ExplicitTrapezoidal),
        fode_factory(caputo.PECE, corrector_iterations=1),
        # FIXME: this does not converge to the correct order with one iteration
        fode_factory(caputo.PEC, corrector_iterations=2),
        fode_factory(caputo.ModifiedPECE, corrector_iterations=1),
        fode_factory(caputo.L1),
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_fode_caputo(
    factory: Callable[
        [float, int],
        FractionalDifferentialEquationMethod[FractionalOperatorT, StateFunction],
    ],
    alpha: float,
) -> None:
    r"""
    Test convergence of the methods for single-order Caputo FODEs.

    The convergence is checked in the :math:`\ell^2` norm for
    :func:`garrappa2009_solution`.
    """

    from pycaputo.events import StepCompleted, StepFailed
    from pycaputo.utils import BlockTimer, EOCRecorder

    eoc = EOCRecorder()
    if not callable(factory):
        # NOTE: this is a pytest.param and we take out the callable
        factory = factory.values[0]

    for n in [32, 64, 128, 256, 512]:
        m = factory(alpha, n)
        dtinit = getattr(m.control, "dt", None)

        with BlockTimer(name=m.name) as bt:
            ts = []
            ys = []
            for event in evolve(m, dtinit=dtinit):
                if isinstance(event, StepFailed):
                    raise ValueError("Step update failed")
                elif isinstance(event, StepCompleted):
                    ts.append(event.t)
                    ys.append(event.y)

        dt = np.max(np.diff(np.array(ts)))

        y_ref = garrappa2009_solution(ts[-1])
        error = la.norm(ys[-1] - y_ref) / la.norm(y_ref)
        log.info("dt %.5f error %.12e (%s)", dt, error, bt)

        eoc.add_data_point(dt, error)

    eoc = replace(eoc, order=m.order)
    log.info("\n%s", eoc)

    if ENABLE_VISUAL:
        t = np.array(ts)
        y = np.array(ys).squeeze()
        y_ref = np.array([garrappa2009_solution(ti) for ti in t]).squeeze()

        from pycaputo.utils import figure

        filename = f"test_fode_caputo_{m.name}_{alpha}"
        with figure(TEST_DIRECTORY / filename, normalize=True) as fig:
            ax = fig.gca()

            ax.plot(t, y, label=f"{m.name}")
            ax.plot(t, y_ref, "k--", label="Reference")
            ax.set_xlabel("$t$")
            ax.set_ylabel("$y$")
            ax.legend()

    if alpha < 0.5 and isinstance(m, caputo.ExplicitTrapezoidal):
        # FIXME: this seems unstable for some reason
        return

    assert eoc.estimated_order > m.order - 0.25


# }}}


# {{{ test_fode_caputo_system


@pytest.mark.parametrize(
    "factory",
    [
        fode_factory(caputo.ForwardEuler, nterms=3),
        fode_factory(caputo.WeightedEuler, theta=0.5, nterms=3),
        fode_factory(caputo.Trapezoidal, nterms=3),
        fode_factory(caputo.ExplicitTrapezoidal, nterms=3),
        fode_factory(caputo.PECE, corrector_iterations=1, nterms=3),
        fode_factory(caputo.ModifiedPECE, corrector_iterations=1, nterms=3),
    ],
)
def test_fode_caputo_system(
    factory: Callable[
        [tuple[float, ...], int],
        FractionalDifferentialEquationMethod[FractionalOperatorT, StateFunction],
    ],
) -> None:
    """
    Test convergence of the methods for single-order Caputo systems FODEs.

    This checks a simple system of 3 equations with the same right-hand side,
    but different fractional order for each equation.
    """

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
        log.info("dt %.5f error %.12e (%s)", dt, error, bt)

        eoc.add_data_point(dt, error)

    eoc = replace(eoc, order=m.order)
    log.info("\n%s", eoc)

    assert eoc.estimated_order > m.order - 0.25


# }}}


# {{{ test_caputo_l1_convergence


def singular_solution(t: float, *, alpha: float) -> Array:
    return np.array([t**alpha])


def singular_source(t: float, y: Array, *, alpha: float) -> Array:
    from math import gamma

    return np.array([gamma(1 + alpha)])


@pytest.mark.parametrize("mesh_type", ["uniform", "graded"])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_singular_caputo_l1(mesh_type: str, alpha: float) -> None:
    """
    Test convergence of the L1 method when the solution is singular.

    The L1 method should degrade from order :math:`2 - \alpha` to order :math:`1`
    on a uniform mesh. The optimal convergence should be recovered on a graded
    mesh.
    """

    from pycaputo.events import StepCompleted, StepFailed
    from pycaputo.utils import BlockTimer, EOCRecorder

    if mesh_type == "uniform":
        order = 1.0
    elif mesh_type == "graded":
        order = 2.0 - alpha
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")

    eoc = EOCRecorder(order=order)
    graded = mesh_type == "graded"

    for n in [32, 64, 128, 256, 512]:
        m = fode_factory(caputo.L1, wrap=False, graded=graded)(alpha, n)
        m = replace(m, source=partial(singular_source, alpha=alpha))

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

        y_ref = singular_solution(ts[-1], alpha=alpha)
        error = la.norm(ys[-1] - y_ref) / la.norm(y_ref)
        log.info("dt %.5f error %.12e (%s)", dt, error, bt)

        eoc.add_data_point(dt, error)

    log.info("\n%s", eoc)

    if ENABLE_VISUAL:
        t = np.array(ts)
        y = np.array(ys).squeeze()
        y_ref = np.array([singular_solution(ti, alpha=alpha) for ti in t]).squeeze()

        from pycaputo.utils import figure

        filename = f"test_fode_caputo_{m.name}_{alpha}_sing"
        with figure(TEST_DIRECTORY / filename, normalize=True) as fig:
            ax = fig.gca()

            ax.plot(t, y, label=f"{m.name}")
            ax.plot(t, y_ref, "k--", label="Reference")
            ax.set_xlabel("$t$")
            ax.set_ylabel("$y$")
            ax.legend()

    assert order - 0.2 < eoc.estimated_order < order + 0.2


# }}}


# {{{ test_variable_order_caputo


@pytest.mark.parametrize(
    ("alpha", "c", "omega"),
    [
        ((0.6, 0.8), 2.0, 1.0),
        ((0.5, 0.9), 1.0, 2.0),
        ((0.9, 0.6), 1.0, 0.5),
    ],
)
def test_variable_order_caputo(
    alpha: tuple[float, float], c: float, omega: float
) -> None:
    """Test convergence of the backward Euler method for variable-order Caputo.

    The tests try to reproduce Table 1 from [Garrappa2023]_.
    """

    from pycaputo.events import StepCompleted
    from pycaputo.utils import BlockTimer, EOCRecorder

    order = 1.0
    eoc = EOCRecorder(order=order)

    from pycaputo.controller import make_fixed_controller
    from pycaputo.derivatives import VariableExponentialCaputoDerivative as D
    from pycaputo.fode import variable_caputo as vo

    func = vo.Relaxation(D(alpha=alpha, c=c), y0=1.0, omega=omega)
    tstart, tfinal = 0.0, 4.0

    for h in 2.0 ** (-np.arange(2, 8)):
        control = make_fixed_controller(float(h), tstart=tstart, tfinal=tfinal)
        m = vo.VariableExponentialBackwardEuler(
            ds=(func.d,),
            source=func.source,
            source_jac=func.source_jac,
            control=control,
            y0=(func(control.tstart),),
        )

        with BlockTimer(name=m.name) as bt:
            ts = []
            ys = []
            for event in evolve(m):
                assert isinstance(event, StepCompleted)
                ts.append(event.t)
                ys.append(event.y)

        # FIXME: we are not matching the first-order from [Garrappa2023] and
        # it's not clear why not. `ys[-1]` matches the output of the MATLAB code
        # pretty much exactly, so only y_ref can be wrong here..
        y_ref = func(tfinal - float(h))
        error = la.norm(ys[-1] - y_ref)
        log.info(
            "dt %.5f y %.12e y_ref %.12e error %.12e (%s)",
            h,
            ys[-1].item(),
            la.norm(y_ref),
            error,
            bt,
        )

        eoc.add_data_point(h, error)

    log.info("\n%s", eoc)

    if ENABLE_VISUAL:
        t = np.array(ts)
        y = np.array(ys).squeeze()
        y_ref = np.array([func(ti) for ti in t]).squeeze()

        from pycaputo.utils import figure

        filename = f"test_fode_variable_caputo_{alpha[0]}_{alpha[1]}"
        with figure(TEST_DIRECTORY / filename, normalize=True) as fig:
            ax = fig.gca()

            ax.plot(t, y, label=f"{m.name}")
            ax.plot(t, y_ref, "k--", label="Reference")
            ax.set_xlabel("$t$")
            ax.set_ylabel("$y$")
            ax.legend()

    assert order - 0.2 < eoc.estimated_order < order + 0.5


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
