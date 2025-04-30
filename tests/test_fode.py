# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pycaputo.controller import Controller
from pycaputo.derivatives import CaputoDerivative
from pycaputo.logging import get_logger
from pycaputo.utils import get_environ_bool, set_recommended_matplotlib

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent
ENABLE_VISUAL = get_environ_bool("ENABLE_VISUAL")

log = get_logger(f"pycaputo.{TEST_FILENAME.stem}")
set_recommended_matplotlib()


# {{{ test_history_growth


def test_history_growth() -> None:
    """
    Test that :class:`ProductIntegrationHistory` can grow its storage accordingly.
    """

    from pycaputo.history import ProductIntegrationHistory

    history = ProductIntegrationHistory.empty(n=32, shape=(2,), dtype=np.float64)

    nsizes = 512
    sizes = np.empty(nsizes, dtype=np.int64)
    sizes[0] = 1

    for i in range(nsizes - 1):
        if i == history.capacity - 1:
            history.resize(i + 1)

        sizes[i + 1] = history.capacity
        assert history.capacity > i

    if not ENABLE_VISUAL:
        return

    from pycaputo.utils import figure

    with figure(TEST_DIRECTORY / "test_history_growth", normalize=True) as fig:
        ax = fig.gca()

        ax.plot(sizes[2:] / sizes[1:-1], "o-")
        ax.axhline(1.15, ls="--", color="k")
        ax.set_xlabel("$i$")
        ax.set_ylabel("Capacity")


# }}}

# {{{ test_history_append


def test_history_append() -> None:
    """
    Test that :class:`ProductIntegrationHistory` can grow and append items.
    """

    from pycaputo.history import ProductIntegrationHistory

    n = 32
    history = ProductIntegrationHistory.empty(n=16, shape=(2,), dtype=np.float64)
    assert not history

    dt = 0.1
    rng = np.random.default_rng(seed=42)

    for i in range(n):
        t = i * dt
        y = rng.random(size=2)
        history.append(t, y)

        assert history
        assert len(history) == i + 1

    for i in range(5):
        r = history[i]
        assert r.t == i * dt

    with pytest.raises(IndexError):
        r = history[n + 1]

    history.clear()
    assert not history
    assert len(history) == 0


# }}}


# {{{ test_fixed_controller


def _check_fixed_controller_evolution(
    c: Controller,
    dtinit: float,
    *,
    is_increasing: bool = True,
    rtol: float = 3.0e-12,
) -> None:
    from pycaputo.fode import caputo

    assert c.tfinal is not None
    assert c.nsteps is not None

    ncomponents = 7
    rng = np.random.default_rng(seed=42)
    y = rng.normal(size=ncomponents)
    yprev = rng.normal(size=ncomponents)

    m = caputo.ForwardEuler(
        ds=(CaputoDerivative(0.8),) * y.size,
        control=c,
        source=lambda t, y: np.zeros_like(y),
        y0=(yprev,),
    )

    from pycaputo.events import StepCompleted
    from pycaputo.stepping import evolve

    n = 0
    t = c.tstart
    dt = dtinit
    for event in evolve(m, dtinit=dtinit):
        assert isinstance(event, StepCompleted)

        t = event.t
        n = event.iteration
        dtnext = event.dt

        if is_increasing:
            # fmt: off
            assert dtnext >= min(dt, c.tfinal - event.t), (
                f"[{n:04d}] dt = {dt:.8e} dtnext {dtnext:.8e} error {dt - dtnext:.8e}"
                )
            # fmt: on

        dt = dtnext

    # fmt: off
    assert n == c.nsteps, c.nsteps - n
    assert abs(t - c.tfinal) < rtol * dt, (
        f"t = {t:.8e} tfinal {c.tfinal:.8e} error {t - c.tfinal:.8e}"
        )
    # fmt: on


def test_fixed_controller() -> None:
    """
    Test that the :class:`FixedController` computes the correct timestep.
    """

    from pycaputo.controller import make_fixed_controller

    dt = 1.0e-2
    with pytest.raises(ValueError, match="Must provide"):
        make_fixed_controller(dt)

    c = make_fixed_controller(dt, tstart=0.0, tfinal=1.0, nsteps=27)
    assert c.tfinal is not None
    assert c.nsteps is not None
    assert abs(c.tfinal - c.tstart - c.dt * c.nsteps) < 1.0e-15

    _check_fixed_controller_evolution(c, c.dtinit)


# }}}


# {{{ test_graded_controller


def test_graded_controller() -> None:
    """
    Test that the :class:`GradedController` computes the correct time steps.
    """

    from pycaputo.controller import evaluate_timestep_accept, make_graded_controller

    with pytest.raises(ValueError, match="Must provide"):
        make_graded_controller()

    with pytest.raises(ValueError, match="Must provide"):
        make_graded_controller(tfinal=1.0)

    with pytest.raises(ValueError, match="Grading estimate"):
        make_graded_controller(tfinal=1.0, nsteps=27, alpha=2.0)

    alpha = 0.75
    r = (2.0 - alpha) / alpha
    c = make_graded_controller(tfinal=1.0, nsteps=27, r=r)
    assert c.tfinal is not None
    assert c.nsteps is not None

    from pycaputo.fode import caputo

    y0 = np.zeros(7)
    m = caputo.ForwardEuler(
        ds=(CaputoDerivative(0.8),) * y0.size,
        control=c,
        source=lambda t, y: np.zeros_like(y0),
        y0=(y0,),
    )

    q = 1.0
    xi = np.arange(c.nsteps + 1) / c.nsteps
    t_ref = c.tstart + (c.tfinal - c.tstart) * xi**r

    t = dt = c.dtinit
    for i in range(1, c.nsteps):
        dt = evaluate_timestep_accept(c, m, q, dt, {"t": t, "n": i - 1})

        assert abs(t - t_ref[i]) < 3.0e-14, (i, abs(t - t_ref[i]))
        t += dt


# }}}


# {{{ test_given_controller


@pytest.mark.parametrize("name", ["random", "graded", "stretch"])
def test_given_controller(name: str) -> None:
    from pycaputo.controller import (
        GivenStepController,
        make_graded_controller,
        make_random_controller,
    )

    rng = np.random.default_rng(seed=42)
    tstart, tfinal = 0.0, 1.0
    nsteps = 31

    c: GivenStepController
    if name == "random":
        rtol = 1.0e-10
        c = make_random_controller(tstart, tfinal, rng=rng)
    elif name == "graded":
        rtol = 5.0e-12
        c = make_graded_controller(tstart, tfinal, nsteps=nsteps, alpha=0.75)
    elif name == "stretch":
        from pycaputo.grid import make_stretched_points

        rtol = 5.0e-13
        p = make_stretched_points(nsteps + 1, tstart, tfinal)
        c = GivenStepController(
            tstart=tstart, tfinal=tfinal, nsteps=nsteps, timesteps=p.dx
        )
    else:
        raise ValueError(f"Unknown controller name: {name!r}")

    _check_fixed_controller_evolution(c, c.dtinit, rtol=rtol, is_increasing=False)


# }}}


# {{{ test_fixed_controller_doubling


def test_fixed_controller_doubling() -> None:
    """
    Test that the :class:`FixedController` doubles times steps consistently.
    """

    from pycaputo.controller import make_fixed_controller

    dt = 1.0e-2
    with pytest.raises(ValueError, match="Must provide"):
        make_fixed_controller(dt)

    c0 = make_fixed_controller(dt, tstart=0.0, tfinal=1.0)
    assert c0.tfinal is not None
    assert c0.nsteps is not None
    assert abs(c0.tfinal - c0.tstart - c0.dt * c0.nsteps) < 1.0e-15

    _check_fixed_controller_evolution(c0, c0.dt, rtol=5.0e-11)

    c1 = make_fixed_controller(dt / 2.0, tstart=0.0, tfinal=1.0)
    assert c1.tfinal is not None
    assert c1.nsteps is not None
    assert abs(c1.tfinal - c1.tstart - c1.dt * c1.nsteps) < 1.0e-15

    _check_fixed_controller_evolution(c1, c1.dt, rtol=5.0e-11)

    assert c1.nsteps == 2 * c0.nsteps


# }}}

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
