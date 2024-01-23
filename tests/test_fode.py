# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pycaputo.controller import Controller
from pycaputo.logging import get_logger
from pycaputo.utils import set_recommended_matplotlib

logger = get_logger("pycaputo.test_misc")
dirname = pathlib.Path(__file__).parent
set_recommended_matplotlib()


# {{{ test_history_growth


def test_history_growth(*, visualize: bool = False) -> None:
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

    if not visualize:
        return

    from pycaputo.utils import figure

    with figure(dirname / "test_history_growth") as fig:
        ax = fig.gca()

        ax.plot(sizes[2:] / sizes[1:-1], "o-")
        ax.axhline(1.15, ls="--", color="k")
        ax.set_xlabel("$i$")
        ax.set_ylabel("Capacity")


# }}}

# {{{ test_history_append


def test_history_append() -> None:
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


def _check_fixed_controller_evolution(c: Controller, dt: float) -> None:
    from pycaputo.controller import (
        evaluate_error_estimate,
        evaluate_timestep_accept,
        evaluate_timestep_factor,
    )
    from pycaputo.fode.caputo import ForwardEuler

    assert c.tfinal is not None
    assert c.nsteps is not None

    rng = np.random.default_rng(seed=42)
    y = rng.normal(size=7)
    yprev = rng.normal(size=7)
    trunc = rng.normal(size=7)

    m = ForwardEuler(
        derivative_order=(0.8,) * y.size,
        control=c,
        source=lambda t, y: np.zeros_like(y),
        y0=(yprev,),
    )

    n, t = 0, c.tstart
    while not c.finished(n, t):
        eest = evaluate_error_estimate(c, m, trunc, y, yprev)
        q = evaluate_timestep_factor(c, m, eest)

        tmp_state = {"t": t, "n": n, "y": yprev}
        dtnext = evaluate_timestep_accept(c, m, q, dt, tmp_state)

        # fmt: off
        assert dtnext >= min(dt, c.tfinal - t), (
            f"[{n:04d}] dt = {dt:.8e} dtnext {dtnext:.8e} error {dt - dtnext:.8e}"
            )
        # fmt: on

        t += dt
        n += 1
        dt = dtnext

    # fmt: off
    assert n == c.nsteps, c.nsteps - n
    assert abs(t - c.tfinal) < 3.0e-12 * dt, (
        f"t = {t:.8e} tfinal {c.tfinal:.8e} error {t - c.tfinal:.8e}"
        )
    # fmt: on


def test_fixed_controller() -> None:
    from pycaputo.controller import make_fixed_controller

    dt = 1.0e-2
    with pytest.raises(ValueError, match="Must provide"):
        make_fixed_controller(dt)

    c = make_fixed_controller(dt, tstart=0.0, tfinal=1.0, nsteps=27)
    assert c.tfinal is not None
    assert c.nsteps is not None
    assert abs(c.tfinal - c.tstart - c.dt * c.nsteps) < 1.0e-15

    _check_fixed_controller_evolution(c, c.dt)


# }}}


# {{{ test_graded_controller


def test_graded_controller() -> None:
    from pycaputo.controller import evaluate_timestep_accept, make_graded_controller

    with pytest.raises(ValueError, match="Must provide"):
        make_graded_controller()

    with pytest.raises(ValueError, match="Must provide"):
        make_graded_controller(tfinal=1.0)

    with pytest.raises(ValueError, match="Grading estimate"):
        make_graded_controller(tfinal=1.0, alpha=2.0)

    c = make_graded_controller(tfinal=1.0, nsteps=27, alpha=0.75)
    assert c.tfinal is not None
    assert c.nsteps is not None

    from pycaputo.fode.caputo import ForwardEuler

    y0 = np.zeros(7)
    m = ForwardEuler(
        derivative_order=(0.8,) * y0.size,
        control=c,
        source=lambda t, y: np.zeros_like(y0),
        y0=(y0,),
    )

    q = 1.0
    n = np.arange(c.nsteps)
    t_ref = c.tstart + (n / (c.nsteps - 1)) ** c.r * (c.tfinal - c.tstart)

    t = dt = c.tstart
    for i in range(c.nsteps):
        dt = evaluate_timestep_accept(c, m, q, dt, {"t": t, "n": i})

        assert abs(t - t_ref[i]) < 3.0e-14, (n, abs(t - t_ref[i]))
        t += dt

    dt = evaluate_timestep_accept(c, m, q, dt, {"t": 0.0, "n": 0})
    _check_fixed_controller_evolution(c, 0.0)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
