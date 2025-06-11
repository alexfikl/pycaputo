# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from functools import partial
from typing import Any

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.fode import caputo_fabrizio
from pycaputo.logging import get_logger
from pycaputo.typing import Array
from pycaputo.utils import get_environ_bool, set_recommended_matplotlib

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent
ENABLE_VISUAL = get_environ_bool("ENABLE_VISUAL")

log = get_logger(f"pycaputo.{TEST_FILENAME.stem}")
set_recommended_matplotlib()


# {{{ solutions


def func(t: float) -> Array:
    return np.array([np.sin(t)])


def func_derivative(t: float, *, alpha: float, t0: float = 0.0) -> Array:
    return np.array([
        (
            alpha * np.cos(t)
            + (1 - alpha) * np.sin(t)
            - np.exp(-alpha / (1 - alpha) * (t - t0))
            * (alpha * np.cos(t0) + (1 - alpha) * np.sin(t0))
        )
        / (1.0 - 2.0 * alpha * (1.0 - alpha))
    ])


def func_source(
    t: float,
    y: Array,
    *,
    alpha: float,
    t0: float = 0.0,
    beta: float = 2.0,
    c: float = 0.0,
) -> Array:
    y_ref = func(t)
    dy_ref = func_derivative(t, alpha=alpha, t0=t0)

    return dy_ref + c * (y**beta - y_ref**beta)


# }}}


# {{{ test_atangana_seda


@pytest.mark.parametrize(
    ("method", "order"),
    [
        ("AtanganaSeda2", 1.0),
        ("AtanganaSeda3", 1.0),
    ],
)
@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75, 0.9])
def test_atangana_seda(method: str, order: float, alpha: float) -> None:
    from pycaputo.derivatives import CaputoFabrizioOperator

    d = CaputoFabrizioOperator(alpha=alpha)

    from pycaputo.events import StepCompleted
    from pycaputo.utils import EOCRecorder

    eoc = EOCRecorder(order=order)

    for h in 2.0 ** (-np.arange(3, 8)):
        from pycaputo.controller import make_fixed_controller

        tstart, tfinal = 0.0, 1.75 * np.pi
        control = make_fixed_controller(float(h), tstart=tstart, tfinal=tfinal)

        m: caputo_fabrizio.AtanganaSeda[Any]
        if method == "AtanganaSeda2":
            m = caputo_fabrizio.AtanganaSeda2(
                ds=(d,),
                control=control,
                source=partial(func_source, alpha=alpha, t0=tstart),
                y0=(func(tstart),),
            )
        elif method == "AtanganaSeda3":
            m = caputo_fabrizio.AtanganaSeda3(
                ds=(d,),
                control=control,
                source=partial(func_source, alpha=alpha, t0=tstart),
                y0=(func(tstart),),
            )
        else:
            raise ValueError(f"Unsupported method: '{method}'")

        from pycaputo.stepping import evolve

        ts = []
        ys = []
        for event in evolve(m):
            assert isinstance(event, StepCompleted)
            ts.append(event.t)
            ys.append(event.y)

        y_ref = func(tfinal)
        error = la.norm(ys[-1] - y_ref) / la.norm(y_ref)
        log.info(
            "dt %.5f y %.12e y_ref %.12e error %.12e",
            h,
            ys[-1].item(),
            la.norm(y_ref),
            error,
        )

        eoc.add_data_point(h, error)

    log.info("\n%s", eoc)

    if ENABLE_VISUAL:
        t = np.array(ts)
        y = np.array(ys).squeeze()
        y_ref = np.array([func(ti) for ti in t]).squeeze()

        from pycaputo.utils import figure

        filename = f"test_fode_caputo_fabrizio_{method}_{alpha}"
        with figure(TEST_DIRECTORY / filename, normalize=True) as fig:
            ax = fig.gca()

            ax.plot(t, y, label=f"{m.name}")
            ax.plot(t, y_ref, "k--", label="Reference")
            ax.set_xlabel("$t$")
            ax.set_ylabel("$y$")
            ax.legend()

    assert order - 0.5 < eoc.estimated_order < order + 0.5


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
