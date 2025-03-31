# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from functools import partial

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


def func_derivative(t: float, *, alpha: float) -> Array:
    return np.array([
        (
            alpha * np.cos(t)
            + (1 - alpha) * np.sin(t)
            - alpha * np.exp(-alpha / (1 - alpha) * t)
        )
        / (1.0 + 2.0 * (alpha - 1.0) * alpha)
    ])


def func_source(t: float, y: Array, *, alpha: float) -> Array:
    y_ref = func(t)
    dy_ref = func_derivative(t, alpha=alpha)

    return dy_ref - (y**3 - y_ref**3)


# }}}


# {{{ test_atangana_seda


@pytest.mark.parametrize(("method", "order"), [("AtanganaSeda2", 2.0)])
@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75, 0.95])
def test_atangana_seda(method: str, order: float, alpha: float) -> None:
    from pycaputo.derivatives import CaputoFabrizioOperator

    d = CaputoFabrizioOperator(alpha=alpha)

    from pycaputo.events import StepCompleted
    from pycaputo.utils import EOCRecorder

    eoc = EOCRecorder(order=order)

    for h in 2.0 ** (-np.arange(2, 8)):
        from pycaputo.controller import make_fixed_controller

        control = make_fixed_controller(h, tstart=0.0, tfinal=1.75 * np.pi)

        if method == "AtanganaSeda2":
            m = caputo_fabrizio.AtanganaSeda2(
                ds=(d,),
                control=control,
                source=partial(func_source, alpha=alpha),
                y0=(func(control.tstart),),
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

        y_ref = func(control.tfinal)
        error = la.norm(ys[-1] - y_ref)
        log.info(
            "dt %.5f y %.12e y_ref %.12e error %.12e",
            h,
            ys[-1],
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

        filename = f"test_fode_caputo_fabrizio_{alpha}"
        with figure(TEST_DIRECTORY / filename, normalize=True) as fig:
            ax = fig.gca()

            ax.plot(t, y, label=f"{m.name}")
            ax.plot(t, y_ref, "k--", label="Reference")
            ax.set_xlabel("$t$")
            ax.set_ylabel("$y$")
            ax.legend()


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
