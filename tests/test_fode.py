# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib
from functools import partial

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.derivatives import CaputoDerivative, Side
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


def fode_solution(t: float) -> Array:
    return np.array([t**2])


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_caputo_fode(alpha: float, *, visualize: bool = True) -> None:
    from pycaputo.fode import CaputoForwardEulerMethod, History, evolve
    from pycaputo.utils import EOCRecorder

    eoc = EOCRecorder(order=1.0)

    for n in [32, 64, 128, 256, 512]:
        y0 = fode_solution(0.0)
        tspan = (0.0, 1.0)
        dt = (tspan[1] - tspan[0]) / n

        m = CaputoForwardEulerMethod(
            d=CaputoDerivative(order=alpha, side=Side.Left),
            predict_time_step=partial(fode_time_step, dt=dt),
            source=partial(fode_source, alpha=alpha),
            tspan=tspan,
            y0=(y0,),
        )

        history = History()
        for _ in evolve(m, history=history, verbose=True):
            pass

        t, y = history.load(-1)
        y_ref = fode_solution(t)
        error = la.norm(y - y_ref) / la.norm(y_ref)
        logger.info("dt %.5f error %.12e", dt, error)

        eoc.add_data_point(dt, error)

    logger.info("\n%s", eoc)

    if not visualize:
        ts = np.array(history.thistory)
        ys = np.array(history.yhistory).squeeze()
        ys_ref = np.array([fode_solution(t) for t in ts]).squeeze()

        from pycaputo.utils import figure

        dirname = pathlib.Path(__file__).parent
        filename = f"test_caputo_fode_{m.name}_{alpha}".replace(".", "_").lower()
        with figure(dirname / filename) as fig:
            ax = fig.gca()

            ax.plot(t, ys)
            ax.plot(t, ys_ref, "k--")
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
