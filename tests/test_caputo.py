# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.logging import get_logger
from pycaputo.utils import Array, set_recommended_matplotlib

logger = get_logger("pycaputo.test_caputo")
set_recommended_matplotlib()

# {{{ test_caputo_l1


@pytest.mark.parametrize(
    "name",
    [
        "CaputoL1Method",
        "CaputoUniformL1Method",
        "CaputoModifiedL1Method",
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_caputo_l1(name: str, alpha: float, visualize: bool = False) -> None:
    import math

    from pycaputo import evaluate, make_diff_method
    from pycaputo.grid import (
        make_stretched_points,
        make_uniform_midpoints,
        make_uniform_points,
    )

    def f(x: Array) -> Array:
        return (1 + x) ** 3

    def df(x: Array) -> Array:
        return np.array(
            3 * x ** (1 - alpha) / math.gamma(2 - alpha)
            + 6 * x ** (2 - alpha) / math.gamma(3 - alpha)
            + 6 * x ** (3 - alpha) / math.gamma(4 - alpha)
        )

    from pycaputo.utils import EOCRecorder, savefig

    diff = make_diff_method(name, alpha)
    eoc = EOCRecorder(order=2 - alpha)

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    for n in [32, 64, 128, 256, 512, 768, 1024]:
        if name == "CaputoL1Method":
            p = make_stretched_points(n, a=0, b=1, strength=4.0)
        elif name == "CaputoUniformL1Method":
            p = make_uniform_points(n, a=0, b=1)
        elif name == "CaputoModifiedL1Method":
            p = make_uniform_midpoints(n, a=0, b=1)
        else:
            raise AssertionError

        df_num = evaluate(diff, f, p)
        df_ref = df(p.x)

        h = np.max(p.dx)
        e = la.norm(df_num[1:] - df_ref[1:], ord=np.inf)
        eoc.add_data_point(h, e)
        logger.info("n %4d h %.5e e %.12e", n, h, e)

        if visualize:
            ax.plot(p.x, df_num)
            # ax.semilogy(p.x, abs(df_num - df_ref))

    logger.info("\n%s", eoc)
    assert eoc.order is not None
    assert eoc.order - 0.25 < eoc.estimated_order < eoc.order

    if visualize:
        ax.plot(p.x, df_ref, "k-")
        ax.set_xlabel("$x$")
        ax.set_ylabel(rf"$D^{{{alpha}}}_C f$")
        # ax.set_ylim([1.0e-16, 1])

        dirname = pathlib.Path(__file__).parent
        savefig(fig, dirname / f"test_caputo_l1_{alpha}".replace(".", "_"))


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
