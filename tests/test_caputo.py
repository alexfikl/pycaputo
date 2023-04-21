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

# {{{ test_caputo_lmethods


@pytest.mark.parametrize(
    ("name", "grid_type"),
    [
        ("CaputoL1Method", "stretch"),
        ("CaputoL1Method", "uniform"),
        ("CaputoModifiedL1Method", "midpoints"),
        ("CaputoL2CMethod", "uniform"),
        ("CaputoL2Method", "uniform"),
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_caputo_lmethods(
    name: str,
    grid_type: str,
    alpha: float,
    visualize: bool = False,
) -> None:
    import math

    from pycaputo import diff, make_diff_from_name
    from pycaputo.grid import make_points_from_name

    if name in ("CaputoL2Method", "CaputoL2CMethod"):
        alpha += 1

    def f(x: Array) -> Array:
        return (0.5 - x) ** 4

    def df(x: Array) -> Array:
        if 0 < alpha < 1:
            return np.array(
                -1 / 2 * x ** (1 - alpha) / math.gamma(2 - alpha)
                + 3 * x ** (2 - alpha) / math.gamma(3 - alpha)
                - 12 * x ** (3 - alpha) / math.gamma(4 - alpha)
                + 24 * x ** (4 - alpha) / math.gamma(5 - alpha)
            )

        if 1 < alpha < 2:
            p = 12 + 8 * (x - 2) * x - 7 * alpha + 4 * alpha * x + alpha**2
            return np.array(3 * x ** (2 - alpha) * p / math.gamma(5 - alpha))

        raise ValueError(f"Unsupported order: {alpha}")

    from pycaputo.utils import EOCRecorder, savefig

    meth = make_diff_from_name(name, alpha)
    eoc = EOCRecorder(order=meth.order)

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    for n in [16, 32, 64, 128, 256, 512, 768, 1024]:
        p = make_points_from_name(grid_type, n, a=0.0, b=1.0)
        df_num = diff(meth, f, p)
        df_ref = df(p.x)

        h = np.max(p.dx)
        e = la.norm(df_num[1:] - df_ref[1:]) / la.norm(df_ref[1:])
        # e = abs(df_num[n // 2] - df_ref[n // 2])
        eoc.add_data_point(h, e)
        logger.info("n %4d h %.5e e %.12e", n, h, e)

        if visualize:
            ax.plot(p.x[1:], df_num[1:])
            # ax.semilogy(p.x, abs(df_num - df_ref))

    logger.info("\n%s", eoc)

    if visualize:
        ax.plot(p.x[1:], df_ref[1:], "k--")
        ax.set_xlabel("$x$")
        ax.set_ylabel(rf"$D^{{{alpha}}}_C f$")
        # ax.set_ylim([1.0e-16, 1])

        dirname = pathlib.Path(__file__).parent
        filename = f"test_caputo_{meth.name}_{alpha}".replace(".", "_")
        savefig(fig, dirname / filename.lower())

    assert eoc.order is not None
    assert eoc.order - 0.25 < eoc.estimated_order < eoc.order + 0.25


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
