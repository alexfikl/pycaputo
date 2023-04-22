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


# {{{ test_riemann_liouville_quad


@pytest.mark.xfail(reason="work in progress")
@pytest.mark.parametrize(
    ("name", "grid_type"),
    [
        ("RiemannLiouvilleRectangularMethod", "uniform"),
        ("RiemannLiouvilleRectangularMethod", "stretch"),
        # ("RiemannLiouvilleTrapezoidalMethod", "uniform"),
        # ("RiemannLiouvilleTrapezoidalMethod", "stretch"),
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0, 2.0, 7.0])
def test_riemann_liouville_quad(
    name: str,
    grid_type: str,
    alpha: float,
    visualize: bool = False,
) -> None:
    from pycaputo import make_quad_from_name, quad
    from pycaputo.grid import make_points_from_name

    def f(x: Array) -> Array:
        return np.zeros_like(x)

    def qf(x: Array) -> Array:
        return np.zeros_like(x)

    from pycaputo.utils import EOCRecorder, savefig

    meth = make_quad_from_name(name, -alpha)
    eoc = EOCRecorder(order=meth.order)

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    for n in [16, 32, 64, 128, 256, 512, 768, 1024]:
        p = make_points_from_name(grid_type, n, a=0.0, b=1.0)
        qf_num = quad(meth, f, p)
        qf_ref = qf(p.x)

        h = np.max(p.dx)
        e = la.norm(qf_num[1:] - qf_ref[1:]) / la.norm(qf_ref[1:])
        eoc.add_data_point(h, e)
        logger.info("n %4d h %.5e e %.12e", n, h, e)

        if visualize:
            ax.plot(p.x[1:], qf_num[1:])

    logger.info("\n%s", eoc)

    if visualize:
        ax.plot(p.x[1:], qf_ref[1:], "k--")
        ax.set_xlabel("$x$")
        ax.set_ylabel(rf"$I^{{{alpha}}}_{{RL}} f$")

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
