# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib

import numpy as np
import pytest

from pycaputo.logging import get_logger
from pycaputo.utils import set_recommended_matplotlib

logger = get_logger("pycaputo.test_points")
set_recommended_matplotlib()


# {{{ test_points


def test_points(*, visualize: bool = False) -> None:
    from pycaputo.grid import REGISTERED_POINTS, make_points_from_name
    from pycaputo.utils import savefig

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()

    for name in REGISTERED_POINTS:
        p = make_points_from_name(name, 32, a=-1.0, b=3.0)
        assert abs(p.a - p.x[0]) < 1.0e-8

        if visualize:
            ax = fig.gca()

            y = np.linspace(p.a, p.b, 128)
            ax.plot(p.x, p.x, "o-")
            ax.plot(y, y, "k--")

            ax.set_xlabel("$x$")
            ax.set_ylabel("$x$")

            dirname = pathlib.Path(__file__).parent
            savefig(fig, dirname / f"test_points_{name}")
            fig.clf()


# }}}

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
