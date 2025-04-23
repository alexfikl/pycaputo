# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pycaputo.logging import get_logger
from pycaputo.utils import get_environ_bool, set_recommended_matplotlib

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent
ENABLE_VISUAL = get_environ_bool("ENABLE_VISUAL")

log = get_logger(f"pycaputo.{TEST_FILENAME.stem}")
set_recommended_matplotlib()


# {{{ test_points


def test_points() -> None:
    """
    Check that grids are correctly constructed.
    """

    from pycaputo.grid import REGISTERED_POINTS, make_points_from_name
    from pycaputo.utils import savefig

    if ENABLE_VISUAL:
        import matplotlib.pyplot as mp

        fig = mp.figure()

    for name in REGISTERED_POINTS:
        p = make_points_from_name(name, 32, a=-1.0, b=3.0)
        assert abs(p.a - p.x[0]) < 1.0e-8

        if name not in {"midpoints"}:
            assert abs(p.b - p.x[-1]) < 1.0e-8

        if ENABLE_VISUAL:
            ax = fig.gca()

            y = np.linspace(p.a, p.b, 32)
            ax.plot(p.x[1:], p.dx, "o-")
            ax.plot(y[1:], np.diff(y), "ko-")

            ax.set_xlabel("$x$")
            ax.set_ylabel(r"$\Delta x$")

            savefig(fig, TEST_DIRECTORY / f"test_points_{name}", normalize=True)
            fig.clf()


# }}}

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
