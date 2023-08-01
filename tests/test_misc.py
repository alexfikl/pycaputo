# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pycaputo.logging import get_logger
from pycaputo.utils import set_recommended_matplotlib

logger = get_logger("pycaputo.test_misc")
dirname = pathlib.Path(__file__).parent
set_recommended_matplotlib()


# {{{ test_lipschitz_uniform_sample


def test_lipschitz_uniform_sample(*, visualize: bool = False) -> None:
    from pycaputo.lipschitz import uniform_diagonal_sample

    a = -1.0
    b = +1.0
    n = 128
    delta = 0.0625 * np.sqrt(2) * (b - a)

    x, y = uniform_diagonal_sample(a, b, n, delta=delta, rng=np.random.default_rng())
    assert np.all(np.abs(x - y) <= delta)
    assert np.all(np.logical_and(a <= x, x <= b))
    assert np.all(np.logical_and(a <= y, y <= b))

    if not visualize:
        return

    from matplotlib.patches import Rectangle

    from pycaputo.utils import figure

    with figure(dirname / "test_lipschitz_uniform_sample") as fig:
        ax = fig.gca()

        mask = np.abs(x - y) <= delta
        ax.plot(x[mask], y[mask], "o")
        ax.plot(x[~mask], y[~mask], "o")

        ax.add_patch(Rectangle((a, a), b - a, b - a, facecolor="none", edgecolor="red"))
        ax.set_xlim([a - delta, b + delta])
        ax.set_ylim([a - delta, b + delta])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
