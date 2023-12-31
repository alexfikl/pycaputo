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
        logger.info("[%4d] capacity %d", i, history.capacity)

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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
