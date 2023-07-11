# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.logging import get_logger
from pycaputo.utils import set_recommended_matplotlib

logger = get_logger("pycaputo.test_generating_functions")
set_recommended_matplotlib()


# {{{ test_lubich_bdf_weights


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.25, 2.5, 5.27])
def test_lubich_bdf_weights(
    order: int, alpha: float, *, visualize: bool = False
) -> None:
    from pycaputo.generating_functions import (
        lubich_bdf_starting_weights,
        lubich_bdf_weights,
    )
    from pycaputo.utils import EOCRecorder, savefig

    eoc = EOCRecorder(order=order)
    a, b = 0.0, 1.0
    s = order

    from math import gamma

    for n in [64, 128, 256, 512, 1024]:
        t = np.linspace(a, b, n)
        h = t[1] - t[0]
        w = lubich_bdf_weights(-alpha, order, n)
        omega = np.fromiter(
            lubich_bdf_starting_weights(w, s, -alpha, beta=1.0),
            dtype=np.dtype((w.dtype, s)),
        ).reshape(-1, s)

        int_ref = t**alpha / gamma(1 + alpha)
        int_bdf = np.empty_like(int_ref)
        int_bdf[:s] = int_ref[:s]
        for k in range(s, n):
            int_bdf[k] = h**alpha * (np.sum(w[: k + 1]) + np.sum(omega[k - s]))

        error = la.norm(int_ref - int_bdf) / la.norm(int_ref)
        logger.info("ref %.12e bdf %.12e error %.12e", int_ref[-1], int_bdf[-1], error)

        eoc.add_data_point(h, error)

    logger.info("\n%s", eoc)

    if not visualize:
        return

    import matplotlib.pyplot as mp

    fig = mp.figure()
    ax = fig.gca()

    # ax.semilogy(t[s:], abs(int_ref - int_bdf)[s:], "k--")
    ax.plot(t, int_ref, "k--")
    ax.plot(t, int_bdf)
    ax.set_xlabel("$t$")
    ax.set_ylabel(f"$I^{{{alpha}}}_{{RL}}[1]$")

    dirname = pathlib.Path(__file__).parent
    filename = f"test_generator_lubich_bdf_{order}_{alpha}".replace(".", "_")
    savefig(fig, dirname / filename)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
