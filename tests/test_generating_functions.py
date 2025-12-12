# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.logging import get_logger
from pycaputo.utils import get_environ_bool, set_recommended_matplotlib

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent
ENABLE_VISUAL = get_environ_bool("ENABLE_VISUAL")

log = get_logger(f"pycaputo.{TEST_FILENAME.stem}")
set_recommended_matplotlib()


# {{{ test_lmm_starting_weights


@pytest.mark.parametrize("name", ["diethelm", "lubich", "garrappa"])
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("alpha", [0.5, 0.9, 1.75])
def test_lmm_starting_weights(name: str, order: int, alpha: float) -> None:
    import pycaputo.generating_functions as lmm

    if name == "diethelm":
        sigma = lmm.diethelm_starting_powers(order, alpha)
    elif name == "lubich":
        sigma = lmm.lubich_starting_powers(order, alpha)
    elif name == "garrappa":
        sigma = lmm.garrappa_starting_powers(order, alpha)
    else:
        raise ValueError(f"Unknown starting weights: {name!r}")

    npoints = 128
    w = lmm.lubich_bdf_weights(alpha, order, npoints)

    for k, ws in lmm.lmm_starting_weights(w, sigma, alpha):
        assert k >= sigma.size
        assert ws.shape == sigma.shape
        assert np.all(np.isfinite(ws))


# }}}


# {{{ test_lubich_bdf_weights


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.25, 2.5, 5.27])
def test_lubich_bdf_weights(order: int, alpha: float) -> None:
    """
    Check that the Lubich weights can integrate a smooth function.
    """

    from pycaputo.generating_functions import lubich_bdf_weights
    from pycaputo.utils import EOCRecorder, savefig

    eoc = EOCRecorder(order=order)
    a, b = 0.0, 1.0

    if ENABLE_VISUAL:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    from math import gamma

    for n in [64, 128, 256, 512, 1024]:
        t = np.linspace(a, b, n)
        h = t[1] - t[0]
        w = lubich_bdf_weights(alpha, order, n)

        int_ref = (t - a) ** alpha / gamma(1 + alpha)
        int_bdf = np.empty_like(int_ref)

        fx = np.ones_like(w)
        for k in range(1, n - 1):
            int_bdf[k] = h**alpha * np.sum(w[: k + 1] * fx[: k + 1])

        error = la.norm(int_ref[5:-5] - int_bdf[5:-5], ord=np.inf) / la.norm(
            int_ref[5:-5], ord=np.inf
        )
        log.info("ref %.12e bdf %.12e error %.12e", int_ref[-5], int_bdf[-5], error)

        eoc.add_data_point(h, error)

        if ENABLE_VISUAL:
            ax.semilogy(t, abs(int_ref - int_bdf) + 1.0e-16)
            # ax.plot(t, int_ref, "k--")
            # ax.plot(t, int_bdf)

    log.info("\n%s", eoc)

    if ENABLE_VISUAL:
        ax.set_xlabel("$t$")
        ax.set_ylabel(f"$I^{{{alpha}}}_{{RL}}[1]$")

        filename = f"test_lmm_lubich_bdf_{order}_{alpha}".replace(".", "_")
        savefig(fig, TEST_DIRECTORY / filename)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
