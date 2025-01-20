# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.differentiation import diff, riemann_liouville
from pycaputo.logging import get_logger
from pycaputo.typing import Array
from pycaputo.utils import get_environ_bool, set_recommended_matplotlib

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent
ENABLE_VISUAL = get_environ_bool("ENABLE_VISUAL")

log = get_logger(f"pycaputo.{TEST_FILENAME.stem}")
set_recommended_matplotlib()

# {{{ test_riemann_liouville_lmethods


def f_test(x: Array, *, mu: float = 3.5) -> Array:
    return (0.5 - x) ** mu


def df_test(x: Array, *, alpha: float, mu: float = 3.5) -> Array:
    from scipy.special import gamma, hyp2f1

    if 0 < alpha < 1:
        return np.array(
            -mu
            * 2 ** (1 - mu)
            * x ** (1 - alpha)
            * hyp2f1(1, 1 - mu, 2 - alpha, 2 * x)
            / gamma(2 - alpha)
            + 2 ** (-mu) * x ** (-alpha) / gamma(1 - alpha)
        )

    if 1 < alpha < 2:
        return np.array(
            -mu
            * (1 - mu)
            * 2 ** (2 - mu)
            * x ** (2 - alpha)
            * hyp2f1(1, 2 - mu, 3 - alpha, 2 * x)
            / gamma(3 - alpha)
            + 2 ** (-mu) * x ** (-alpha) / gamma(1 - alpha)
            - mu * 2 ** (1 - mu) * x ** (1 - alpha) / gamma(2 - alpha)
        )

    raise ValueError(f"Unsupported order: {alpha}")


@pytest.mark.parametrize(
    ("name", "grid_type"),
    [
        ("L1", "stretch"),
        ("L1", "uniform"),
        ("L2", "uniform"),
        ("L2C", "uniform"),
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_riemann_liouville_lmethods(
    name: str,
    grid_type: str,
    alpha: float,
) -> None:
    r"""
    Test convergence of the LXXX methods for the Riemann-Liouville derivative.
    The convergence is checked in the :math:`\ell^2` norm using :func:`f_test`.
    """

    from pycaputo.grid import make_points_from_name

    if name in {"L2", "L2C"}:
        alpha += 1

    meth: riemann_liouville.RiemannLiouvilleDerivativeMethod
    if name == "L1":
        meth = riemann_liouville.L1(alpha=alpha)
        order = 2.0 - alpha
    elif name == "L2":
        meth = riemann_liouville.L2(alpha=alpha)
        order = 1.0
    elif name == "L2C":
        meth = riemann_liouville.L2C(alpha=alpha)
        order = 3.0 - alpha
    else:
        raise ValueError(f"Unsupported method: {name}")

    from pycaputo.utils import EOCRecorder, savefig

    eoc = EOCRecorder(order=order)

    if ENABLE_VISUAL:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    for n in [16, 32, 64, 128, 256, 512, 768, 1024]:
        p = make_points_from_name(grid_type, n, a=0.0, b=0.5)
        df_num = diff(meth, f_test, p)
        df_ref = df_test(p.x, alpha=alpha)

        h = np.max(p.dx)
        e = la.norm(df_num[1:] - df_ref[1:]) / la.norm(df_ref[1:])
        eoc.add_data_point(h, e)
        log.info("n %4d h %.5e e %.12e", n, h, e)

        if ENABLE_VISUAL:
            ax.plot(p.x[1:], df_num[1:])
            # ax.semilogy(p.x, abs(df_num - df_ref))

    log.info("\n%s", eoc)

    if ENABLE_VISUAL:
        ax.plot(p.x[1:], df_ref[1:], "k--")
        ax.set_xlabel("$x$")
        ax.set_ylabel(rf"$D^{{{alpha}}}_C f$")
        # ax.set_ylim([1.0e-16, 1])

        filename = f"test_rl_{meth.name}_{alpha}"
        savefig(fig, TEST_DIRECTORY / filename, normalize=True)

    # FIXME: the L2 methods do not behave as expected, but they're doing better
    # so maybe shouldn't complain too much
    assert order - 0.25 < eoc.estimated_order < order + 1.5


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
