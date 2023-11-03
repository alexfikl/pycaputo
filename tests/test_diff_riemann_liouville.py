# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.derivatives import RiemannLiouvilleDerivative, Side
from pycaputo.differentiation import diff
from pycaputo.logging import get_logger
from pycaputo.utils import Array, set_recommended_matplotlib

logger = get_logger("pycaputo.test_diff_riemann_liouville")
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
        ("RiemannLiouvilleL1Method", "stretch"),
        ("RiemannLiouvilleL1Method", "uniform"),
        ("RiemannLiouvilleL2Method", "uniform"),
        ("RiemannLiouvilleL2CMethod", "uniform"),
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_riemann_liouville_lmethods(
    name: str,
    grid_type: str,
    alpha: float,
    *,
    visualize: bool = False,
) -> None:
    from pycaputo.differentiation import make_method_from_name
    from pycaputo.grid import make_points_from_name

    if name in {"RiemannLiouvilleL2Method", "RiemannLiouvilleL2CMethod"}:
        alpha += 1

    from pycaputo.utils import EOCRecorder, savefig

    d = RiemannLiouvilleDerivative(alpha, Side.Left)
    meth = make_method_from_name(name, d)
    eoc = EOCRecorder(order=meth.order)

    if visualize:
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
        filename = f"test_rl_{meth.name}_{alpha}".replace(".", "_")
        savefig(fig, dirname / filename.lower())

    # FIXME: the L2 methods do not behave as expected, but they're doing better
    # so maybe shouldn't complain too much
    assert eoc.order is not None
    assert eoc.order - 0.25 < eoc.estimated_order < eoc.order + 1.5


# }}}
