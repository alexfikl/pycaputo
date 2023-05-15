# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import math
import pathlib

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.logging import get_logger
from pycaputo.utils import Array, set_recommended_matplotlib

logger = get_logger("pycaputo.test_quadrature")
set_recommended_matplotlib()


# {{{ test_riemann_liouville_quad


def f_test(x: Array) -> Array:
    return (0.5 - x) ** 4


def qf_test(x: Array, *, alpha: float) -> Array:
    return np.array(
        0.0625 * x**alpha / math.gamma(1 + alpha)
        - 0.5 * x ** (1 + alpha) / math.gamma(2 + alpha)
        + 3 * x ** (2 + alpha) / math.gamma(3 + alpha)
        - 12 * x ** (3 + alpha) / math.gamma(4 + alpha)
        + 24 * x ** (4 + alpha) / math.gamma(5 + alpha)
    )


@pytest.mark.xfail(reason="work in progress")
@pytest.mark.parametrize(
    ("name", "grid_type"),
    [
        ("RiemannLiouvilleRectangularMethod", "uniform"),
        ("RiemannLiouvilleRectangularMethod", "stynes"),
        # ("RiemannLiouvilleTrapezoidalMethod", "uniform"),
        ("RiemannLiouvilleTrapezoidalMethod", "stretch"),
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.25, 2.5, 7.75])
def test_riemann_liouville_quad(
    name: str,
    grid_type: str,
    alpha: float,
    *,
    visualize: bool = False,
) -> None:
    from pycaputo import make_quad_from_name, quad
    from pycaputo.grid import make_points_from_name
    from pycaputo.utils import EOCRecorder, savefig

    meth = make_quad_from_name(name, -alpha)
    eoc = EOCRecorder(order=meth.order)

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    for n in [16, 32, 64, 128, 256, 512, 768, 1024]:
        p = make_points_from_name(grid_type, n, a=0.0, b=1.0)
        qf_num = quad(meth, f_test, p)
        qf_ref = qf_test(p.x, alpha=alpha)

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
        filename = f"test_rl_quadrature_{meth.name}_{alpha}".replace(".", "_")
        savefig(fig, dirname / filename.lower())

    assert eoc.order is not None
    assert eoc.order - 0.25 < eoc.estimated_order < eoc.order + 0.25


# }}}


# {{{ test_riemann_liouville_quad_spectral


@pytest.mark.parametrize(
    ("j_alpha", "j_beta"),
    [
        # Legendre polynomials
        (0.0, 0.0),
        # Chebyshev polynomials
        (-0.5, -0.5),
        # Other? Not really of any interest
        (1.0, 1.0),
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.25, 2.5, 7.75])
def test_riemann_liouville_quad_spectral(
    j_alpha: float,
    j_beta: float,
    alpha: float,
    *,
    visualize: bool = False,
) -> None:
    from pycaputo import (
        RiemannLiouvilleDerivative,
        RiemannLiouvilleSpectralMethod,
        Side,
        quad,
    )
    from pycaputo.grid import make_jacobi_gauss_lobatto_points
    from pycaputo.utils import EOCRecorder, savefig

    d = RiemannLiouvilleDerivative(order=-alpha, side=Side.Left)
    meth = RiemannLiouvilleSpectralMethod(d=d)
    eoc = EOCRecorder(order=meth.order)

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    for n in [8, 12, 16, 24, 32]:
        p = make_jacobi_gauss_lobatto_points(
            n,
            a=0.0,
            b=1.0,
            alpha=j_alpha,
            beta=j_beta,
        )
        qf_num = quad(meth, f_test, p)
        qf_ref = qf_test(p.x, alpha=alpha)

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
        filename = f"test_rl_quadrature_{meth.name}_{alpha}".replace(".", "_")
        savefig(fig, dirname / filename.lower())

    # NOTE: the test function is polynomial, so this is always exact
    assert eoc.max_error < 1.0e-13


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
