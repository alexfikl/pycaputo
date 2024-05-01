# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.logging import get_logger
from pycaputo.quadrature import quad, riemann_liouville
from pycaputo.utils import Array, set_recommended_matplotlib

logger = get_logger("pycaputo.test_quad_riemann_liouville")
set_recommended_matplotlib()


# {{{ test_riemann_liouville_quad


def f_test(x: Array, d: int = 0, *, mu: float = 3.5) -> Array:
    if d == 0:
        return (0.5 - x) ** mu
    elif d == 1:
        return -mu * (0.5 - x) ** (mu - 1)
    else:
        raise NotImplementedError


def qf_test(x: Array, *, alpha: float, mu: float = 3.5) -> Array:
    from scipy.special import gamma, hyp2f1

    return np.array(
        2**-mu * x**alpha * hyp2f1(1, -mu, 1 + alpha, 2 * x) / gamma(1 + alpha)
    )


@pytest.mark.parametrize(
    ("name", "grid_type"),
    [
        ("Rectangular", "uniform"),
        ("Rectangular", "stynes"),
        ("Trapezoidal", "uniform"),
        ("Trapezoidal", "stretch"),
        ("Simpson", "uniform"),
        ("CubicHermite", "uniform"),
        ("Lubich1", "uniform"),
        # ("Lubich2", "uniform"),
        # ("Lubich3", "uniform"),
        # ("Lubich4", "uniform"),
        # ("Lubich5", "uniform"),
        # ("Lubich6", "uniform"),
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
    from pycaputo.grid import make_points_from_name
    from pycaputo.utils import EOCRecorder, savefig

    meth: riemann_liouville.RiemannLiouvilleMethod
    if name == "Rectangular":
        meth = riemann_liouville.Rectangular(-alpha, theta=0.5)
        order = min(2.0, 1 + alpha) if meth.theta == 0.5 else 1.0
    elif name == "Trapezoidal":
        meth = riemann_liouville.Trapezoidal(-alpha)
        order = 2.0
    elif name == "Simpson":
        meth = riemann_liouville.Simpson(-alpha)
        order = 3.0
    elif name == "CubicHermite":
        meth = riemann_liouville.CubicHermite(-alpha)
        order = 4.0
    elif name.startswith("Lubich"):
        order = int(name[6:])
        meth = riemann_liouville.Lubich(-alpha, quad_order=order, beta=np.inf)
    else:
        raise ValueError(f"Unsupported method: '{name}'")

    eoc = EOCRecorder(order=order)

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    if meth.name == "RLCubicHermite":
        # FIXME: errors start to grow at finer meshes; not clear why?
        resolutions = [8, 12, 16, 20, 24, 28, 32, 48, 64]
    else:
        resolutions = [16, 32, 64, 128, 256, 512, 768, 1024]

    for n in resolutions:
        p = make_points_from_name(grid_type, n, a=0.0, b=0.5)
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

    assert order - 0.25 < eoc.estimated_order < order + 1.0


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
    from pycaputo.grid import make_jacobi_gauss_lobatto_points
    from pycaputo.utils import EOCRecorder, savefig

    meth = riemann_liouville.SpectralJacobi(-alpha)
    eoc = EOCRecorder()

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    for n in [8, 12, 16, 24, 32]:
        p = make_jacobi_gauss_lobatto_points(
            n,
            a=0.0,
            b=0.5,
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

    # FIXME: what's the expected behavior here? This just checks that the code
    # doesn't start doing something else all of a sudden..
    assert eoc.estimated_order > 7.0


# }}}


# {{{ test_riemann_liouville_spline


@pytest.mark.parametrize("npoints", [2, 4, 6, 8])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.25, 2.5, 7.75])
def test_riemann_liouville_spline(
    npoints: int,
    alpha: float,
    *,
    visualize: bool = False,
) -> None:
    from pycaputo.grid import make_uniform_points
    from pycaputo.utils import EOCRecorder, savefig

    # FIXME: this order is not right? where is it coming from?
    order = 1.0 * min(npoints, 4.0)
    order = order + min(order, alpha) - 1.0
    meth = riemann_liouville.SplineLagrange(alpha=-alpha, npoints=npoints)

    eoc = EOCRecorder(order=order)

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    for n in [8, 12, 16, 24, 32]:
        p = make_uniform_points(n, a=0.0, b=0.5)
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

    from pycaputo.lagrange import vandermonde

    kappa = la.cond(vandermonde(meth.xi))
    assert eoc.estimated_order > order or eoc.max_error < 1.0e-15 * kappa


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
