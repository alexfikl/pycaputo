# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.logging import get_logger
from pycaputo.quadrature import QuadratureMethod
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


def make_rl_conv_factory(order: int) -> Callable[[float], QuadratureMethod]:
    from pycaputo.derivatives import RiemannLiouvilleDerivative, Side
    from pycaputo.quadrature import RiemannLiouvilleConvolutionMethod

    def wrapper(alpha: float) -> RiemannLiouvilleConvolutionMethod:
        d = RiemannLiouvilleDerivative(order=alpha, side=Side.Left)
        return RiemannLiouvilleConvolutionMethod(d=d, quad_order=order, beta=np.inf)

    return wrapper


@pytest.mark.parametrize(
    ("factory", "grid_type"),
    [
        ("RiemannLiouvilleRectangularMethod", "uniform"),
        ("RiemannLiouvilleRectangularMethod", "stynes"),
        ("RiemannLiouvilleTrapezoidalMethod", "uniform"),
        ("RiemannLiouvilleTrapezoidalMethod", "stretch"),
        ("RiemannLiouvilleSimpsonMethod", "uniform"),
        ("RiemannLiouvilleCubicHermiteMethod", "uniform"),
        (make_rl_conv_factory(1), "uniform"),
        # (make_rl_conv_factory(2), "uniform"),
        # (make_rl_conv_factory(3), "uniform"),
        # (make_rl_conv_factory(4), "uniform"),
        # (make_rl_conv_factory(5), "uniform"),
        # (make_rl_conv_factory(6), "uniform"),
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.25, 2.5, 7.75])
def test_riemann_liouville_quad(
    factory: str | Callable[[float], QuadratureMethod],
    grid_type: str,
    alpha: float,
    *,
    visualize: bool = False,
) -> None:
    from pycaputo.grid import make_points_from_name
    from pycaputo.quadrature import make_method_from_name, quad
    from pycaputo.utils import EOCRecorder, savefig

    if callable(factory):
        meth = factory(-alpha)
    else:
        meth = make_method_from_name(factory, -alpha)

    eoc = EOCRecorder(order=meth.order)

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    if meth.name == "RLCHermite":
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

    assert eoc.order is not None
    assert eoc.order - 0.25 < eoc.estimated_order < eoc.order + 1.0


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
    from pycaputo.derivatives import RiemannLiouvilleDerivative, Side
    from pycaputo.grid import make_jacobi_gauss_lobatto_points
    from pycaputo.quadrature import RiemannLiouvilleSpectralMethod, quad

    d = RiemannLiouvilleDerivative(order=-alpha, side=Side.Left)
    meth = RiemannLiouvilleSpectralMethod(d=d)

    from pycaputo.utils import EOCRecorder, savefig

    eoc = EOCRecorder(order=meth.order)

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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
