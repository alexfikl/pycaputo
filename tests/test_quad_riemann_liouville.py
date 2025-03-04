# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.logging import get_logger
from pycaputo.quadrature import quad, riemann_liouville, variable_riemann_liouville
from pycaputo.typing import Array, ScalarFunction
from pycaputo.utils import get_environ_bool, set_recommended_matplotlib

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent
ENABLE_VISUAL = get_environ_bool("ENABLE_VISUAL")

log = get_logger(f"pycaputo.{TEST_FILENAME.stem}")
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
) -> None:
    r"""
    Check the convergence of approximations for the Riemann--Liouville integral.
    The convergence is checked in the :math:`\ell^2` norm using :func:`f_test`.
    """

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

    if ENABLE_VISUAL:
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
        log.info("n %4d h %.5e e %.12e", n, h, e)

        if ENABLE_VISUAL:
            ax.plot(p.x[1:], qf_num[1:])

    log.info("\n%s", eoc)

    if ENABLE_VISUAL:
        ax.plot(p.x[1:], qf_ref[1:], "k--")
        ax.set_xlabel("$x$")
        ax.set_ylabel(rf"$I^{{{alpha}}}_{{RL}} f$")

        filename = f"test_rl_quadrature_{meth.name}_{alpha}"
        savefig(fig, TEST_DIRECTORY / filename, normalize=True)

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
) -> None:
    r"""
    Test convergence of the spectral methods for the RL integral.
    The convergence is checked in the :math:`\ell^2` norm using :func:`f_test`.

    This method is tested separately from :func:`test_riemann_liouville_quad` because
    it requires a slightly different setup.
    """

    from pycaputo.grid import make_jacobi_gauss_lobatto_points
    from pycaputo.utils import EOCRecorder, savefig

    meth = riemann_liouville.SpectralJacobi(-alpha)
    eoc = EOCRecorder()

    if ENABLE_VISUAL:
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
        log.info("n %4d h %.5e e %.12e", n, h, e)

        if ENABLE_VISUAL:
            ax.plot(p.x[1:], qf_num[1:])

    log.info("\n%s", eoc)

    if ENABLE_VISUAL:
        ax.plot(p.x[1:], qf_ref[1:], "k--")
        ax.set_xlabel("$x$")
        ax.set_ylabel(rf"$I^{{{alpha}}}_{{RL}} f$")

        filename = f"test_rl_quadrature_{meth.name}_{alpha}"
        savefig(fig, TEST_DIRECTORY / filename, normalize=True)

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
) -> None:
    r"""
    Test convergence of the Lagrange spline methods for the RL integral.
    The convergence is checked in the :math:`\ell^2` norm using :func:`f_test`.

    This method is tested separately from :func:`test_riemann_liouville_quad` because
    it requires a slightly different setup.
    """

    from pycaputo.grid import make_uniform_points
    from pycaputo.utils import EOCRecorder, savefig

    # FIXME: this order is not right? where is it coming from?
    order = 1.0 * min(npoints, 4.0)
    order = order + min(order, alpha) - 1.0
    meth = riemann_liouville.SplineLagrange(alpha=-alpha, npoints=npoints)

    eoc = EOCRecorder(order=order)

    if ENABLE_VISUAL:
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
        log.info("n %4d h %.5e e %.12e", n, h, e)

        if ENABLE_VISUAL:
            ax.plot(p.x[1:], qf_num[1:])

    log.info("\n%s", eoc)

    if ENABLE_VISUAL:
        ax.plot(p.x[1:], qf_ref[1:], "k--")
        ax.set_xlabel("$x$")
        ax.set_ylabel(rf"$I^{{{alpha}}}_{{RL}} f$")

        filename = f"test_rl_quadrature_{meth.name}_{alpha}"
        savefig(fig, TEST_DIRECTORY / filename, normalize=True)

    from pycaputo.lagrange import vandermonde

    kappa = la.cond(vandermonde(meth.xi))
    assert eoc.estimated_order > order or eoc.max_error < 1.0e-15 * kappa


# }}}


# {{{ test_riemann_liouville_diffusive


@pytest.mark.parametrize(
    ("name", "grid_type"),
    [
        ("YuanAgrawal", "uniform"),
        ("YuanAgrawal", "stynes"),
        ("Diethelm", "uniform"),
        ("Diethelm", "stynes"),
        ("BirkSong", "uniform"),
        ("BirkSong", "stynes"),
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.75, 0.95])
def test_riemann_liouville_diffusive(
    name: str,
    grid_type: str,
    alpha: float,
) -> None:
    r"""
    Check the convergence of diffusive approximations.
    The convergence is checked in the :math:`\ell^2` norm using :func:`f_test`.
    """

    from pycaputo.grid import make_points_from_name
    from pycaputo.utils import EOCRecorder, savefig

    meth: riemann_liouville.DiffusiveRiemannLiouvilleMethod
    resolutions = [8, 16, 24, 32, 48, 64]

    n = 128
    p = make_points_from_name(grid_type, n, a=0.0, b=0.5)
    qf_ref = qf_test(p.x, alpha=alpha)

    eoc = EOCRecorder()
    if ENABLE_VISUAL:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    for quad_order in resolutions:
        if name == "YuanAgrawal":
            meth = riemann_liouville.YuanAgrawal(
                -alpha,
                quad_order=quad_order,
                method="Radau",
            )
            beta = 1.0 - 2.0 * alpha
            order = 1.0 - beta
        elif name == "Diethelm":
            meth = riemann_liouville.Diethelm(
                -alpha, quad_order=quad_order, method="Radau"
            )
            order = None
        elif name == "BirkSong":
            meth = riemann_liouville.BirkSong(
                -alpha, quad_order=quad_order, method="Radau"
            )
            order = None
        else:
            raise ValueError(f"Unsupported method: '{name}'")

        qf_num = quad(meth, f_test, p)

        h = 1.0 / quad_order
        e = la.norm(qf_num[1:] - qf_ref[1:]) / la.norm(qf_ref[1:])
        eoc.add_data_point(h, e)
        log.info("n %4d h %.5e e %.12e", n, h, e)

        if ENABLE_VISUAL:
            ax.plot(p.x[1:], qf_num[1:])

    from dataclasses import replace

    eoc = replace(eoc, order=order)
    log.info("\n%s", eoc)

    if ENABLE_VISUAL:
        ax.plot(p.x[1:], qf_ref[1:], "k--")
        ax.set_xlabel("$x$")
        ax.set_ylabel(rf"$I^{{{alpha}}}_{{RL}} f$")

        filename = f"test_rl_quadrature_{meth.name}_{alpha}"
        savefig(fig, TEST_DIRECTORY / filename, normalize=True)

    if order is not None:
        assert order - 0.25 < eoc.estimated_order < order + 1.0
    else:
        assert eoc.estimated_order > 2.0


# }}}


# {{{ test_variable_riemann_liouville


def make_variable_riemann_liouville_f(
    alpha: tuple[float, float],
    c: float,
    *,
    a: float,
) -> tuple[ScalarFunction, ScalarFunction]:
    # The solutions are constructed by numerically computing the inverse
    # Laplace transform. Given
    #
    #       F(t) = I^{\alpha_1, \alpha_2}[f](t)
    #   =>  LT[F] = LT[Psi] LT[f]
    #
    # where LT[Psi] is the known Laplace transform of the kernel and (f, LT[f])
    # are also taken as some known functions. Then, we can compute the solution
    #
    #   F(t) = LT^{-1}[LT[Psi](s) LT[f](s)]
    #
    # and :tada:, we have a solution to test against!

    import mpmath  # type: ignore[import-untyped]
    from scipy.special import jv

    def f_vo(x: Array) -> Array:
        # NOTE: using the Bessel function of the first-kind
        return np.array(jv(0, 2.0 * np.sqrt(a * x)))

    def f_vo_laplace_transform(s: mpmath.mpf) -> mpmath.mpf:
        return mpmath.exp(-a / s) / s

    def psi_laplace_transform(s: mpmath.mpf) -> mpmath.mpf:
        sA = (alpha[0] * s + alpha[1] * c) / (s + c)
        return s ** (-sA)

    def qf_vo_laplace_transform(s: mpmath.mpf) -> mpmath.mpf:
        return psi_laplace_transform(s) * f_vo_laplace_transform(s)

    def qf_vo(x: Array) -> Array:
        qf = np.empty_like(x)
        for i in range(1, x.size):
            qf[i] = mpmath.invertlaplace(qf_vo_laplace_transform, x[i], method="talbot")

        return qf

    return f_vo, qf_vo


@pytest.mark.parametrize(
    ("alpha", "c"),
    [
        ((0.85, 0.85), 2.0),
        ((0.6, 0.8), 2.0),
        ((0.5, 0.9), 1.0),
        ((0.9, 0.6), 1.0),
    ],
)
def test_variable_riemann_liouville(alpha: tuple[float, float], c: float) -> None:
    r"""
    Check the convergence of approximations for the Riemann--Liouville integral.
    The convergence is checked in the :math:`\ell^2` norm using :func:`f_test`.
    """

    from pycaputo.derivatives import VariableExponentialCaputoDerivative
    from pycaputo.grid import make_points_from_name
    from pycaputo.utils import EOCRecorder, savefig

    d = VariableExponentialCaputoDerivative(alpha=(-alpha[0], -alpha[1]), c=c)
    meth = variable_riemann_liouville.ExponentialRectangular(
        # NOTE: parameters match MATLAB code exactly
        d,
        tau=1.0e-12,
        r=0.99,
        safety_factor=0.01,
    )
    order = 1

    eoc = EOCRecorder(order=order)
    f_test, qf_test = make_variable_riemann_liouville_f(alpha, c, a=1.0)

    if ENABLE_VISUAL:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    resolutions = [16, 32, 64, 128, 256]

    for n in resolutions:
        p = make_points_from_name("uniform", n, a=0.0, b=0.5)
        qf_num = quad(meth, f_test, p)
        qf_ref = qf_test(p.x)

        h = np.max(p.dx)
        e = la.norm(qf_num[1:] - qf_ref[1:]) / la.norm(qf_ref[1:])
        eoc.add_data_point(h, e)
        log.info("n %4d h %.5e e %.12e", n, h, e)

        if ENABLE_VISUAL:
            ax.plot(p.x[1:], qf_num[1:])

    log.info("\n%s", eoc)

    if ENABLE_VISUAL:
        ax.plot(p.x[1:], qf_ref[1:], "k--")
        ax.set_xlabel("$x$")
        ax.set_ylabel(rf"$I^{{{alpha}}}_{{RL}} f$")

        filename = f"test_rl_quadrature_{meth.name}_{alpha[0]}_{alpha[1]}"
        savefig(fig, TEST_DIRECTORY / filename, normalize=True)

    assert order - 0.25 < eoc.estimated_order < order + 1.0


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
