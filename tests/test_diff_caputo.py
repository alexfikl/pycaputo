# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.differentiation import caputo, diff
from pycaputo.grid import Points
from pycaputo.logging import get_logger
from pycaputo.typing import Array, ScalarFunction
from pycaputo.utils import get_environ_bool, set_recommended_matplotlib

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent
ENABLE_VISUAL = get_environ_bool("ENABLE_VISUAL")

log = get_logger(f"pycaputo.{TEST_FILENAME.stem}")
set_recommended_matplotlib()

# {{{ utils


def f_test(x: Array, d: int = 0, *, mu: float = 3.5) -> Array:
    if d == 0:
        return (0.5 - x) ** mu
    elif d == 1:
        return -mu * (0.5 - x) ** (mu - 1)
    elif d == 2:
        return mu * (mu - 1) * (0.5 - x) ** (mu - 2)
    else:
        raise NotImplementedError


def df_test(x: Array, *, alpha: float, mu: float = 3.5) -> Array:
    from scipy.special import gamma, hyp2f1

    if 0 < alpha < 1:
        return np.array(
            -mu
            * 2 ** (1 - mu)
            * x ** (1 - alpha)
            * hyp2f1(1, 1 - mu, 2 - alpha, 2 * x)
            / gamma(2 - alpha)
        )

    if 1 < alpha < 2:
        return np.array(
            -mu
            * (1 - mu)
            * 2 ** (2 - mu)
            * x ** (2 - alpha)
            * hyp2f1(1, 2 - mu, 3 - alpha, 2 * x)
            / gamma(3 - alpha)
        )

    raise ValueError(f"Unsupported order: {alpha}")


def make_test_points_from_name(
    name: str,
    n: int,
    *,
    j_alpha: float = 0.0,
    j_beta: float = 0.0,
) -> Points:
    from pycaputo.grid import make_jacobi_gauss_lobatto_points, make_points_from_name

    p: Points
    if name == "jacobi":
        p = make_jacobi_gauss_lobatto_points(
            n,
            a=0.0,
            b=0.5,
            alpha=j_alpha,
            beta=j_beta,
        )
    else:
        p = make_points_from_name(name, n, a=0.0, b=0.5)

    return p


def make_method_from_name(
    name: str,
    alpha: float,
    *,
    quad_order: int | None = None,
) -> tuple[caputo.CaputoMethod, float]:
    meth: caputo.CaputoMethod
    if name == "L1":
        meth = caputo.L1(alpha=alpha)
        order = 2.0 - alpha
    elif name == "L1D":
        meth = caputo.LXD(alpha=alpha)
        order = 2.0 - alpha
    elif name == "ModifiedL1":
        meth = caputo.ModifiedL1(alpha=alpha)
        order = 2 - alpha
    elif name == "L2":
        meth = caputo.L2(alpha=alpha)
        # FIXME: this is wrong
        order = 1.0
    elif name == "L2C":
        meth = caputo.L2C(alpha=alpha)
        order = 3.0 - alpha
    elif name == "L2F":
        from pycaputo.derivatives import Side

        meth = caputo.L2F(alpha=alpha, side=Side.Left)
        order = 1.0
    elif name == "L2D":
        meth = caputo.LXD(alpha=alpha)
        order = 3.0 - alpha
    elif name == "Jacobi":
        meth = caputo.Jacobi(alpha=alpha)
        order = 5.0
    elif name == "YuanAgrawal":
        assert quad_order is not None
        meth = caputo.YuanAgrawal(
            alpha,
            quad_order=quad_order,
            method="Radau",
        )
        beta = 2.0 * alpha - 1.0
        order = 1.0 - beta
    elif name == "Diethelm":
        assert quad_order is not None
        meth = caputo.Diethelm(alpha, quad_order=quad_order, method="Radau")
        order = 0
    elif name == "BirkSong":
        assert quad_order is not None
        meth = caputo.BirkSong(alpha, quad_order=quad_order, method="Radau")
        order = 0
    else:
        raise ValueError(f"Unsupported method: '{name}'")

    return meth, order


# }}}


# {{{ test_caputo_lmethods


@pytest.mark.parametrize(
    ("name", "grid_type"),
    [
        ("L1", "stretch"),
        ("L1", "uniform"),
        ("L1D", "uniform"),
        ("L2", "uniform"),
        ("L2C", "uniform"),
        ("L2D", "uniform"),
        ("L2F", "uniform"),
        ("ModifiedL1", "midpoints"),
        ("ModifiedL1", "stretch"),
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_caputo_lmethods(
    name: str,
    grid_type: str,
    alpha: float,
) -> None:
    r"""
    Test convergence of the LXXX methods for the Caputo derivative.
    The convergence is checked in the :math:`\ell^2` norm using :func:`f_test`.
    """

    if name in {"L2", "L2C", "L2D", "L2F"}:
        alpha += 1

    from pycaputo.utils import EOCRecorder, savefig, stringify_eoc

    meth, order = make_method_from_name(name, alpha)
    eoc = EOCRecorder(order=order)

    if ENABLE_VISUAL:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    for n in [16, 32, 64, 128, 256, 512, 768, 1024]:
        p = make_test_points_from_name(grid_type, n)
        if name == "ModifiedL1" and grid_type != "midpoints":
            from pycaputo.grid import make_midpoints_from

            p_ref: Points = make_midpoints_from(p)
        else:
            p_ref = p

        df_num = diff(meth, f_test, p)
        df_ref = df_test(p_ref.x, alpha=alpha)

        h = np.max(p.dx)
        e = la.norm(df_num[1:] - df_ref[1:]) / la.norm(df_ref[1:])
        eoc.add_data_point(h, e)
        log.info("n %4d h %.5e e %.12e", n, h, e)

        if ENABLE_VISUAL:
            ax.plot(p_ref.x[1:], df_num[1:])
            # ax.semilogy(p_ref.x, abs(df_num - df_ref))

    log.info("\n%s", stringify_eoc(eoc))

    if ENABLE_VISUAL:
        ax.plot(p_ref.x[1:], df_ref[1:], "k--")
        ax.set_xlabel("$x$")
        ax.set_ylabel(rf"$D^{{{alpha}}}_C f$")
        # ax.set_ylim([1.0e-16, 1])

        filename = f"test_caputo_{meth.name}_{alpha}"
        savefig(fig, TEST_DIRECTORY / filename, normalize=True)

    assert order - 0.25 < eoc.estimated_order < order + 0.25


# }}}


# {{{ test_caputo_spectral


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
@pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_caputo_spectral(
    j_alpha: float,
    j_beta: float,
    alpha: float,
) -> None:
    r"""
    Test convergence of the spectral methods for the Caputo derivative.
    The convergence is checked in the :math:`\ell^2` norm using :func:`f_test`.

    This method is tested separately from :func:`test_caputo_lmethods` because
    it requires a slightly different setup.
    """

    from pycaputo.utils import EOCRecorder, savefig

    meth, _ = make_method_from_name("Jacobi", alpha)
    eoc = EOCRecorder()

    if ENABLE_VISUAL:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    for n in [8, 12, 16, 24, 32]:
        p = make_test_points_from_name("jacobi", n, j_alpha=j_alpha, j_beta=j_beta)
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

        filename = f"test_caputo_{meth.name}_{j_alpha}_{j_beta}_{alpha}"
        savefig(fig, TEST_DIRECTORY / filename, normalize=True)

    assert eoc.estimated_order > 5.0


# }}}


# {{{ test_caputo_diffusive


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
def test_caputo_diffusive(
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

    resolutions = [8, 16, 24, 32, 48, 64]

    n = 128
    p = make_points_from_name(grid_type, n, a=0.0, b=0.5)
    df_ref = df_test(p.x, alpha=alpha)

    eoc = EOCRecorder()
    if ENABLE_VISUAL:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    for quad_order in resolutions:
        meth, order = make_method_from_name(name, alpha, quad_order=quad_order)
        df_num = diff(meth, f_test, p)

        h = 1.0 / quad_order
        e = la.norm(df_num[1:] - df_ref[1:]) / la.norm(df_ref[1:])
        eoc.add_data_point(h, e)
        log.info("n %4d h %.5e e %.12e", n, h, e)

        if ENABLE_VISUAL:
            ax.plot(p.x[1:], df_num[1:])

    from dataclasses import replace

    eoc = replace(eoc, order=order)
    log.info("\n%s", eoc)

    if ENABLE_VISUAL:
        ax.plot(p.x[1:], df_ref[1:], "k--")
        ax.set_xlabel("$x$")
        ax.set_ylabel(rf"$D^{{{alpha}}}_{{C}} f$")

        filename = f"test_caputo_{meth.name}_{alpha}"
        savefig(fig, TEST_DIRECTORY / filename, normalize=True)

    if order > 0:
        assert order - 0.25 < eoc.estimated_order < order + 1.0
    else:
        assert eoc.estimated_order > 1.9


# }}}


# {{{ test_caputo_vs_differint


@dataclass(frozen=True)
class DifferIntCaputoL1(caputo.CaputoMethod):
    pass


@diff.register(DifferIntCaputoL1)
def _diff_differint_l1(m: DifferIntCaputoL1, f: ScalarFunction, p: Points) -> Array:
    from differint.differint import CaputoL1point

    df = np.empty_like(p.x)
    df[0] = np.nan

    for n in range(1, df.size):
        df[n] = CaputoL1point(
            m.alpha,
            f,
            domain_start=p.a,
            domain_end=p.x[n],
            num_points=n + 1,
        )

    return df


@dataclass(frozen=True)
class DifferIntCaputoL2(caputo.CaputoMethod):
    pass


@diff.register(DifferIntCaputoL2)
def _diff_differint_l2(m: DifferIntCaputoL2, f: ScalarFunction, p: Points) -> Array:
    from differint.differint import CaputoL2point

    df = np.empty_like(p.x)
    df[0] = np.nan

    for n in range(1, df.size):
        df[n] = CaputoL2point(
            m.alpha,
            f,
            domain_start=p.a,
            domain_end=p.x[n],
            num_points=n + 1,
        )

    return df


@dataclass(frozen=True)
class DifferIntCaputoL2C(caputo.CaputoMethod):
    pass


@diff.register(DifferIntCaputoL2C)
def _diff_differint_l2c(m: DifferIntCaputoL2C, f: ScalarFunction, p: Points) -> Array:
    from differint.differint import CaputoL2Cpoint

    df = np.empty_like(p.x)
    df[0] = np.nan

    for n in range(2, df.size - 1):
        df[n] = CaputoL2Cpoint(
            m.alpha,
            f,
            domain_start=p.a,
            domain_end=p.x[n],
            num_points=n + 1,
        )

    return df


@pytest.mark.parametrize("name", ["L1", "L2", "L2C"])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_caputo_vs_differint(
    name: str,
    alpha: float,
) -> None:
    """
    Compare the Caputo derivative approximations with the :mod:`differint` library.

    The L2 and L2C methods are implemented differently by us, so those will not
    compare well. In particular, boundary terms use a biased stencil instead of
    the same centered stencil.
    """

    pytest.importorskip("differint")

    if name in {"L2", "L2C"}:
        alpha += 1

    meth: caputo.CaputoMethod
    if name == "L1":
        meth = caputo.L1(alpha=alpha)
        differint_meth: caputo.CaputoMethod = DifferIntCaputoL1(alpha=alpha)
    elif name == "L2":
        from pycaputo.derivatives import Side

        meth = caputo.L2F(alpha=alpha, side=Side.Right)
        differint_meth = DifferIntCaputoL2(alpha=alpha)
    elif name == "L2C":
        meth = caputo.L2C(alpha=alpha)
        differint_meth = DifferIntCaputoL2C(alpha=alpha)
    else:
        raise ValueError(f"Unknown method: '{name}'")

    from pycaputo.grid import make_points_from_name

    # NOTE: b must be < 0.5 so that f(p.b + dx) is well defined for the test function
    p = make_points_from_name("uniform", 512, a=0.0, b=0.25)

    df_ref = df_test(p.x, alpha=alpha)
    df_num = diff(meth, f_test, p)
    df_num_di = diff(differint_meth, f_test, p)

    error_vs_ref = la.norm(df_num[1:] - df_ref[1:]) / la.norm(df_ref[1:])
    error_di_vs_ref = la.norm(df_num_di[1:] - df_ref[1:]) / la.norm(df_ref[1:])
    error_vs_di = la.norm(df_num[1:] - df_num_di[1:]) / la.norm(df_num_di[1:])

    log.info(
        "error: vs ref %.12e vs differint %.12e (differint vs ref %.12e)",
        error_vs_ref,
        error_vs_di,
        error_di_vs_ref,
    )

    if ENABLE_VISUAL:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

        ax.semilogy(p.x[1:], abs(df_num[1:] - df_num_di[1:]), label="Comparison")
        ax.semilogy(p.x[1:], abs(df_num_di[1:] - df_ref[1:]), "ko:", label="differint")
        ax.semilogy(p.x[1:], abs(df_num[1:] - df_ref[1:]), "--", label="pycaputo")

        ax.set_xlabel("$x$")
        ax.set_ylabel(rf"$D^{{{alpha}}}_C f$")
        ax.legend()

        from pycaputo.utils import savefig

        filename = f"test_caputo_differint_{meth.name}_{alpha}"
        savefig(fig, TEST_DIRECTORY / filename, normalize=True)

    assert error_vs_ref < 1.0e-2
    if name == "L1":
        assert error_vs_di < 1.0e-12
    elif name == "L2":
        assert error_vs_di < 2.0e-11
    elif name == "L2C":
        # FIXME: need to implement a L2CF similar to L2F to better compare
        assert error_vs_di < 7.0e-2


# }}}


# {{{ test_caputo_consistency


@pytest.mark.parametrize(
    ("name", "grid_type"),
    [
        ("L1", "stretch"),
        ("L2", "stretch"),
    ],
)
@pytest.mark.parametrize("alpha", [0.3, 0.9])
def test_caputo_consistency(
    name: str,
    grid_type: str,
    alpha: float,
) -> None:
    """
    Tests that the `quadrature_weights`, `differentiation_matrix`, `diffs` and
    `diff` methods actually give the same results.
    """
    from pycaputo.differentiation import (
        differentiation_matrix,
        diffs,
        quadrature_weights,
    )

    if name in {"L2", "L2F", "L2D"}:
        alpha += 1

    meth, _ = make_method_from_name(name, alpha)
    p = make_test_points_from_name(grid_type, 256)

    # check quadrature_weights vs differentiation_matrix
    W_weights = np.zeros((p.size, p.size), dtype=p.dtype)
    for n in range(1, p.size):
        w = quadrature_weights(meth, p, n)
        W_weights[n, : w.size] = w

    W_mat = differentiation_matrix(meth, p)

    error = la.norm(W_weights[1:, :] - W_mat[1:, :])
    log.info("quadrature_weights vs differentiation_matrix")
    log.info("  error %.12e", error)
    assert error < 1.0e-15

    # check quadrature_weights vs diffs
    log.info("quadrature_weights vs diffs")
    fx = f_test(p.x)
    for n in [0, 1, p.size // 2 + 3, p.size - 1]:
        df_from_diffs = diffs(meth, fx, p, n)

        if n == 0:
            w = quadrature_weights(meth, p, n)
            assert w.shape == (0,)
            assert df_from_diffs.shape == (1,)
            assert np.isnan(df_from_diffs)
        else:
            w = quadrature_weights(meth, p, n)
            df_from_weights = w @ fx[: w.size]
            log.info(
                "   value %.12e %.12e (ref %.12e) w shape %s",
                df_from_weights,
                df_from_diffs,
                df_test(p.x[n], alpha=alpha),
                w.shape,
            )

            error = la.norm(df_from_weights - df_from_diffs)
            log.info("   n %d error %.12e", n, error)

            # FIXME: this seems to fail (sometimes) on Python 3.11 and 3.12 on the
            # CI. There's no randomness in here, so not clear why that would be
            # assert error < 6.0e-14
            assert error < 1.0e-11

    # check diff vs differentiation_matrix
    df = diff(meth, fx, p)
    df_mat = W_mat @ fx

    error = la.norm(df[1:] - df_mat[1:]) / la.norm(df_mat[1:])
    log.info("diff vs differentiation_matrix")
    log.info("   error %.12e", error)
    assert error < 1.0e-12

    # check diff vs diffs
    log.info("diff vs diffs")
    df_ref = df
    assert isinstance(df_ref, np.ndarray)
    for n in [1, p.size // 2 - 5, p.size - 1]:
        df_from_diffs = diffs(meth, fx, p, n)

        error = la.norm(df_from_diffs - df_ref[n]) / la.norm(df_ref[n])
        log.info("   error %.12e", error)
        assert error < 6.0e-12


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
