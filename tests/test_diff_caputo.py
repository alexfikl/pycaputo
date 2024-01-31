# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.differentiation import CaputoDerivativeMethod, diff
from pycaputo.grid import Points
from pycaputo.logging import get_logger
from pycaputo.utils import Array, ScalarFunction, set_recommended_matplotlib

dirname = pathlib.Path(__file__).parent
logger = get_logger("pycaputo.test_diff_caputo")
set_recommended_matplotlib()

# {{{ test_caputo_lmethods


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


@pytest.mark.parametrize(
    ("name", "grid_type"),
    [
        ("CaputoL1Method", "stretch"),
        ("CaputoL1Method", "uniform"),
        ("CaputoModifiedL1Method", "midpoints"),
        ("CaputoL2CMethod", "uniform"),
        ("CaputoL2Method", "uniform"),
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_caputo_lmethods(
    name: str,
    grid_type: str,
    alpha: float,
    *,
    visualize: bool = False,
) -> None:
    from pycaputo.differentiation import make_method_from_name
    from pycaputo.grid import make_points_from_name

    if name in {"CaputoL2Method", "CaputoL2CMethod"}:
        alpha += 1

    from pycaputo.utils import EOCRecorder, savefig

    meth = make_method_from_name(name, alpha)
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

        filename = f"test_caputo_{meth.name}_{alpha}".replace(".", "_")
        savefig(fig, dirname / filename.lower())

    assert eoc.order is not None
    assert eoc.order - 0.25 < eoc.estimated_order < eoc.order + 0.25


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
    *,
    visualize: bool = False,
) -> None:
    from pycaputo.derivatives import CaputoDerivative, Side
    from pycaputo.differentiation import CaputoSpectralMethod
    from pycaputo.grid import make_jacobi_gauss_lobatto_points

    d = CaputoDerivative(alpha, side=Side.Left)
    meth = CaputoSpectralMethod(d)

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

        filename = f"test_caputo_{meth.name}_{j_alpha}_{j_beta}_{alpha}"
        savefig(fig, dirname / filename.replace(".", "_").replace("-", "m").lower())

    assert eoc.estimated_order > 5.0


# }}}


# {{{ test_caputo_vs_differint


@dataclass(frozen=True)
class DifferIntCaputoL1Method(CaputoDerivativeMethod):
    @property
    def order(self) -> float:
        return 2 - self.d.order


@diff.register(DifferIntCaputoL1Method)
def _diff_differint_l1(
    m: DifferIntCaputoL1Method, f: ScalarFunction, p: Points
) -> Array:
    from differint.differint import CaputoL1point

    df = np.empty_like(p.x)
    df[0] = np.nan

    for n in range(1, df.size):
        df[n] = CaputoL1point(
            m.d.order,
            f,
            domain_start=p.a,
            domain_end=p.x[n],
            num_points=n + 1,
        )

    return df


@dataclass(frozen=True)
class DifferIntCaputoL2Method(CaputoDerivativeMethod):
    @property
    def order(self) -> float:
        return 3 - self.d.order


@diff.register(DifferIntCaputoL2Method)
def _diff_differint_l2(
    m: DifferIntCaputoL2Method, f: ScalarFunction, p: Points
) -> Array:
    from differint.differint import CaputoL2point

    df = np.empty_like(p.x)
    df[0] = np.nan

    for n in range(1, df.size):
        df[n] = CaputoL2point(
            m.d.order,
            f,
            domain_start=p.a,
            domain_end=p.x[n],
            num_points=n + 1,
        )

    return df


@dataclass(frozen=True)
class DifferIntCaputoL2CMethod(CaputoDerivativeMethod):
    @property
    def order(self) -> float:
        return 3 - self.d.order


@diff.register(DifferIntCaputoL2CMethod)
def _diff_differint_l2c(
    m: DifferIntCaputoL2CMethod, f: ScalarFunction, p: Points
) -> Array:
    from differint.differint import CaputoL2Cpoint

    df = np.empty_like(p.x)
    df[0] = np.nan

    for n in range(2, df.size - 1):
        df[n] = CaputoL2Cpoint(
            m.d.order,
            f,
            domain_start=p.a,
            domain_end=p.x[n],
            num_points=n + 1,
        )

    return df


@pytest.mark.xfail()
@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("CaputoL1Method", DifferIntCaputoL1Method),
        ("CaputoL2Method", DifferIntCaputoL2Method),
        ("CaputoL2CMethod", DifferIntCaputoL2CMethod),
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_caputo_vs_differint(
    name: str,
    cls: type[CaputoDerivativeMethod],
    alpha: float,
    *,
    visualize: bool = False,
) -> None:
    pytest.importorskip("differint")

    from pycaputo.differentiation import CaputoDerivativeMethod, make_method_from_name

    if name in {"CaputoL2Method", "CaputoL2CMethod"}:
        alpha += 1

    meth = make_method_from_name(name, alpha)
    assert isinstance(meth, CaputoDerivativeMethod)

    differint_meth = cls(d=meth.d)

    from pycaputo.grid import make_points_from_name

    p = make_points_from_name("uniform", 512, a=0.0, b=0.5)

    df_ref = df_test(p.x, alpha=alpha)
    df_num = diff(meth, f_test, p)
    df_num_di = diff(differint_meth, f_test, p)

    error_vs_ref = la.norm(df_num[1:] - df_ref[1:]) / la.norm(df_ref[1:])
    error_di_vs_ref = la.norm(df_num_di[1:] - df_ref[1:]) / la.norm(df_ref[1:])
    error_vs_di = la.norm(df_num[4:-4] - df_num_di[4:-4]) / la.norm(df_num_di[4:-4])
    logger.info(
        "error: vs ref %.12e vs differint %.12e differint vs ref %.12e",
        error_vs_ref,
        error_vs_di,
        error_di_vs_ref,
    )

    if visualize:
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

        filename = f"test_caputo_differint_{meth.name}_{alpha}".replace(".", "_")
        savefig(fig, dirname / filename.lower())

    assert error_vs_ref < 1.0e-2
    assert error_vs_di < 1.0e-12


# }}}

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
