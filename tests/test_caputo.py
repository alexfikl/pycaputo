# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib
from dataclasses import dataclass
from typing import Type

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.differentiation import CaputoDerivativeMethod, diff
from pycaputo.grid import Points
from pycaputo.logging import get_logger
from pycaputo.utils import Array, ScalarFunction, set_recommended_matplotlib

logger = get_logger("pycaputo.test_caputo")
set_recommended_matplotlib()

# {{{ test_caputo_lmethods


def f_test(x: Array) -> Array:
    return (0.5 - x) ** 4


def df_test(x: Array, *, alpha: float) -> Array:
    import math

    if 0 < alpha < 1:
        return np.array(
            -1 / 2 * x ** (1 - alpha) / math.gamma(2 - alpha)
            + 3 * x ** (2 - alpha) / math.gamma(3 - alpha)
            - 12 * x ** (3 - alpha) / math.gamma(4 - alpha)
            + 24 * x ** (4 - alpha) / math.gamma(5 - alpha)
        )

    if 1 < alpha < 2:
        p = 12 + 8 * (x - 2) * x - 7 * alpha + 4 * alpha * x + alpha**2
        return np.array(3 * x ** (2 - alpha) * p / math.gamma(5 - alpha))

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
    visualize: bool = False,
) -> None:
    from pycaputo import make_diff_from_name
    from pycaputo.grid import make_points_from_name

    if name in ("CaputoL2Method", "CaputoL2CMethod"):
        alpha += 1

    from pycaputo.utils import EOCRecorder, savefig

    meth = make_diff_from_name(name, alpha)
    eoc = EOCRecorder(order=meth.order)

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    for n in [16, 32, 64, 128, 256, 512, 768, 1024]:
        p = make_points_from_name(grid_type, n, a=0.0, b=1.0)
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
        filename = f"test_caputo_{meth.name}_{alpha}".replace(".", "_")
        savefig(fig, dirname / filename.lower())

    assert eoc.order is not None
    assert eoc.order - 0.25 < eoc.estimated_order < eoc.order + 0.25


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


@pytest.mark.xfail
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
    cls: Type[CaputoDerivativeMethod],
    alpha: float,
    visualize: bool = False,
) -> None:
    from pycaputo import make_diff_from_name

    if name in ("CaputoL2Method", "CaputoL2CMethod"):
        alpha += 1

    meth = make_diff_from_name(name, alpha)
    differint_meth = cls(d=meth.d)  # type: ignore[attr-defined]

    from pycaputo.grid import make_points_from_name

    p = make_points_from_name("uniform", 512, a=0.0, b=1.0)

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

        dirname = pathlib.Path(__file__).parent
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
