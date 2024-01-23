# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pytest

from pycaputo.logging import get_logger, stringify_table
from pycaputo.utils import Array, ScalarFunction, set_recommended_matplotlib

logger = get_logger("pycaputo.test_misc")
dirname = pathlib.Path(__file__).parent
set_recommended_matplotlib()


# {{{ test_lipschitz_uniform_sample


def test_lipschitz_uniform_sample(*, visualize: bool = False) -> None:
    from pycaputo.lipschitz import uniform_diagonal_sample

    a = -1.0
    b = +1.0
    n = 128
    delta = 0.0625 * np.sqrt(2) * (b - a)

    rng = np.random.default_rng(seed=42)
    x, y = uniform_diagonal_sample(a, b, n, delta=delta, rng=rng)
    assert np.all(np.abs(x - y) <= delta)
    assert np.all(np.logical_and(a <= x, x <= b))
    assert np.all(np.logical_and(a <= y, y <= b))

    if not visualize:
        return

    from matplotlib.patches import Rectangle

    from pycaputo.utils import figure

    with figure(dirname / "test_lipschitz_uniform_sample") as fig:
        ax = fig.gca()

        mask = np.abs(x - y) <= delta
        ax.plot(x[mask], y[mask], "o")
        ax.plot(x[~mask], y[~mask], "o")

        ax.add_patch(Rectangle((a, a), b - a, b - a, facecolor="none", edgecolor="red"))
        ax.set_xlim([a - delta, b + delta])
        ax.set_ylim([a - delta, b + delta])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")


# }}}


# {{{ test_estimate_lischitz_constant


def f_wood_1(x: Array) -> Array:
    return np.array(x - x**3 / 3)


def f_wood_2(x: Array) -> Array:
    return np.array(np.sin(x) + np.sin(2 * x / 3))


def f_wood_3(x: Array) -> Array:
    return -np.array(sum(np.sin((i + 1) * x + i) for i in range(1, 6)))


@pytest.mark.parametrize(
    ("f", "L", "a", "b"),
    [
        (f_wood_1, 1.0, -1.0, 1.0),
        (f_wood_2, 1.67, 3.1, 20.4),
        # NOTE: Wood1996 says this function has L = 67, but that's not true
        (f_wood_3, 19.2, -10.0, 10.0),
    ],
)
def test_estimate_lischitz_constant(
    f: ScalarFunction,
    L: float,
    a: float,
    b: float,
    *,
    visualize: bool = False,
) -> None:
    nbatches = [25, 50, 75, 100]
    nslopes = [3, 5, 7, 9, 11]
    delta = 0.05

    from rich.table import Table

    table = Table(f"L = {L}", *[f"m = {m}" for m in nbatches])
    lipschitz_approx = np.empty(len(nbatches))

    from pycaputo.lipschitz import estimate_lipschitz_constant

    for n in nslopes:
        for i, m in enumerate(nbatches):
            rng = np.random.default_rng(seed=42)
            lipschitz_approx[i] = estimate_lipschitz_constant(
                f,
                a,
                b,
                delta=delta,
                nslopes=n,
                nbatches=m,
                rng=rng,
            )

            error = abs(L - lipschitz_approx[i]) / abs(L)
            logger.info(
                "n %3d m %3d L %.5e Lapprox %.5e error %.5e",
                n,
                m,
                L,
                lipschitz_approx[i],
                error,
            )
            assert error < 6.0e-2

        table.add_row(f"n = {n}", *[f"{Lapprox:.5e}" for Lapprox in lipschitz_approx])

    logger.info("Results:\n%s", stringify_table(table))


# }}}


# {{{


class MyClass:
    def __init__(self, value: float) -> None:
        self.value = value


@dataclass(frozen=True)
class MyFrozenClass:
    value: float


@pytest.mark.parametrize("cls", [MyClass, MyFrozenClass])
def test_cached_on_first_arg(cls: type) -> None:
    from pycaputo.utils import cached_on_first_arg

    flag = [0]

    @cached_on_first_arg
    def cached_func(o: object) -> int:
        flag[0] = flag[0] + 1
        return flag[0]

    o = cls(3.0)
    cached_func(o)
    assert flag[0] == 1

    cached_func(o)
    assert flag[0] == 1

    cached_func(o)
    assert flag[0] == 1

    cached_func.clear_cached(o)  # type: ignore[attr-defined]
    cached_func(o)
    assert flag[0] == 2


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
