# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import numpy.linalg as la
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from pycaputo.differentiation import caputo, diff
from pycaputo.typing import Array


def func(x: Array, *, alpha: float) -> Array:
    return np.array(x**8 - 3 * x ** (4 + alpha / 2) + 9 / 4 * x**alpha)


def func_der_ref(x: Array, *, alpha: float) -> Array:
    from math import gamma

    return np.array(
        40320 / gamma(9 - alpha) * x ** (8 - alpha)
        - 3 * gamma(5 + alpha / 2) / gamma(5 - alpha / 2) * x ** (4 - alpha / 2)
        + 9 / 4 * gamma(1 + alpha)
        + (3 / 2 * x ** (alpha / 2) - x**4) ** 3
        - func(x, alpha=alpha) ** (3 / 2)
    )


@pytest.mark.benchmark(group="diff")
@pytest.mark.parametrize(
    ("name", "grid_type"),
    [
        ("L1", "stretch"),
        ("L1", "uniform"),
        ("L2", "uniform"),
        ("L2C", "uniform"),
        ("ModifiedL1", "midpoints"),
    ],
)
def test_caputo_diff(name: str, grid_type: str, benchmark: BenchmarkFixture) -> None:
    from pycaputo.grid import make_points_from_name

    alpha = 0.9
    n = 1024

    if name in {"L2", "L2C"}:
        alpha += 1

    meth: caputo.CaputoMethod
    if name == "L1":
        meth = caputo.L1(alpha=alpha)
    elif name == "ModifiedL1":
        meth = caputo.ModifiedL1(alpha=alpha)
    elif name == "L2":
        meth = caputo.L2(alpha=alpha)
    elif name == "L2C":
        meth = caputo.L2C(alpha=alpha)
    else:
        raise ValueError(f"Unsupported method: {name}")

    p = make_points_from_name(grid_type, n, a=0.0, b=1.0)

    f_test = func(p.x, alpha=alpha)
    df_ref = func_der_ref(p.x, alpha=alpha)

    def run_diff() -> None:
        for _ in range(32):
            df_test = diff(meth, f_test, p)

        error = la.norm(df_test[1:] - df_ref[1:]) / la.norm(df_ref[1:])
        assert error < 1.0e-2

    benchmark.extra_info["alpha"] = alpha
    benchmark.extra_info["n"] = n
    benchmark(run_diff)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
