# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.logging import get_logger
from pycaputo.utils import get_environ_bool, set_recommended_matplotlib

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent
ENABLE_VISUAL = get_environ_bool("ENABLE_VISUAL")

log = get_logger(f"pycaputo.{TEST_FILENAME.stem}")
set_recommended_matplotlib()


# {{{ test_vandermonde_matrix


def test_vandermonde_matrix() -> None:
    """
    Check that we correctly compute the Vandermonde matrix.
    """

    from pycaputo.grid import make_jacobi_gauss_lobatto_points

    n = 24
    a, b = -1.0, 1.0
    p = make_jacobi_gauss_lobatto_points(n, a, b).x

    from pycaputo.lagrange import lagrange_polynomials, vandermonde_inverse

    rng = np.random.default_rng(seed=42)
    A = vandermonde_inverse(p)
    kappa = la.cond(A)
    log.info("kappa: %.12e", kappa)

    for i, poly in enumerate(lagrange_polynomials(p)):
        x = rng.uniform(a, b, size=32)
        mu = np.arange(n).reshape(-1, 1)

        f_lagrange = poly(x)
        f_vdm = A[:, i] @ (x**mu)

        error = la.norm(f_vdm - f_lagrange) / la.norm(f_lagrange)
        log.info("Error: L_%d %.12e", i, error)
        assert error < 5.0e-16 * kappa


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
