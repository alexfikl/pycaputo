# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.logging import get_logger
from pycaputo.utils import set_recommended_matplotlib

logger = get_logger("pycaputo.test_caputo")
set_recommended_matplotlib()

# {{{ test_jacobi_polynomials


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [
        (0.0, 0.0),
        (-0.5, -0.5),
        (1.0, 1.0),
        (2.0, 0.5),
    ],
)
def test_jacobi_polynomials(alpha: float, beta: float, rtol: float = 1.0e-13) -> None:
    from scipy.special import jacobi

    from pycaputo.grid import make_jacobi_gauss_lobatto_points, make_uniform_points
    from pycaputo.jacobi import jacobi_gamma, jacobi_polynomial

    N = 32

    # check vs scipy at Jacobi-Gauss-Lobatto points
    p = make_jacobi_gauss_lobatto_points(N, a=-1, b=1, alpha=alpha, beta=beta)

    for n, Pn in enumerate(jacobi_polynomial(p, N, alpha=alpha, beta=beta)):
        Pn_ref = jacobi(n, alpha, beta)(p.x)

        p_norm = np.sum(Pn[1:-1] * p.w[1:-1]) / jacobi_gamma(0, alpha, beta)
        error = la.norm(Pn - Pn_ref) / la.norm(Pn_ref)
        logger.info("order %3d error %.12e norm %.12e", n, error, p_norm)
        assert error < rtol
        assert p_norm < rtol or abs(p_norm - 1.0) < rtol

    # check vs scipy at uniform points
    q = make_uniform_points(N, a=-1, b=1)

    for n, Pn in enumerate(jacobi_polynomial(q, N, alpha=alpha, beta=beta)):
        Pn_ref = jacobi(n, alpha, beta)(q.x)

        error = la.norm(Pn - Pn_ref) / la.norm(Pn_ref)
        logger.info("order %3d error %.12e", n, error)
        assert error < rtol


# }}}


# {{{ test_jacobi_project


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [
        (0.0, 0.0),
        (-0.5, -0.5),
        (1.0, 1.0),
        (2.0, 0.5),
    ],
)
def test_jacobi_project(alpha: float, beta: float, rtol: float = 5.0e-13) -> None:
    from pycaputo.grid import make_jacobi_gauss_lobatto_points
    from pycaputo.jacobi import jacobi_polynomial, jacobi_project

    N = 32

    # check vs scipy at Jacobi-Gauss-Lobatto points
    p = make_jacobi_gauss_lobatto_points(N, a=-1, b=1, alpha=alpha, beta=beta)

    for n, Pn in enumerate(jacobi_polynomial(p, N - 3, alpha=alpha, beta=beta)):
        Phat = jacobi_project(Pn, p)
        Phat_ref = np.zeros_like(Phat)
        Phat_ref[n] = 1.0

        error = la.norm(Phat - Phat_ref)
        logger.info("order %3d error %.12e", n, error)
        assert error < rtol


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
