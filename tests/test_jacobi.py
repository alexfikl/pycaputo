# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.logging import get_logger
from pycaputo.utils import set_recommended_matplotlib

logger = get_logger("pycaputo.test_jacobi")
set_recommended_matplotlib()


# {{{ test_jacobi_weights


def quad_monomial(k: int, a: float, b: float) -> float:
    return (b ** (k + 1) - a ** (k + 1)) / (k + 1)


def test_jacobi_weights() -> None:
    from pycaputo.grid import make_jacobi_gauss_lobatto_points

    a = -4.0
    b = np.pi

    for n in [4, 5, 8, 9, 16, 17]:
        p = make_jacobi_gauss_lobatto_points(n, a=a, b=b)

        # NOTE: JGL should be exact up to 2n - 3, so 2n - 2 is approximate
        for k in range(2 * n - 1):
            q_ref = quad_monomial(k, a=a, b=b)
            q = np.sum(p.x**k * p.w)

            error = abs(q - q_ref) / abs(q_ref)
            logger.error(
                "order %3d monomial %3d %+.12e ref %+.12e error %.12e",
                n,
                k,
                q,
                q_ref,
                error,
            )

            if k == 2 * n - 2:
                assert error > 1.0e-10
            else:
                assert error < 2.0e-14


# }}}


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

    N = 31

    # check vs scipy at Jacobi-Gauss-Lobatto points
    p = make_jacobi_gauss_lobatto_points(N, a=-1, b=1, alpha=alpha, beta=beta)

    for n, Pn in enumerate(jacobi_polynomial(p, N, alpha=alpha, beta=beta)):
        Pn_ref = jacobi(n, alpha, beta)(p.x)

        pn_int = np.sum(Pn * p.w) / jacobi_gamma(0, alpha, beta)
        error = la.norm(Pn - Pn_ref) / la.norm(Pn_ref)
        logger.info("order %3d error %.12e integral %.12e", n, error, pn_int)
        assert error < rtol
        assert pn_int < rtol or abs(pn_int - 1.0) < rtol

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


# {{{ test_jacobi_riemann_liouville_integral


def test_jacobi_riemann_liouville_integral(*, visualize: bool = False) -> None:
    from pycaputo.grid import make_jacobi_gauss_lobatto_points
    from pycaputo.jacobi import jacobi_riemann_liouville_integral

    N = 8
    alpha = beta = 0.0
    p = make_jacobi_gauss_lobatto_points(N, a=-1, b=1, alpha=alpha, beta=beta)

    from scipy.special import poch

    alpha = 2.0

    # NOTE: these are taken from mathematica
    # FIXME: can we make sympy do this? Seems unlikely, but should try it
    x = p.x
    xa = (1 + x) ** alpha
    Phat_ref = [
        # P0
        xa / poch(alpha, 1),
        # P1
        xa * (x - alpha) / poch(alpha, 2),
        # P2
        (xa * (alpha**2 + 3 * x**2 - 3 * alpha * x - 1)) / poch(alpha, 3),
        # P3
        xa
        / poch(alpha, 4)
        * (
            15 * x**3
            - alpha**3
            + 6 * x * alpha**2
            - 15 * x**2 * alpha
            - 9 * x
            + 4 * alpha
        ),
        # P4
        xa
        / poch(alpha, 5)
        * (
            9
            + 105 * x**4
            - 105 * alpha * x**3
            - 10 * alpha**2
            + alpha**4
            + 5 * alpha * x * (11 - 2 * alpha**2)
            + 45 * x**2 * (alpha**2 - 2)
        ),
        # P5
        xa
        / poch(alpha, 6)
        * (
            -64 * alpha
            + 20 * alpha**3
            - alpha**5
            + 225 * x
            - 195 * alpha**2 * x
            + 15 * alpha**4 * x
            + 735 * alpha * x**2
            - 105 * alpha**3 * x**2
            - 1050 * x**3
            + 420 * alpha**2 * x**3
            - 945 * alpha * x**4
            + 945 * x**5
        ),
        # P6
        xa
        / poch(alpha, 7)
        * (
            -225
            + 259 * alpha**2
            - 35 * alpha**4
            + alpha**6
            - 21 * alpha * (99 - 25 * alpha**2 + alpha**4) * x
            + 105 * (45 - 32 * alpha**2 + 2 * alpha**4) * x**2
            - 630 * alpha * (-17 + 2 * alpha**2) * x**3
            + 4725 * (-3 + alpha**2) * x**4
            - 10395 * alpha * x**5
            + 10395 * x**6
        ),
        # P7
        xa
        / poch(alpha, 8)
        * (
            -(
                (-6 + alpha)
                * (-4 + alpha)
                * (-2 + alpha)
                * alpha
                * -(2 + alpha)
                * (4 + alpha)
                * (6 + alpha)
            )
            + 7 * (-1575 + 1516 * alpha**2 - 170 * alpha**4 + 4 * alpha**6) * x
            - 189 * alpha * (283 - 60 * alpha**2 + 2 * alpha**4) * x**2
            + 1575 * (63 - 38 * alpha**2 + 2 * alpha**4) * x**3
            - 17325 * alpha * (-10 + alpha**2) * x**4
            + 31185 * (-7 + 2 * alpha**2) * x**5
            - 135135 * alpha * x**6
            + 135135 * x**7
        ),
    ]

    for n, Phat in enumerate(jacobi_riemann_liouville_integral(p, alpha=alpha)):
        error = la.norm(Phat - Phat_ref[n]) / la.norm(Phat_ref[n])
        logger.info("order %3d error %.12e", n, error)

        if visualize:
            import matplotlib.pyplot as mp

            fig = mp.figure()
            ax = fig.gca()

            ax.plot(p.x, Phat)
            ax.plot(p.x, Phat_ref[n], "k--")
            ax.set_xlabel("$x$")
            ax.set_ylabel(rf"$\hat{{P}}^{{{p.alpha}, {p.beta}}}_{{{n}}}$")

            from pycaputo.utils import savefig

            dirname = pathlib.Path(__file__).parent
            filename = f"test_jacobi_riemann_liouville_integral_{n}"
            savefig(fig, dirname / filename)


# }}}


# {{{ test_jacobi_caputo_derivative


def test_jacobi_caputo_derivative(*, visualize: bool = False) -> None:
    from pycaputo.grid import make_jacobi_gauss_lobatto_points
    from pycaputo.jacobi import jacobi_caputo_derivative

    N = 8
    alpha = beta = 0.0
    p = make_jacobi_gauss_lobatto_points(N, a=-1, b=1, alpha=alpha, beta=beta)

    from scipy.special import gamma

    alpha = 1.5
    m = int(np.ceil(alpha))

    # NOTE: these are also integrated with mathematica
    Dhat_ref = [
        # D0
        0,
        # D1
        0,
        # D2
        6 * np.sqrt(1 + p.x) / gamma(m - alpha),
        # D3
        10 * np.sqrt(1 + p.x) * (2 * p.x - 1) / gamma(m - alpha),
        # D4
        2 * np.sqrt(1 + p.x) * (28 * p.x**2 - 14 * p.x + 3) / gamma(m - alpha),
        # D5
        2
        * np.sqrt(1 + p.x)
        * (72 * p.x**3 - 36 * p.x**2 - 8 * p.x - 5)
        / gamma(m - alpha),
        # D6
        4
        * np.sqrt(1 + p.x)
        * (88 * p.x**4 - 44 * p.x**3 - 30 * p.x**2 + 4 * p.x + 7)
        / gamma(m - alpha),
        # D7
        4
        * np.sqrt(1 + p.x)
        * (208 * p.x**5 - 104 * p.x**4 - 120 * p.x**3 + 34 * p.x**2 + 22 * p.x - 9)
        / gamma(m - alpha),
    ]

    for n, Dhat in jacobi_caputo_derivative(p, alpha=alpha):
        assert n >= m

        error = la.norm(Dhat - Dhat_ref[n]) / la.norm(Dhat_ref[n])
        logger.info("order %3d error %.12e", n, error)

        if visualize:
            import matplotlib.pyplot as mp

            fig = mp.figure()
            ax = fig.gca()

            ax.plot(p.x, Dhat)
            ax.plot(p.x, Dhat_ref[n], "k--")
            ax.set_xlabel("$x$")
            ax.set_ylabel(rf"$\hat{{D}}^{{{p.alpha}, {p.beta}, {alpha}}}_{{{n}}}$")

            from pycaputo.utils import savefig

            dirname = pathlib.Path(__file__).parent
            filename = f"test_jacobi_caputo_derivative_{n}"
            savefig(fig, dirname / filename)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
