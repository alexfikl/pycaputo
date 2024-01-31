# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib
from functools import partial
from typing import Callable

import numpy as np
import pytest

from pycaputo import mittagleffler as ml
from pycaputo.logging import get_logger
from pycaputo.utils import set_recommended_matplotlib

dirname = pathlib.Path(__file__).parent
logger = get_logger("pycaputo.test_mittag_leffler")
set_recommended_matplotlib()


# {{{ test_mittag_leffler_series


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [
        (0, 1),
        (0, 3),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (0.5, 1),
        (1, 2),
        (2, 2),
    ],
)
def test_mittag_leffler_series(alpha: float, beta: float) -> None:
    rng = np.random.default_rng(seed=42)
    z = rng.random(128)

    result_ref = ml.mittag_leffler(z, alpha=alpha, beta=beta)
    result = np.vectorize(
        lambda zi: 0j + ml.mittag_leffler_series(zi, alpha=alpha, beta=beta)
    )(z)

    error = np.linalg.norm(result - result_ref) / (1 + np.linalg.norm(result_ref))
    logger.info("Error E[%g, %g]: %.12e", alpha, beta, error)
    assert error < 2.0e-15


# }}}


# {{{ test_mittag_leffler_diethelm


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [
        (0, 1),
        (0, 3),
        (1, 1),
        # FIXME: typo?
        # (2, 1),
        (3, 1),
        (4, 1),
        (0.5, 1),
        (1, 2),
        (2, 2),
    ],
)
def test_mittag_leffler_diethelm(alpha: float, beta: float) -> None:
    rng = np.random.default_rng(seed=42)
    z = rng.random(128)

    result_ref = ml.mittag_leffler(z, alpha=alpha, beta=beta)
    result = np.vectorize(
        lambda zi: 0j + ml.mittag_leffler_diethelm(zi, alpha=alpha, beta=beta)
    )(z)

    error = np.linalg.norm(result - result_ref) / (1 + np.linalg.norm(result_ref))
    logger.info("Error E[%g, %g]: %.12e", alpha, beta, error)

    # NOTE: due to the numerical quadrature, this can fail sometimes
    assert error < 2.0e-11


# }}}


# {{{ test_mittag_leffler_mathematica


@pytest.mark.parametrize("iref", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("alg", [ml.Algorithm.Series, ml.Algorithm.Diethelm])
def test_mittag_leffler_mathematica(iref: int, alg: ml.Algorithm) -> None:
    from mittag_leffler_ref import MATHEMATICA_RESULTS

    ref = MATHEMATICA_RESULTS[iref]
    is_on_unit_disk = np.all(np.abs(ref.z) <= 1.0)

    if alg == ml.Algorithm.Series and not is_on_unit_disk:
        pytest.skip("Series representation is not valid for z >= 1")

    result = ml.mittag_leffler(ref.z, alpha=ref.alpha, beta=ref.beta)
    error = np.linalg.norm(result - ref.result) / np.linalg.norm(ref.result)
    logger.info("Error E[%g, %g]: %.12e", ref.alpha, ref.beta, error)
    assert error < 1.0e-5


# }}}


# {{{ test_mittag_leffler_opt


def opt_find_bracket(
    f: Callable[[float], float], a: float = 0.0, b: float = 10.0
) -> tuple[float, float]:
    t = np.linspace(a, b, 32)

    fprev = f(t[0])
    n = 1
    while n < t.size:
        fnext = f(t[n])
        if fprev * fnext < 0.0:
            return t[n - 1], t[n]

        n += 1

    return a, b


def opt_func(t: float, a: float, b: float, *, alpha: float) -> float:
    r"""
    .. math::

        f(t) = a - b * E_\alpha(-t^\alpha)
    """
    # result = a - b * mittleff(alpha, 1.0, -(t**alpha))
    result = a - b * ml.mittag_leffler(
        -(t**alpha),
        alpha=alpha,
        beta=1.0,
        alg=ml.Algorithm.Diethelm,
    )
    return float(np.real_if_close(result))


@pytest.mark.parametrize("alpha", [0.9, 0.95])
def test_mittag_leffler_opt(alpha: float, *, visualize: bool = False) -> None:
    rng = np.random.default_rng(seed=42)

    import scipy.optimize as so

    for a, m in [
        # FIXME: none of these seem to go over the unit disk, so they don't really
        # test a good chunk of the function values.
        (430, -58),
        (342, -50),
        (152, -46),
        (230, -58),
        (40, -58),
        (285, -47),
        (100, -48),
    ]:
        b = a - rng.uniform(m - 2, m + 2)
        f = partial(opt_func, a=a, b=b, alpha=alpha)
        bracket = opt_find_bracket(f)
        result = so.root_scalar(f, x0=(bracket[0] + bracket[1]) / 2, bracket=bracket)

        fstar = f(result.root)
        logger.info(
            "f(t) = %+.12e t = %.12e bracket [%.8e, %.8e]",
            fstar,
            result.root,
            *bracket,
        )
        assert abs(fstar) < 1.0e-10

    if visualize:
        from pycaputo.utils import figure

        t = np.linspace(bracket[0], bracket[1], 256)
        f = np.vectorize(opt_func)(t, a, b, alpha=alpha)

        suffix = str(alpha).replace(".", "_")
        with figure(dirname / f"test_mittag_leffler_opt_{suffix}") as fig:
            ax = fig.gca()
            ax.plot(t, f)
            ax.plot(result.root, fstar, "ro", ms=10)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
