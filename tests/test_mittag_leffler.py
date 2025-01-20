# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from collections.abc import Callable
from functools import partial

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
    from pymittagleffler import mittag_leffler

    # result = a - b * mittleff(alpha, 1.0, -(t**alpha))
    result = a - b * mittag_leffler(-(t**alpha), alpha=alpha, beta=1.0)
    return float(np.real_if_close(result, tol=1000))


@pytest.mark.parametrize("alpha", [0.9, 0.95])
def test_mittag_leffler_opt(alpha: float) -> None:
    """
    Test an optimization problem with the Mittag-Leffler function.

    This is used in :mod:`pycaputo.integrate_fire.lif` to get the first spike
    time for the Leaky model.
    """

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
        log.info(
            "f(t) = %+.12e t = %.12e bracket [%.8e, %.8e]",
            fstar,
            result.root,
            *bracket,
        )
        assert abs(fstar) < 1.0e-10

    if ENABLE_VISUAL:
        from pycaputo.utils import figure

        t = np.linspace(bracket[0], bracket[1], 256)
        f = np.vectorize(opt_func)(t, a, b, alpha=alpha)

        filename = f"test_mittag_leffler_opt_{alpha}"
        with figure(TEST_DIRECTORY / filename, normalize=True) as fig:
            ax = fig.gca()
            ax.plot(t, f)
            ax.plot(result.root, fstar, "ro", ms=10)


# }}}


# {{{ test_mittag_leffler_sine_mathematica


@pytest.mark.parametrize("iref", [0, 1])
def test_mittag_leffler_sine_mathematica(iref: int) -> None:
    """
    Check the evaluation of the Caputo derivative of the Sine function against
    known results from Mathematica.
    """
    from mittag_leffler_ref import MATHEMATICA_SINE_RESULTS

    from pycaputo.derivatives import CaputoDerivative, Side
    from pycaputo.special import _sin_derivative_caputo  # noqa: PLC2701

    ref = MATHEMATICA_SINE_RESULTS[iref]

    d = CaputoDerivative(ref.alpha, side=Side.Left)
    result = _sin_derivative_caputo(d, ref.z, omega=1.0)

    error = la.norm(result - ref.result) / la.norm(ref.result)
    log.info("Error D^%g[sin]: %.12e", ref.alpha, error)

    if ENABLE_VISUAL:
        from pycaputo.utils import figure

        filename = f"test_mittag_leffler_sine_{ref.alpha}"
        with figure(TEST_DIRECTORY / filename, normalize=True) as fig:
            ax = fig.gca()

            i = np.argsort(ref.z)
            ax.semilogy(ref.z[i], np.abs(result[i] - ref.result[i]))

            ax.set_xlabel("$t$")
            ax.set_ylabel("$D[sin]$")

    assert error < 2.0e-11


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
