# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from functools import partial

import numpy as np
import pytest

from pycaputo.logging import get_logger
from pycaputo.utils import Array, set_recommended_matplotlib

logger = get_logger("pycaputo.test_integrate_fire")
dirname = pathlib.Path(__file__).parent
set_recommended_matplotlib()

# {{{ test_ad_ex_parameters


def test_ad_ex_parameters() -> None:
    from pycaputo.integrate_fire.ad_ex import AD_EX_PARAMS, AdEx

    alpha = (0.77, 0.31)
    for name, dim in AD_EX_PARAMS.items():
        ad_ex = AdEx.from_dimensional(dim, alpha)
        assert all(np.all(np.isfinite(np.array(v))) for v in ad_ex)
        assert str(dim)
        assert str(ad_ex)

        logger.info("[%s] Parameters:\n%s\n%s", name, dim, ad_ex)


# }}}


# {{{ test_ad_ex_lambert_arg


def ad_ex_lambert_zero_roots(
    a: float, tau_w: float, alpha: float
) -> tuple[float, float]:
    r"""Computes the zeros of :func:`ad_ex_zero` below.

    This is only valid if :math:`h \in [0, 1]`. In other case, we will get
    negative values that do not make any sense.
    """
    from math import sqrt

    a = 1 + a
    b = 1 + tau_w
    c = tau_w

    if a == 0:
        return -c / b, -c / b

    h_min = -b / (2 * a)
    delta = b**2 - 4 * a * c
    if delta <= 0:
        return np.nan, np.nan

    delta = sqrt(delta) / (2 * a)
    return h_min - delta, h_min + delta


def ad_ex_zero(a: float, tau_w: float, h: Array) -> Array:
    return (1 + (1 + a * h / (h + tau_w)) * h) / h


def test_ad_ex_lambert_arg(*, visualize: bool = True) -> None:
    from math import gamma

    from pycaputo.integrate_fire.ad_ex import AD_EX_PARAMS, AdEx

    dirname = pathlib.Path(__file__).parent
    alpha = 0.9
    a, b = 0, gamma(2 - alpha)

    for name, dim in AD_EX_PARAMS.items():
        ad_ex = AdEx.from_dimensional(dim, alpha)
        logger.info("[%s] Parameters: a %.12e tau_w %.12e", name, ad_ex.a, ad_ex.tau_w)

        hm, hp = ad_ex_lambert_zero_roots(ad_ex.a, ad_ex.tau_w, alpha)
        hmin = (hp + hm) / 2.0
        logger.info("[%s] hp %.12e hm %.12e hmin %.12e", name, hp, hm, hmin)
        logger.info("[%s] Works? %s", name, a < hm < b or a < hp < b)

        func = partial(ad_ex_zero, ad_ex.a, ad_ex.tau_w)
        assert np.isnan(hm) or abs(func(hm)) <= 5.0e-14, func(hm)
        assert np.isnan(hp) or abs(func(hp)) <= 5.0e-14, func(hp)

        if visualize:
            from pycaputo.utils import figure

            if np.isnan(hm):
                hm = hp = hmin = 1.0

            h = np.linspace(min(hm, 0), max(hp, 1), 256)
            with figure(dirname / f"test_ad_ex_lambert_arg_{name}") as fig:
                ax = fig.gca()

                ax.plot(h, func(h))
                ax.plot([hm, hmin, hp], [func(hm), func(hmin), func(hp)], "ro")


# }}}


# {{{ test_ad_ex_solve


def test_ad_ex_solve() -> None:
    from pycaputo.integrate_fire.ad_ex import (
        AD_EX_PARAMS,
        AdExModel,
        ad_ex_solve,
        get_ad_ex_parameters,
    )

    rng = np.random.default_rng(seed=42)
    dt = 1.0e-2
    alpha = (0.9, 0.71)

    from math import gamma

    for name in AD_EX_PARAMS:
        param = get_ad_ex_parameters(name, alpha)
        ad_ex = AdExModel(param)

        for _ in range(16):
            t = dt
            y0 = np.array([rng.uniform(param.v_reset, param.v_peak), rng.uniform()])
            h = np.array([gamma(2 - a) * dt**a for a in alpha])
            r = np.array([dt**a / gamma(1 - a) * yi for a, yi in zip(alpha, y0)])

            y = ad_ex_solve(ad_ex, t, y0, h, r)
            error = y - h * ad_ex.source(t, y) - r
            e_imag = np.linalg.norm(error.imag)
            e_real = np.linalg.norm(error.real)

            assert e_imag < 1.0e-15
            assert e_real < 1.0e-15

        logger.info("[%s] real %.12e imag %.12e", name, e_real, e_imag)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
