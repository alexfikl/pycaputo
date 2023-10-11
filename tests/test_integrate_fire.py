# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pycaputo.logging import get_logger
from pycaputo.utils import set_recommended_matplotlib

logger = get_logger("pycaputo.test_integrate_fire")
dirname = pathlib.Path(__file__).parent
set_recommended_matplotlib()

# {{{ test_ad_ex_naud_parameters


def test_ad_ex_naud_parameters() -> None:
    from pycaputo.integrate_fire.ad_ex import AD_EX_PARAMS, AdEx, get_lambert_time_step

    alpha = (0.77, 0.31)
    for name, dim in AD_EX_PARAMS.items():
        ad_ex = AdEx.from_dimensional(dim, alpha)
        assert all(np.all(np.isfinite(np.array(v))) for v in ad_ex)
        assert str(dim)
        assert str(ad_ex)
        # logger.info("Parameters:\n%s\n%s", dim, ad_ex)

        dt = get_lambert_time_step(ad_ex)
        logger.info("[%s] Time step: %.12e", name, -1.0 if dt is None else dt)
        assert dt is None or np.isfinite(dt)


# }}}


# {{{ test_ad_ex_solve


def test_ad_ex_solve() -> None:
    from pycaputo.integrate_fire.ad_ex import (
        AdExModel,
        ad_ex_solve,
        get_ad_ex_parameters,
    )

    rng = np.random.default_rng(seed=42)
    dt = 1.0e-2
    alpha = 0.9

    from math import gamma

    param = get_ad_ex_parameters("Naud4h", alpha)
    ad_ex = AdExModel(param)

    t = dt
    y0 = np.array([rng.uniform(param.v_reset, param.v_peak), rng.uniform()])
    c = np.array([gamma(2 - alpha) * dt**alpha, gamma(2 - alpha) * dt**alpha])
    r = dt**alpha / gamma(1 - alpha) * y0

    y = ad_ex_solve(ad_ex, t, y0, c, r)
    error = y - c * ad_ex.source(t, y) - r
    error_imag = np.linalg.norm(error.imag)
    error_real = np.linalg.norm(error.real)
    logger.info("Error: %s (real %.12e imag %.12e)", error.real, error_real, error_imag)
    assert error_imag < 1.0e-15
    assert error_real < 7.0e-2


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
