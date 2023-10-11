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

# {{{


def test_ad_ex_naud_parameters() -> None:
    from pycaputo.integrate_fire.ad_ex import AD_EX_PARAMS, AdEx, get_lambert_time_step

    alpha = (0.77, 0.31)
    for name, dim in AD_EX_PARAMS.items():

        ad_ex = AdEx.from_dimensional(dim, alpha)
        assert all(np.all(np.isfinite(v)) for v in ad_ex)
        assert str(dim)
        assert str(ad_ex)
        # logger.info("Parameters:\n%s\n%s", dim, ad_ex)

        dt = get_lambert_time_step(ad_ex)
        logger.info("[%s] Time step: %.12e", name, -1.0 if dt is None else dt)
        assert dt is None or np.isfinite(dt)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
