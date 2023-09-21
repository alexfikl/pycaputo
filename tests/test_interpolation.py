# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo.interpolation import (
    InterpStencil,
    apply_interpolation,
    make_lagrange_approximation,
)
from pycaputo.logging import get_logger
from pycaputo.utils import set_recommended_matplotlib

logger = get_logger("pycaputo.test_interpolation")
set_recommended_matplotlib()


# {{{ test_interpolation_lagrange


def interpolation_convergence(s: InterpStencil) -> float:
    from pycaputo.utils import EOCRecorder

    eoc = EOCRecorder()

    k = np.s_[abs(s.offsets[0]) + 1 : -abs(s.offsets[-1]) - 1]
    for n in [32, 64, 128, 256, 512]:
        theta = np.linspace(0.0, 2.0 * np.pi, n, dtype=s.coeffs.dtype)
        theta_m = (theta[1:] + theta[:-1]) / 2.0
        h = theta[1] - theta[0]

        f = np.sin(theta)
        fhat = apply_interpolation(s, f)[:-1]
        f_ref = np.sin(theta_m)

        error = np.linalg.norm(fhat[k] - f_ref[k]) / np.linalg.norm(f_ref[k])
        eoc.add_data_point(h, error)

    logger.info("\n%s", eoc)
    return eoc.estimated_order


def test_interpolation_lagrange(*, visualize: bool = False) -> None:
    stencils = [
        # (
        #     make_lagrange_approximation((0, 1), 0.5),
        #     np.array([1 / 2, 1 / 2]),
        #     2,
        # ),
        # (
        #     make_lagrange_approximation((-1, 1), 0.5),
        #     np.array([-1 / 8, 6 / 8, 3 / 8]),
        #     3,
        # ),
        (
            make_lagrange_approximation((-1, 2), -0.5),
            np.array([5 / 16, 15 / 16, -5 / 16, 1 / 16]),
            4,
        )
    ]

    for s, a, order in stencils:
        logger.info("stencil:\n%r", s)

        assert np.allclose(np.sum(s.coeffs), 1.0)
        assert np.allclose(s.coeffs, np.array(a, dtype=s.coeffs.dtype))
        assert s.order == order

        estimated_order = interpolation_convergence(s)
        assert estimated_order >= order - 0.25


# }}}
