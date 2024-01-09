# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

from pycaputo.finite_difference import (
    DiffStencil,
    apply_derivative,
    determine_stencil_truncation_error,
    make_taylor_approximation,
    modified_wavenumber,
)
from pycaputo.logging import get_logger
from pycaputo.utils import savefig, set_recommended_matplotlib

logger = get_logger("pycaputo.test_finite_difference")
set_recommended_matplotlib()

# {{{ test_finite_difference_taylor


def finite_difference_convergence(d: DiffStencil) -> float:
    from pycaputo.utils import EOCRecorder

    eoc = EOCRecorder()

    s = np.s_[abs(d.offsets[0]) + 1 : -abs(d.offsets[-1]) - 1]
    for n in [32, 64, 128, 256, 512]:
        theta = np.linspace(0.0, 2.0 * np.pi, n, dtype=d.coeffs.dtype)
        h = theta[1] - theta[0]

        f = np.sin(theta)
        num_df_dx = apply_derivative(d, f, h)

        df = np.cos(theta) if d.derivative % 2 == 1 else np.sin(theta)
        df_dx = (-1.0) ** ((d.derivative - 1) // 2 + 1) * df

        error = np.linalg.norm(df_dx[s] - num_df_dx[s]) / np.linalg.norm(df_dx[s])
        eoc.add_data_point(h, error)

    logger.info("\n%s", eoc)
    return eoc.estimated_order


def test_finite_difference_taylor_stencil(*, visualize: bool = False) -> None:
    stencils = [
        (
            make_taylor_approximation(1, (-2, 2)),
            np.array([1 / 12, -8 / 12, 0.0, 8 / 12, -1 / 12]),
            4,
            -1 / 30,
        ),
        (
            make_taylor_approximation(1, (-2, 1)),
            np.array([1 / 6, -6 / 6, 3 / 6, 2 / 6]),
            3,
            1 / 12,
        ),
        (
            make_taylor_approximation(1, (-1, 2)),
            np.array([-2 / 6, -3 / 6, 6 / 6, -1 / 6]),
            3,
            -1 / 12,
        ),
        (
            make_taylor_approximation(2, (-2, 1)),
            np.array([0.0, 1.0, -2.0, 1.0]),
            2,
            1 / 12,
        ),
        (
            make_taylor_approximation(2, (-2, 2)),
            np.array([-1 / 12, 16 / 12, -30 / 12, 16 / 12, -1 / 12]),
            4,
            -1 / 90,
        ),
        (
            make_taylor_approximation(3, (-2, 2)),
            np.array([-1 / 2, 2 / 2, 0.0, -2 / 2, 1 / 2]),
            2,
            1 / 4,
        ),
        (
            make_taylor_approximation(4, (-2, 2)),
            np.array([1.0, -4.0, 6.0, -4.0, 1.0]),
            2,
            1 / 6,
        ),
    ]

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()

    for s, a, order, coefficient in stencils:
        logger.info("stencil:\n%r", s)

        assert np.allclose(np.sum(s.coeffs), 0.0)
        assert np.allclose(s.coeffs, np.array(a, dtype=s.coeffs.dtype))
        assert np.allclose(s.trunc.error, coefficient)
        assert s.trunc.order == order

        estimated_order = finite_difference_convergence(s)
        assert estimated_order >= order - 0.25

        if visualize:
            part = np.real if s.derivative % 2 == 0 else np.imag

            k = np.linspace(0.0, np.pi, 128)
            km = part(modified_wavenumber(s, k))
            sign = part(1.0j**s.derivative)

            ax = fig.gca()
            ax.plot(k, km)
            ax.plot(k, sign * k**s.derivative, "k--")

            ax.set_xlabel("$k h$")
            ax.set_ylabel(r"$\tilde{k} h$")
            ax.set_xlim(0.0, float(np.pi))
            ax.set_ylim(0.0, float(sign * np.pi**s.derivative))

            dirname = pathlib.Path(__file__).parent
            filename = f"test_diff_fd_{s.derivative}_{s.trunc.order}"
            savefig(fig, dirname / filename)

    if visualize:
        mp.close(fig)

    a = np.array([
        -0.02651995,
        0.18941314,
        -0.79926643,
        0.0,
        0.79926643,
        -0.18941314,
        0.02651995,
    ])
    offsets = np.arange(-3, 4)
    s = DiffStencil(derivative=1, coeffs=a, offsets=offsets)

    order, c = determine_stencil_truncation_error(s, atol=1.0e-6)
    assert order == 4
    assert np.allclose(c, 0.01970656333333333)


# }}}
