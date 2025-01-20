# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.differentiation import diff
from pycaputo.differentiation import grunwald_letnikov as gl
from pycaputo.logging import get_logger
from pycaputo.typing import Array
from pycaputo.utils import get_environ_bool, set_recommended_matplotlib

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent
ENABLE_VISUAL = get_environ_bool("ENABLE_VISUAL")

log = get_logger(f"pycaputo.{TEST_FILENAME.stem}")
set_recommended_matplotlib()

# {{{ test_grunwald_letnikov


def f_test(x: Array, *, mu: float = 3.5) -> Array:
    return np.array(np.cos(mu * x))


def df_test(x: Array, *, alpha: float, mu: float = 3.5) -> Array:
    from pycaputo.derivatives import GrunwaldLetnikovDerivative, Side
    from pycaputo.special import cos_derivative

    d = GrunwaldLetnikovDerivative(alpha=alpha, side=Side.Left)
    return cos_derivative(d, x, t0=x[0], omega=mu)


@pytest.mark.parametrize(
    "name",
    [
        "GrunwaldLetnikov",
        "ShiftedGrunwaldLetnikov",
        "TianZhouDeng2",
        # "TianZhouDeng3",
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_grunwald_letnikov(
    name: str,
    alpha: float,
) -> None:
    r"""
    Test convergence of the LXXX methods for the Riemann-Liouville derivative.
    The convergence is checked in the :math:`\ell^2` norm using :func:`f_test`.
    """

    from pycaputo.grid import make_uniform_points

    meth: gl.GrunwaldLetnikovMethod
    resolutions = [16, 32, 64, 128, 256, 512, 768, 1024]
    if name == "GrunwaldLetnikov":
        meth = gl.GrunwaldLetnikov(alpha=alpha)
        order = 1.0
    elif name == "ShiftedGrunwaldLetnikov":
        shift = gl.ShiftedGrunwaldLetnikov.recommended_shift_for_alpha(alpha)
        assert shift is not None

        meth = gl.ShiftedGrunwaldLetnikov(alpha=alpha, shift=shift)
        order = 2.0
    elif name == "TianZhouDeng2":
        shift2 = gl.TianZhouDeng2.recommended_shift_for_alpha(alpha)
        assert shift2 is not None

        meth = gl.TianZhouDeng2(alpha=alpha, shift=shift2)
        order = 2.0
    elif name == "TianZhouDeng3":
        shift3 = gl.TianZhouDeng3.recommended_shift_for_alpha(alpha)
        assert shift3 is not None

        meth = gl.TianZhouDeng3(alpha=alpha, shift=shift3)
        order = 3.0

        # FIXME: there's a good chance the third order method is buggy and this
        # just works around it. Will need to look at it more carefully!
        resolutions = resolutions[:4]
    else:
        raise ValueError(f"Unsupported method: {name}")

    from pycaputo.utils import EOCRecorder, savefig

    eoc = EOCRecorder(order=order)

    if ENABLE_VISUAL:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    for n in resolutions:
        p = make_uniform_points(n, a=0.0, b=1.0)
        df_num = diff(meth, f_test, p)
        df_ref = df_test(p.x, alpha=alpha)

        h = np.max(p.dx)
        e = la.norm(df_num[1:] - df_ref[1:]) / la.norm(df_ref[1:])
        eoc.add_data_point(h, e)
        log.info("n %4d h %.5e e %.12e", n, h, e)

        if ENABLE_VISUAL:
            ax.plot(p.x[1:], df_num[1:])
            # ax.semilogy(p.x, abs(df_num - df_ref))

    log.info("\n%s", eoc)

    if ENABLE_VISUAL:
        ax.plot(p.x[1:], df_ref[1:], "k--")
        ax.set_xlabel("$x$")
        ax.set_ylabel(rf"$D^{{{alpha}}}_C f$")
        # ax.set_ylim([1.0e-16, 1])

        filename = f"test_rl_{meth.name}_{alpha}"
        savefig(fig, TEST_DIRECTORY / filename, normalize=True)

    assert order - 0.5 < eoc.estimated_order < order + 0.5


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
