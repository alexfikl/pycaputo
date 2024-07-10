# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.differentiation import diff, grunwald_letnikov
from pycaputo.logging import get_logger
from pycaputo.utils import Array, set_recommended_matplotlib

logger = get_logger("pycaputo.test_diff_grunwald_letnikov")
set_recommended_matplotlib()


# {{{ test_grunwald_letnikov


def f_test(x: Array, *, mu: float = 3.5) -> Array:
    # NOTE: this is a smooth function so that the GL methods get optimal order
    return x**mu


def df_test(x: Array, *, alpha: float, mu: float = 3.5) -> Array:
    from scipy.special import gamma

    return np.array(gamma(1 + mu) / gamma(mu - alpha + 1) * x ** (mu - alpha))


@pytest.mark.parametrize(
    "name",
    [
        "GrunwaldLetnikov",
        "ShiftedGrunwaldLetnikov",
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_grunwald_letnikov(
    name: str,
    alpha: float,
    *,
    visualize: bool = False,
) -> None:
    r"""
    Test convergence of the LXXX methods for the Riemann-Liouville derivative.
    The convergence is checked in the :math:`\ell^2` norm using :func:`f_test`.
    """

    from pycaputo.grid import make_uniform_points

    if name in {"L2", "L2C"}:
        alpha += 1

    meth: grunwald_letnikov.GrunwaldLetnikovMethod
    if name == "GrunwaldLetnikov":
        meth = grunwald_letnikov.GrunwaldLetnikov(alpha=alpha)
        order = 1.0
    elif name == "ShiftedGrunwaldLetnikov":
        shift = grunwald_letnikov.ShiftedGrunwaldLetnikov.optimal_shift_for_alpha(alpha)
        assert shift is not None

        meth = grunwald_letnikov.ShiftedGrunwaldLetnikov(alpha=alpha, shift=shift)
        order = 2.0
    else:
        raise ValueError(f"Unsupported method: {name}")

    from pycaputo.utils import EOCRecorder, savefig

    eoc = EOCRecorder(order=order)

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    for n in [16, 32, 64, 128, 256, 512, 768, 1024]:
        p = make_uniform_points(n, a=0.0, b=0.5)
        df_num = diff(meth, f_test, p)
        df_ref = df_test(p.x, alpha=alpha)

        h = np.max(p.dx)
        e = la.norm(df_num[1:] - df_ref[1:]) / la.norm(df_ref[1:])
        eoc.add_data_point(h, e)
        logger.info("n %4d h %.5e e %.12e", n, h, e)

        if visualize:
            ax.plot(p.x[1:], df_num[1:])
            # ax.semilogy(p.x, abs(df_num - df_ref))

    logger.info("\n%s", eoc)

    if visualize:
        ax.plot(p.x[1:], df_ref[1:], "k--")
        ax.set_xlabel("$x$")
        ax.set_ylabel(rf"$D^{{{alpha}}}_C f$")
        # ax.set_ylim([1.0e-16, 1])

        dirname = pathlib.Path(__file__).parent
        filename = f"test_rl_{meth.name}_{alpha}".replace(".", "_")
        savefig(fig, dirname / filename.lower())

    assert order - 0.25 < eoc.estimated_order < order + 0.25


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
