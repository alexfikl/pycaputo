# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.logging import get_logger
from pycaputo.utils import Array, set_recommended_matplotlib

logger = get_logger("pycaputo.test_caputo")
set_recommended_matplotlib()

# {{{ test_caputo_l1


@pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_caputo_l1(alpha: float, visualize: bool = True) -> None:
    import math

    from pycaputo import CaputoDerivative, CaputoL1Method, Side, evaluate
    from pycaputo.grid import make_uniform_points

    def f(x: Array) -> Array:
        return (1 + x) ** 3

    def df(x: Array) -> Array:
        return np.array(
            3 * x ** (1 - alpha) / math.gamma(2 - alpha)
            + 6 * x ** (2 - alpha) / math.gamma(3 - alpha)
            + 6 * x ** (3 - alpha) / math.gamma(4 - alpha)
        )

    from pycaputo.utils import EOCRecorder, savefig

    side = Side.Left
    diff = CaputoL1Method(d=CaputoDerivative(order=alpha, side=side))
    eoc = EOCRecorder()

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    for n in [32, 64, 128, 256, 512, 768, 1024]:
        p = make_uniform_points(n, a=0, b=1)
        df_num = evaluate(diff, f, p)
        df_ref = df(p.x)

        e = la.norm(df_num - df_ref) / la.norm(df_ref)
        eoc.add_data_point(e, p.dx[0])
        logger.info("n %4d h %.5e e %.12e", n, p.dx[0], e)

        if visualize:
            ax.plot(p.x, df_num)
            # ax.semilogy(p.x, abs(df_num - df_ref))

    logger.info("\n%s", eoc)

    if visualize:
        ax.plot(p.x, df_ref, "k-")
        ax.set_xlabel("$x$")
        ax.set_ylabel(rf"$D^{{{alpha}}}_C f$")
        # ax.set_ylim([1.0e-16, 1])

        savefig(fig, f"test_caputo_l1_{alpha}")


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
