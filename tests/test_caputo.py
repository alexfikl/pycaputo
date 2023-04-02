import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.logging import get_logger

logger = get_logger("pycaputo.test_caputo")

# {{{ test_caputo_l1


@pytest.mark.parametrize("order", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_caputo_l1(order: float) -> None:
    from pycaputo import CaputoDerivative, CaputoL1Method, Side, evaluate
    from pycaputo.grid import make_uniform_points

    def f(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    def df(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    side = Side.Left
    diff = CaputoL1Method(CaputoDerivative(order=order, side=side))

    from pycaputo.utils import EOCRecorder

    eoc = EOCRecorder()

    for n in [32, 64, 128, 256, 512]:
        p = make_uniform_points(n)
        df_num = evaluate(diff, f, p)
        df_ref = df(p.x)

        e = la.norm(df_num - df_ref) / la.norm(df_ref)
        eoc.add_data_point(e, p.dx[0])
        logger.info("n %4d h %.5e e %.12e", n, p.dx[0], e)

    logger.info("\n%s", eoc)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
