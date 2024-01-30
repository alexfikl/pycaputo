# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from pycaputo import mittagleffler as ml
from pycaputo.logging import get_logger
from pycaputo.utils import set_recommended_matplotlib

logger = get_logger("pycaputo.test_mittag_leffler")
set_recommended_matplotlib()


# {{{ test_mittag_leffler_series


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [
        (0, 1),
        (0, 3),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (0.5, 1),
        (1, 2),
        (2, 2),
    ],
)
def test_mittag_leffler_series(alpha: float, beta: float) -> None:
    rng = np.random.default_rng(seed=42)
    z = rng.random(128)

    result_ref = ml.mittag_leffler(z, alpha=alpha, beta=beta)
    result = np.vectorize(
        lambda zi: 0j + ml.mittag_leffler_series(zi, alpha=alpha, beta=beta)
    )(z)

    error = np.linalg.norm(result - result_ref) / (1 + np.linalg.norm(result_ref))
    logger.info("Error E[%g, %g]: %.12e", alpha, beta, error)
    assert error < 2.0e-15


# }}}


# {{{ test_mittag_leffler_diethelm


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [
        (0, 1),
        (0, 3),
        (1, 1),
        # FIXME: typo?
        # (2, 1),
        (3, 1),
        (4, 1),
        (0.5, 1),
        (1, 2),
        (2, 2),
    ],
)
def test_mittag_leffler_diethelm(alpha: float, beta: float) -> None:
    rng = np.random.default_rng(seed=42)
    z = rng.random(128)

    result_ref = ml.mittag_leffler(z, alpha=alpha, beta=beta)
    result = np.vectorize(
        lambda zi: 0j + ml.mittag_leffler_diethelm(zi, alpha=alpha, beta=beta)
    )(z)

    error = np.linalg.norm(result - result_ref) / (1 + np.linalg.norm(result_ref))
    logger.info("Error E[%g, %g]: %.12e", alpha, beta, error)

    # NOTE: due to the numerical quadrature, this can fail sometimes
    assert error < 2.0e-11


# }}}


# {{{


@pytest.mark.parametrize("iref", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("alg", [ml.Algorithm.Series, ml.Algorithm.Diethelm])
def test_mittag_leffler_mathematica(
    iref: int, alg: ml.Algorithm, *, visualize: bool = True
) -> None:
    from mittag_leffler_ref import MATHEMATICA_RESULTS

    ref = MATHEMATICA_RESULTS[iref]
    is_on_unit_disk = np.all(np.abs(ref.z) <= 1.0)

    if alg == ml.Algorithm.Series and not is_on_unit_disk:
        pytest.skip("Series representation is not valid for z >= 1")

    result = ml.mittag_leffler(ref.z, alpha=ref.alpha, beta=ref.beta)
    error = np.linalg.norm(result - ref.result) / np.linalg.norm(ref.result)
    logger.info("Error E[%g, %g]: %.12e", ref.alpha, ref.beta, error)
    assert error < 1.0e-5


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
