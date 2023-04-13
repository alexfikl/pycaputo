# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from pycaputo.logging import get_logger

logger = get_logger("pycaputo.test_mittag_leffler")


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
    from pycaputo.mittagleffler import _mittag_leffler_series, mittag_leffler

    rng = np.random.default_rng(seed=42)
    z = rng.random(128)

    result_ref = mittag_leffler(z, alpha=alpha, beta=beta)
    result = np.vectorize(_mittag_leffler_series)(z, alpha=alpha, beta=beta)

    error = np.linalg.norm(result - result_ref) / np.linalg.norm(result_ref)
    logger.info("Error E[%g, %g]: %.12e", alpha, beta, error)
    assert error < 2.0e-15


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
