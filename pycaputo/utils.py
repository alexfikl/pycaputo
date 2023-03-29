# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from typing import Protocol

import numpy as np


class ScalarFunction(Protocol):
    """A generic callable that can be evaluated at :math:`x`."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        ...
