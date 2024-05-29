# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""This example showcases how to create a new differentiation method.

The definition is given by registering a new method using ``diff.register``.
"""

from dataclasses import dataclass

import numpy as np

from pycaputo.derivatives import RiemannLiouvilleDerivative, Side
from pycaputo.differentiation import DerivativeMethod, diff
from pycaputo.grid import Points
from pycaputo.utils import Array, ArrayOrScalarFunction


@dataclass(frozen=True)
class RiemannLiouvilleDerivativeMethod(DerivativeMethod):
    alpha: float

    @property
    def name(self) -> str:
        return "RLdiff"

    @property
    def d(self) -> RiemannLiouvilleDerivative:
        return RiemannLiouvilleDerivative(self.alpha, side=Side.Left)


@diff.register(RiemannLiouvilleDerivativeMethod)
def _diff_rl(
    m: RiemannLiouvilleDerivativeMethod,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    fx = f(p.x) if callable(f) else f
    return np.zeros_like(fx)


m = RiemannLiouvilleDerivativeMethod(alpha=0.9)
