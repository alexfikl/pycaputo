# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

import numpy as np

from pycaputo.derivatives import RiemannLiouvilleDerivative
from pycaputo.differentiation import DerivativeMethod, diff, make_method_from_name
from pycaputo.grid import Points
from pycaputo.utils import Array, ArrayOrScalarFunction


@dataclass(frozen=True)
class RiemannLiouvilleDerivativeMethod(DerivativeMethod):
    d: RiemannLiouvilleDerivative

    @property
    def name(self) -> str:
        return "RLdiff"

    @property
    def order(self) -> float:
        return 1.0

    def supports(self, alpha: float) -> bool:
        return 0.0 < self.d.order < 1.0


@diff.register(RiemannLiouvilleDerivativeMethod)
def _diff_rl(
    m: RiemannLiouvilleDerivativeMethod,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    fx = f(p.x) if callable(f) else f
    return np.zeros_like(fx)


m = make_method_from_name("RiemannLiouvilleDerivativeMethod", 0.5)
