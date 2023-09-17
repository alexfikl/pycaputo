# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

import numpy as np

from pycaputo.derivatives import HadamardDerivative
from pycaputo.grid import Points
from pycaputo.quadrature import QuadratureMethod, make_method_from_name, quad
from pycaputo.utils import Array, ArrayOrScalarFunction


@dataclass(frozen=True)
class HadamardQuadratureMethod(QuadratureMethod):
    d: HadamardDerivative

    @property
    def name(self) -> str:
        return "Hadamard"

    @property
    def order(self) -> float:
        return 1.0


@quad.register(HadamardQuadratureMethod)
def _quad_hadamard(
    m: HadamardQuadratureMethod,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    fx = f(p.x) if callable(f) else f
    return np.zeros_like(fx)


m = make_method_from_name("HadamardQuadratureMethod", -1.5)
