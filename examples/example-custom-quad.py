# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""This example showcases how to create a new quadrature method.

The definition is given by registering a new method using ``quad.register``.
"""

from dataclasses import dataclass

import numpy as np

from pycaputo.derivatives import HadamardDerivative, Side
from pycaputo.grid import Points
from pycaputo.quadrature import QuadratureMethod, quad
from pycaputo.utils import Array, ArrayOrScalarFunction


@dataclass(frozen=True)
class HadamardQuadratureMethod(QuadratureMethod):
    alpha: float

    @property
    def name(self) -> str:
        return "Hadamard"

    @property
    def d(self) -> HadamardDerivative:
        return HadamardDerivative(self.alpha, side=Side.Left)


@quad.register(HadamardQuadratureMethod)
def _quad_hadamard(
    m: HadamardQuadratureMethod,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    fx = f(p.x) if callable(f) else f
    return np.zeros_like(fx)


d = HadamardQuadratureMethod(0.9)
