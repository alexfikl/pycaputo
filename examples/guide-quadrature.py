# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""This example showcases how to create a new quadrature method.

The definition is given by registering a new method using ``quad.register``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pycaputo.derivatives import HadamardDerivative, Side
from pycaputo.typing import Array, ArrayOrScalarFunction, is_scalar_function

if TYPE_CHECKING:
    from pycaputo.grid import Points

# {{{

# [class-definition-start]
from pycaputo.quadrature import QuadratureMethod


@dataclass(frozen=True)
class HadamardQuadratureMethod(QuadratureMethod):
    alpha: float

    @property
    def name(self) -> str:
        return "Hadamard"

    @property
    def d(self) -> HadamardDerivative:
        return HadamardDerivative(self.alpha, side=Side.Left)
        # [class-definition-end]


# }}}


# {{{

# [register-start]
from pycaputo.quadrature import quad


@quad.register(HadamardQuadratureMethod)
def _quad_hadamard(
    m: HadamardQuadratureMethod,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    fx = f(p.x) if is_scalar_function(f) else f
    # ... add an actual implementation here ...
    return np.zeros_like(fx)
    # [register-end]


# }}}

d = HadamardQuadratureMethod(0.9)
