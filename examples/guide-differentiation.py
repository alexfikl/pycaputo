# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""This example showcases how to create a new differentiation method.

The definition is given by registering a new method using ``diff.register``.
"""

from __future__ import annotations  # noqa: I001

from dataclasses import dataclass

import numpy as np

from pycaputo.derivatives import RiemannLiouvilleDerivative, Side
from pycaputo.grid import Points
from pycaputo.typing import Array, ArrayOrScalarFunction, Scalar

# {{{

# [class-definition-start]
from pycaputo.differentiation import DerivativeMethod


@dataclass(frozen=True)
class RiemannLiouvilleDerivativeMethod(DerivativeMethod):
    alpha: float

    @property
    def name(self) -> str:
        return "RLdiff"

    @property
    def d(self) -> RiemannLiouvilleDerivative:
        return RiemannLiouvilleDerivative(self.alpha, side=Side.Left)
        # [class-definition-end]


# }}}


# {{{

# [register-start]
from pycaputo.differentiation import (
    diff,
    differentiation_matrix,
    diffs,
    quadrature_weights,
)


@quadrature_weights.register(RiemannLiouvilleDerivativeMethod)
def _quadrature_weights_rl(
    m: RiemannLiouvilleDerivativeMethod,
    p: Points,
    n: int,
) -> Array:
    # ... add an actual implementation here ...
    return np.zeros(n, dtype=p.x.dtype)


@differentiation_matrix.register(RiemannLiouvilleDerivativeMethod)
def _differentiation_matrix_rl(
    m: RiemannLiouvilleDerivativeMethod,
    p: Points,
) -> Array:
    # ... add an actual implementation here ...
    return np.zeros((p.size, p.size), dtype=p.x.dtype)


@diffs.register(RiemannLiouvilleDerivativeMethod)
def _diffs_rl(
    m: RiemannLiouvilleDerivativeMethod,
    f: ArrayOrScalarFunction,
    p: Points,
    n: int,
) -> Scalar:
    # ... add an actual implementation here ...
    return np.array(0.0)


@diff.register(RiemannLiouvilleDerivativeMethod)
def _diff_rl(
    m: RiemannLiouvilleDerivativeMethod,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    fx = f(p.x) if callable(f) else f
    # ... add an actual implementation here ...
    return np.zeros_like(fx)
    # [register-end]


# }}}


m = RiemannLiouvilleDerivativeMethod(alpha=0.9)
