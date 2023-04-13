# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import math
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from pycaputo.derivatives import CaputoDerivative
from pycaputo.utils import Array

# {{{ Caputo derivatives


class CaputoFunctionDerivative(CaputoDerivative, ABC):
    @abstractmethod
    def f(self, x: Array) -> Array:
        pass

    @abstractmethod
    def df(self, x: Array) -> Array:
        pass


class CaputoConstantDerivative(CaputoFunctionDerivative):
    value: float

    def f(self, x: Array) -> Array:
        return np.full_like(x, self.value)

    def df(self, x: Array) -> Array:
        return np.zeros_like(x)


class CaputoPolynomialDerivative(CaputoFunctionDerivative):
    a: Tuple[Tuple[float, float], ...]

    def f(self, x: Array) -> Array:
        return sum([a * x**p for a, p in self.a], 0)

    def df(self, x: Array) -> Array:
        alpha = self.order
        n = self.n

        # NOTE: this also handles constants
        # FIXME: is this right? Found some cases where `p \notin N`?
        return sum(
            [
                a
                * (
                    0
                    if p < n
                    else math.gamma(p) / math.gamma(p - alpha) * x ** (p - alpha - 1)
                )
                for a, p in self.p
            ],
            0,
        )


# TODO:
# * sine and cosine
# * exponential
# * Mittag-Leffler

# }}}
