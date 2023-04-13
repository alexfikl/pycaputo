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
    """A function and its Caputo fractional order derivative.

    .. automethod:: __call__
    """

    @abstractmethod
    def diff(self, x: Array) -> Array:
        """Evaluate the Caputo fractional order derivative at *x*."""

    @abstractmethod
    def __call__(self, x: Array) -> Array:
        """Evaluate the function at *x*."""


class CaputoConstantDerivative(CaputoFunctionDerivative):
    """The constant function :math:`f(x) = a`."""

    #: Value of the constant function.
    value: float

    def __call__(self, x: Array) -> Array:
        return np.full_like(x, self.value)

    def diff(self, x: Array) -> Array:
        return np.zeros_like(x)


class CaputoPolynomialDerivative(CaputoFunctionDerivative):
    """A polynomial of integer order :math:`f(x) = a_i x^{p_i}`."""

    #: A sequence of ``(coefficient, power)`` tuples that define the polynomial.
    a: Tuple[Tuple[float, float], ...]

    def __call__(self, x: Array) -> Array:
        return sum([a * x**p for a, p in self.a], np.zeros_like(x))

    def diff(self, x: Array) -> Array:
        alpha = self.order
        n = self.n

        # NOTE: this also handles constants
        # FIXME: is this right? Found some cases where `p \notin N`?
        return sum(
            [
                a
                * (
                    np.zeros_like(x)
                    if p < n
                    else math.gamma(p) / math.gamma(p - alpha) * x ** (p - alpha - 1)
                )
                for a, p in self.a
            ],
            np.zeros_like(x),
        )


# TODO:
# * sine and cosine
# * exponential
# * Mittag-Leffler

# }}}
