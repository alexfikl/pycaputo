# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import enum
import math
from dataclasses import dataclass


class Side(enum.Enum):
    """Side of a fractional derivative, if any."""

    #: Denotes the left side, e.g. :math:`[a, x]` for intervals around :math:`x`.
    Left = enum.auto()
    #: Denotes the right side, e.g. :math:`[x, b]` for intervals around :math:`x`.
    Right = enum.auto()


@dataclass(frozen=True)
class FractionalDerivative:
    """Generic type of fractional derivative."""

    #: Order of the fractional derivative, as an integer in :math:[0, \infty]`.
    order: float

    @property
    def n(self) -> int:
        return math.ceil(self.order)


@dataclass(frozen=True)
class RiemannLiouvilleDerivative(FractionalDerivative):
    """Riemann-Liouville fractional order derivatives."""

    #: Side on which to compute the derivative
    side: Side


@dataclass(frozen=True)
class CaputoDerivative(FractionalDerivative):
    """Caputo fractional order derivatives."""

    #: Side on which to compute the derivative
    side: Side
