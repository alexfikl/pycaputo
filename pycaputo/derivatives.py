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
    r"""Riemann-Liouville fractional order derivatives.

    For an order :math:`n - 1 < \alpha \le n`, where :math:`n \in \mathbb{N}_+`,
    the lower Riemann-Liouville fractional derivative of a function
    :math:`f: [a, b] \to \mathbb{R}` is given by (see e.g. [Li2020]_)

    .. math::

        D_{RL}^\alpha[f](x) = \frac{1}{\Gamma(n - \alpha)}
            \frac{\mathrm{d}^n}{\mathrm{d} x^n} \int_a^x
            \frac{f(s)}{(x - s)^{\alpha + 1 - n}} \,\mathrm{d}s,

    while the upper Riemann-Liouville fractional derivative is integrated on
    :math:`[x, b]`.
    """

    #: Side on which to compute the derivative
    side: Side


@dataclass(frozen=True)
class CaputoDerivative(FractionalDerivative):
    r"""Caputo fractional order derivatives.

    For an order :math:`n - 1 < \alpha \le n`, where :math:`n \in \mathbb{N}_+`,
    the lower Caputo fractional derivative of a function
    :math:`f: [a, b] \to \mathbb{R}` is given by (see e.g. [Li2020]_)

    .. math::

        D_C^\alpha[f](x) = \frac{1}{\Gamma(n - \alpha)} \int_a^x
            \frac{f^{(n)}(s)}{(x - s)^{\alpha + 1 - n}} \,\mathrm{d}s,

    while the upper Caputo fractional derivative is integrated on :math:`[x, b]`.
    """

    #: Side on which to compute the derivative
    side: Side
