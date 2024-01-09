# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

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
class FractionalOperator:
    """Generic type of a fractional order operator.

    Subclasses can define any form of a fractional order derivative or integral.
    This includes classic derivatives, such as the Riemann-Liouville derivative,
    or more modern operators that do not necessarily satisfy all the fractional
    derivative axioms (or can even be reduced to integer order derivatives).

    For a recent review of these operators see [SalesTeodoro2019]_.
    """

    #: Order of the fractional operator, as a real number
    #: :math:`\alpha \in \mathbb{R}`. A positive number would denote a derivative,
    #: while a negative number would denote a fractional integral, as supported.
    order: float

    @property
    def n(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class RiemannLiouvilleDerivative(FractionalOperator):
    r"""Riemann-Liouville fractional order derivative.

    For an order :math:`n - 1 \le \alpha < n`, where :math:`n \in \mathbb{Z}`,
    the lower Riemann-Liouville fractional derivative of a function
    :math:`f: [a, b] \to \mathbb{R}` is given by (see e.g. [Li2020]_)

    .. math::

        D_{RL}^\alpha[f](x) = \frac{1}{\Gamma(n - \alpha)}
            \frac{\mathrm{d}^n}{\mathrm{d} x^n} \int_a^x
            \frac{f(s)}{(x - s)^{\alpha + 1 - n}} \,\mathrm{d}s,

    while the upper Riemann-Liouville fractional derivative is integrated on
    :math:`[x, b]` with a factor of :math:`(-1)^n`.
    """

    #: Side on which to compute the derivative.
    side: Side

    @property
    def n(self) -> int:
        r"""Integer part of the :attr:`~FractionalOperator.order`, i.e.
        :math:`n - 1 \le \text{order} < n`.
        """
        return math.floor(self.order + 1)


@dataclass(frozen=True)
class CaputoDerivative(FractionalOperator):
    r"""Caputo fractional order derivative.

    For an order :math:`n - 1 < \alpha \le n`, where :math:`n \in \mathbb{Z}`,
    the lower Caputo fractional derivative of a function
    :math:`f: [a, b] \to \mathbb{R}` is given by (see e.g. [Li2020]_)

    .. math::

        D_C^\alpha[f](x) = \frac{1}{\Gamma(n - \alpha)} \int_a^x
            \frac{f^{(n)}(s)}{(x - s)^{\alpha + 1 - n}} \,\mathrm{d}s,

    while the upper Caputo fractional derivative is integrated on :math:`[x, b]`.
    Note that for negative orders, the Caputo derivative is defined as the
    Riemann-Liouville integral.
    """

    #: Side on which to compute the derivative.
    side: Side

    @property
    def n(self) -> int:
        r"""Integer part of the :attr:`~FractionalOperator.order`, i.e.
        :math:`n - 1 < \text{order} \le n`.
        """
        return math.ceil(self.order)


@dataclass(frozen=True)
class GrunwaldLetnikovDerivative(FractionalOperator):
    r"""Grünwald-Letnikov fractional order derivative.

    For an order :math:`n - 1 \le \alpha < n`, where :math:`n \in \mathbb{Z}`,
    the lower Grünwald-Letnikov fractional derivative of a function
    :math:`f: [a, b] \to \mathbb{R}` is given by (see e.g. [SalesTeodoro2019]_)

    .. math::

        D_{GL}^\alpha[f](x) = \lim_{h \to 0^+} \frac{1}{h^\alpha}
            \sum_{k = 0}^{N(h)} (-1)^k
            \frac{\Gamma(\alpha + 1)}{\Gamma(k + 1) \Gamma(\alpha + 1 - k)}
            f(x - k h),

    where :math:`N(h) = (x - a) / h`. The upper derivative is similarly defined.
    """

    #: Side on which to compute the derivative.
    side: Side

    @property
    def n(self) -> int:
        r"""Integer part of the :attr:`~FractionalOperator.order`, i.e.
        :math:`n - 1 \le \text{order} < n`.
        """
        return math.floor(self.order + 1)


@dataclass(frozen=True)
class HadamardDerivative(FractionalOperator):
    r"""Hadamard fractional order derivative.

    For an order :math:`n - 1 < \alpha \le n`, where :math:`n \in \mathbb{Z}`,
    the Hadamard fractional derivative of a function :math:`f: [a, b] \to \mathbb{R}`
    is given by (see e.g. [SalesTeodoro2019]_)

    .. math::

        D_{H}^\alpha[f](x) = \frac{\alpha}{\Gamma(1 - \alpha)}
            \frac{\mathrm{d}^n}{\mathrm{d} x^n}
            \int_a^x (\log x - \log s)^{2 + n - \alpha} \frac{f(s)}{s} \,\mathrm{d}s.
    """

    #: Side on which to compute the derivative.
    side: Side
