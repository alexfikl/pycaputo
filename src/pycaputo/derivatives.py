# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import enum
import math
from dataclasses import dataclass


class Side(enum.Enum):
    """Side of a fractional derivative, if any."""

    Left = enum.auto()
    """Denotes the left side, e.g. :math:`[a, x]` for intervals around :math:`x`."""
    Right = enum.auto()
    """Denotes the right side, e.g. :math:`[x, b]` for intervals around :math:`x`."""


@dataclass(frozen=True)
class FractionalOperator:
    r"""Generic type of a fractional order operator.

    Subclasses can define any form of a fractional order derivative or integral.
    This includes classic derivatives, such as the Riemann-Liouville derivative,
    or more modern operators that do not necessarily satisfy all the fractional
    derivative axioms (or can even be reduced to integer order derivatives).

    For a recent review of these operators see [SalesTeodoro2019]_.
    """


@dataclass(frozen=True)
class RiemannLiouvilleDerivative(FractionalOperator):
    r"""Riemann-Liouville fractional order derivative.

    For an order :math:`n - 1 \le \alpha < n`, where :math:`n \in \mathbb{Z}`,
    the lower Riemann-Liouville fractional derivative of a function
    :math:`f: [a, b] \to \mathbb{R}` is given by (see e.g. [Li2020]_)

    .. math::

        D_{RL}^\alpha[f](x) = \frac{1}{\Gamma(n - \alpha)}
            \frac{\mathrm{d}^n}{\mathrm{d} x^n} \int_a^x
            \frac{f(s)}{(x - s)^{\alpha - n + 1}} \,\mathrm{d}s,

    while the upper Riemann-Liouville fractional derivative is integrated on
    :math:`[x, b]` with a factor of :math:`(-1)^n`.
    """

    alpha: float
    """Order of the Riemann-Liouville derivative."""

    side: Side
    """Side on which to compute the derivative."""

    @property
    def n(self) -> int:
        r"""Integer part of the :attr:`~RiemannLiouvilleDerivative.alpha`, i.e.
        :math:`n - 1 \le \alpha < n`.
        """
        return math.floor(self.alpha + 1)


@dataclass(frozen=True)
class CaputoDerivative(FractionalOperator):
    r"""Caputo fractional order derivative.

    For an order :math:`n - 1 < \alpha \le n`, where :math:`n \in \mathbb{Z}`,
    the lower Caputo fractional derivative of a function
    :math:`f: [a, b] \to \mathbb{R}` is given by (see e.g. [Li2020]_)

    .. math::

        D_C^\alpha[f](x) = \frac{1}{\Gamma(n - \alpha)} \int_a^x
            \frac{f^{(n)}(s)}{(x - s)^{\alpha - n + 1}} \,\mathrm{d}s,

    while the upper Caputo fractional derivative is integrated on :math:`[x, b]`.
    Note that for negative orders, the Caputo derivative is defined as the
    Riemann-Liouville integral.
    """

    alpha: float
    """Order of the Caputo derivative."""

    side: Side
    """Side on which to compute the derivative."""

    @property
    def n(self) -> int:
        r"""Integer part of the :attr:`~CaputoDerivative.alpha`, i.e.
        :math:`n - 1 < \alpha \le n`.
        """
        return math.ceil(self.alpha)


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

    alpha: float
    """Order of the Grünwald-Letnikov derivative."""

    side: Side
    """Side on which to compute the derivative."""

    @property
    def n(self) -> int:
        r"""Integer part of the :attr:`~GrunwaldLetnikovDerivative.alpha`, i.e.
        :math:`n - 1 \le \alpha < n`.
        """
        return math.floor(self.alpha + 1)


@dataclass(frozen=True)
class HadamardDerivative(FractionalOperator):
    r"""Hadamard fractional order derivative.

    For an order :math:`n - 1 \le \alpha < n`, where :math:`n \in \mathbb{Z}`,
    the Hadamard fractional derivative of a function :math:`f: [a, b] \to \mathbb{R}`
    is given by (see e.g. [SalesTeodoro2019]_)

    .. math::

        D_{H}^\alpha[f](x) = \frac{1}{\Gamma(n - \alpha)}
            \left(x \frac{\mathrm{d}}{\mathrm{d} x}\right)^n
            \int_a^x \frac{(\log x - \log s)^{n + 1 - \alpha}}{s} f(s) \,\mathrm{d}s.
    """

    alpha: float
    """Order of the Hadamard derivative."""

    side: Side
    """Side on which to compute the derivative."""

    @property
    def n(self) -> int:
        r"""Integer part of the :attr:`~HadamardDerivative.alpha`, i.e.
        :math:`n - 1 \le \alpha < n`.
        """
        return math.floor(self.alpha + 1)


@dataclass(frozen=True)
class CaputoHadamardDerivative(FractionalOperator):
    r"""Caputo-Hadamard fractional order derivative.

    For an order :math:`n - 1 < \alpha \le n`, where :math:`n \in \mathbb{Z}`,
    the Caputo-Hadamard fractional derivative of a function :math:`f: [a, b]
    \to \mathbb{R}` is given by

    .. math::

        D_{CH}^\alpha[f](x) = \frac{1}{\Gamma(n - \alpha)}
            \int_a^x \frac{(\log x - \log s)^{n + 1 - \alpha}}{s}
            \left(s \frac{\mathrm{d}}{\mathrm{d} s}\right)^n f(s) \,\mathrm{d}s.
    """

    alpha: float
    """Order of the Caputo-Hadamard derivative."""

    side: Side
    """Side on which to compute the derivative."""

    @property
    def n(self) -> int:
        r"""Integer part of the :attr:`~CaputoHadamardDerivative.alpha`, i.e.
        :math:`n - 1 < \alpha \le n`.
        """
        return math.ceil(self.alpha)
