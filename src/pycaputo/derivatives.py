# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import TypeVar


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


FractionalOperatorT = TypeVar("FractionalOperatorT", bound=FractionalOperator)
"""A :class:`~typing.TypeVar` bound to :class:`FractionalOperator`."""


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

    side: Side = Side.Left
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

    side: Side = Side.Left
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
            \sum_{k = 0}^{N(h)} (-1)^k \binom{\alpha}{k} f(x - k h),

    where :math:`N(h)` is a function that goes to infinity as :math:`h \to 0^+`.
    The upper derivative is similarly defined.
    """

    alpha: float
    """Order of the Grünwald-Letnikov derivative."""

    side: Side = Side.Left
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

    side: Side = Side.Left
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

    side: Side = Side.Left
    """Side on which to compute the derivative."""

    @property
    def n(self) -> int:
        r"""Integer part of the :attr:`~CaputoHadamardDerivative.alpha`, i.e.
        :math:`n - 1 < \alpha \le n`.
        """
        return math.ceil(self.alpha)


@dataclass(frozen=True)
class CaputoFabrizioOperator(FractionalOperator):
    r"""Caputo-Fabrizio fractional operator from [Caputo2015]_.

    For an order :math:`n - 1 < \alpha \le n`, where :math:`n \in \mathbb{Z}`,
    the Caputo-Fabrizio fractional operator applied to a function
    :math:`f: [a, b] \to \mathbb{R}` is given by (see e.g. [Caputo2015]_)

    .. math::

        D_{CF}^\alpha[f](x) = \frac{M(\alpha)}{n - \alpha} \int_a^x
            \exp \left(-\frac{\alpha - n + 1}{n - \alpha} (x - s)\right)
            f^{(n)}(s) \,\mathrm{d}s,

    Note that, unlike :class:`CaputoDerivative`, this operator has a very different
    kernel. In particular, it is a smooth kernel with different properties. For
    a discussion on fractional operators with smooth kernels see [SalesTeodoro2019]_.
    The corresponding integral is given by

    .. math::

        I_{CF}^\alpha[f](x) =
            \frac{1 - \alpha}{M(\alpha)} f(x)
            + \frac{\alpha}{M(\alpha)} \int_{a}^x f(s) \,\mathrm{d}s,

    We can see that the integral does not have a kernel at all, so it cannot be
    considered a fractional integral in the same sense as those of
    Riemann-Liouville type.

    .. [Caputo2015] M. Caputo, M. Fabrizio,
        *A New Definition of Fractional Derivative Without Singular Kernel*,
        Progress in Fractional Differentiation & Applications, Vol. 1, pp. 73--85, 2015.
        `URL <https://digitalcommons.aaru.edu.jo/pfda/vol1/iss2/1/>`__.
    """

    alpha: float
    """Parameter in the Caputo-Fabrizio operator that corresponds to a
    derivative order."""

    @property
    def n(self) -> int:
        r"""Integer part of the :attr:`~CaputoFabrizioOperator.alpha`, i.e.
        :math:`n - 1 < \alpha \le n`.
        """
        return math.ceil(self.alpha)

    def normalization(self) -> float:
        r"""Normalization :math:`M(\alpha)` used in the definition of the operator.

        From [Caputo2015]_, the normalization must satisfy
        :math:`M(n - 1) = M(n) = 1`, so that the limits to the integer order limits
        are satisfied. By default we take :math:`M(\alpha) = 1`.
        """
        return 1.0


@dataclass(frozen=True)
class VariableExponentialCaputoDerivative(FractionalOperator):
    r"""Variable-order exponential Caputo fractional-order derivative.

    The variable-order exponential Caputo derivative is described in [Garrappa2023]_.
    It considers a fractional order :math:`\alpha(t)` and its Laplace transform
    :math:`A(s)` as

    .. math::

        \alpha(t) = \alpha_1 + (\alpha_0 - \alpha_1) e^{-c t}
        \implies A(s) = \frac{\alpha_0 s + \alpha_1 c}{s (s + c)},

    where :math:`0 < \alpha_0, \alpha_1 < 1` and :math:`c > 0`. Using the Laplace
    transform :math:`A(s)`, we define the variable-order kernel :math:`\phi_\alpha`
    as the Laplace transform of

    .. math::

        \Phi_\alpha(s) = s^{s A(s) - 1}.

    Unfortunately, there is no analytic expression for :math:`\phi_\alpha` even
    for this simple case. Finally, the derivative is given by

    .. math::

        D_{VC}^{\alpha(x)}[f](x) =
            \int_0^x \phi_\alpha(x - s) f'(s) \,\mathrm{d}s.

    The corresponding integral is also given in [Garrappa2023]_.
    """

    alpha: tuple[float, float]
    """Asymptotic orders of the variable Caputo derivative."""
    c: float
    r"""Transition rate from :math:`\alpha_0` to :math:`\alpha_1`."""

    if __debug__:

        def __post_init__(self) -> None:
            if not -1 < self.alpha[0] < 1:
                raise ValueError(f"'alpha_0' is not in (0, 1): {self.alpha}")

            if not -1 < self.alpha[1] < 1:
                raise ValueError(f"'alpha_1' is not in (0, 1): {self.alpha}")

            if self.c <= 0:
                raise ValueError(f"'c' cannot be negative: {self.c}")

    @property
    def side(self) -> Side:
        return Side.Left
