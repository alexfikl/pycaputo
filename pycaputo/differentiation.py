# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Dict, Type

import numpy as np

from pycaputo.derivatives import CaputoDerivative, Side
from pycaputo.grid import Points, UniformMidpoints, UniformPoints
from pycaputo.logging import get_logger
from pycaputo.utils import Array, ScalarFunction

logger = get_logger(__name__)

# {{{ interface


@dataclass(frozen=True)
class DerivativeMethod(ABC):
    """A generic method used to evaluate a fractional derivative at a set of points."""

    @property
    @abstractmethod
    def name(self) -> str:
        """An identifier for the method."""

    @property
    @abstractmethod
    def order(self) -> float:
        """Expected order of convergence of the method."""

    @abstractmethod
    def supports(self, alpha: float) -> bool:
        """
        :returns: *True* if the method supports computing the fractional
            order derivative of order *alpha* and *False* otherwise.
        """


@singledispatch
def diff(m: DerivativeMethod, f: ScalarFunction, x: Points) -> Array:
    """Evaluate the fractional derivative of *f* at *x*.

    Note that not all numerical methods can evaluate the derivative at all
    points in *x*. For example, the :class:`CaputoL1Method` cannot evaluate the
    derivative at ``x[0]``.

    :arg m: method used to evaluate the derivative.
    :arg f: a simple function for which to evaluate the derivative.
    :arg x: an array of points at which to evaluate the derivative.
    """
    raise NotImplementedError(
        f"Cannot evaluate derivative with method '{type(m).__name__}'"
    )


# }}}


# {{{ Caputo L1 Method


@dataclass(frozen=True)
class CaputoL1Method(DerivativeMethod):
    r"""Implements the L1 method for the Caputo fractional derivative
    of order :math:`\alpha \in (0, 1)`.

    This method is defined in Section 4.1.1 (II) from [Li2020]_ for general
    non-uniform grids. Note that it cannot compute the derivative at the
    starting point, i.e. :math:`D_C^\alpha[f](a)` is undefined.

    This method is of order :math:`\mathcal{O}(h^{2 - \alpha})`.
    """

    #: The type of the Caputo derivative.
    d: CaputoDerivative

    if __debug__:

        def __post_init__(self) -> None:
            if not self.supports(self.d.order):
                raise ValueError(
                    f"{type(self).__name__} only supports orders in (0, 1): "
                    f"got order '{self.d.order}'"
                )

    @property
    def name(self) -> str:
        return "L1"

    @property
    def order(self) -> float:
        return 2 - self.d.order

    def supports(self, alpha: float) -> bool:
        return 0 < alpha < 1


@diff.register(CaputoL1Method)
def _diff_l1method(m: CaputoL1Method, f: ScalarFunction, p: Points) -> Array:
    # precompute variables
    x = p.x
    fx = f(x)

    alpha = m.d.order
    w0 = 1 / math.gamma(2 - alpha)

    # NOTE: [Li2020] Equation 4.20
    df = np.zeros_like(x)
    df[0] = np.nan

    if isinstance(p, UniformPoints):
        w0 = w0 / p.dx[0] ** alpha
        k = np.arange(fx.size - 1)

        for n in range(1, df.size):
            w = (n - k[:n]) ** (1 - alpha) - (n - k[:n] - 1) ** (1 - alpha)
            df[n] = w0 * np.sum(w * np.diff(fx[: n + 1]))
    else:
        for n in range(1, df.size):
            w = (
                (x[n] - x[:n]) ** (1 - alpha) - (x[n] - x[1 : n + 1]) ** (1 - alpha)
            ) / p.dx[:n]
            df[n] = w0 * np.sum(w * np.diff(fx[: n + 1]))

    return df


@dataclass(frozen=True)
class CaputoModifiedL1Method(CaputoL1Method):
    r"""Implements the modified L1 method for the Caputo fractional derivative
    of order :math:`\alpha \in (0, 1)`.

    This method is defined in Section 4.1.1 (III) from [Li2020]_ for quasi-uniform
    grids. Note that it cannot compute the derivative at the starting point, i.e.
    :math:`D_C^\alpha[f](a)` is undefined.

    This method is of order :math:`\mathcal{O}(h^{2 - \alpha})`.
    """

    @property
    def name(self) -> str:
        return "L1M"


@diff.register(CaputoModifiedL1Method)
def _diff_modified_l1method(
    m: CaputoModifiedL1Method, f: ScalarFunction, p: Points
) -> Array:
    # precompute variables
    x = p.x
    fx = f(x)

    alpha = m.d.order
    w0 = 1 / math.gamma(2 - alpha)

    # NOTE: [Li2020] Equation 4.51
    df = np.empty_like(x)
    df[0] = np.nan

    if isinstance(p, UniformMidpoints):
        w0 = w0 / p.dx[-1] ** alpha
        k = np.arange(fx.size)

        # FIXME: this does not use the formula from the book; any benefit to it?
        w = 2 * w0 * ((k[:-1] + 0.5) ** (1 - alpha) - k[:-1] ** (1 - alpha))
        df[1:] = w * (fx[1] - fx[0])

        for n in range(1, df.size):
            w = (n - k[1:n]) ** (1 - alpha) - (n - k[1:n] - 1) ** (1 - alpha)
            df[n] += w0 * np.sum(w * np.diff(fx[1 : n + 1]))
    else:
        raise NotImplementedError(
            f"'{type(m).__name__}' not implemented for '{type(p).__name__}' grids"
        )

    return df


# }}}


# {{{ Caputo L2 Method


@dataclass(frozen=True)
class CaputoL2Method(DerivativeMethod):
    r"""Implements the L2 method for the Caputo fractional derivative
    of order :math:`\alpha \in (1, 2)`.

    This method is defined in Section 4.1.2 from [Li2020]_. Note that
    it cannot compute the derivative at the starting point, i.e.
    :math:`D_C^\alpha[f](a)` is undefined.

    This method is of order :math:`\mathcal{O}(h^{3 - \alpha})`.
    """

    #: The type of the Caputo derivative.
    d: CaputoDerivative

    if __debug__:

        def __post_init__(self) -> None:
            if not self.supports(self.d.order):
                raise ValueError(
                    f"{type(self).__name__} only supports orders in (1, 2): "
                    f"got order '{self.d.order}'"
                )

    @property
    def name(self) -> str:
        return "L2"

    @property
    def order(self) -> float:
        return 1

    def supports(self, alpha: float) -> bool:
        return 1 < alpha < 2


def l2uweights(alpha: float, i: Any, k: Any) -> Array:
    return np.array((i - k) ** (2 - alpha) - (i - k - 1) ** (2 - alpha))


@diff.register(CaputoL2Method)
def _diff_l2method(m: CaputoL2Method, f: ScalarFunction, p: Points) -> Array:
    # precompute variables
    x = p.x
    fx = f(x)

    alpha = m.d.order
    w0 = 1 / math.gamma(3 - alpha)

    # NOTE: [Li2020] Section 4.2
    # NOTE: the method is not written as in [Li2020] and has several tweaks:
    # * terms are written as `sum(w * f'')` instead of `sum(w * f)`, which
    #   makes it easier to express w
    # * boundary terms are approximated with a biased stencil.

    df = np.empty_like(x)
    df[0] = np.nan

    if isinstance(p, UniformPoints):
        w0 = w0 / p.dx[0] ** alpha
        k = np.arange(fx.size)

        ddf = np.zeros(fx.size - 1, dtype=fx.dtype)
        ddf[:-1] = fx[2:] - 2 * fx[1:-1] + fx[:-2]
        ddf[-1] = 2 * fx[-1] - 5 * fx[-2] + 4 * fx[-3] - fx[-4]

        for n in range(1, df.size):
            df[n] = w0 * np.sum(l2uweights(alpha, n, k[:n]) * ddf[:n])
    else:
        raise NotImplementedError(
            f"'{type(m).__name__}' not implemented for '{type(p).__name__}' grids"
        )

    return df


@dataclass(frozen=True)
class CaputoL2CMethod(CaputoL2Method):
    r"""Implements the L2C method for the Caputo fractional derivative
    of order :math:`\alpha \in (1, 2)`.

    This method is defined in Section 4.1.2 from [Li2020]_. Note that
    it cannot compute the derivative at the starting point, i.e.
    :math:`D_C^\alpha[f](a)` is undefined.

    This method is of order :math:`\mathcal{O}(h^{3 - \alpha})`.
    """

    @property
    def name(self) -> str:
        return "L2C"

    @property
    def order(self) -> float:
        return 3 - self.d.order


@diff.register(CaputoL2CMethod)
def _diff_uniform_l2cmethod(m: CaputoL2CMethod, f: ScalarFunction, p: Points) -> Array:
    # precompute variables
    x = p.x
    fx = f(x)

    alpha = m.d.order
    w0 = 1 / math.gamma(3 - alpha)

    # NOTE: [Li2020] Section 4.2
    df = np.empty_like(x)
    df[0] = np.nan

    if isinstance(p, UniformPoints):
        w0 = w0 / (2 * p.dx[0] ** alpha)
        k = np.arange(fx.size)

        ddf = np.zeros(fx.size - 1, dtype=fx.dtype)
        ddf[1:-1] = (fx[3:] - fx[2:-1]) - (fx[1:-2] - fx[:-3])
        ddf[0] = 3 * fx[0] - 7 * fx[1] + 5 * fx[2] - fx[3]
        ddf[-1] = 3 * fx[-1] - 7 * fx[-2] + 5 * fx[-3] - fx[-4]

        for n in range(1, df.size):
            df[n] = w0 * np.sum(l2uweights(alpha, n, k[:n]) * ddf[:n])
    else:
        raise NotImplementedError(
            f"'{type(m).__name__}' not implemented for '{type(p).__name__}'"
        )

    return df


# }}}


# {{{ make


REGISTERED_METHODS: Dict[str, Type[DerivativeMethod]] = {
    "CaputoL1Method": CaputoL1Method,
    "CaputoL2CMethod": CaputoL2CMethod,
    "CaputoL2Method": CaputoL2Method,
    "CaputoModifiedL1Method": CaputoModifiedL1Method,
}


def make_diff_from_name(
    name: str,
    order: float,
    *,
    side: Side = Side.Left,
) -> DerivativeMethod:
    if name not in REGISTERED_METHODS:
        raise ValueError(
            "Unknown differentiation method '{}'. Known methods are '{}'".format(
                name, "', '".join(REGISTERED_METHODS)
            )
        )

    d = CaputoDerivative(order=order, side=side)
    method = REGISTERED_METHODS[name](d)  # type: ignore[call-arg]

    if not method.supports(order):
        raise ValueError(f"Method '{name}' does not support derivative order '{order}'")

    return method


# }}}
