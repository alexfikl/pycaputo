# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import singledispatch

import numpy as np

from pycaputo.derivatives import CaputoDerivative, Side
from pycaputo.grid import Points
from pycaputo.logging import get_logger
from pycaputo.utils import Array, ScalarFunction

logger = get_logger(__name__)

# {{{ interface


@dataclass(frozen=True)
class DerivativeMethod(ABC):
    """A generic method used to evaluate a fractional derivative at a set of points."""

    @property
    @abstractmethod
    def order(self) -> float:
        """Expected order of convergence of the method."""

    @abstractmethod
    def supports(self, alpha: float) -> bool:
        """
        :returns: *True* if the method supports computing the fractional
            order derivative of order *alpha*.
        """


@singledispatch
def evaluate(m: DerivativeMethod, f: ScalarFunction, x: Points) -> Array:
    """Evaluate the fractional derivative of *f*.

    :arg m: method used to evaluate the derivative.
    :arg f: a simple function for which to evaluate the derivative.
    :arg x: an array of points at which to evaluate the derivative.
    """
    raise NotImplementedError(
        f"Cannot evaluate function with method '{type(m).__name__}'"
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
    def order(self) -> float:
        return 2 - self.d.order

    def supports(self, alpha: float) -> bool:
        return 0 < alpha < 1


@evaluate.register(CaputoL1Method)
def _evaluate_l1method(m: CaputoL1Method, f: ScalarFunction, p: Points) -> Array:
    import math

    x = p.x
    fx = f(x)
    alpha = m.d.order

    # NOTE: this method cannot compute the derivative at x[0], since it relies
    # on approximating an integral, better luck elsewhere :(
    df = np.zeros_like(x)
    df[0] = np.nan

    # TODO: How to do this convolution faster??
    c = math.gamma(2 - alpha)

    # NOTE: [Li2020] Equation 4.20
    for n in range(1, df.size):
        omega = (
            (x[n] - x[:n]) ** (1 - alpha) - (x[n] - x[1 : n + 1]) ** (1 - alpha)
        ) / p.dx[:n]
        df[n] = np.sum(omega * np.diff(fx[: n + 1])) / c

    return df


@dataclass(frozen=True)
class CaputoUniformL1Method(CaputoL1Method):
    r"""Implements the uniform L1 method for the Caputo fractional derivative
    of order :math:`\alpha \in (0, 1)`.

    This method is defined in Section 4.1.1 (I) from [Li2020]_ for uniform
    grids. Note that it cannot compute the derivative at the starting point,
    i.e. :math:`D_C^\alpha[f](a)` is undefined.

    If :attr:`modified` is *True*, a variant of the :class:`CaputoModifiedL1Method`
    is used with

    .. math::

        f\left(\frac{x_{k} + x_{k - 1}}{2}\right) \approx
        \frac{f(x_{k}) + f(x_{k - 1})}{2}.
    """

    #: Flag to denote the modified L1 method.
    modified: bool


@evaluate.register(CaputoUniformL1Method)
def _evaluate_uniform_l1method(
    m: CaputoUniformL1Method, f: ScalarFunction, p: Points
) -> Array:
    from pycaputo.grid import UniformPoints

    assert isinstance(p, UniformPoints)

    import math

    x = p.x
    fx = f(x)
    alpha = m.d.order

    # NOTE: this method cannot compute the derivative at x[0], since it relies
    # on approximating an integral, better luck elsewhere :(
    df = np.zeros_like(x)
    df[0] = np.nan

    c = p.dx[0] ** alpha * math.gamma(2 - alpha)
    k = np.arange(df.size - 1)

    if m.modified:  # pylint: disable=no-else-raise
        # NOTE: [Li2020] Equation 4.53
        raise NotImplementedError
    else:
        # NOTE: [Li2020] Equation 4.3
        for n in range(1, df.size):
            w = (n - k[:n]) ** (1 - alpha) - (n - k[:n] - 1) ** (1 - alpha)
            df[n] = np.sum(w * np.diff(fx[: n + 1])) / c

    return df


@dataclass(frozen=True)
class CaputoModifiedL1Method(CaputoL1Method):
    r"""Implements the modified L1 method for the Caputo fractional derivative
    of order :math:`\alpha \in (0, 1)`.

    This method is defined in Section 4.1.1 (III) from [Li2020]_ for quasi-uniform
    grids. Note that it cannot compute the derivative at the starting point, i.e.
    :math:`D_C^\alpha[f](a)` is undefined.
    """


@evaluate.register(CaputoModifiedL1Method)
def _evaluate_modified_l1method(
    m: CaputoModifiedL1Method, f: ScalarFunction, p: Points
) -> Array:
    from pycaputo.grid import UniformMidpoints

    assert isinstance(p, UniformMidpoints)

    import math

    x = p.x
    h = p.dx[1]
    fx = f(x)
    alpha = m.d.order

    # NOTE: this method cannot compute the derivative at x[0], since it relies
    # on approximating an integral, better luck elsewhere :(
    df = np.empty_like(x)
    df[0] = np.nan

    c = h**alpha * math.gamma(2 - alpha)
    k = np.arange(df.size)

    # NOTE: [Li2020] Equation 4.51
    # FIXME: this does not use the formula from the book; any benefit to it?
    w = 2 / c * ((k[:-1] + 0.5) ** (1 - alpha) - k[:-1] ** (1 - alpha))
    df[1:] = w * (fx[1] - fx[0])

    for n in range(1, df.size):
        w = (n - k[1:n]) ** (1 - alpha) - (n - k[1:n] - 1) ** (1 - alpha)
        df[n] += np.sum(w * np.diff(fx[1 : n + 1])) / c

    return df


# }}}


# {{{ Caputo L2 Method


@dataclass(frozen=True)
class CaputoUniformL2Method(DerivativeMethod):
    r"""Implements the uniform L2 method for the Caputo fractional derivative
    of order :math:`\alpha \in (1, 2)`.

    This method is defined in Section 4.1.2 from [Li2020]_. Note that
    it cannot compute the derivative at the starting point, i.e.
    :math:`D_C^\alpha[f](a)` is undefined.
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
    def order(self) -> float:
        return 3 - self.d.order

    def supports(self, alpha: float) -> bool:
        return 1 < alpha < 2


@evaluate.register(CaputoUniformL2Method)
def _evaluate_uniform_l2method(
    m: CaputoUniformL2Method, f: ScalarFunction, p: Points
) -> Array:
    from pycaputo.grid import UniformPoints

    assert isinstance(p, UniformPoints)

    import math

    x = p.x
    h = p.dx[1]
    fx = f(x)
    alpha = m.d.order

    # NOTE: this method cannot compute the derivative at x[0], since it relies
    # on approximating an integral, better luck elsewhere :(
    df = np.empty_like(x)
    df[0] = np.nan

    # NOTE: [Li2020] Equation 4.51
    # NOTE: the boundary terms are not taken as in [Li2020]; instead we use a
    # upwind / downwind biased stencils to obtain an approximation of the same order

    c = 1 / (h**alpha * math.gamma(3 - alpha))
    k = np.arange(df.size)

    df[1:] = c * (2 * fx[1:] - 5 * fx[0:-1] + 4 * fx[0:-1] - fx[0:-1])
    df[-1] += c * (2 * fx[0] - 5 * fx[1] + 4 * fx[2] - fx[3])

    for n in range(1, df.size - 2):
        omega = c * (k[1:n + 1] ** (2 - alpha) - k[:n] ** (2 - alpha))
        F = fx[2:n + 2] - 2 * fx[1:n + 1] + fx[0:n]
        df[n] += np.sum(omega * F[::-1])

    return df


# }}}


# {{{ make


REGISTERED_METHODS = {
    "CaputoL1Method": CaputoL1Method,
    "CaputoUniformL1Method": CaputoUniformL1Method,
    "CaputoModifiedL1Method": CaputoModifiedL1Method,
    "CaputoUniformL2Method": CaputoUniformL2Method,
}


def make_diff_method(
    name: str,
    order: float,
    *,
    side: Side = Side.Left,
    modified: bool = False,
) -> DerivativeMethod:
    if name not in REGISTERED_METHODS:
        raise ValueError(
            "Unknown differentiation method '{}'. Known methods are '{}'".format(
                name, "', '".join(REGISTERED_METHODS)
            )
        )

    d = CaputoDerivative(order=order, side=side)

    if name == "CaputoL1Method":
        method = CaputoL1Method(d)
    elif name == "CaputoUniformL1Method":
        method = CaputoUniformL1Method(d, modified=modified)
    elif name == "CaputoModifiedL1Method":
        method = CaputoModifiedL1Method(d)
    elif name == "CaputoUniformL2Method":
        method = CaputoUniformL2Method(d)
    else:
        raise AssertionError

    if not method.supports(order):
        raise ValueError(f"Method '{name}' does not support derivative order '{order}'")

    return method


# }}}
