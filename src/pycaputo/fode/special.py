# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from pycaputo.utils import Array, gamma


@dataclass(frozen=True)
class Solution(ABC):
    r"""An abstract solution to a fractional differential equation of the form

    .. math::

        D^\alpha[y](t) = f(t, y(t)),

    with appropriate initial conditions. The solution is given by :meth:`function`
    and its fractional derivative is given by :meth:`derivative`. Then, the
    right-hand side term is given by :meth:`source` and its Jacobian with
    respect to the second argument is given by :meth:`source_jac`.
    """

    def __call__(self, t: float) -> Array:
        return self.function(t)

    @abstractmethod
    def function(self, t: float) -> Array:
        """Evaluates the reference solution at *t*."""

    @abstractmethod
    def derivative(self, t: float) -> Array:
        """Evaluates the fractional derivative of the reference solution at *t*."""

    @abstractmethod
    def source(self, t: float, y: Array) -> Array:
        """Evaluates the right-hand side of the fractional equation."""

    @abstractmethod
    def source_jac(self, t: float, y: Array) -> Array:
        """Evaluates the Jacobian of :meth:`source` with respect to *y*."""


# {{{ monomials


@dataclass(frozen=True)
class CaputoMonomial(Solution):
    r"""A monomial solution to the Caputo fractional differential equation.

    .. math::

        y_{\text{ref}}(t) = \sum_{i = 0}^n Y_{\nu_i} (t - t_0)^{\nu_i}.

    The equation is set up as follows

    .. math::

        D^\alpha_C[y](t) = D^\alpha_C[y_{\text{ref}}](t)
                           + c (y_{\text{ref}}^\beta(t) - y^\beta(t)),

    where the last term is added to make the equation more complex. Note that
    if :math:`\beta` is not a even positive integer, the solution can become
    complex and fail.
    """

    Yv: Array
    """Coefficient array for the reference solution."""
    nu: Array
    """Powers of the monomials in the reference solution."""

    t0: float
    """The starting time for the equation."""
    alpha: float
    """Order of the fractional derivative."""
    beta: float
    """Power used in the right-hand side source term. A lower value results in
    less smoothness in the right-hand side and can give larger errors."""
    c: float
    """Factor for the included term in the right-hand side."""

    if __debug__:

        def __post_init__(self) -> None:
            if self.Yv.size != self.nu.size:
                raise ValueError(
                    "Array size mismatch between 'Yv' and 'nu': "
                    f"{self.Yv.shape} and {self.nu.shape}"
                )

            if self.alpha <= 0:
                raise ValueError(f"'alpha' must be positive: {self.alpha}")

            if self.beta <= 0:
                raise ValueError(f"'beta' must be positive: {self.beta}")

    def function(self, t: float) -> Array:
        result = np.sum(self.Yv * (t - self.t0) ** self.nu)
        return np.array([result])

    def derivative(self, t: float) -> Array:
        mask = self.nu > 0.0
        gYv = gamma(1 + self.nu) / gamma(1 + self.nu - self.alpha) * self.Yv
        result = np.sum(gYv[mask] * (t - self.t0) ** (self.nu[mask] - self.alpha))

        return np.array([result])

    def source(self, t: float, y: Array) -> Array:
        y_ref = self.function(t)
        dy_ref = self.derivative(t)

        result = dy_ref + self.c * (y_ref**self.beta - y**self.beta)
        return np.array(result)

    def source_jac(self, t: float, y: Array) -> Array:
        if self.c == 0:
            return np.zeros_like(y)

        result = -self.c * self.beta * y ** (self.beta - 1.0)
        return result


# }}}


# {{{ sine


@dataclass(frozen=True)
class CaputoSine(Solution):
    r"""A (smooth) sine solution to the fractional differential equation.

    .. math::

        y_{\text{ref}}(t) = \sin t.

    The equation is then given by

    .. math::

        D^\alpha_C[y](t) = D^\alpha_C[y_{\text{ref}}](t).
    """

    t0: float
    """The starting time for the equation."""
    alpha: float
    """Order of the fractional derivative."""

    def __call__(self, t: float) -> Array:
        return self.function(t)

    def function(self, t: float) -> Array:
        return np.array([np.sin(t)])

    def derivative(self, t: float) -> Array:
        from pycaputo.mittagleffler import caputo_derivative_sine

        result = caputo_derivative_sine(t - self.t0, self.alpha)
        return np.array([result])

    def source(self, t: float, y: Array) -> Array:
        dy_ref = self.function(t)
        return dy_ref

    def source_jac(self, t: float, y: Array) -> Array:
        return np.zeros_like(y)


# }}}
