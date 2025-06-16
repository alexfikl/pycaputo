# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from pycaputo.derivatives import FractionalOperator
from pycaputo.typing import Array


@dataclass(frozen=True)
class Function(ABC):
    r"""Abstract class for right-hand side functions of differential equations

    .. math::

        D^\alpha[y](t) = f(t, y(t)),

    The right-hand side term is given by :meth:`source` and its Jacobian with
    respect to the second argument is given by :meth:`source_jac`.
    """

    @abstractmethod
    def source(self, t: float, y: Array) -> Array:
        """Evaluates the right-hand side of the fractional equation."""

    @abstractmethod
    def source_jac(self, t: float, y: Array) -> Array:
        """Evaluates the Jacobian of :meth:`source` with respect to *y*."""


@dataclass(frozen=True)
class Solution(Function):
    r"""An abstract solution to a fractional differential equation of the form

    .. math::

        D^\alpha[y](t) = f(t, y(t)),

    with appropriate initial conditions. The solution is given by :meth:`function`
    and its fractional derivative is given by :meth:`derivative`. This is a
    subclass of :class:`Function` and can also be used to define the equation.
    """

    def __call__(self, t: float | Array) -> Array:
        return self.function(t)

    @abstractmethod
    def function(self, t: float | Array) -> Array:
        """Evaluates the reference solution at *t*.

        :returns: an array of shape ``(n, d)``, where *n* is the size of *t*
            and *d* is the size of the system.
        """

    @abstractmethod
    def derivative(self, t: float | Array) -> Array:
        """Evaluates the fractional derivative of the reference solution at *t*.

        :returns: an array of shape ``(n, d)``, where *n* is the size of *t*
            and *d* is the size of the system.
        """


# {{{ monomials


@dataclass(frozen=True)
class Monomial(Solution):
    r"""A monomial solution to a fractional differential equation.

    .. math::

        y_{\text{ref}}(t) = \sum_{i = 0}^n Y_{\nu_i} (t - t_0)^{\nu_i}.

    The equation is set up as follows

    .. math::

        D^\alpha_*[y](t) = D^\alpha_*[y_{\text{ref}}](t)
                           + c (y_{\text{ref}}^\beta(t) - y^\beta(t)),

    where the last term is added to make the equation more complex. Note that
    if :math:`\beta` is not a even positive integer, the solution can become
    complex and fail.
    """

    d: FractionalOperator
    """The fractional operator used in the fractional differential equation."""

    Yv: Array
    """Coefficient array for the reference solution."""
    nu: Array
    """Powers of the monomials in the reference solution."""

    t0: float
    """Order of the fractional derivative."""
    beta: float
    """Power used in the right-hand side source term. A lower value results in
    less smoothness in the right-hand side and can give larger errors.
    """
    c: float
    """Factor for the included term in the right-hand side."""

    if __debug__:

        def __post_init__(self) -> None:
            if self.Yv.size != self.nu.size:
                raise ValueError(
                    "Array size mismatch between 'Yv' and 'nu': "
                    f"{self.Yv.shape} and {self.nu.shape}"
                )

            if np.any(self.nu < 0.0):
                raise ValueError(f"Powers 'nu' cannot be negative: {self.nu}")

            if self.beta <= 0:
                raise ValueError(f"'beta' must be positive: {self.beta}")

    def function(self, t: float | Array) -> Array:
        result = np.sum(self.Yv * (t - self.t0) ** self.nu)
        return np.array([result])

    def derivative(self, t: float | Array) -> Array:
        from pycaputo.special import pow_derivative

        result = np.sum([
            self.Yv[i] * pow_derivative(self.d, t, t0=self.t0, omega=self.nu[i])
            for i in range(self.nu.size)
        ])

        return np.array([result])

    def source(self, t: float, y: Array) -> Array:
        if self.c == 0:
            dy_ref = self.derivative(t)
            result = dy_ref
        else:
            y_ref = self.function(t)
            dy_ref = self.derivative(t)
            result = dy_ref + self.c * (y_ref**self.beta - y**self.beta)

        return np.array(result)

    def source_jac(self, t: float, y: Array) -> Array:
        if self.c == 0:
            return np.zeros_like(y)

        result = -self.c * self.beta * y ** (self.beta - 1.0)
        return result

    @classmethod
    def random(
        cls,
        alpha: float,
        t0: float = 0.0,
        *,
        p: int | None = None,
        Ymin: float = 0.0,
        Ymax: float = 3.0,
        beta: float = 2.0,
        c: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> Monomial:
        r"""Constructs a solution where the leading terms blow up in the
        derivative.

        This mimics the exact solution of fractional equations, where the
        it is known to have an asymptotic expansion of this fashion in a
        neighborhood of :math:`t_0`. For example, see Equation 5 in
        [Garrappa2015b]_ which gives

        .. math::

            y(t) = \sum_{k = 0}^{m - 1} \frac{(t - t_0)^k}{\Gamma(1 + k)} y_0^{(k)}
                    + \sum_{\nu \in \mathcal{A}_{p, m}} Y_{\nu} (t - t_0)^\nu
                    + \mathcal{O}((t - t_0)^p)

        where

        .. math::

            \mathcal{A}_{p, m} =
                \{i + j \alpha \mid i, j \in \mathbb{N}, i + j \alpha < p\}
                \setminus \{0, \dots, m - 1\},

        and :math:`m - 1 \le \alpha < m`. The coefficients :math:`Y_{\nu}`
        are randomly generated and sorted in decreasing order, such that
        the term :math:`(t - t_0)^\alpha` has the largest coefficient.

        .. note::

            As can be seen from the expansion, the first-order derivative of the
            solution will not be singular for :math:`\alpha > 1`. However, some
            higher-order derivatives will always be singular.

        :arg p: the order of the expansion, taken to be :math:`m + 2` by default.
        :arg Ymin: minimum value of the coefficients.
        :arg Ymax: maximum value of the coefficients.
        """
        from pycaputo.derivatives import CaputoDerivative, Side

        d = CaputoDerivative(alpha, side=Side.Left)
        m = d.n
        if p is None:
            p = m + 2

        if rng is None:
            rng = np.random.default_rng()

        imax = p
        jmax = int(np.ceil(p / alpha))

        nu = np.array([i + alpha * j for i in range(imax) for j in range(jmax)])
        Ap = nu[nu < p]
        Apm = np.setdiff1d(Ap, np.arange(m, dtype=nu.dtype))
        Yv = rng.uniform(Ymin, Ymax, size=Apm.size)

        return Monomial(
            d=d,
            Yv=np.sort(Yv)[::-1],
            nu=np.sort(Apm),
            t0=t0,
            beta=beta,
            c=c,
        )


# }}}


# {{{ Sine


@dataclass(frozen=True)
class Sine(Solution):
    r"""A (smooth) sine solution to the fractional differential equation.

    .. math::

        y_{\text{ref}}(t) = \sin t.

    The equation is then given by

    .. math::

        D^\alpha_C[y](t) =
            D^\alpha_C[y_{\text{ref}}](t)
            + \zeta (y_{\text{ref}}^\beta(t) - y^\beta(t)).
    """

    d: FractionalOperator
    """Fractional operator used in the fractional differential equation."""

    t0: float
    """The starting time for the equation."""
    zeta: float
    """Nonlinear forcing term magnitude."""
    beta: float
    """Nonlinear forcing term power."""

    def function(self, t: float | Array) -> Array:
        return np.array([np.sin(t)]).T  # type: ignore[no-any-return]

    def derivative(self, t: float | Array) -> Array:
        from pycaputo.special import sin_derivative

        result = sin_derivative(self.d, t, t0=self.t0, omega=1.0)
        return np.array([result])

    def source(self, t: float, y: Array) -> Array:
        f = self.derivative(t)
        yref = self.function(t)

        return f + self.zeta * (yref**self.beta - y**self.beta)

    def source_jac(self, t: float, y: Array) -> Array:
        if self.beta == 1:
            dy = np.full_like(y, -self.zeta)
        else:
            dy = -self.zeta * self.beta * y ** (self.beta - 1)

        return np.diag(dy).squeeze()


# }}}
