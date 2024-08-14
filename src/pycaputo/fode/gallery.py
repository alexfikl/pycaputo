# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycaputo.logging import get_logger
from pycaputo.typing import Array

from .special import Function

logger = get_logger(__name__)


# {{{ Brusselator


@dataclass(frozen=True)
class Brusselator(Function):
    r"""Implements the right-hand side of the Brusselator system.

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            a - (\mu + 1) x + x^2 y + A \cos \omega t, \\
        D^\alpha[y](t) & =
            \mu x - x^2 y.
        \end{aligned}
    """

    a: float
    """Parameter in the Brusselator model."""
    mu: float
    """Parameter in the Brusselator model."""

    amplitude: float
    """Forcing amplitude."""
    omega: float
    """Angular velocity of the forcing."""

    if __debug__:

        def __post_init__(self) -> None:
            if self.a < 0:
                raise ValueError(f"'a' must be positive: {self.a}")

            if self.mu < 0:
                raise ValueError(f"'mu' must be positive: {self.mu}")

    def source(self, t: float, y: Array) -> Array:
        f = self.amplitude * np.cos(self.omega * t)
        return np.array([
            self.a - (self.mu + 1) * y[0] + y[0] ** 2 * y[1] + f,
            self.mu * y[0] - y[0] ** 2 * y[1],
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        return np.array([
            [-(self.mu + 1) + 2 * y[0] * y[1], y[0] ** 2],
            [self.mu - 2 * y[0] * y[1], -(y[0] ** 2)],
        ])


# }}}


# {{{ Duffing


@dataclass(frozen=True)
class Duffing(Function):
    r"""Implements the right-hand side of the Duffing system (see Equation 5.46
    from [Petras2011]_).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            y, \\
        D^\alpha[y](t) & =
            x - x^3 - \alpha y + A \cos \omega t.
        \end{aligned}
    """

    alpha: float
    """Parameter in the Duffing system."""

    amplitude: float
    """Forcing amplitude."""
    omega: float
    """Angular velocity of the forcing."""

    def source(self, t: float, y: Array) -> Array:
        f = self.amplitude * np.cos(self.omega * t)
        return np.array([
            y[1],
            y[0] - y[0] ** 3 - self.alpha * y[1] + f,
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        return np.array([[0, 1], [1 - 3 * y[0], -self.alpha]])


# }}}

# {{{


@dataclass(frozen=True)
class Lorenz(Function):
    r"""Implements the right-hand side of the Lorenz system (see Equation 5.52
    from [Petras2011]_).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            \sigma (y - x), \\
        D^\alpha[y](t) & =
            x (\rho - z) - y, \\
        D^\alpha[z](t) & =
            x y - \beta z.
        \end{aligned}
    """

    sigma: float
    """Parameter in the Lorenz system. This is proportional to the Prandtl number
    in the fluids derivation of the system.
    """

    rho: float
    """Parameter in the Lorenz system. This is proportional to the Rayleigh number
    in the fluids derivation of the system.
    """

    beta: float
    """Parameter in the Lorenz system. This parameter is related to the height
    of the fluid layer in the fluids derivation of the system.
    """

    if __debug__:

        def __post_init__(self) -> None:
            if self.sigma <= 0:
                raise ValueError(f"Parameter 'sigma' must be positive: {self.sigma}")

            if self.rho <= 0:
                raise ValueError(f"Parameter 'rho' must be positive: {self.rho}")

            if self.beta <= 0:
                raise ValueError(f"Parameter 'beta' must be positive: {self.beta}")

    def source(self, t: float, y: Array) -> Array:
        return np.array([
            self.sigma * (y[1] - y[0]),
            y[0] * (self.rho - y[2]) - y[1],
            y[0] * y[1] - self.beta * y[2],
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        return np.array([
            [-self.sigma, self.sigma, 0],
            [self.rho - y[2], -1, -y[0]],
            [y[1], y[0], -self.beta],
        ])


# }}}


# {{{ van der Pol


@dataclass(frozen=True)
class VanDerPol(Function):
    r"""Implements the right-hand side of the van der Pol system (see Equation 5.40
    from [Petras2011]_).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            y, \\
        D^\alpha[y](t) & =
            \mu (1 - x^2) y - x + A \sin (\omega t).
        \end{aligned}
    """

    mu: float
    r"""Parameter indicating the strength of the nonlinearity and damping."""

    amplitude: float
    """Forcing amplitude."""
    omega: float
    """Angular velocity of the forcing."""

    if __debug__:

        def __post_init__(self) -> None:
            if self.mu < 0:
                raise ValueError(f"'mu' must be positive: {self.mu}")

    def source(self, t: float, y: Array) -> Array:
        f = self.amplitude * np.cos(self.omega * t)
        return np.array([y[1], self.mu * (1.0 - y[0] ** 2) * y[1] - y[0] + f])

    def source_jac(self, t: float, y: Array) -> Array:
        return np.array([
            [0.0, 1.0],
            [-2.0 * self.mu * y[0] * y[1] - 1.0, self.mu * (1.0 - y[0] ** 2)],
        ])


# }}}
