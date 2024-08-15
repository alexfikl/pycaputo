# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycaputo.logging import get_logger
from pycaputo.typing import Array

from .special import Function

logger = get_logger(__name__)


# {{{ Arneodo


@dataclass(frozen=True)
class Arneodo(Function):
    r"""Implements the right-hand side of the Arneodo system (see Equation 5.69
    from [Petras2011]_).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            y, \\
        D^\alpha[y](t) & =
            z, \\
        D^\alpha[z](t) & =
            \beta_4 x^3 - \beta_1 x - \beta_2 y - \beta_3 z
        \end{aligned}

    This system is similar to :class:`GenesioTesi`, but has a cubic nonlinearity.
    """

    beta: tuple[float, float, float, float]

    def source(self, t: float, y: Array) -> Array:
        beta1, beta2, beta3, beta4 = self.beta
        return np.array([
            y[1],
            y[2],
            beta4 * y[0] ** 3 - beta1 * y[0] - beta2 * y[1] - beta3 * y[2],
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        beta1, beta2, beta3, beta4 = self.beta
        return np.array([
            [0, 1, 0],
            [0, 0, 1],
            [3 * beta4 * y[0] ** 2 - beta1, -beta2, -beta3],
        ])


# }}}


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


# {{{ Chen


@dataclass(frozen=True)
class Chen(Function):
    r"""Implements the right-hand side of the Chen system (see Equation 5.54
    from [Petras2011]_).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            a (y - x), \\
        D^\alpha[y](t) & =
            (c - a) x - x z + c y, \\
        D^\alpha[z](t) & =
            x y -  b z.
        \end{aligned}
    """

    a: float
    """Parameter in the Chen system."""
    b: float
    """Parameter in the Chen system."""
    c: float
    """Parameter in the Chen system."""

    def source(self, t: float, y: Array) -> Array:
        return np.array([
            self.a * (y[1] - y[0]),
            (self.c - self.a) * y[0] - y[0] * y[2] + self.c * y[1],
            y[0] * y[1] - self.b * y[2],
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        return np.array([
            [-self.a, self.a, 0.0],
            [self.c - self.a - y[2], self.c, -y[0]],
            [y[1], y[0], -self.b],
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


# {{{ Genesio-Tesi


@dataclass(frozen=True)
class GenesioTesi(Function):
    r"""Implements the right-hand side of the Genesio-Tesi system (see Equation
    5.65 from [Petras2011]_).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            y, \\
        D^\alpha[y](t) & =
            z, \\
        D^\alpha[z](t) & =
            \beta_4 x^2 - \beta_1 x - \beta_2 y - \beta_3 z.
        \end{aligned}

    This system is similar to :class:`Arneodo`, but has a quadratic nonlinearity.
    """

    beta: tuple[float, float, float, float]

    if __debug__:

        def __post_init__(self) -> None:
            if any(beta < 0 for beta in self.beta):
                raise ValueError(f"Parameters must be positive: {self.beta}")

            c, b, a, _ = self.beta
            if a * b > c:
                raise ValueError(
                    "Parameters must satisfy beta[1] beta[2] < beta[0]: "
                    f"'beta = {self.beta}', where 'beta[1] beta[2] = {a * b}'"
                )

    def source(self, t: float, y: Array) -> Array:
        beta1, beta2, beta3, beta4 = self.beta
        return np.array([
            y[1],
            y[2],
            beta4 * y[0] ** 2 - beta1 * y[0] - beta2 * y[1] - beta3 * y[2],
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        beta1, beta2, beta3, beta4 = self.beta
        return np.array([
            [0, 1, 0],
            [0, 0, 1],
            [2 * beta4 * y[0] - beta1, -beta2, -beta3],
        ])


# }}}


# {{{ Lorenz


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


# {{{ Liu


@dataclass(frozen=True)
class Liu(Function):
    r"""Implements the right-hand side of the Liu system (see Equation 5.61 from
    [Petras2011]_).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            -a x - e y^2, \\
        D^\alpha[y](t) & =
            b y - k x z, \\
        D^\alpha[z](t) & =
            m x y - c z.
        \end{aligned}
    """

    a: float
    """Parameter in the Liu system."""
    b: float
    """Parameter in the Liu system."""
    c: float
    """Parameter in the Liu system."""
    e: float
    """Parameter in the Liu system."""
    k: float
    """Parameter in the Liu system."""
    m: float
    """Parameter in the Liu system."""

    def source(self, t: float, y: Array) -> Array:
        return np.array([
            -self.a * y[0] - self.e * y[1] ** 2,
            self.b * y[1] - self.k * y[0] * y[2],
            self.m * y[0] * y[1] - self.c * y[2],
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        return np.array([
            [-self.a, -2.0 * self.e * y[1], 0.0],
            [-self.k * y[2], self.b, -self.k * y[0]],
            [self.m * y[1], self.m * y[0], -self.c],
        ])


# }}}


# {{{ Lü


@dataclass(frozen=True)
class Lu(Function):
    r"""Implements the right-hand side of the Lü system (see Equation 5.58
    from [Petras2011]_).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            a (y - x), \\
        D^\alpha[y](t) & =
            -x z + c y, \\
        D^\alpha[z](t) & =
            x y - b z.
        \end{aligned}

    This system is very similar to :class:`Chen` and :class:`Lorenz`.
    """

    a: float
    """Parameter in the Lu system."""
    b: float
    """Parameter in the Lu system."""
    c: float
    """Parameter in the Lu system."""

    def source(self, t: float, y: Array) -> Array:
        return np.array([
            self.a * (y[1] - y[0]),
            -y[0] * y[2] + self.c * y[1],
            y[0] * y[1] - self.b * y[2],
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        return np.array([
            [-self.a, self.a, 0.0],
            [-y[2], self.c, -y[0]],
            [y[1], y[0], -self.b],
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
