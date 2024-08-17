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


# {{{ Cellular Neural Network


@dataclass(frozen=True)
class CellularNeuralNetwork3(Function):
    r"""Implements the right-hand side of the three-cell system (see Equation
    5.93 from [Petras2011]_).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            -x + p_1 f(x) - s f(y) - s f(z), \\
        D^\alpha[y](t) & =
            -y - s f(x) + p_2 f(y) - r f(z), \\
        D^\alpha[z](t) & =
            -z - s f(x) + r f(y) + p_3 f(z).
        \end{aligned}

    where the activation function :math:`f` is given by

    .. math::

        f(s) = \frac{1}{2} (|s + 1| - |s - 1|).
    """

    p: tuple[float, float, float]
    """Parameters for the CNN system diagonal."""
    r: float
    """Parameter for the CNN system."""
    s: float
    """Parameter for the CNN system."""

    def source(self, t: float, y: Array) -> Array:
        f = (abs(y + 1) - abs(y - 1)) / 2.0
        p, r, s = self.p, self.r, self.s
        return np.array([
            -y[0] + p[0] * f[0] - s * f[1] - s * f[2],
            -y[1] - s * f[0] + p[1] * f[1] - r * f[2],
            -y[2] - s * f[0] + r * f[1] + p[2] * f[2],
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        df = np.where((y >= -1.0) & (y <= 1.0), 1, 0)
        p, r, s = self.p, self.r, self.s
        return np.array([
            [-1.0 + p[0] * df[0], -s * df[1], -s * df[2]],
            [-s * df[0], -1.0 + p[1] * df[1], -r * df[2]],
            [-s * df[0], r * df[1], -1.0 + p[2] * df[2]],
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


# {{{ Lotka-Volterra


@dataclass(frozen=True)
class LotkaVolterra2(Function):
    r"""Implements the right-hand side of the Lotka-Volterra system (see Equation
    5.82 from [Petras2011]_).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            x (\alpha - r x - \beta y), \\
        D^\alpha[y](t) & =
            -y (\gamma - \delta x).
        \end{aligned}

    The Lotka-Volterra model is usually referred to as a predator-prey model, where
    :math:`x` denotes the prey and :math:`y` denotes the predator.
    """

    alpha: float
    """Parameter in the Lotka-Volterra system, which represents the prey per
    capita growth rate.
    """
    beta: float
    """Parameter in the Lotka-Volterra system, which represents the effect of
    the presence of predators on the prey death rate.
    """
    gamma: float
    """Parameter in the Lotka-Volterray system, which represents the predator
    per capita death rate.
    """
    delta: float
    """Parameter in the Lotka-Volterray system, which represents the effect of
    presence of prey on the predator's growth rate.
    """
    r: float
    """Parameter in the Lotka-Volterra system, which is taken as :math:`r = 0`
    in the standard system.
    """

    if __debug__:

        def __post_init__(self) -> None:
            if self.alpha < 0.0:
                raise ValueError(f"'alpha' must be positive: {self.alpha}")

            if self.beta < 0.0:
                raise ValueError(f"'beta' must be positive: {self.beta}")

            if self.gamma < 0.0:
                raise ValueError(f"'gamma' must be positive: {self.gamma}")

            if self.delta < 0.0:
                raise ValueError(f"'delta' must be positive: {self.delta}")

            if self.r < 0.0:
                raise ValueError(f"'r' must be positive: {self.r}")

    def source(self, t: float, y: Array) -> Array:
        return np.array([
            y[0] * (self.alpha - self.r * y[0] - self.beta * y[1]),
            -y[1] * (self.gamma - self.delta * y[0]),
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        return np.array([
            [self.alpha - 2 * self.r * y[0] - self.beta * y[1], -self.beta * y[0]],
            [self.delta * y[1], self.delta * y[0] - self.gamma],
        ])


@dataclass(frozen=True)
class LotkaVolterra3(Function):
    r"""Implements the right-hand side of a Lotka-Volterra system with two
    predators (see Equation 5.83 from [Petras2011]_).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            a x - b x y + e x^2 - s x^2 z, \\
        D^\alpha[y](t) & =
            -c y + d x y, \\
        D^\alpha[z](t) & =
            -p z + s x^2 z.
        \end{aligned}

    When taking :math:`p = s = 0` and :math:`e = -r`, we obtain the standard
    predator-prey model :class:`LotkaVolterra2`.
    """

    a: float
    """Parameter in the Lotka-Volterra system."""
    b: float
    """Parameter in the Lotka-Volterra system."""
    c: float
    """Parameter in the Lotka-Volterra system."""
    d: float
    """Parameter in the Lotka-Volterra system."""
    e: float
    """Parameter in the Lotka-Volterra system."""
    p: float
    """Parameter in the Lotka-Volterra system."""
    s: float
    """Parameter in the Lotka-Volterra system."""

    def source(self, t: float, y: Array) -> Array:
        a, b, c, d, e = self.a, self.b, self.c, self.d, self.e
        p, s = self.p, self.s
        return np.array([
            a * y[0] - b * y[0] * y[1] + e * y[0] ** 2 - s * y[0] ** 2 * y[2],
            -c * y[1] + d * y[0] * y[1],
            -p * y[2] + s * y[0] ** 2 * y[2],
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        a, b, c, d, e = self.a, self.b, self.c, self.d, self.e
        p, s = self.p, self.s
        return np.array([
            [
                a - b * y[1] + 2 * e * y[0] - 2 * s * y[0] * y[2],
                -b * y[0],
                -s * y[0] ** 2,
            ],
            [d * y[1], -c + d * y[0], 0],
            [2 * s * y[0] * y[2], 0, -p + s * y[0] ** 2],
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


# {{{ Ma-Chen


@dataclass(frozen=True)
class MaChen(Function):
    r"""Implements the right-hand side of the Ma-Chen financial system (see
    Equation 5.88 from [Petras2011]_).

    .. math::
        \begin{aligned}
        D^\alpha[x](t) & =
            z + (y - a) x, \\
        D^\alpha[y](t) & =
            1 - b y - x^2, \\
        D^\alpha[z](t) & =
            -x - c z.
        \end{aligned}

    where :math:`x` represents the interest rate, :math:`y` represents the
    investment demand, and :math:`z` represents the price index.
    """

    a: float
    """Savings amount."""
    b: float
    """Cost per inverstment."""
    c: float
    """Elasticity of demand of the commercial market."""

    if __debug__:

        def __post_init__(self) -> None:
            if self.a < 0:
                raise ValueError(f"'a' must be positive: {self.a}")

            if self.b < 0:
                raise ValueError(f"'b' must be positive: {self.b}")

            if self.c < 0:
                raise ValueError(f"'c' must be positive: {self.c}")

    def source(self, t: float, y: Array) -> Array:
        return np.array([
            y[2] + (y[1] - self.a) * y[0],
            1.0 - self.b * y[1] - y[0] ** 2,
            -y[0] - self.c * y[2],
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        return np.array([
            [y[1] - self.a, y[0], 1.0],
            [2.0 * y[0], -self.b, 0.0],
            [-1.0, 0.0, -self.c],
        ])


# }}}


# {{{ Newton-Leipnik


@dataclass(frozen=True)
class NewtonLeipnik(Function):
    r"""Implements the right-hand side of the Newton-Leipnik system (see Equation
    5.79 from [Petras2011]_).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            -a x + y + 10 y z, \\
        D^\alpha[y](t) & =
            -x - \frac{4}{10} y + 5 x z, \\
        D^\alpha[z](t) & =
            b z - 5 x y.
        \end{aligned}
    """

    a: float
    """Parameter for the Newton-Leipnik system."""
    b: float
    """Parameter for the Newton-Leipnik system."""

    def source(self, t: float, y: Array) -> Array:
        return np.array([
            -self.a * y[0] + y[1] + 10.0 * y[1] * y[2],
            -y[0] - 0.4 * y[1] + 5.0 * y[0] * y[2],
            self.b * y[2] - 5.0 * y[0] * y[1],
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        return np.array([
            [-self.a, 1.0 + 10.0 * y[2], 10.0 * y[1]],
            [-1.0 + 5.0 * y[2], -0.4, 5.0 * y[0]],
            [-5.0 * y[1], -5.0 * y[0], self.b],
        ])


# }}}


# {{{ Rössler


@dataclass(frozen=True)
class Rossler(Function):
    r"""Implements the right-hand side of the Rössler system (see Equation 5.75
    from [Petras2011]_).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            -y - z, \\
        D^\alpha[y](t) & =
            x + a y, \\
        D^\alpha[z](t) & =
            b + z (x - c).
        \end{aligned}
    """

    a: float
    """Parameter in the Rössler system."""
    b: float
    """Parameter in the Rössler system."""
    c: float
    """Parameter in the Rössler system."""

    def source(self, t: float, y: Array) -> Array:
        return np.array([
            -y[1] - y[2],
            y[0] + self.a * y[1],
            self.b + y[2] * (y[0] - self.c),
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        return np.array([
            [0, -1.0, -1.0],
            [1.0, self.a, 0.0],
            [y[2], 0.0, y[0] - self.c],
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


# {{{ Volta


@dataclass(frozen=True)
class Volta(Function):
    r"""Implements the right-hand side of the Volta system (see Equation 5.98
    from [Petras2011]_).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            -x - a y - y z, \\
        D^\alpha[y](t) & =
            -y - b x - x z, \\
        D^\alpha[y](t) & =
            1.0 + c z + x y.
        \end{aligned}
    """

    a: float
    """Parameter in the Volta system."""
    b: float
    """Parameter in the Volta system."""
    c: float
    """Parameter in the Volta system."""

    def source(self, t: float, y: Array) -> Array:
        return np.array([
            -y[0] - self.a * y[1] - y[1] * y[2],
            -y[1] - self.b * y[0] - y[0] * y[2],
            1.0 + self.c * y[2] + y[0] * y[1],
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        return np.array([
            [-1.0, -self.a - y[2], -y[1]],
            [-self.b - y[2], -1.0, -y[0]],
            [y[1], y[0], self.c],
        ])


# }}}
