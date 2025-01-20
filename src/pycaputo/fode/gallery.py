# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from pycaputo.logging import get_logger
from pycaputo.typing import Array

from .special import Function

log = get_logger(__name__)


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


# {{{ Chua


@dataclass(frozen=True)
class Chua(Function):
    r"""Implements the right-hand side of the Chua system (see Equation 5.13
    from [Petras2011]_).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            \alpha (y - x + f(x)), \\
        D^\alpha[y](t) & =
            x - y + z, \\
        D^\alpha[z](t) & =
            -\beta y - \gamma z
        \end{aligned}
    """

    alpha: float
    """Non-dimensional parameter in the Chua system."""
    beta: float
    """Non-dimensional parameter in the Chua system."""
    gamma: float
    """Non-dimensional parameter in the Chua system."""
    m: tuple[float, float]
    """Parameters used to define describing the diode (Equation 5.5)."""

    def source(self, t: float, y: Array) -> Array:
        m0, m1 = self.m
        fx = m1 * y[0] + 0.5 * (m0 - m1) * (abs(y[0] + 1) - abs(y[0] - 1))

        return np.array([
            self.alpha * (y[1] - y[0] - fx),
            y[0] - y[1] + y[2],
            -self.beta * y[1] - self.gamma * y[2],
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        m0, m1 = self.m
        df = m0 if abs(y[0]) < 1.0 else m1

        return np.array([
            [-self.alpha * (df + 1.0), self.alpha, 0.0],
            [1.0, -1.0, 1.0],
            [0.0, -self.beta, -self.gamma],
        ])

    @classmethod
    def from_dim(
        cls,
        C1: float,
        C2: float,
        R2: float,
        RL: float,
        L1: float,
        Ga: float,
        Gb: float,
    ) -> Chua:
        """Construct the non-dimensional system from the standard dimensional
        parameters of the Chua system.

        :arg C1: capacitance (in nano Farad) of first capacitor.
        :arg C2: capacitance (in nano Farad) of second capacitor.
        :arg R2: resistance (in Ohm).
        :arg RL: resistance (in Ohm).
        :arg L1: inductance (in mili Henry).
        :arg Ga: slope of nonlinear function (in mili Ampere per Volt).
        :arg Gb: slope of nonlinear function (in mili Ampere per Volt).
        """
        G = 1.0 / R2
        alpha = C2 / C1
        beta = C2 / (L1 * G**2)
        gamma = C2 * R2 / (L1 * G)
        m0 = Ga / G
        m1 = Gb / G

        return Chua(alpha=alpha, beta=beta, gamma=gamma, m=(m0, m1))


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


# {{{ FitzHugh-Rinzel


class FitzHughRinzelParameter(NamedTuple):
    """Parameters for the :class:`FitzHughRinzel` system."""

    current: float
    """External current applied to the system."""
    a: float
    """Parameters in the FitzHugh-Rinzel system."""
    b: float
    """Parameters in the FitzHugh-Rinzel system."""
    c: float
    """Parameters in the FitzHugh-Rinzel system."""
    d: float
    """Parameters in the FitzHugh-Rinzel system."""
    delta: float
    """Parameters in the FitzHugh-Rinzel system."""
    mu: float
    """Parameters in the FitzHugh-Rinzel system."""

    @classmethod
    def from_name(cls, name: str) -> FitzHughRinzelParameter:
        """Parameters from [Mondal2019]_ are available and are named like
        ``MondalSetXX``.
        """

        return FITZHUGH_RINZEL_PARAMETERS[name]


FITZHUGH_RINZEL_PARAMETERS: dict[str, FitzHughRinzelParameter] = {
    "MondalSetI": FitzHughRinzelParameter(
        current=0.3125,
        a=0.7,
        b=0.8,
        c=-0.775,
        d=1.0,
        delta=0.08,
        mu=0.0001,
    ),
    "MondalSetII": FitzHughRinzelParameter(
        current=0.4,
        a=0.7,
        b=0.8,
        c=-0.775,
        d=1.0,
        delta=0.08,
        mu=0.0001,
    ),
    "MondalSetIII": FitzHughRinzelParameter(
        current=3.0,
        a=0.7,
        b=0.8,
        c=-0.775,
        d=1.0,
        delta=0.08,
        mu=0.18,
    ),
    "MondalSetIV": FitzHughRinzelParameter(
        current=0.3125,
        a=0.7,
        b=0.8,
        c=1.3,
        d=1.0,
        delta=0.08,
        mu=0.0001,
    ),
    "MondalSetV": FitzHughRinzelParameter(
        current=0.3125,
        a=0.7,
        b=0.8,
        c=-0.908,
        d=1.0,
        delta=0.08,
        mu=0.002,
    ),
}


@dataclass(frozen=True)
class FitzHughRinzel(Function):
    r"""Implements the right-hand side of the FitzHugh-Rinzel system (see
    Equation 1 from [Mondal2019]_).

    .. math::

        \begin{aligned}
        D^\alpha[v](t) & =
            v - \frac{v^3}{3} - w + y + I, \\
        D^\alpha[w](t) & =
            \delta (a + v - b w), \\
        D^\alpha[y](t) & =
            \mu (c - v - d y)
        \end{aligned}

    where :math:`v` represents the membrane voltage, :math:`w` represents the
    recovery variable and :math:`y` represents the slow modulation of the
    current.

    .. [Mondal2019] A. Mondal, S. K. Sharma, R. K. Upadhyay, A. Mondal,
        *Firing Activities of a Fractional-Order FitzHugh-Rinzel Bursting
        Neuron Model and Its Coupled Dynamics*,
        Scientific Reports, Vol. 9, 2019,
        `DOI <https://doi.org/10.1038/s41598-019-52061-4>`__.
    """

    p: FitzHughRinzelParameter
    """Parameters in the FitzHugh-Rinzel system."""

    def source(self, t: float, y: Array) -> Array:
        I, a, b, c, d, delta, mu = self.p  # noqa: E741

        return np.array([
            y[0] - y[0] ** 3 / 3 - y[1] + y[2] + I,
            delta * (a + y[0] - b * y[1]),
            mu * (c - y[0] - d * y[2]),
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        _, _, b, _, d, delta, mu = self.p
        return np.array([
            [1.0 - y[0] ** 2, -1.0, 1.0],
            [delta, -b * delta, 0.0],
            [-mu, 0.0, -d * mu],
        ])


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


# {{{ Labyrinth


@dataclass(frozen=True)
class Labyrinth(Function):
    r"""Implements the right-hand side of the Labyrinth system (see Equation 3
    from `here <https://doi.org/10.1063/1.1772551>`__).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            \sin y - b x, \\
        D^\alpha[y](t) & =
            \sin z - b y, \\
        D^\alpha[z](t) & =
            \sin x - b z.
        \end{aligned}
    """

    b: float
    """Parameter in the Labyrinth system."""

    def source(self, t: float, y: Array) -> Array:
        b = self.b
        return np.array([
            np.sin(y[1]) - b * y[0],
            np.sin(y[2]) - b * y[1],
            np.sin(y[0]) - b * y[2],
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        b = self.b
        return np.array([
            [-b, np.cos(y[1]), 0],
            [0, -b, np.cos(y[2])],
            [np.cos(y[0]), 0, -b],
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


# {{{ Lorenz84


@dataclass(frozen=True)
class Lorenz84(Function):
    r"""Implements the right-hand side of the Lorenz-84 system (see Equations 1-3
    from [Lorenz1984]_). This is also known as the Hadley system.

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            -y^2 - z^2 - a x + a F, \\
        D^\alpha[y](t) & =
            x y - b x z - y + G, \\
        D^\alpha[z](t) & =
            b x y + x z - z.
        \end{aligned}

    .. [Lorenz1984] E. N. Lorenz,
        *Irregularity: A Fundamental Property of the Atmosphere*,
        Tellus A, Vol. 36A, pp. 98--110, 1984,
        `DOI <https://doi.org/10.1111/j.1600-0870.1984.tb00230.x>`__.
    """

    a: float
    """Parameter in the Lorenz-84 system."""
    b: float
    """Parameter in the Lorenz-84 system."""
    F: float
    """Parameter in the Lorenz-84 system."""
    G: float
    """Parameter in the Lorenz-84 system."""

    def source(self, t: float, y: Array) -> Array:
        return np.array([
            -(y[1] ** 2) - y[2] ** 2 - self.a * y[0] + self.a * self.F,
            y[0] * y[1] - self.b * y[0] * y[2] - y[1] + self.G,
            self.b * y[0] * y[1] + y[0] * y[2] - y[2],
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        return np.array([
            [-self.a, -2.0 * y[1], -2.0 * y[2]],
            [y[1] - self.b * y[2], y[0] - 1.0, -self.b * y[0]],
            [self.b * y[1] + y[2], self.b * y[0], y[0] - 1.0],
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


# {{{ Qi


@dataclass(frozen=True)
class Qi(Function):
    r"""Implements the right-hand side of the Qi system (see Equation 6
    from [Qi2005]_).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            a (y - x) + y z, \\
        D^\alpha[y](t) & =
            c x - y - x z, \\
        D^\alpha[z](t) & =
            x y - b z.
        \end{aligned}

    .. [Qi2005] G. Qi, G. Chen, S. Du, Z. Chen, Z. Yuan,
        *Analysis of a New Chaotic System*,
        Physica A: Statistical Mechanics and Its Applications,
        Vol. 352, pp. 295--308, 2005,
        `DOI <https://doi.org/10.1016/j.physa.2004.12.040>`__.
    """

    a: float
    """Parameter in the Qi system."""
    b: float
    """Parameter in the Qi system."""
    c: float
    """Parameter in the Qi system."""

    def source(self, t: float, y: Array) -> Array:
        return np.array([
            self.a * (y[1] - y[0]) + y[1] * y[2],
            self.c * y[0] - y[1] - y[0] * y[2],
            y[0] * y[1] - self.b * y[2],
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        return np.array([
            [-self.a, self.a + y[2], y[1]],
            [self.c - y[2], -1.0, -y[0]],
            [y[1], y[0], -self.b],
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
