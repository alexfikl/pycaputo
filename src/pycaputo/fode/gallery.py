# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, NamedTuple, overload

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
            \beta_4 x^3 - \beta_1 x - \beta_2 y - \beta_3 z.
        \end{aligned}

    This system is similar to :class:`GenesioTesi`, but has a cubic nonlinearity.
    """

    beta: tuple[float, float, float, float]
    """Parameters for the Arneodo system."""

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
            -\beta y - \gamma z.
        \end{aligned}

    The :math:`f` function is given by Equation 5.5 in [Petras2011]_ as

    .. math::

        f(x) = m_1 x + \frac{1}{2} (m_0 - m_1) (|x + 1| - |x - 1|).
    """

    alpha: float
    """Non-dimensional parameter in the Chua system."""
    beta: float
    """Non-dimensional parameter in the Chua system."""
    gamma: float
    """Non-dimensional parameter in the Chua system."""
    m: tuple[float, float]
    """Parameters used to define describing the function :math:`f`."""

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
        parameters of the Chua system given by Equation 5.12 from [Petras2011]_.

        The non-dimensionalization uses the change of variables given in
        Equation 5.3 from [Petras2011]_ with the units mentioned below.

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


# {{{ FitzHugh-Nagumo


@dataclass(frozen=True)
class FitzHughNagumo(Function):
    r"""Implements the right-hand side of the FitzHugh-Nagumo system (see
    Equation 8 in [Brandibur2018]_).

    .. math::

        \begin{aligned}
        D^\alpha[v](t) & =
            v - \frac{v^3}{3} - w + I, \\
        D^\alpha[w](t) & =
            r (v + c - d w),
        \end{aligned}

    where :math:`v` represents the membrane potential and :math:`w` represents
    the recovery variable.

    .. [Brandibur2018] O. Brandibur, E. Kaslik,
        *Stability of Two-component Incommensurate Fractional-order Systems
        and Applications to the Investigation of a FitzHugh-Nagumo Neuronal Model*,
        Mathematical Methods in the Applied Sciences, Vol. 41, pp. 7182--7194, 2018,
        `DOI <https://doi.org/10.1002/mma.4768>`__.
    """

    current: float
    """External excitation current :math:`I`."""
    r: float
    """The time scale parameter in the model. This usually sets the time-scale
    separation between :math:`v` (fast) and :math:`w` (slow).
    """
    c: float
    """An offset parameter in the model. Theoretically, this sets the position
    of the :math:`w`-nullcline relative to the :math:`v`-nullcline.
    """
    d: float
    """A scaling parameter for :math:`w`. Functionally, this influences the
    damping of the excitation.
    """

    def source(self, t: float, y: Array) -> Array:
        r, c, d = self.r, self.c, self.d
        return np.array([
            y[0] - y[0] ** 3 / 3 - y[1] + self.current,
            r * (y[0] + c - d * y[1]),
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        r, _, d = self.r, self.c, self.d
        return np.array([
            [1.0 - y[0] ** 2, -1.0],
            [r, -r * d],
        ])


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
    """Parameters for the Genesio-Tesi system."""

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


# {{{ Hindmarsh-Rose


@dataclass(frozen=True)
class HindmarshRose2(Function):
    r"""Implements the right-hand side of the two-dimensional Hindmarsh-Rose
    system (see Equation 3.1 from [Kaslik2017]_).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            y - a x^3 + b x^2 + I, \\
        D^\alpha[y](t) & =
            c - d x^2 - y.
        \end{aligned}

    .. [Kaslik2017] E. Kaslik,
        *Analysis of Two- And Three-Dimensional Fractional-Order Hindmarsh-Rose
        Type Neuronal Models*,
        Fractional Calculus and Applied Analysis, Vol. 20, pp. 623--645, 2017,
        `DOI <https://doi.org/10.1515/fca-2017-0033>`__.
    """

    current: float
    """External forcing current."""
    a: float
    """Parameter in the Hindmarsh-Rose model."""
    b: float
    """Parameter in the Hindmarsh-Rose model."""
    c: float
    """Parameter in the Hindmarsh-Rose model."""
    d: float
    """Parameter in the Hindmarsh-Rose model."""

    if __debug__:

        def __post_init__(self) -> None:
            if self.a <= 0:
                raise ValueError(f"Parameter 'a' must be positive: {self.a}")

            if self.b <= 0:
                raise ValueError(f"Parameter 'b' must be positive: {self.b}")

            if self.c <= 0:
                raise ValueError(f"Parameter 'c' must be positive: {self.c}")

            if self.d <= 0:
                raise ValueError(f"Parameter 'd' must be positive: {self.d}")

    @classmethod
    def get_resting_potential(cls, a: float, b: float, c: float, d: float) -> float:
        r0 = c / a
        p = (b - d) / a

        h = np.polynomial.Polynomial([-r0, 0.0, -p, 1.0], symbol="x")
        x = h.roots()
        x0 = np.min(x[np.isreal(x)].real)
        assert x0 < min(0.0, 2.0 / 3.0 * p)

        return float(x0)

    def source(self, t: float, y: Array) -> Array:
        return np.array([
            y[1] - self.a * y[0] ** 3 + self.b * y[0] ** 2 + self.current,
            self.c - self.d * y[0] ** 2 - y[1],
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        return np.array([
            [-3.0 * self.a * y[0] ** 2 + 2.0 * self.b * y[0], 1.0],
            [-2.0 * self.d * y[0], -1.0],
        ])


@dataclass(frozen=True)
class HindmarshRose3(HindmarshRose2):
    r"""Implements the right-hand side of the three-dimensional Hindmarsh-Rose
    system (see Equation 4.1 from [Kaslik2017]_).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            y - a x^3 + b x^2 - z + I, \\
        D^\alpha[y](t) & =
            c - d x^2 - y, \\
        D^\alpha[z](t) & =
            \epsilon (s (x - x_0) - z),
        \end{aligned}

    where the third equation adds a slow adaptation current that acts as a
    bursting variable.
    """

    epsilon: float
    """Parameter in the Hindmarsh-Rose system."""
    s: float
    """Parameter in the Hindmarsh-Rose system."""
    x0: float
    """Parameter in the Hindmarsh-Rose system. """

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()

            if not 0.0 < self.epsilon < 1:
                raise ValueError(f"Parameter 'epsilon' not in (0, 1): {self.epsilon}")

            if self.s <= 0:
                raise ValueError(f"Parameter 's' must be positive: {self.s}")

    def source(self, t: float, y: Array) -> Array:
        return np.array([
            y[1] - self.a * y[0] ** 3 + self.b * y[0] ** 2 - y[2] + self.current,
            self.c - self.d * y[0] ** 2 - y[1],
            self.epsilon * (self.s * (y[0] - self.x0) - y[2]),
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        return np.array([
            [-3.0 * self.a * y[0] ** 2 + 2.0 * self.b * y[0], 1.0, -1.0],
            [-2.0 * self.d * y[0], -1.0, 0.0],
            [self.epsilon * self.s, 0.0, -self.epsilon],
        ])


@dataclass(frozen=True)
class HindmarshRose4(HindmarshRose3):
    r"""Implements the right-hand side of the extended four-dimensional
    Hindmarsh-Rose system (see Equation 2.11 from [Giresse2019]_).

    .. math::

        \begin{aligned}
        D^\alpha[x](t) & =
            y - a x^3 + b x^2 - z + I, \\
        D^\alpha[y](t) & =
            c - d x^2 - y - e w, \\
        D^\alpha[z](t) & =
            \epsilon (s (x - x_0) - z), \\
        D^\alpha[w](t) & =
            h (f(y + g) - p w),
        \end{aligned}

    where the fourth equation adds another slow variable. This variable is meant
    to model the slow intracellular exchange of Calcium ions between the cytoplasm
    and its store.

    .. [Giresse2019] T. A. Giresse, K. T. Crepin, T. Martin,
        *Generalized Synchronization of the Extended Hindmarsh-Rose Neuronal
        Model With Fractional Order Derivative*,
        Chaos, Solitons &Amp; Fractals, Vol. 118, pp. 311--319, 2019,
        `DOI <https://doi.org/10.1016/j.chaos.2018.11.028>`__.
    """

    e: float
    """Parameter in the Hindmarsh-Rose system."""
    f: float
    """Parameter in the Hindmarsh-Rose system."""
    g: float
    """Parameter in the Hindmarsh-Rose system. """
    h: float
    """Parameter in the Hindmarsh-Rose system."""
    p: float
    """Parameter in the Hindmarsh-Rose system. """

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()

            # TODO: verify if these assumptions are necessary? It's unclear from
            # the papers what these variables represent and what ranges are allowed
            if self.e <= 0:
                raise ValueError(f"Parameter 'e' should be positive: {self.e}")

            if self.f <= 0:
                raise ValueError(f"Parameter 'f' should be positive: {self.f}")

            if self.g <= 0:
                raise ValueError(f"Parameter 'g' should be positive: {self.g}")

            if self.h > self.epsilon or self.h < 0:
                raise ValueError(
                    "Parameter 'h' should be smaller than 'epsilon': "
                    f"h = {self.h} and epsilon = {self.epsilon}"
                )

            if self.p <= 0:
                raise ValueError(f"Parameter 'p' should be positive: {self.p}")

    def source(self, t: float, y: Array) -> Array:
        return np.array([
            y[1] - self.a * y[0] ** 3 + self.b * y[0] ** 2 - y[2] + self.current,
            self.c - self.d * y[0] ** 2 - y[1] - self.e * y[3],
            self.epsilon * (self.s * (y[0] - self.x0) - y[2]),
            self.h * (self.f * (y[1] - self.g) - self.p * y[3]),
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        return np.array([
            [-3.0 * self.a * y[0] ** 2 + 2.0 * self.b * y[0], 1.0, -1.0, 0.0],
            [-2.0 * self.d * y[0], -1.0, 0.0, -self.e],
            [self.epsilon * self.s, 0.0, -self.epsilon, 0.0],
            [0.0, self.h * self.f, 0.0, -self.h * self.p],
        ])


# }}}


# {{{ Hodgkin-Huxley


@dataclass(frozen=True)
class HodgkinHuxleyParameter(ABC):
    r"""Parameters for the Hodgkin-Huxley system defined in [Nagy2014]_.

    This class should be subclassed to implement the rate functions
    :math:`(\alpha_i(V), \beta_i(V))` for the model as well as their first-order
    derivatives. The derivatives are used when computing the Jacobian in
    implicit methods.
    """

    current: float
    """External current input."""
    C: float
    """Membrane capacitance."""

    g_Na: float  # noqa: N815
    r"""Maximum conductance (mS/cm^2) of the :math:`\mathrm{Na}^+` ionic current."""
    g_K: float  # noqa: N815
    r"""Maximum conductance (mS/cm^2) of the :math:`\mathrm{K}^+` ionic current."""
    g_L: float  # noqa: N815
    """Maximum conductance (mS/cm^2) of the leak current."""

    V_Na: float
    r"""Equilibrium potential (mV) for the :math:`\mathrm{Na}^{+}` ion channel."""
    V_K: float
    r"""Equilibrium potential (mV) for the :math:`\mathrm{K}^{+}` ion channel."""
    V_L: float
    """Equilibrium potential (mV) for the leak current."""

    if __debug__:

        def __post_init__(self) -> None:
            if self.C <= 0.0:
                raise ValueError(f"Capacitance 'C' most be positive: {self.C}")

            if self.g_Na <= 0.0:
                raise ValueError(f"Conductance 'g_Na' most be positive: {self.g_Na}")

            if self.g_K <= 0.0:
                raise ValueError(f"Conductance 'g_K' most be positive: {self.g_K}")

            if self.g_L <= 0.0:
                raise ValueError(f"Conductance 'g_L' most be positive: {self.g_L}")

    def get_steady_state(self, V0: float) -> tuple[float, float, float]:
        r"""For each variable, the steady state is achieved with

        .. math::

            p(V) = \frac{\alpha_p(V)}{\alpha_p(V) + \beta_p(V)}

        where :math:`p = n, m, h`.
        """
        alpha_n, alpha_m, alpha_h = self.alpha(V0)
        beta_n, beta_m, beta_h = self.beta(V0)
        return (
            alpha_n / (alpha_n + beta_n),
            alpha_m / (alpha_m + beta_m),
            alpha_h / (alpha_h + beta_h),
        )

    @overload
    def alpha(self, V: float) -> tuple[float, float, float]:
        pass

    @overload
    def alpha(self, V: Array) -> tuple[Array, Array, Array]:
        pass

    @abstractmethod
    def alpha(self, V: Any) -> tuple[Any, Any, Any]:
        r"""Rate function for all variables."""

    @overload
    def beta(self, V: float) -> tuple[float, float, float]:
        pass

    @overload
    def beta(self, V: Array) -> tuple[Array, Array, Array]:
        pass

    @abstractmethod
    def beta(self, V: Any) -> tuple[Any, Any, Any]:
        r"""Rate function for all variables."""

    @abstractmethod
    def alpha_derivative(self, V: float) -> tuple[float, float, float]:
        r"""Derivative of the rate function for all variables."""

    @abstractmethod
    def beta_derivative(self, V: float) -> tuple[float, float, float]:
        r"""Derivative of the rate function for all variables."""

    @classmethod
    def from_name(cls, name: str) -> HodgkinHuxleyParameter:
        """Get a parameter set from a given paper.

        * Sets named ``NagySetXX`` from [Nagy2014]_. Currently only a single set
          of parameters is available.
        """

        if name in HODGKIN_HUXLEY_PARAMETERS:
            return HODGKIN_HUXLEY_PARAMETERS[name]

        return HodgkinHuxleyParameter.from_name(name)


@dataclass(frozen=True)
class StandardHodgkinHuxleyParameter(HodgkinHuxleyParameter):
    r"""Implements the standard Hodgkin-Huxley rate functions from [Nagy2014]_.

    .. math::

        \begin{aligned}
        \alpha_n(V) & =
            a_n (A_n - V) \left[\exp\left(\frac{A_n - V}{A_0}\right) - 1\right]^{-1}, \\
        \alpha_m(V) & =
            a_m (A_m - V) \left[\exp\left(\frac{A_m - V}{A_1}\right) - 1\right]^{-1}, \\
        \alpha_h(V) & =
            a_h \exp\left(\frac{A_h - V}{A_2}\right), \\
        \beta_n(V) & =
            b_n \exp\left(\frac{B_n - V}{B_0}\right), \\
        \beta_m(V) & =
            b_m \exp\left(\frac{B_m - V}{B_1}\right), \\
        \beta_h(V) & =
            b_h \left[\exp\left(\frac{B_h - V}{B_2}\right) + 1\right].
        \end{aligned}
    """

    a: tuple[float, float, float]
    """Amplitudes of the rate functions."""
    Ap: tuple[float, float, float]
    r"""Offsets of the potential in the :math:`\alpha` rate functions."""
    Ai: tuple[float, float, float]
    r"""Scaling of the potential in the :math:`\alpha` rate functions."""
    b: tuple[float, float, float]
    """Amplitudes of the rate functions."""
    Bp: tuple[float, float, float]
    r"""Offsets of the potential in the :math:`\beta` rate functions."""
    Bi: tuple[float, float, float]
    r"""Scaling of the potential in the :math:`\beta` rate functions."""

    def alpha(self, V: Any) -> tuple[Any, Any, Any]:
        an, am, ah = self.a
        An, Am, Ah = self.Ap
        Sn, Sm, Sh = self.Ai

        En = np.exp((An - V) / Sn)
        Em = np.exp((Am - V) / Sm)
        Eh = np.exp((Ah - V) / Sh)

        return (
            an * (An - V) / (En - 1),
            am * (Am - V) / (Em - 1),
            ah * Eh,
        )

    def alpha_derivative(self, V: float) -> tuple[float, float, float]:
        an, am, ah = self.a
        An, Am, Ah = self.Ap
        Sn, Sm, Sh = self.Ai

        En = np.exp((An - V) / Sn)
        Em = np.exp((Am - V) / Sm)
        Eh = np.exp((Ah - V) / Sh)

        return (
            -an / (En - 1) + an * (An - V) * En / Sn / (En - 1) ** 2,
            -am / (Em - 1) + am * (Am - V) * Em / Sm / (Em - 1) ** 2,
            -ah * Eh / Sh,
        )

    def beta(self, V: Any) -> tuple[Any, Any, Any]:
        bn, bm, bh = self.b
        Bn, Bm, Bh = self.Bp
        Sn, Sm, Sh = self.Bi

        return (
            bn * np.exp((Bn - V) / Sn),
            bm * np.exp((Bm - V) / Sm),
            bh / (np.exp((Bh - V) / Sh) + 1),
        )

    def beta_derivative(self, V: float) -> tuple[float, float, float]:
        bn, bm, bh = self.b
        Bn, Bm, Bh = self.Bp
        Sn, Sm, Sh = self.Bi

        En = np.exp((Bn - V) / Sn)
        Em = np.exp((Bm - V) / Sm)
        Eh = np.exp((Bh - V) / Sh)

        return (
            -bn * En / Sn,
            -bm * Em / Sm,
            bh * Eh / Sh / (Eh + 1) ** 2,
        )


HODGKIN_HUXLEY_PARAMETERS: dict[str, HodgkinHuxleyParameter] = {
    "HodgkinHuxley": StandardHodgkinHuxleyParameter(
        current=0.0,
        C=1.0,
        g_Na=120.0,
        g_K=36.0,
        g_L=0.3,
        V_Na=120.0,
        V_K=-12.0,
        V_L=10.6,
        a=(-0.01, -0.1, 0.07),
        Ap=(-10.0, -25.0, 0.0),
        Ai=(-10.0, -10.0, -20.0),
        b=(0.125, 4.0, 1.0),
        Bp=(0.0, 0.0, -30.0),
        Bi=(-80.0, -18.0, -10.0),
    ),
    "NagyFigure4": StandardHodgkinHuxleyParameter(
        current=0.0,
        C=1.0,
        g_Na=120.0,
        g_K=36.0,
        g_L=0.3,
        V_Na=120.0,
        V_K=-12.0,
        V_L=10.6,
        a=(0.01, 0.1, 0.07),
        Ap=(10.0, 25.0, 0.0),
        Ai=(10.0, 10.0, 20.0),
        b=(0.125, 4.0, 1.0),
        Bp=(0.0, 0.0, 30.0),
        Bi=(80.0, 18.0, 10.0),
    ),
}


@dataclass(frozen=True)
class HodgkinHuxley(Function):
    r"""Implements the right-hand side of the Hodgkin-Huxley system (see
    Equation 13 and Equation 15 in [Nagy2014]_).

    .. math::

        \begin{aligned}
        D^\alpha[V](t) & =
            I
            - g_{\mathrm{Na}} m^3 h (V - V_{\mathrm{Na}})
            - g_{\mathrm{K}} n^4 (V - V_{\mathrm{K}})
            - g_{\mathrm{L}} (V - V_{\mathrm{L}}), \\
        D^\alpha[n](t) & =
            \alpha_n(V) (1 - n) - \beta_n(V) n, \\
        D^\alpha[m](t) & =
            \alpha_m(V) (1 - m) - \beta_m(V) m, \\
        D^\alpha[h](t) & =
            \alpha_h(V) (1 - h) - \beta_h(V) h, \\
        \end{aligned}

    where the rate functions :math:`(\alpha_i, \beta_i)` are given by the
    specific implementation of :class:`HodgkinHuxleyParameter`.

    .. [Nagy2014] A. M. Nagy, N. H. Sweilam,
        *An Efficient Method for Solving Fractional Hodgkin-Huxley Model*,
        Physics Letters A, Vol. 378, pp. 1980--1984, 2014,
        `DOI <https://doi.org/10.1016/j.physleta.2014.06.012>`__.
    """

    p: HodgkinHuxleyParameter
    """Parameters for the Hodgkin-Huxley model."""

    def source(self, t: float, y: Array) -> Array:
        g_Na, g_K, g_L = self.p.g_Na, self.p.g_K, self.p.g_L
        V_Na, V_K, V_L = self.p.V_Na, self.p.V_K, self.p.V_L
        Iext, C = self.p.current, self.p.C

        alpha_n, alpha_m, alpha_h = self.p.alpha(y[0])
        beta_n, beta_m, beta_h = self.p.beta(y[0])

        return np.array([
            (
                Iext
                - g_Na * y[2] ** 3 * y[3] * (y[0] - V_Na)
                - g_K * y[1] ** 4 * (y[0] - V_K)
                - g_L * (y[0] - V_L)
            )
            / C,
            alpha_n * (1 - y[1]) - beta_n * y[1],
            alpha_m * (1 - y[2]) - beta_m * y[2],
            alpha_h * (1 - y[3]) - beta_h * y[3],
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        g_Na, g_K, g_L = self.p.g_Na, self.p.g_K, self.p.g_L
        V_Na, V_K = self.p.V_Na, self.p.V_K
        C = self.p.C

        alpha_n, alpha_m, alpha_h = self.p.alpha(y[0])
        d_alpha_n, d_alpha_m, d_alpha_h = self.p.alpha_derivative(y[0])
        beta_n, beta_m, beta_h = self.p.beta(y[0])
        d_beta_n, d_beta_m, d_beta_h = self.p.beta_derivative(y[0])

        return np.array([
            [
                -(g_Na * y[2] ** 3 * y[3] + g_K * y[1] ** 4 + g_L) / C,
                -4.0 * g_K * y[1] ** 3 * (y[0] - V_K) / C,
                -3.0 * g_Na * y[2] ** 2 * y[3] * (y[0] - V_Na) / C,
                -g_Na * y[2] ** 3 * (y[0] - V_Na) / C,
            ],
            [d_alpha_n * (1 - y[1]) - d_beta_n * y[1], -alpha_n - beta_n, 0, 0],
            [d_alpha_m * (1 - y[2]) - d_beta_m * y[2], 0, -alpha_m - beta_m, 0],
            [d_alpha_h * (1 - y[3]) - d_beta_h * y[3], 0, 0, -alpha_h - beta_h],
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
            -x - c z,
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


# {{{ Morris-Lecar


@dataclass(frozen=True)
class MorrisLecarParameter(ABC):
    r"""Parameters for the Morris-Lecar system, as defined in [Shi2014]_.

    This class should be subclassed to implement the modulation functions
    :meth:`w_inf`, :meth:`m_inf` and :meth:`tau_w` as well as their
    first-order derivatives. The derivatives are used for computing the Jacobian
    in implicit methods.
    """

    C: float
    """Membrane capacitance."""
    g_Ca: float  # noqa: N815
    r"""Maximum conductance of the :math:`\mathrm{Ca}^{2+}` ionic current."""
    g_K: float  # noqa: N815
    r"""Maximum conductance of the :math:`\mathrm{K}^{+}` ionic current."""
    g_L: float  # noqa: N815
    r"""Maximum conductance of the leak current."""
    v_Ca: float  # noqa: N815
    r"""Equilibrium potential for the :math:`\mathrm{Ca}^{2+}` ion channel."""
    v_K: float  # noqa: N815
    r"""Equilibrium potential for the :math:`\mathrm{K}^{+}` ion channel."""
    v_L: float  # noqa: N815
    r"""Equilibrium potential for the leak current."""

    epsilon: float
    """A small parameter that controls the time scale between spiking and modulation."""
    v_0: float
    """An initial current applied at the starting time."""

    if __debug__:

        def __post_init__(self) -> None:
            if self.C <= 0.0:
                raise ValueError(f"Capacitance 'C' most be positive: {self.C}")

            if self.g_Ca <= 0.0:
                raise ValueError(f"Conductance 'g_Ca' most be positive: {self.g_Ca}")

            if self.g_K <= 0.0:
                raise ValueError(f"Conductance 'g_K' most be positive: {self.g_K}")

            if self.g_L <= 0.0:
                raise ValueError(f"Conductance 'g_L' most be positive: {self.g_L}")

    @abstractmethod
    def m_inf(self, V: float) -> float:
        r"""Modulation of the :math:`g_{\mathrm{Ca}}` conductance."""

    def m_inf_derivative(self, V: float) -> float:
        r"""Derivative of :math:`m_\infty`."""
        raise NotImplementedError

    @abstractmethod
    def w_inf(self, V: float) -> float:
        r"""Modulation of the :math:`g_{\mathrm{K}}` conductance."""

    def w_inf_derivative(self, V: float) -> float:
        r"""Derivative of :math:`w_\infty`."""
        raise NotImplementedError

    @abstractmethod
    def tau_w(self, V: float) -> float:
        r"""Modulation of the time scale of the activation variable :math:`w`."""

    def tau_w_derivative(self, V: float) -> float:
        r"""Derivative of :math:`\tau_w`."""
        raise NotImplementedError


@dataclass(frozen=True)
class StandardMorrisLecarParameter(MorrisLecarParameter):
    r"""Implements the standard Morris-Lecar modulation functions.

    .. math::

        \begin{aligned}
        m_\infty(V) & =
            \frac{1}{2} \left(1 + \tanh\left(\frac{V - V_1}{V_2}\right)\right), \\
        w_\infty(V) & =
            \frac{1}{2} \left(1 + \tanh\left(\frac{V - V_3}{V_4}\right)\right), \\
        \tau_w(V) & =
            \left(\phi \cosh\left(\frac{V - V_3}{2 V_4}\right)\right)^{-1}.
        \end{aligned}
    """

    phi: float
    r"""Reference frequency. Taken to be :math:`\phi = 1/3` in [Shi2014]_."""
    vinf: tuple[float, float, float, float]
    """A tuple of 4 parameters used to define the modulation functions."""

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()

            if self.phi <= 0 or self.phi > 1.0:
                raise ValueError(f"Frequency 'phi' must be in [0, 1]: {self.phi}")

    def m_inf(self, V: float) -> float:
        V = (V - self.vinf[0]) / self.vinf[1]
        return (1.0 + np.tanh(V)) / 2.0  # type: ignore[no-any-return]

    def m_inf_derivative(self, V: float) -> float:
        V = (V - self.vinf[0]) / self.vinf[1]
        return 1.0 / np.cosh(V) ** 2 / (2.0 * self.vinf[1])  # type: ignore[no-any-return]

    def w_inf(self, V: float) -> float:
        V = (V - self.vinf[2]) / self.vinf[3]
        return (1.0 + np.tanh(V)) / 2.0  # type: ignore[no-any-return]

    def w_inf_derivative(self, V: float) -> float:
        V = (V - self.vinf[2]) / self.vinf[3]
        return 1.0 / np.cosh(V) ** 2 / (2.0 * self.vinf[3])  # type: ignore[no-any-return]

    def tau_w(self, V: float) -> float:
        V = (V - self.vinf[2]) / self.vinf[3]
        return 1.0 / (self.phi * np.cosh(V / 2.0))  # type: ignore[no-any-return]

    def tau_w_derivative(self, V: float) -> float:
        V = (V - self.vinf[2]) / self.vinf[3]
        return (  # type: ignore[no-any-return]
            -2.0 / (self.phi * self.vinf[3]) * np.sinh(V / 2.0) ** 3 / np.sinh(V) ** 2
        )


@dataclass(frozen=True)
class MorrisLecar(Function):
    r"""Implements the right-hand side of the Morris-Lecar system (see Equation 8
    in [Shi2014]_).

    .. math::

        \begin{aligned}
        C D^\alpha[V](t) & =
            g_{\mathrm{Ca}} m_\infty(V) (V_{\mathrm{Ca}} - V)
            + g_{\mathrm{K}} w (V_{\mathrm{K}} - V)
            + g_{\mathrm{L}} (V_{\mathrm{L}} - V) + I, \\
        D^\alpha[w](t) & =
            \frac{w_\infty(V) - w}{\tau_w(V)}, \\
        D^\alpha[I](t) & =
            - \epsilon (V_0 + V),
        \end{aligned}

    where :math:`V` is membrane potential, :math:`w` is the activation variable
    for :math:`\mathrm{K}^+`, and :math:`I` is the current. The additional
    parameters and functions can be customized using the :class:`MorrisLecarParameter`
    class.

    Note that this is not the standard Morris-Lecar model, as it has an extra
    third equation proposed by Izhikevich. The original form can be retrieved by
    setting :math:`\epsilon = 0` in the above equation.

    .. [Shi2014] M. Shi, Z. Wang,
        *Abundant Bursting Patterns of a Fractional-Order Morris-Lecar Neuron Model*,
        Communications in Nonlinear Science and Numerical Simulation, Vol. 19, pp.
        1956--1969, 2014,
        `DOI <https://doi.org/10.1016/j.cnsns.2013.10.032>`__.
    """

    p: MorrisLecarParameter
    """Parameters for the Morris-Lecar system."""

    def source(self, t: float, y: Array) -> Array:
        p = self.p
        g_Ca = p.g_Ca / p.C
        g_K = p.g_K / p.C
        g_L = p.g_L / p.C

        m_inf = p.m_inf(y[0])
        w_inf = p.w_inf(y[0])
        tau_w = p.tau_w(y[0])

        return np.array([
            g_Ca * m_inf * (p.v_Ca - y[0])
            + g_K * y[1] * (p.v_K - y[0])
            + g_L * (p.v_L - y[0])
            + y[2] / p.C,
            (w_inf - y[1]) / tau_w,
            -p.epsilon * (p.v_0 + y[0]),
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        p = self.p
        g_Ca = p.g_Ca / p.C
        g_K = p.g_K / p.C
        g_L = p.g_L / p.C

        m_inf, d_m_inf = p.m_inf(y[0]), p.m_inf_derivative(y[0])
        w_inf, d_w_inf = p.w_inf(y[0]), p.w_inf_derivative(y[0])
        tau_w, d_tau_w = p.tau_w(y[0]), p.tau_w_derivative(y[0])

        return np.array([
            [
                g_Ca * d_m_inf * (p.v_Ca - y[0]) - g_Ca * m_inf - g_K * y[1] - g_L,
                p.g_K * (p.v_K - y[0]) / p.C,
                1.0 / p.C,
            ],
            [d_w_inf / tau_w - (w_inf - y[1]) / tau_w**2 * d_tau_w, -1.0 / tau_w, 0.0],
            [-p.epsilon, 0.0, 0.0],
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
