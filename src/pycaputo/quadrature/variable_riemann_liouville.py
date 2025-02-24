# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, overload

import numpy as np

from pycaputo.derivatives import VariableExponentialCaputoDerivative
from pycaputo.grid import Points, UniformPoints
from pycaputo.typing import Array, ArrayOrScalarFunction

from .base import QuadratureMethod, quad


@dataclass(frozen=True)
class VariableOrderMethod(QuadratureMethod):
    """Quadrature method for variable-order fractional operators."""


# {{{ rectangular


@dataclass(frozen=True)
class ExponentialRectangular(VariableOrderMethod):
    """This approximates the integral corresponding to
    :class:`~pycaputo.derivatives.VariableExponentialCaputoDerivative`.

    The method is described in section 4.1 from [Garrappa2023]_.
    """

    dd: VariableExponentialCaputoDerivative
    """Operator being used to describe the variable-order kernel."""

    tau: float | None = None
    r"""Tolerance used to determine the quadrature weights (see Section 4.4 in
    [Garrappa2023]_). This is set to :math:`10^4 \epsilon` by default.
    """
    r: float | None = None
    """Radius used to determine the quadrature weights (see Section 4.4 in
    [Garrappa2023]_). This is set to :math:`1 - 1.1 h` by default.
    """
    safety_factor: float | None = None
    """A safety factor used to determine the quadrature weights (see Section 4.4
    in [Garrappa2023]_). This is set to :math:`0.01` by default.
    """

    if __debug__:

        def __post_init__(self) -> None:
            if not isinstance(self.d, VariableExponentialCaputoDerivative):
                raise TypeError(f"Unsupported derivative type: {type(self.d)}")

            if self.tau is not None and not 0 < self.tau < 1:
                raise ValueError(f"Tolerance 'tau' no in (0, 1): {self.tau}")

            if self.r is not None and not 0 < self.r < 1:
                raise ValueError(f"Radius 'r' not in (0, 1): {self.r}")

            if self.safety_factor is not None and not 0 < self.safety_factor < 1:
                raise ValueError(f"Safety factor not in (0, 1): {self.safety_factor}")

    @property
    def d(self) -> VariableExponentialCaputoDerivative:
        return self.dd


def gl_scarpi_exp_weights(
    d: VariableExponentialCaputoDerivative,
    p: UniformPoints,
    *,
    tau: float | None = None,
    r: float | None = None,
    safety_factor: float | None = None,
    psi_lim: float = 10.0,
    psi_points: int = 128,
) -> Array:
    alpha0, alpha1 = -d.alpha[0], -d.alpha[1]
    c = d.c

    n = p.size
    h = p.dx[0]
    logh = np.log(h)

    eps = float(np.finfo(p.x.dtype).eps)
    if tau is None:
        # NOTE: [Garrappa2023] recommends 10^-12 ~ 10^-13 and the MATLAB code
        # uses 10^-12 exactly. This should work nicely for floats and doubles
        tau = 1.0e4 * eps
    assert tau > eps

    if r is None:
        # NOTE: We need h < 1 - r from theory, so r should be very close to 1.
        # The MATLAB code uses r = 0.99 fixed for every time step, which would fail
        # for h >= 0.1. That's a pretty large time step, so probably not worth it
        r = 1.0 - 1.1 * h
    assert r is not None

    # if h > 1 - r:
    #     raise ValueError(
    #         f"Invalid radius given (must be h < 1 - r): r = {r} and h = {h}"
    #     )

    @overload
    def Psi(z: float) -> float: ...

    @overload
    def Psi(z: Array) -> Array: ...

    def Psi(z: Any) -> Any:  # noqa: N802
        return np.exp(
            -(alpha1 * c * h + alpha0 - alpha0 * z)
            / (c * h + 1 - z)
            * (np.log(1 - z) - logh)
        )

    # estimate rho on the basis of the round-off error
    Fs = 0.01 if safety_factor is None else safety_factor
    M_r = np.abs(Psi(r))
    rho = np.exp(-1.0 / n * np.log((tau / eps) * (Fs / M_r)))
    rho_n_inv = 1.0 / rho**n

    if rho > r:
        r = (1.0 + rho) / 2.0
    assert r is not None

    # estimate number of nodes
    from math import ceil

    x = np.linspace(-psi_lim, psi_lim, psi_points).reshape(-1, 1)
    y = np.linspace(-np.log(r / rho), psi_lim, psi_points)
    M_r_rho = np.max(np.abs(Psi(x + 1j * y)))
    N = ceil(
        (np.log(M_r_rho / rho_n_inv + tau) - np.log(tau)) / (np.log(r) - np.log(rho))
    )

    omega = 2j * np.pi * np.arange(N) / N
    psi_k = Psi(rho * np.exp(omega))

    w = np.empty(n + 1, dtype=psi_k.dtype)
    for j in range(n + 1):
        w[j] = rho_n_inv / N * np.sum(psi_k * np.exp(-j * omega))

    return w.real


@quad.register(ExponentialRectangular)
def _quad_vo_exp_rect(
    m: ExponentialRectangular,
    f: ArrayOrScalarFunction,
    p: Points,
) -> Array:
    if not isinstance(p, UniformPoints):
        raise TypeError(f"Only uniform points are supported: {type(p).__name__}")

    fx = f(p.x) if callable(f) else f
    w = gl_scarpi_exp_weights(m.d, p, tau=m.tau, r=m.r, safety_factor=m.safety_factor)

    qf = np.empty_like(fx)
    qf[0] = np.nan

    for n in range(1, qf.size):
        qf[n] = np.sum(w[:n][::-1] * fx[:n])

    return fx


# }}}
