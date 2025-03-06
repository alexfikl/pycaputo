# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, overload

import numpy as np

from pycaputo.derivatives import VariableExponentialCaputoDerivative
from pycaputo.grid import Points, UniformPoints
from pycaputo.logging import get_logger
from pycaputo.typing import Array, ArrayOrScalarFunction

from .base import QuadratureMethod, quad

log = get_logger(__name__)


@dataclass(frozen=True)
class VariableOrderMethod(QuadratureMethod):
    """Quadrature method for variable-order fractional operators."""


# {{{ rectangular


@dataclass(frozen=True)
class ExponentialRectangular(VariableOrderMethod):
    """This approximates the integral corresponding to
    :class:`~pycaputo.derivatives.VariableExponentialCaputoDerivative`.

    The method is described in Section 4.1 from [Garrappa2023]_.
    """

    dd: VariableExponentialCaputoDerivative
    """Operator being used to describe the variable-order kernel."""

    tau: float | None = None
    r"""Tolerance used to determine the quadrature weights (see Section 4.4 in
    [Garrappa2023]_). This is set to :math:`10^4 \epsilon` by default, where
    :math:`\epsilon` is the precision of the given points.
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


def _gl_scarpi_exp_weights(
    n: int,
    h: float,
    alpha0: float,
    alpha1: float,
    c: float,
    *,
    tau: float | None = None,
    r: float | None = None,
    safety_factor: float | None = None,
    psi_lim: float = 5.0,
    psi_points: int = 100,
) -> Array:
    """
    :arg n: number of points in the discretization.
    :arg h: step size.
    :arg alpha0: see :attr:`~pycaputo.derivatives.VariableExponentialCaputoDerivative`.
    :arg alpha1: see :attr:`~pycaputo.derivatives.VariableExponentialCaputoDerivative`.
    :arg c: see :attr:`~pycaputo.derivatives.VariableExponentialCaputoDerivative`.

    :arg psi_lim: bound `[-psi_lim, psi_lim]` used when determining the :math:`M_r`
        bound from the paper. This defaults to the same values as the MATLAB code.
    :arg psi_points: number of points in the interval for *psi_lim*.
    """
    logh = np.log(h)

    eps = float(np.finfo(np.array(h).dtype).eps)
    if tau is None:
        # NOTE: [Garrappa2023] recommends 10^-12 ~ 10^-13 and the MATLAB code
        # uses 10^-12 exactly. This should work nicely for floats and doubles
        tau = 1.0e4 * eps
    assert tau > 0

    if r is None:
        # NOTE: We need h < 1 - r from theory, so r should be very close to 1.
        # The MATLAB code uses r = 0.99 fixed for every time step, which would fail
        # for h >= 0.1. That's a pretty large time step, so probably not worth it
        r = 1.0 - 1.1 * h
    assert r is not None

    if h > 1 - r:
        # FIXME: Should this be a hard error? Does not seem like a hard requirement
        # in the MATLAB code, and would likely not be hit in practice..
        log.warning("Invalid radius given (must be r < 1 - h): r = %g and h = %g", r, h)

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
        (np.log(M_r_rho * rho_n_inv + tau) - np.log(tau)) / (np.log(r) - np.log(rho))
    )

    # compute weights
    if __debug__:
        log.info("Predicted round-off error: %.12e", rho_n_inv * M_r * tau)

    omega = 2j * np.pi * np.arange(N) / N
    psi_k = Psi(rho * np.exp(omega))

    w = np.empty(n + 1, dtype=psi_k.dtype)
    for j in range(n + 1):
        w[j] = np.sum(psi_k * np.exp(-j * omega)) / (rho**j * N)

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
    w = _gl_scarpi_exp_weights(
        p.size,
        p.dx[0],
        -m.d.alpha[0],
        -m.d.alpha[1],
        m.d.c,
        tau=m.tau,
        r=m.r,
        safety_factor=m.safety_factor,
    )

    log.info("%r", np.linalg.norm(w))

    qf = np.empty_like(fx)
    qf[0] = np.nan

    for n in range(1, qf.size):
        qf[n] = np.sum(w[:n][::-1] * fx[:n])

    return qf


# }}}
