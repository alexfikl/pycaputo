# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.linalg as la

from pycaputo.controller import FixedController
from pycaputo.derivatives import VariableExponentialCaputoDerivative
from pycaputo.events import Event
from pycaputo.history import ProductIntegrationHistory
from pycaputo.logging import get_logger
from pycaputo.stepping import (
    FractionalDifferentialEquationMethod,
    advance,
    evolve,
    make_initial_condition,
)
from pycaputo.typing import Array, StateFunctionT
from pycaputo.utils import cached_on_first_arg

from .product_integration import AdvanceResult
from .special import Function

log = get_logger(__name__)


# {{{ gallery


@dataclass(frozen=True)
class Relaxation(Function):
    r"""Implements the right-hand side of the relaxation problem Equation 12 in
    [Garrappa2023]_.

    .. math::

        D^{\alpha(t)}[y](t) = -\omega y,

    The exact solution is also computed in :meth:`__call__` using an approximate
    inverse Laplace transform.

    .. automethod:: __call__
    """

    d: VariableExponentialCaputoDerivative
    """Description of the variable-order Caputo derivative used."""

    y0: float
    """Initial condition for the equation."""
    omega: float
    """Relaxation parameter (i.e. eigenvalue of the operator)."""

    def __call__(self, t: float) -> Array:
        """Evaluate the exact solution at time *t*."""

        if t == 0.0:
            return np.array([self.y0])

        import mpmath  # type: ignore[import-untyped]

        alpha1, alpha2 = self.d.alpha
        c = self.d.c

        def y_laplace_transform(s: mpmath.mpf) -> mpmath.mpf:
            # Equation 13 from [Garrappa2023]
            sA = (alpha1 * s + alpha2 * c) / (s + c)
            H = 1.0 / (s + self.omega * s ** (1 - sA))
            return H * self.y0

        return np.array([mpmath.invertlaplace(y_laplace_transform, t, method="talbot")])

    def source(self, t: float, y: Array) -> Array:
        return -self.omega * y

    def source_jac(self, t: float, y: Array) -> Array:
        return np.array(-self.omega)


# }}}


# {{{ VariableExponentialBackwardEuler


@dataclass(frozen=True)
class VariableExponentialBackwardEuler(
    FractionalDifferentialEquationMethod[
        VariableExponentialCaputoDerivative, StateFunctionT
    ]
):
    source_jac: StateFunctionT | None
    r"""Jacobian of
    :attr:`~pycaputo.stepping.FractionalDifferentialEquationMethod.source`.
    By default, :mod:`scipy` is used to provide root finding algorithms, so setting
    this to *None* will fall back to finite difference approximations.
    """

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()

            if not isinstance(self.control, FixedController):
                raise TypeError(f"Only 'FixedController' is supported: {self.control}")

    @property
    def order(self) -> float:
        return 1.0

    @cached_property
    def derivative_order(self) -> tuple[float, ...]:
        return tuple([max(d.alpha) for d in self.ds])

    def make_default_history(self) -> ProductIntegrationHistory:
        nsteps = self.control.nsteps
        return ProductIntegrationHistory.empty_like(
            # NOTE: All product integration rules just store the right-hand side
            # `f`, which are the same size and dtype as `y0`
            self.y0[0],
            n=512 if nsteps is None else nsteps,
        )

    # NOTE: this is cleverly copy pasted from CaputoImplicitProductIntegrationMethod

    # NOTE: `_get_kwargs` is meant to be overwritten for testing purposes or
    # some specific application (undocumented for now).

    def _get_kwargs(self, *, scalar: bool = True) -> dict[str, object]:
        """
        :returns: additional keyword arguments for :func:`scipy.optimize.root_scalar`.
            or :func:`scipy.optimize.root`.
        """
        if scalar:
            return {}
        else:
            # NOTE: the default hybr does not use derivatives, so use lm instead
            # FIXME: will need to maybe benchmark these a bit?
            return {"method": "lm" if self.source_jac else None}

    def solve(self, t: float, y0: Array, c: Array, r: Array) -> Array:
        """Wrapper around :func:`~pycaputo.implicit.solve` to solve the
        implicit equation.

        This function should be overwritten for specific applications if better
        solvers are known. For example, many problems can be solved explicitly
        or approximated to a very good degree to provide a better *y0*.
        """
        from pycaputo.implicit import solve

        result = solve(
            self.source,
            self.source_jac,
            t,
            y0,
            c,
            r,
            **self._get_kwargs(scalar=y0.size == 1),
        )

        if __debug__:
            error = la.norm(result - c * self.source(t, result) - r)
            rtol = 1.0e-6 * la.norm(result)
            assert error < rtol, (error, rtol)

        return result


@cached_on_first_arg
def variable_caputo_backwards_euler_weights(
    m: VariableExponentialBackwardEuler[StateFunctionT],
) -> Array:
    from pycaputo.quadrature.variable_riemann_liouville import _gl_scarpi_exp_weights

    c = m.control
    assert isinstance(c, FixedController)
    n = c.nsteps
    assert n is not None
    h = c.dt

    # FIXME: ideally, `_gl_scarpi_exp_weights` would support vector alpha
    omega = np.empty((n + 1, len(m.ds)), dtype=np.array(h).dtype)
    for i, d in enumerate(m.ds):
        omega[:, i] = _gl_scarpi_exp_weights(
            n,
            h,
            d.alpha[0],
            d.alpha[1],
            d.c,
            # FIXME: Allow passing in additional parameters for the weights?
            # These are taken from the MATLAB code to match the results
            tau=1.0e-12,
            r=0.99,
            safety_factor=0.01,
        )

    return omega


@make_initial_condition.register(VariableExponentialBackwardEuler)
def _make_initial_condition_vo_caputo_backward_euler(  # type: ignore[misc]
    m: VariableExponentialBackwardEuler[StateFunctionT],
) -> Array:
    return m.y0[0]


@advance.register(VariableExponentialBackwardEuler)
def _advance_vo_caputo_backward_euler(  # type: ignore[misc]
    m: VariableExponentialBackwardEuler[StateFunctionT],
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> AdvanceResult:
    # set next time step
    n = len(history)
    t = history.ts[n] = history.ts[n - 1] + dt

    # initialize solution
    # FIXME: VariableExponentialCaputoDerivative only supports 0 < alpha < 1
    # so the initial condition always simplifies to just this
    fnext = m.y0[0].copy()

    # add history
    omega = variable_caputo_backwards_euler_weights(m)[: n + 1][::-1]
    fnext += np.einsum("ij,ij->j", omega[:-1], history.storage[:n])

    # solve `ynext = fac * f(t, ynext) + fnext`
    ynext = m.solve(t, y, omega[-1], fnext)

    return AdvanceResult(ynext, np.zeros_like(ynext), m.source(t, ynext))


@evolve.register(VariableExponentialBackwardEuler)
def _evolve_vo_caputo_backward_euler(  # type: ignore[misc]
    m: VariableExponentialBackwardEuler[StateFunctionT],
    *,
    history: ProductIntegrationHistory | None = None,
    dtinit: float | None = None,
) -> Iterator[Event]:
    if history is None:
        history = m.make_default_history()

    # initialize
    c = m.control
    assert isinstance(c, FixedController)

    if dtinit is not None:
        log.warning("'dtinit' is ignored for fixed step size controller: %g", dtinit)
    dt = c.dt

    n = 0
    t = c.tstart
    y = make_initial_condition(m)
    history.append(t, m.source(t, y))

    # evolve
    from pycaputo.events import StepAccepted

    trunc = np.zeros_like(y)
    yield StepAccepted(t=t, iteration=n, dt=dt, y=y, eest=0.0, q=1.0, trunc=trunc)

    while not c.finished(n, t):
        # advance solution
        y, trunc, storage = advance(m, history, y, dt)

        # store solution
        history.append(t, storage)

        # advance iteration
        n += 1
        t += dt

        yield StepAccepted(t=t, iteration=n, dt=dt, y=y, eest=0.0, q=1.0, trunc=trunc)

    return


# }}}
