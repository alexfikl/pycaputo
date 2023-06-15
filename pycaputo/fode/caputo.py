# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from pycaputo.derivatives import FractionalOperator
from pycaputo.fode.base import (
    FractionalDifferentialEquationMethod,
    advance,
    make_initial_condition,
)
from pycaputo.fode.history import History, SourceHistory
from pycaputo.logging import get_logger
from pycaputo.utils import Array, StateFunction

logger = get_logger(__name__)


@dataclass(frozen=True)
class CaputoDifferentialEquationMethod(FractionalDifferentialEquationMethod):
    r"""A generic method used to solve fractional ordinary differential
    equations (FODE) with the Caputo derivative.
    """

    @cached_property
    def d(self) -> FractionalOperator:
        from pycaputo.derivatives import CaputoDerivative, Side

        return CaputoDerivative(self.order, side=Side.Left)


@make_initial_condition.register(CaputoDifferentialEquationMethod)
def _make_initial_condition_caputo(
    m: CaputoDifferentialEquationMethod,
    t: float,
    y0: tuple[Array, ...],
) -> Array:
    return y0[0]


# {{{ forward Euler


@dataclass(frozen=True)
class CaputoForwardEulerMethod(CaputoDifferentialEquationMethod):
    """The first-order forward Euler discretization of the Caputo derivative."""

    @property
    def order(self) -> float:
        return 1.0


@advance.register(CaputoForwardEulerMethod)
def _advance_caputo_forward_euler(
    m: CaputoForwardEulerMethod,
    history: History,
    t: float,
    y: Array,
) -> Array:
    history.ts.append(t)
    if not history:
        history.append(SourceHistory(t=t, f=m.source(t, y)))
        return y

    from math import gamma

    n = len(history)
    alpha = m.derivative_order

    # add initial conditions
    ynext = np.zeros_like(y)
    for k, y0k in enumerate(m.y0):
        ynext += (t - m.tspan[0]) ** k / gamma(k + 1) * y0k

    # add history term
    ts = history.ts
    for k in range(n):
        yk = history[k]
        assert isinstance(yk, SourceHistory)

        omega = ((t - ts[k]) ** alpha - (t - ts[k + 1]) ** alpha) / gamma(1 + alpha)
        ynext += omega * yk.f

    history.append(SourceHistory(t=ts[-1], f=m.source(ts[-1], ynext)))
    return ynext


# }}}


# {{{ Crank-Nicolson


@dataclass(frozen=True)
class CaputoCrankNicolsonMethod(CaputoDifferentialEquationMethod):
    r"""The Crank-Nicolson discretization of the Caputo derivative.

    The Crank-Nicolson method is a convex combination of the forward Euler
    and the backward Euler method. This implementation uses a parameter
    :attr:`theta` to interpolate between the two.

    Note that for :math:`\theta = 0` we get the forward Euler method, which
    is first order, for :math:`\theta = 1` we get the backward Euler method,
    which is first order, and for :math:`\theta = 1/2` we get the Crank-Nicolson
    method, which is order :math:`1 + \alpha`. This method only becomes second
    order in the limit of :math:`\alpha \to 1`.
    """

    #: Parameter weight between the forward and backward Euler methods. The value
    #: of :math:`\theta = 1/2` gives the standard Crank-Nicolson method.
    theta: float

    source_jac: StateFunction | None

    if __debug__:

        def __post_init__(self) -> None:
            if not 0.0 <= self.theta <= 1.0:
                raise ValueError(
                    f"'theta' parameter must be in [0, 1]: got {self.theta}"
                )

    @property
    def order(self) -> float:
        return (1.0 + self.d.order) if self.theta == 0.5 else 1.0

    def solve(self, t: float, y0: Array, c: float, r: Array) -> Array:
        """Solves an implicit update formula.

        This function will solve an equation of the form

        .. math::

            y - c * f(t, y) = r

        for the solution :math:`y`. This is specific to first-order FODEs.

        :arg t: time at which the solution *y* is evaluated.
        :arg y: unknown solution at time *t*.
        :arg c: constant for the source term *f* that corresponds to
            :attr:`FractionalDifferentialEquationMethod.source`.
        :arg r: right-hand side term.

        :returns: solution :math:`y^*` of the above root finding problem.
        """

        def func(y: Array) -> Array:
            return np.array(y - c * self.source(t, y) - r)

        def jac(y: Array) -> Array:
            assert self.source_jac is not None
            return np.array(1 - c * self.source_jac(t, y))

        import scipy.optimize as so

        result = so.root(
            func,
            y0,
            jac=jac if self.source_jac is not None else None,
            method="lm",
            options={"ftol": 1.0e-10},
        )

        return np.array(result.x)


@advance.register(CaputoCrankNicolsonMethod)
def _advance_caputo_crank_nicolson(
    m: CaputoCrankNicolsonMethod,
    history: History,
    t: float,
    y: Array,
) -> Array:
    history.ts.append(t)
    if not history:
        history.append(SourceHistory(t=t, f=m.source(t, y)))
        return y

    from math import gamma

    n = len(history)
    alpha = m.derivative_order

    # add initial conditions
    fnext = np.zeros_like(y)
    for k, y0k in enumerate(m.y0):
        fnext += (t - m.tspan[0]) ** k / gamma(k + 1) * y0k

    # compute explicit memory term
    ts = history.ts
    for k in range(n - 1):
        omega = ((t - ts[k]) ** alpha - (t - ts[k + 1]) ** alpha) / gamma(1 + alpha)

        # add forward term
        if m.theta != 0.0:
            yk = history[k]
            assert isinstance(yk, SourceHistory)
            fnext += omega * m.theta * yk.f

        # add backward term
        if m.theta != 1.0:
            yk = history[k + 1]
            assert isinstance(yk, SourceHistory)
            fnext += omega * (1 - m.theta) * yk.f

    # add last forward
    k = n - 1
    omega = ((t - ts[k]) ** alpha - (t - ts[k + 1]) ** alpha) / gamma(1 + alpha)

    if m.theta != 0.0:
        yk = history[k]
        assert isinstance(yk, SourceHistory)
        fnext += omega * m.theta * yk.f

    # solve implicit equation
    if m.theta != 1.0:
        ynext = m.solve(ts[-1], y, omega * (1 - m.theta), fnext)
    else:
        ynext = fnext

    history.append(SourceHistory(t=ts[-1], f=m.source(ts[-1], ynext)))
    return ynext


# }}}


# {{{ Predictor-Corector (PEC and PECE)


@dataclass(frozen=True)
class CaputoPredictorCorrectorMethod(CaputoDifferentialEquationMethod):
    r"""The Predictor-Corrector discretization of the Caputo derivative.

    In their classic forms (see e.g. [Diethelm2002]_), these are methods of
    order :math:`1 + \alpha` with good stability properties.

    In general, the corrector step can be repeated multiple times to achieve
    convergence using :attr:`corrector_iterations`. In the limit of
    :math:`k \to \infty`, it is equivalent to a Adams-Moulton method solved by
    fixed point iteration.

    Note that using a high number of corrector iterations is not recommended, as
    the fixed point iteration is not guaranteed to converge, e.g. for very stiff
    problems. In that case it is better to use an implicit method and, e.g.,
    a Newton iteration to solve the root finding problem.
    """

    #: Number of repetitions of the corrector step.
    corrector_iterations: int

    if __debug__:

        def __post_init__(self) -> None:
            if self.corrector_iterations < 1:
                raise ValueError(
                    "More than one corrector iteration is required:"
                    f" {self.corrector_iterations}"
                )

    @property
    def order(self) -> float:
        return 1.0 + self.d.order

    @classmethod
    def corrector_iterations_from_order(
        cls, alpha: float, *, is_d_c2: bool = True
    ) -> int:
        r"""Guess an optimal number of corrector iterations.

        A detailed analysis in [Garrappa2010]_ has shown that the Predictor-Corrector
        method can achieve a maximum convergence order of 2 with a well-chosen
        number of iterations.

        :arg alpha: fractional order of the Caputo derivative.
        :arg is_d_c2: if *True* assume that :math:`D_C^\alpha[y] \in \mathcal{C}^2`,
            otherwise assume that :math:`y \in \mathcal{C}^2`. If neither of
            these is assumptions is true a maximum order of :math:`1 + \alpha`
            can be achieved regardlss of the number of iterations.

        :returns: a recommended number of corrector iterations.
        """
        from math import ceil

        return ceil(1 / alpha) if is_d_c2 else ceil(1 / alpha - 1)


@dataclass(frozen=True)
class CaputoPECEMethod(CaputoPredictorCorrectorMethod):
    """The Predict-Evaluate-Correct-Evaluate (PECE) discretization of the
    Caputo Derivative.

    This method is described in [Diethelm2002]_ in its simplest case with a
    single corrector step, which effectively gives the so-called PECE scheme.
    The corrector step can be repeated any number of times to give the
    :math:`PE(CE)^k` methods (see
    :attr:`CaputoPredictorCorrectorMethod.corrector_iterations`).
    """


@dataclass(frozen=True)
class CaputoPECMethod(CaputoPredictorCorrectorMethod):
    """The Predict-Evaluate-Correct (PEC) discretization of the Caputo derivative.

    This is a predictor-corrector similar to :class:`CaputoPECEMethod`, where
    the previous evaluation of the predictor is used to avoid an additional
    right-hand side call. Like the PECE method, the corrector step can be
    repeated multiple times for improved error results.
    """


@advance.register(CaputoPredictorCorrectorMethod)
def _advance_caputo_predictor_corrector(
    m: CaputoPredictorCorrectorMethod,
    history: History,
    t: float,
    y: Array,
) -> Array:
    history.ts.append(t)
    if not history:
        history.append(SourceHistory(t=t, f=m.source(t, y)))
        return y

    from math import gamma

    n = len(history)
    alpha = m.derivative_order
    ts = history.ts
    gamma1 = gamma(1 + alpha)
    gamma2 = gamma(2 + alpha)

    # add initial conditions
    y0 = np.zeros_like(y)
    for k, y0k in enumerate(m.y0):
        y0 += (t - m.tspan[0]) ** k / gamma(k + 1) * y0k

    # predictor step (forward Euler)
    yp = np.copy(y0)
    omega_e = np.empty(n)
    for k in range(n):
        yk = history[k]
        assert isinstance(yk, SourceHistory)

        omega_e[k] = ((t - ts[k]) ** alpha - (t - ts[k + 1]) ** alpha) / gamma1
        yp += omega_e[k] * yk.f

    # corrector step (Adams-Bashforth 2)
    yexplicit = np.copy(y0)
    for k in range(n - 1):
        yk = history[k]
        assert isinstance(yk, SourceHistory)

        dt = ts[k + 1] - ts[k]
        omega = (
            (t - ts[k + 1]) ** (alpha + 1) / gamma2 / dt
            - (t - ts[k]) ** (alpha + 1) / gamma2 / dt
            + (t - ts[k]) ** alpha / gamma1
        )
        yexplicit += omega * yk.f

        yk = history[k + 1]
        assert isinstance(yk, SourceHistory)

        omega = (
            (t - ts[k]) ** (alpha + 1) / gamma2 / dt
            - (t - ts[k + 1]) ** (alpha + 1) / gamma2 / dt
            - (t - ts[k + 1]) ** alpha / gamma1
        )
        yexplicit += omega * yk.f

    k = n - 1
    yk = history[k]
    assert isinstance(yk, SourceHistory)

    dt = ts[k + 1] - ts[k]
    omega = (
        (t - ts[k + 1]) ** (alpha + 1) / gamma2 / dt
        - (t - ts[k]) ** (alpha + 1) / gamma2 / dt
        + (t - ts[k]) ** alpha / gamma1
    )
    yexplicit += omega * yk.f

    # corrector iterations
    omega = (
        (t - ts[k]) ** (alpha + 1) / gamma2 / dt
        - (t - ts[k + 1]) ** (alpha + 1) / gamma2 / dt
        - (t - ts[k + 1]) ** alpha / gamma1
    )
    for _ in range(m.corrector_iterations):
        fp = m.source(t, yp)
        yp = yexplicit + omega * fp

    ynext = yp
    f = fp if isinstance(m, CaputoPECMethod) else m.source(ts[-1], ynext)
    history.append(SourceHistory(t=ts[-1], f=f))

    return ynext


# }}}

# }}}
