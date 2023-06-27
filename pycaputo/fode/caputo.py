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

        return CaputoDerivative(self.derivative_order, side=Side.Left)


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


def _update_caputo_initial_condition(
    dy: Array, t: float, y0: tuple[Array, ...]
) -> Array:
    from math import gamma

    for k, y0k in enumerate(y0):
        dy += t**k / gamma(k + 1) * y0k

    return dy


def _update_caputo_forward_euler(
    dy: Array,
    history: History,
    alpha: float,
    *,
    n: int | None = None,
) -> Array:
    from math import gamma

    n = len(history) if n is None else n
    dt = history.ts[-1] - np.array(history.ts)
    assert n is not None

    for k in range(n):
        yk = history[k]
        assert isinstance(yk, SourceHistory)

        omega = (dt[k] ** alpha - dt[k + 1] ** alpha) / gamma(1 + alpha)
        dy += omega * yk.f

    return dy


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

    alpha = m.derivative_order

    dy = np.zeros_like(y)
    dy = _update_caputo_initial_condition(dy, t - m.tspan[0], m.y0)
    dy = _update_caputo_forward_euler(dy, history, alpha)

    history.append(SourceHistory(t=t, f=m.source(t, dy)))
    return dy


# }}}


# {{{ weighted Euler


@dataclass(frozen=True)
class CaputoWeightedEulerMethod(CaputoDifferentialEquationMethod):
    r"""The weighted Euler discretization of the Caputo derivative.

    The weighted Euler method is a convex combination of the forward Euler
    and the backward Euler method. This implementation uses a parameter
    :attr:`theta` to interpolate between the two (see Section 3.3 in [Li2015]_).

    Note that for :math:`\theta = 0` we get the forward Euler method, which
    is first order, for :math:`\theta = 1` we get the backward Euler method,
    which is first order, and for :math:`\theta = 1/2` we get the Crank-Nicolson
    method, which is order :math:`1 + \alpha`. This method only becomes second
    order in the limit of :math:`\alpha \to 1`.
    """

    #: Parameter weight between the forward and backward Euler methods. The value
    #: of :math:`\theta = 1/2` gives the standard Crank-Nicolson method.
    theta: float

    #: Jacobian of :attr:`~pycaputo.fode.FractionalDifferentialEquationMethod.source`.
    #: By default, implicit methods use :mod:`scipy` for their root finding,
    #: which defines the Jacobian as :math:`J_{ij} = \partial f_i / \partial y_j`.
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
        r"""Solves an implicit update formula.

        This function will meant to solve implicit equations of the form

        .. math::

            y_{n + 1} = \sum_{k = 0}^{n + 1} c_k f(t_k, y_k).

        Rearranging the implicit terms, we can write

        .. math::

            y_{n + 1} - c_{n + 1} f(t_{n + 1}, y_{n + 1}) = r_n,

        and solve for the solution :math:`y^{n + 1}`, where :math:`r_n`
        contains all the explicit terms. This is done by a root finding algorithm
        provided by :func:`scipy.optimize.root`.

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
            return np.array(np.eye(y.size) - c * self.source_jac(t, y))

        import scipy.optimize as so

        if y0.size == 1:
            result = so.root_scalar(
                f=lambda y: func(y).squeeze(),
                x0=y0,
                fprime=(
                    (lambda y: jac(y).squeeze())
                    if self.source_jac is not None
                    else None
                ),
                # method="newton",
            )
            solution = np.array(result.root)
        else:
            result = so.root(
                func,
                y0,
                jac=jac if self.source_jac is not None else None,
                # NOTE: the default hybr does not use derivatives, so use lm instead
                # FIXME: will need to maybe benchmark these a bit?
                method="lm",
            )

            solution = np.array(result.x)

        return solution


@advance.register(CaputoWeightedEulerMethod)
def _advance_caputo_weighted_euler(
    m: CaputoWeightedEulerMethod,
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
    fnext = _update_caputo_initial_condition(fnext, t - m.tspan[0], m.y0)

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


@dataclass(frozen=True)
class CaputoPECEMethod(CaputoPredictorCorrectorMethod):
    """The Predict-Evaluate-Correct-Evaluate (PECE) discretization of the
    Caputo derivative.

    This method is described in [Diethelm2002]_ in its simplest case with a
    single corrector step, which effectively gives the so-called PECE scheme.
    The corrector step can be repeated any number of times to give the
    :math:`PE(CE)^k` methods (see
    :attr:`CaputoPredictorCorrectorMethod.corrector_iterations`).
    """

    @classmethod
    def corrector_iterations_from_order(
        cls, alpha: float, *, is_d_c2: bool = True
    ) -> int:
        r"""Guess an optimal number of corrector iterations.

        A detailed analysis in [Garrappa2010]_ has shown that the Predictor-Corrector
        method can achieve a maximum convergence order of 2 with a well-chosen
        number of iterations.

        :arg alpha: fractional order of the Caputo derivative assumed in :math:`(0, 1)`.
        :arg is_d_c2: if *True* assume that :math:`D_C^\alpha[y] \in \mathcal{C}^2`,
            otherwise assume that :math:`y \in \mathcal{C}^2`. If neither of
            these is assumptions is true a maximum order of :math:`1 + \alpha`
            can be achieved regardlss of the number of iterations.

        :returns: a recommended number of corrector iterations.
        """
        from math import ceil

        return ceil(1 / alpha) if is_d_c2 else ceil(1 / alpha - 1)


@dataclass(frozen=True)
class CaputoPECMethod(CaputoPredictorCorrectorMethod):
    """The Predict-Evaluate-Correct (PEC) discretization of the Caputo derivative.

    This is a predictor-corrector similar to :class:`CaputoPECEMethod`, where
    the previous evaluation of the predictor is used to avoid an additional
    right-hand side call. Like the PECE method, the corrector step can be
    repeated multiple times for improved error results.
    """


def _update_caputo_adams_bashforth2(
    dy: Array,
    history: History,
    alpha: float,
    *,
    n: int | None = None,
) -> tuple[Array, float]:
    from math import gamma

    is_n = n is not None
    n = len(history) if n is None else n

    gamma1 = gamma(1 + alpha)
    gamma2 = gamma(2 + alpha)
    assert n is not None

    ts = np.array(history.ts)
    dt = np.diff(ts)
    ts = history.ts[-1] - ts

    for k in range(n - 1):
        yk = history[k]
        assert isinstance(yk, SourceHistory)

        omega = (
            ts[k + 1] ** (alpha + 1) / gamma2 / dt[k]
            - ts[k] ** (alpha + 1) / gamma2 / dt[k]
            + ts[k] ** alpha / gamma1
        )
        dy += omega * yk.f

        yk = history[k + 1]
        assert isinstance(yk, SourceHistory)

        omega = (
            ts[k] ** (alpha + 1) / gamma2 / dt[k]
            - ts[k + 1] ** (alpha + 1) / gamma2 / dt[k]
            - ts[k + 1] ** alpha / gamma1
        )
        dy += omega * yk.f

    if not is_n and n == len(history):
        k = n - 1
        yk = history[k]
        assert isinstance(yk, SourceHistory)

        omega = (
            ts[k + 1] ** (alpha + 1) / gamma2 / dt[k]
            - ts[k] ** (alpha + 1) / gamma2 / dt[k]
            + ts[k] ** alpha / gamma1
        )
        dy += omega * yk.f

    # NOTE: always compute the weight for the last step in the history
    k = len(history) - 1
    omega = (
        ts[k] ** (alpha + 1) / gamma2 / dt[k]
        - ts[k + 1] ** (alpha + 1) / gamma2 / dt[k]
        - ts[k + 1] ** alpha / gamma1
    )

    return dy, omega


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

    alpha = m.derivative_order

    # add initial conditions
    y0 = np.zeros_like(y)
    y0 = _update_caputo_initial_condition(y0, t - m.tspan[0], m.y0)

    # predictor step (forward Euler)
    yp = np.copy(y0)
    yp = _update_caputo_forward_euler(yp, history, alpha)

    # corrector step (Adams-Bashforth 2)
    yexplicit = np.copy(y0)
    yexplicit, omega = _update_caputo_adams_bashforth2(yexplicit, history, alpha)

    # corrector iterations
    for _ in range(m.corrector_iterations):
        fp = m.source(t, yp)
        yp = yexplicit + omega * fp

    ynext = yp
    f = fp if isinstance(m, CaputoPECMethod) else m.source(t, ynext)
    history.append(SourceHistory(t=t, f=f))

    return ynext


# }}}


# {{{ modified Predictor-Corrector


@dataclass(frozen=True)
class CaputoModifiedPECEMethod(CaputoPredictorCorrectorMethod):
    r"""A modified Predict-Evaluate-Correct-Evaluate (PECE) discretization of the
    Caputo derivative.

    This method is described in [Garrappa2010]_ as a modification to the standard
    :class:`CaputoPECEMethod` with improved performance due to reusing the
    convolution weights.

    Note that this method has an improved order, i.e. it achieves
    second-order with a single corrector iteration for all :math:`\alpha`, but
    a smaller stability region.
    """


@advance.register(CaputoModifiedPECEMethod)
def _advance_caputo_modified_pece(
    m: CaputoModifiedPECEMethod,
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
    gamma2 = gamma(2 + alpha)
    gamma1 = gamma(1 + alpha)
    ts = history.ts

    # compute common terms
    dy = np.zeros_like(y)
    dy = _update_caputo_initial_condition(dy, t - m.tspan[0], m.y0)

    if n == 1:
        yp = _update_caputo_forward_euler(dy, history, alpha)
    else:
        dy, _ = _update_caputo_adams_bashforth2(dy, history, alpha, n=n)

        # compute predictor
        yp = np.copy(dy)

        k = n - 1
        yk = history[k - 1]
        assert isinstance(yk, SourceHistory)

        # fmt: off
        omega = (
            (t - ts[k]) ** (alpha + 1) / gamma2 / (ts[k] - ts[k - 1])
            + (t - ts[k]) ** alpha / gamma1
            )
        yp += omega * yk.f
        # fmt: on

        yk = history[k]
        assert isinstance(yk, SourceHistory)

        omega = -((t - ts[k]) ** (alpha + 1)) / gamma2 / (ts[k] - ts[k - 1])
        yp += omega * yk.f

    # compute corrector
    ynext = np.copy(dy)

    k = n - 1
    yk = history[k]
    assert isinstance(yk, SourceHistory)

    dt = ts[k + 1] - ts[k]
    omega = (
        (t - ts[k + 1]) ** (alpha + 1) / gamma2 / dt
        - (t - ts[k]) ** (alpha + 1) / gamma2 / dt
        + (t - ts[k]) ** alpha / gamma1
    )
    ynext += omega * yk.f

    # corrector iterations
    omega = (
        (t - ts[k]) ** (alpha + 1) / gamma2 / dt
        - (t - ts[k + 1]) ** (alpha + 1) / gamma2 / dt
        - (t - ts[k + 1]) ** alpha / gamma1
    )
    for _ in range(m.corrector_iterations):
        fp = m.source(t, yp)
        yp = ynext + omega * fp

    ynext = yp
    history.append(SourceHistory(t=t, f=m.source(t, ynext)))

    return ynext


# }}}
