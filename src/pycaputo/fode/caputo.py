# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

from pycaputo.fode.base import AdvanceResult, advance
from pycaputo.fode.product_integration import CaputoProductIntegrationMethod
from pycaputo.history import ProductIntegrationHistory
from pycaputo.logging import get_logger
from pycaputo.utils import Array, StateFunction, gamma

logger = get_logger(__name__)


def _update_caputo_initial_condition(
    out: Array, m: CaputoProductIntegrationMethod, t: float
) -> Array:
    """Adds the appropriate initial conditions to *dy*."""
    t = t - m.control.tstart
    for k, y0k in enumerate(m.y0):
        out += t**k / gamma(k + 1) * y0k

    return out


def _truncation_error(
    m: CaputoProductIntegrationMethod, t: float, y: Array, tprev: float, yprev: Array
) -> Array:
    from pycaputo.controller import JannelliIntegralController

    alpha = m.alpha
    assert t > tprev

    if isinstance(m.control, JannelliIntegralController):
        trunc = np.array(
            m.gamma1p
            * (t - tprev) ** alpha
            / (t**alpha - tprev**alpha)
            * np.abs(y - yprev)
        )
    else:
        trunc = np.zeros_like(y)

    return trunc


# {{{ forward Euler


@dataclass(frozen=True)
class CaputoForwardEulerMethod(CaputoProductIntegrationMethod):
    """The first-order forward Euler discretization of the Caputo derivative."""

    @property
    def order(self) -> float:
        return 1.0


def _update_caputo_forward_euler(
    dy: Array,
    m: CaputoProductIntegrationMethod,
    history: ProductIntegrationHistory,
    n: int,
) -> Array:
    """Adds the Forward Euler right-hand side to *dy*."""
    # get time history
    assert 0 < n <= len(history)
    ts = history.ts[n] - history.ts[: n + 1]

    # sum up convolution
    alpha = m.alpha.reshape(-1, 1)
    gamma1p = m.gamma1p.reshape(-1, 1)
    omega = (ts[:-1] ** alpha - ts[1:] ** alpha) / gamma1p
    dy += np.einsum("ij,ij->j", omega.T, history.storage[:n])

    return dy


@advance.register(CaputoForwardEulerMethod)
def _advance_caputo_forward_euler(
    m: CaputoForwardEulerMethod,
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> AdvanceResult:
    # set next time step
    n = len(history)
    t = history.ts[n] = history.ts[n - 1] + dt

    # compute solution
    ynext = np.zeros_like(y)
    ynext = _update_caputo_initial_condition(ynext, m, t)
    ynext = _update_caputo_forward_euler(ynext, m, history, n)

    trunc = _truncation_error(m, t, ynext, t - dt, y)
    return ynext, trunc, m.source(t, ynext)


# }}}


# {{{ weighted Euler


@dataclass(frozen=True)
class CaputoWeightedEulerMethod(CaputoProductIntegrationMethod):
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
        return (1.0 + self.smallest_derivative_order) if self.theta == 0.5 else 1.0

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

        return solve(
            self.source,
            self.source_jac,
            t,
            y0,
            c,
            r,
            **self._get_kwargs(scalar=y0.size == 1),
        )


def _update_caputo_weighted_euler(
    dy: Array,
    m: CaputoWeightedEulerMethod,
    history: ProductIntegrationHistory,
    n: int,
) -> tuple[Array, Array]:
    """Adds the weighted Euler right-hand side to *dy*."""
    # NOTE: this is implicit so we never want to compute the last term
    assert 0 < n <= len(history)

    ts = history.ts[n] - history.ts[: n + 1]
    theta = m.theta

    # add explicit terms
    alpha = m.alpha.reshape(-1, 1)
    gamma1p = m.gamma1p.reshape(-1, 1)
    omega = ((ts[:-1] ** alpha - ts[1:] ** alpha) / gamma1p).T

    # add forward terms
    fs = history.storage[:n]
    if theta != 0.0:
        dy += theta * np.einsum("ij,ij->j", omega, fs)

    # add backwards terms
    if theta != 1.0:
        dy += (1 - theta) * np.einsum("ij,ij->j", omega[:-1], fs[1:])

    return dy, omega[-1].squeeze()


@advance.register(CaputoWeightedEulerMethod)
def _advance_caputo_weighted_euler(
    m: CaputoWeightedEulerMethod,
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> AdvanceResult:
    # set next time step
    n = len(history)
    t = history.ts[n] = history.ts[n - 1] + dt

    # add explicit terms
    fnext = np.zeros_like(y)
    fnext = _update_caputo_initial_condition(fnext, m, t)
    fnext, omega = _update_caputo_weighted_euler(fnext, m, history, n)

    # solve implicit equation
    if m.theta != 1.0:  # noqa: SIM108
        ynext = m.solve(t, y, omega * (1 - m.theta), fnext)
    else:
        ynext = fnext

    trunc = _truncation_error(m, t, ynext, t - dt, y)
    return ynext, trunc, m.source(t, ynext)


# }}}


# {{{ Predictor-Corector (PEC and PECE)


@dataclass(frozen=True)
class CaputoPredictorCorrectorMethod(CaputoProductIntegrationMethod):
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
        return 1.0 + self.smallest_derivative_order


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
    history: ProductIntegrationHistory,
    alpha: float,
    *,
    n: int | None = None,
) -> tuple[Array, float]:
    is_n = n is not None
    n = len(history) if n is None else n

    gamma1 = gamma(1 + alpha)
    gamma2 = gamma(2 + alpha)
    assert n is not None

    ts = history.ts[: n + 1]
    dt = np.diff(ts)
    ts = history.ts[n] - ts

    for k in range(n - 1):
        omega = (
            ts[k + 1] ** (alpha + 1) / gamma2 / dt[k]
            - ts[k] ** (alpha + 1) / gamma2 / dt[k]
            + ts[k] ** alpha / gamma1
        )
        dy += omega * history.storage[k]

        omega = (
            ts[k] ** (alpha + 1) / gamma2 / dt[k]
            - ts[k + 1] ** (alpha + 1) / gamma2 / dt[k]
            - ts[k + 1] ** alpha / gamma1
        )
        dy += omega * history.storage[k + 1]

    if not is_n and n == len(history):
        k = n - 1

        omega = (
            ts[k + 1] ** (alpha + 1) / gamma2 / dt[k]
            - ts[k] ** (alpha + 1) / gamma2 / dt[k]
            + ts[k] ** alpha / gamma1
        )
        dy += omega * history.storage[k]

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
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> AdvanceResult:
    from pycaputo.utils import single_valued

    n = len(history)
    alpha = single_valued(m.derivative_order)

    # set next time step
    t = history.ts[n] = history.ts[n - 1] + dt

    # add initial conditions
    y0 = np.zeros_like(y)
    y0 = _update_caputo_initial_condition(y0, m, t)

    # predictor step (forward Euler)
    yp = np.copy(y0)
    yp = _update_caputo_forward_euler(yp, m, history, len(history))

    # corrector step (Adams-Bashforth 2)
    yc_explicit = np.copy(y0)
    yc_explicit, omega = _update_caputo_adams_bashforth2(yc_explicit, history, alpha)

    # corrector iterations
    yc = yp
    for _ in range(m.corrector_iterations):
        fp = m.source(t, yc)
        yc = yc_explicit + omega * fp

    ynext = yc
    f = fp if isinstance(m, CaputoPECMethod) else m.source(t, ynext)

    trunc = _truncation_error(m, t, ynext, t - dt, y)
    return ynext, trunc, f


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
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> AdvanceResult:
    from pycaputo.utils import single_valued

    n = len(history)
    alpha = single_valued(m.derivative_order)
    gamma2 = gamma(2 + alpha)
    gamma1 = gamma(1 + alpha)

    # set next time step
    t = history.ts[n] = history.ts[n - 1] + dt

    ts = history.ts[: n + 1]
    ds = np.diff(ts)
    ts = history.ts[n] - ts

    # compute common terms
    dy = np.zeros_like(y)
    dy = _update_caputo_initial_condition(dy, m, t)

    if n == 1:
        yp = _update_caputo_forward_euler(dy, m, history, len(history))
    else:
        dy, _ = _update_caputo_adams_bashforth2(dy, history, alpha, n=n)

        # compute predictor
        yp = np.copy(dy)

        k = n - 1

        # fmt: off
        omega = (
            ts[k] ** (alpha + 1) / gamma2 / ds[k]
            + ts[k] ** alpha / gamma1
            )
        yp += omega * history.storage[k - 1]
        # fmt: on

        omega = -(ts[k] ** (alpha + 1)) / gamma2 / ds[k]
        yp += omega * history.storage[k]

    # compute corrector
    ynext = np.copy(dy)

    k = n - 1
    omega = (
        ts[k + 1] ** (alpha + 1) / gamma2 / ds[k]
        - ts[k] ** (alpha + 1) / gamma2 / ds[k]
        + ts[k] ** alpha / gamma1
    )
    ynext += omega * history.storage[k]

    # corrector iterations
    omega = (
        ts[k] ** (alpha + 1) / gamma2 / ds[k]
        - ts[k + 1] ** (alpha + 1) / gamma2 / ds[k]
        - ts[k + 1] ** alpha / gamma1
    )
    yc = yp
    for _ in range(m.corrector_iterations):
        fp = m.source(t, yc)
        yc = ynext + omega * fp
    ynext = yc

    trunc = _truncation_error(m, t, ynext, t - dt, y)
    return ynext, trunc, m.source(t, ynext)


# }}}
