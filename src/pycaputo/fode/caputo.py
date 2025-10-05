# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import ceil

import numpy as np
import numpy.linalg as la
from scipy.special import gamma

from pycaputo.controller import Controller
from pycaputo.derivatives import CaputoDerivative
from pycaputo.history import ProductIntegrationHistory
from pycaputo.logging import get_logger
from pycaputo.stepping import advance, make_initial_condition
from pycaputo.typing import Array, StateFunctionT

from .product_integration import (
    AdvanceResult,
    ProductIntegrationMethod,
)

log = get_logger(__name__)


# {{{ base


def _update_caputo_initial_condition(
    out: Array, y0: tuple[Array, ...], t: float
) -> Array:
    """Adds the appropriate initial conditions to *out*."""
    for k, y0k in enumerate(y0):
        out += t**k / gamma(k + 1) * y0k

    return out


def _truncation_error(
    c: Controller, alpha: Array, t: float, y: Array, tprev: float, yprev: Array
) -> Array:
    if not c.is_adaptive:
        return np.zeros_like(y)

    from pycaputo.controller import JannelliIntegralController

    # FIXME: this should not be our job: either let the controller or the method
    # figure out how to compute the truncation error estimates.
    assert t > tprev
    if isinstance(c, JannelliIntegralController):
        trunc = np.array(
            gamma(1 + alpha)
            * (t - tprev) ** alpha
            / (t**alpha - tprev**alpha)
            * np.abs(y - yprev)
        )
    else:
        raise TypeError(f"Unsupported controller type: {type(c)}")

    return trunc


# }}}


# {{{ CaputoProductIntegrationMethod


@dataclass(frozen=True)
class CaputoProductIntegrationMethod(
    ProductIntegrationMethod[CaputoDerivative, StateFunctionT]
):
    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()

            if not all(isinstance(d, CaputoDerivative) for d in self.ds):
                raise TypeError(f"Expected 'CaputoDerivative' operators: {self.ds}")

    @cached_property
    def derivative_order(self) -> tuple[float, ...]:
        return tuple([d.alpha for d in self.ds])

    @cached_property
    def alpha(self) -> Array:
        return np.array([d.alpha for d in self.ds])


@dataclass(frozen=True)
class CaputoImplicitProductIntegrationMethod(
    CaputoProductIntegrationMethod[StateFunctionT]
):
    source_jac: StateFunctionT | None
    r"""Jacobian of
    :attr:`~pycaputo.stepping.FractionalDifferentialEquationMethod.source`.
    By default, implicit methods use :mod:`scipy` for their root finding,
    which defines the Jacobian as :math:`J_{ij} = \partial f_i / \partial y_j`.
    """

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


@make_initial_condition.register(CaputoProductIntegrationMethod)
def _make_initial_condition_caputo(  # type: ignore[misc]
    m: CaputoProductIntegrationMethod[StateFunctionT],
) -> Array:
    return m.y0[0]


# }}}


# {{{ forward Euler


@dataclass(frozen=True)
class ForwardEuler(CaputoProductIntegrationMethod[StateFunctionT]):
    """The first-order forward Euler discretization of the Caputo derivative."""

    @property
    def order(self) -> float:
        return 1.0


def _weights_quadrature_rectangular(
    m: CaputoProductIntegrationMethod[StateFunctionT],
    t: Array,
    n: int,
) -> Array:
    # get time history
    ts = (t[n] - t[: n + 1]).reshape(-1, 1)

    alpha = m.alpha
    g1p = gamma(1 + alpha)
    omega = (ts[:-1] ** alpha - ts[1:] ** alpha) / g1p

    return np.array(omega)


def _update_caputo_forward_euler(
    out: Array,
    m: CaputoProductIntegrationMethod[StateFunctionT],
    history: ProductIntegrationHistory,
    n: int,
    *,
    offset: int = 0,
) -> Array:
    """Adds the Forward Euler right-hand side to *out*."""
    assert 0 < n <= len(history)
    omega = _weights_quadrature_rectangular(m, history.ts, n)

    out += np.einsum("ij,ij->j", omega, history.storage[:n])
    return out


@advance.register(ForwardEuler)
def _advance_caputo_forward_euler(  # type: ignore[misc]
    m: ForwardEuler[StateFunctionT],
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> AdvanceResult:
    # set next time step
    n = len(history)
    tstart = m.control.tstart
    t = history.ts[n] = history.ts[n - 1] + dt

    # compute solution
    ynext = np.zeros_like(y)
    ynext = _update_caputo_initial_condition(ynext, m.y0, t - tstart)
    ynext = _update_caputo_forward_euler(ynext, m, history, n)

    trunc = _truncation_error(m.control, m.alpha, t, ynext, t - dt, y)
    return AdvanceResult(ynext, trunc, m.source(t, ynext))


# }}}


# {{{ backward Euler


@dataclass(frozen=True)
class BackwardEuler(CaputoImplicitProductIntegrationMethod[StateFunctionT]):
    """The first-order backward Euler discretization of the Caputo derivative."""

    @property
    def order(self) -> float:
        return 1.0


@advance.register(BackwardEuler)
def _advance_caputo_backward_euler(  # type: ignore[misc]
    m: BackwardEuler[StateFunctionT],
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> AdvanceResult:
    # set next time step
    n = len(history)
    tstart = m.control.tstart
    t = history.ts[n] = history.ts[n - 1] + dt

    # initialize solution
    fnext = np.zeros_like(y)
    fnext = _update_caputo_initial_condition(fnext, m.y0, t - tstart)

    # add history
    omega = _weights_quadrature_rectangular(m, history.ts, n)
    fnext += np.einsum("ij,ij->j", omega[:-1], history.storage[1:n])

    # solve `ynext = fac * f(t, ynext) + fnext`
    ynext = m.solve(t, y, omega[-1], fnext)

    trunc = _truncation_error(m.control, m.alpha, t, ynext, t - dt, y)
    return AdvanceResult(ynext, trunc, m.source(t, ynext))


# }}}


# {{{ weighted Euler


@dataclass(frozen=True)
class WeightedEuler(CaputoImplicitProductIntegrationMethod[StateFunctionT]):
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

    theta: float
    r"""Parameter weight between the forward and backward Euler methods. The value
    of :math:`\theta = 1/2` gives the standard Crank-Nicolson method.
    """

    if __debug__:

        def __post_init__(self) -> None:
            if not 0.0 <= self.theta <= 1.0:
                raise ValueError(
                    f"'theta' parameter must be in [0, 1]: got {self.theta}"
                )

    @property
    def order(self) -> float:
        return (1.0 + self.smallest_derivative_order) if self.theta == 0.5 else 1.0


def _update_caputo_weighted_euler(
    out: Array,
    m: WeightedEuler[StateFunctionT],
    history: ProductIntegrationHistory,
    n: int,
) -> tuple[Array, Array]:
    """Adds the weighted Euler right-hand side to *out*."""
    assert 0 < n <= len(history)
    omega = _weights_quadrature_rectangular(m, history.ts, n)

    # add forward terms
    theta = m.theta
    fs = history.storage[:n]
    if theta != 0.0:
        out += theta * np.einsum("ij,ij->j", omega, fs)

    # add backwards terms
    if theta != 1.0:
        # NOTE: this is implicit so we do not add the last term
        out += (1 - theta) * np.einsum("ij,ij->j", omega[:-1], fs[1:])

    return out, (1 - theta) * omega[-1].squeeze()


@advance.register(WeightedEuler)
def _advance_caputo_weighted_euler(  # type: ignore[misc]
    m: WeightedEuler[StateFunctionT],
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> AdvanceResult:
    # set next time step
    n = len(history)
    tstart = m.control.tstart
    t = history.ts[n] = history.ts[n - 1] + dt

    # add explicit terms
    fnext = np.zeros_like(y)
    fnext = _update_caputo_initial_condition(fnext, m.y0, t - tstart)
    fnext, fac = _update_caputo_weighted_euler(fnext, m, history, n)

    if m.theta != 1.0:  # noqa: SIM108
        # NOTE: solve `y = fac * f(t, y) + fnext`
        ynext = m.solve(t, y, fac, fnext)
    else:
        ynext = fnext

    trunc = _truncation_error(m.control, m.alpha, t, ynext, t - dt, y)
    return AdvanceResult(ynext, trunc, m.source(t, ynext))


# }}}


# {{{ Trapezoidal


@dataclass(frozen=True)
class Trapezoidal(CaputoImplicitProductIntegrationMethod[StateFunctionT]):
    """The trapezoidal method for discretizing the Caputo derivative.

    This is an implicit method described in [Garrappa2015b]_. It uses a linear
    interpolant on each time step.
    """

    @property
    def order(self) -> float:
        return 2.0


def _weights_quadrature_trapezoidal_single(
    m: CaputoProductIntegrationMethod[StateFunctionT],
    t: Array,
    n: int,
    p: int,
) -> Array:
    # get time history
    ts = (t[n] - t[: p + 1]).reshape(-1, 1)
    dt = np.diff(t[: p + 1]).reshape(-1, 1)

    alpha = m.alpha
    a1p = 1 + alpha
    g1p = gamma(a1p)
    g2p = a1p * g1p

    # compute integrals
    I00 = ts[0] ** alpha / g1p
    I1 = ts ** (1.0 + alpha) / g2p

    # compute weights
    omega = np.empty((p + 1, alpha.size), dtype=I1.dtype)
    omega[0] = I1[1] / dt[0] + I00 - I1[0] / dt[0]
    omega[1:p] = (I1[2:] - I1[1:-1]) / dt[1:] + (I1[:-2] - I1[1:-1]) / dt[:-1]

    if p < n:
        I01 = ts[-1] ** alpha / g1p
        omega[p] = I1[-2] / dt[-1] - I01 - I1[-1] / dt[-1]
    else:
        omega[p] = I1[-2] / dt[-1]

    return omega


def _update_caputo_trapezoidal(
    out: Array,
    m: CaputoProductIntegrationMethod[StateFunctionT],
    history: ProductIntegrationHistory,
    n: int,
    p: int,
) -> tuple[Array, Array]:
    assert 0 < n <= len(history)
    assert 0 < p <= n
    omega = _weights_quadrature_trapezoidal_single(m, history.ts, n, p)

    fs = history.storage[:p]
    out += np.einsum("ij,ij->j", omega[:-1], fs)

    if p < n:
        out += omega[-1] * history.storage[p]

    return out, omega[-1].squeeze()


@advance.register(Trapezoidal)
def _advance_caputo_trapezoidal(  # type: ignore[misc]
    m: Trapezoidal[StateFunctionT],
    history: ProductIntegrationHistory,
    y: Array,
    dt: Array,
) -> AdvanceResult:
    # set next time step
    n = len(history)
    tstart = m.control.tstart
    t = history.ts[n] = history.ts[n - 1] + dt

    # compute solution
    fnext = np.zeros_like(y)
    fnext = _update_caputo_initial_condition(fnext, m.y0, t - tstart)
    fnext, fac = _update_caputo_trapezoidal(fnext, m, history, n, n)

    # solve `ynext = fac * f(t, ynext) + fnext`
    ynext = m.solve(t, y, fac, fnext)

    trunc = _truncation_error(m.control, m.alpha, t, ynext, t - dt, y)
    return AdvanceResult(ynext, trunc, m.source(t, ynext))


@dataclass(frozen=True)
class ExplicitTrapezoidal(CaputoProductIntegrationMethod[StateFunctionT]):
    r"""An explicit trapezoidal method for discretizing the Caputo derivative.

    This is an explicit method described in [Garrappa2010]_. Unlike
    :class:`Trapezoidal`, the last step is estimated by extrapolation, making this
    an explicit method instead with decreased stability.

    .. warning::

        The extrapolation from [Garrappa2010]_ does not match the one that is
        used here even in the uniform case. To our knowledge, the paper has an
        error and the last term in the second line from Equation 8 should be

        .. math::

            (2 \boldsymbol{+ \beta}) \alpha_0 f(t_{n - 1}, y_{n - 1}).
    """

    @property
    def order(self) -> float:
        return 1.0 + self.smallest_derivative_order


def _update_caputo_trapezoidal_extrapolation(
    out: Array,
    m: CaputoProductIntegrationMethod[StateFunctionT],
    history: ProductIntegrationHistory,
    n: int,
) -> Array:
    alpha = m.alpha
    ts1 = history.ts[n] - history.ts[n - 1]
    dt2 = history.ts[n - 1] - history.ts[n - 2]
    fm1 = history.storage[n - 1]
    fm2 = history.storage[n - 2]

    # fmt: off
    omegal = -(ts1 ** (1 + alpha)) / gamma(2 + alpha) / dt2
    out += omegal * fm2
    omegar = (
        ts1 ** (1 + alpha) / gamma(2 + alpha) / dt2
        + ts1 ** alpha / gamma(1 + alpha))
    out += omegar * fm1
    # fmt: on

    return out


@advance.register(ExplicitTrapezoidal)
def _advance_caputo_explicit_trapezoidal(  # type: ignore[misc]
    m: ExplicitTrapezoidal[StateFunctionT],
    history: ProductIntegrationHistory,
    y: Array,
    dt: Array,
) -> AdvanceResult:
    # set next time step
    n = len(history)
    tstart = m.control.tstart
    t = history.ts[n] = history.ts[n - 1] + dt

    # compute solution
    ynext = np.zeros_like(y)
    ynext = _update_caputo_initial_condition(ynext, m.y0, t - tstart)

    if n <= 1:
        ynext = _update_caputo_forward_euler(ynext, m, history, n)
    else:
        ynext, _ = _update_caputo_trapezoidal(ynext, m, history, n, n - 1)
        ynext = _update_caputo_trapezoidal_extrapolation(ynext, m, history, n)

    trunc = _truncation_error(m.control, m.alpha, t, ynext, t - dt, y)
    return AdvanceResult(ynext, trunc, m.source(t, ynext))


# }}}


# {{{ Predictor-Corrector (PEC and PECE)


@dataclass(frozen=True)
class CaputoPredictorCorrectorMethod(CaputoProductIntegrationMethod[StateFunctionT]):
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

    corrector_iterations: int
    """Number of repetitions of the corrector step."""

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
class PECE(CaputoPredictorCorrectorMethod[StateFunctionT]):
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
class PEC(CaputoPredictorCorrectorMethod[StateFunctionT]):
    """The Predict-Evaluate-Correct (PEC) discretization of the Caputo derivative.

    This is a predictor-corrector similar to :class:`PECE`, where
    the previous evaluation of the predictor is used to avoid an additional
    right-hand side call. Like the PECE method, the corrector step can be
    repeated multiple times for improved error results.
    """


@advance.register(CaputoPredictorCorrectorMethod)
def _advance_caputo_predictor_corrector(  # type: ignore[misc]
    m: CaputoPredictorCorrectorMethod[StateFunctionT],
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> AdvanceResult:
    # set next time step
    n = len(history)
    tstart = m.control.tstart
    t = history.ts[n] = history.ts[n - 1] + dt

    # add initial conditions
    y0 = np.zeros_like(y)
    y0 = _update_caputo_initial_condition(y0, m.y0, t - tstart)

    # predictor step (forward Euler)
    yp = np.copy(y0)
    yp = _update_caputo_forward_euler(yp, m, history, n)

    # corrector step (Adams-Bashforth 2 i.e. Trapezoidal)
    yc_explicit = np.copy(y0)
    yc_explicit, fac = _update_caputo_trapezoidal(yc_explicit, m, history, n, n)

    # corrector iterations
    yc = yp
    for _ in range(m.corrector_iterations):
        fp = m.source(t, yc)
        yc = yc_explicit + fac * fp

    ynext = yc
    f = fp if isinstance(m, PEC) else m.source(t, ynext)

    trunc = _truncation_error(m.control, m.alpha, t, ynext, t - dt, y)
    return AdvanceResult(ynext, trunc, f)


# }}}


# {{{ modified Predictor-Corrector


@dataclass(frozen=True)
class ModifiedPECE(CaputoPredictorCorrectorMethod[StateFunctionT]):
    r"""A modified Predict-Evaluate-Correct-Evaluate (PECE) discretization of the
    Caputo derivative.

    This method is described in [Garrappa2010]_ Equation 8 as a modification to
    the standard :class:`PECE` with improved performance due to reusing the
    convolution weights.

    Note that this method has an improved order, i.e. it achieves
    second-order with a single corrector iteration for all :math:`\alpha`, but
    a smaller stability region.
    """


def _weights_quadrature_trapezoidal(
    m: CaputoProductIntegrationMethod[StateFunctionT],
    t: Array,
    n: int,
    p: int,
) -> tuple[Array, Array]:
    # get time history
    ts = (t[n] - t[: p + 1]).reshape(-1, 1)
    dt = np.diff(t[: p + 1]).reshape(-1, 1)

    alpha = m.alpha
    a1p = 1 + alpha
    g1p = gamma(a1p)
    g2p = a1p * g1p

    I0 = ts**alpha / g1p
    I1 = ts ** (1.0 + alpha) / g2p

    omegal = I1[1:] / dt + I0[:-1] - I1[:-1] / dt
    omegar = I1[:-1] / dt - I0[1:] - I1[1:] / dt

    return omegal, omegar


@advance.register(ModifiedPECE)
def _advance_caputo_modified_pece(  # type: ignore[misc]
    m: ModifiedPECE[StateFunctionT],
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> AdvanceResult:
    n = len(history)

    # set next time step
    tstart = m.control.tstart
    t = history.ts[n] = history.ts[n - 1] + dt

    # compute common terms
    dy = np.zeros_like(y)
    dy = _update_caputo_initial_condition(dy, m.y0, t - tstart)

    # compute predictor
    omegal, omegar = _weights_quadrature_trapezoidal(m, history.ts, n, n)
    fs = history.storage[:n]

    if n == 1:
        yp = _update_caputo_forward_euler(dy, m, history, n)
    else:
        dy += np.einsum("ij,ij->j", omegal[:-1], fs[:-1])
        dy += np.einsum("ij,ij->j", omegar[:-1], fs[1:])

        yp = np.copy(dy)
        yp = _update_caputo_trapezoidal_extrapolation(yp, m, history, n)

    # compute corrector
    ynext = np.copy(dy)
    ynext += omegal[-1] * fs[-1]

    # corrector iterations
    omega = omegar[-1].squeeze()
    yc = yp
    for _ in range(m.corrector_iterations):
        fp = m.source(t, yc)
        yc = ynext + omega * fp
    ynext = yc

    trunc = _truncation_error(m.control, m.alpha, t, ynext, t - dt, y)
    return AdvanceResult(ynext, trunc, m.source(t, ynext))


# }}}


# {{{ L1 method


@dataclass(frozen=True)
class L1(CaputoImplicitProductIntegrationMethod[StateFunctionT]):
    """The first-order implicit L1 discretization of the Caputo derivative.

    Note that, unlike the :class:`ForwardEuler` method, the L1 method discretizes
    the Caputo derivative directly and does not use the Volterra formulation of
    the equation.
    """

    @property
    def order(self) -> float:
        return 2.0 - self.largest_derivative_order


def _update_caputo_l1(
    dy: Array,
    m: CaputoProductIntegrationMethod[StateFunctionT],
    history: ProductIntegrationHistory,
    n: int,
    *,
    diff: bool = True,
) -> tuple[Array, Array]:
    d = dy.size

    assert 0 < n <= len(history)
    ts = history.ts[n] - history.ts[: n + 1]

    # compute convolution coefficients
    alpha = m.alpha.reshape(-1, 1)
    g2m = gamma(2 - alpha)

    omega = (ts[:-1] ** (1 - alpha) - ts[1:] ** (1 - alpha)) / g2m
    h = (omega / np.diff(history.ts[: n + 1])).T
    assert h.shape == (n, d)

    if diff:
        r: Array = history.storage[1 : n + 1] - history.storage[:n]
    else:
        # NOTE: can be used to handle discontinuous data stored as
        # [y^-_n, y^+_n] on the two sides of t_n
        r = history.storage[1 : n + 1, :d] - history.storage[:n, d:]

    dy += np.einsum("ij,ij->j", h[:-1], r[:-1])

    return dy, 1.0 / h[-1]


@advance.register(L1)
def _advance_caputo_l1(  # type: ignore[misc]
    m: L1[StateFunctionT],
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> AdvanceResult:
    # set next time step
    n = len(history)
    t = history.ts[n] = history.ts[n - 1] + dt

    if n == 1:
        # FIXME: the history stored for the first step is f(t, y0), which is not
        # what we want here because we need to just store y0
        history.storage[0] = y

    r = np.zeros_like(y)
    r, h = _update_caputo_l1(r, m, history, len(history))

    ynext = m.solve(t, y, h, y - h * r)
    trunc = _truncation_error(m.control, m.alpha, t, ynext, t - dt, y)

    return AdvanceResult(ynext, trunc, ynext)


# }}}
