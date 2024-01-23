# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, field
from functools import singledispatch
from typing import TYPE_CHECKING, Any

import numpy as np

from pycaputo.utils import Array, StateFunction, StateFunctionT

if TYPE_CHECKING:
    # NOTE: avoid cyclic import
    from pycaputo.stepping import FractionalDifferentialEquationMethod

# {{{ utils


def _hairer_norm(x: Array, *, p: int | float | str = 2) -> float:
    r"""Computes the norm from Equation 4.11 in [Hairer2008]_

    .. math::

        \|\mathbf{x}\| \triangleq \sqrt{
            \frac{1}{n} \sum_{i = 0}^{n - 1} x^2_i
        }
    """
    if p == 2:
        return float(np.sqrt(np.sum(x**2) / x.size))

    if p in {np.inf, "inf"}:
        return float(np.max(np.abs(x)))

    if p in {-np.inf, "-inf"}:
        return float(np.min(np.abs(x)))

    from numbers import Number

    if isinstance(p, (int, float, Number)):
        return float((np.sum(x**p) / x.size) ** (1.0 / p))

    raise ValueError(f"Unknown norm order: {p!r}")


def _normalize_time_span_triple(
    dt: float,
    tstart: float = 0.0,
    tfinal: float | None = None,
    nsteps: int | None = None,
) -> tuple[float, float | None, int | None]:
    """Attempts to normalize the ``(dt, tfinal, nsteps)`` triple such that
    ``tfinal = tstart + nsteps * dt``.
    """
    if tfinal is not None and nsteps is not None:
        # NOTE: if both tfinal and nsteps are given, we simply take the smallest
        # [tstart, tfinal] range that we can construct from them
        tf = tstart + nsteps * dt
        if tf < tfinal:
            tfinal = tf
        else:
            nsteps = int((tfinal - tstart) / dt) + 1
            dt = (tfinal - tstart) / nsteps
    elif tfinal is not None:
        nsteps = int((tfinal - tstart) / dt) + 1
        dt = (tfinal - tstart) / nsteps
    elif nsteps is not None:
        tfinal = tstart + nsteps * dt
    else:
        raise ValueError("Must provide either 'tfinal' or 'nsteps' or both")

    return dt, tfinal, nsteps


def make_fixed_controller(
    dt: float,
    tstart: float = 0.0,
    tfinal: float | None = None,
    nsteps: int | None = None,
) -> FixedController:
    """Create a controller with a fixed time step.

    This ensures that the following relation holds for all given values

    .. code:: python

        tfinal = tstart + nsteps * dt

    This is achieved by small modifications to either *nsteps*, *dt* or *tfinal*.
    If both *nsteps* and *tfinal* are given, then the smallest of
    ``tstart + nsteps * dt`` and ``tfinal`` is taken as the final time and the
    values are recalculated.

    :arg dt: desired time step (chosen time step may be slightly smaller).
    :arg tstart: start of the time span.
    :arg tfinal: end of the time span.
    :arg nsteps: number of time steps in the span.
    """
    if tfinal is None and nsteps is None:
        raise ValueError("Must provide either 'tfinal' or 'nsteps' or both")

    dt, tfinal, nsteps = _normalize_time_span_triple(dt, tstart, tfinal, nsteps)
    return FixedController(tstart=tstart, tfinal=tfinal, nsteps=nsteps, dt=dt)


def make_graded_controller(
    dt: float | None = None,
    tstart: float = 0.0,
    tfinal: float | None = None,
    nsteps: int | None = None,
    *,
    alpha: float | None = None,
    r: float | None = None,
) -> GradedController:
    """Create a controller with a graded time step (see :func:`make_fixed_controller`).

    :arg alpha: order of the fractional operator. The order is used to choose an
        optimal grading *r* according to [Stynes2017]_.
    :arg r: the degree of grading in the time step (see :class:`GradedController`).
    """
    if tfinal is None and nsteps is None:
        raise ValueError("Must provide either 'tfinal' or 'nsteps' or both")

    if r is None and alpha is None:
        raise ValueError("Must provide either 'alpha' or 'r' or both to define grading")

    if r is None:
        assert alpha is not None
        if 0.0 < alpha <= 1.0:
            r = (2 - alpha) / alpha
        else:
            raise ValueError("Grading estimate is only valid for 'alpha' in (0, 1)")

    if dt is None:
        if tfinal is None or nsteps is None:
            raise ValueError("Must provide both 'tfinal' and 'nsteps' with no 'dt'")
    else:
        dt, tfinal, nsteps = _normalize_time_span_triple(dt, tstart, tfinal, nsteps)
    return GradedController(tstart=tstart, tfinal=tfinal, nsteps=nsteps, r=r)


def estimate_initial_time_step(
    t0: float,
    y0: Array,
    f: StateFunction,
    alpha: float,
    *,
    trunc: float | None = None,
    abstol: float = 1.0e-4,
    reltol: float = 1.0e-8,
) -> float:
    """Give an estimate for the initial time step.

    Note that this time step is based on a forward Euler method for fractional
    ODEs inspired by Equation 4.14 in [Hairer2008]_. It is recommended to take
    a smaller time step and use an adaptive controller for accurate results.

    :arg t0: initial time corresponding to the initial condition.
    :arg y0: initial condition for the evolution equation.
    :arg f: right-hand side of the fractional equation.
    :arg alpha: fractional order of the operator. If the system has different
        orders for each equation, this can be the largest order.
    :arg trunc: an estimate of the local truncation error order of the method
        this time step will be used with. If not provided, the Euler method is
        assumed.

    :returns: an estimate of the initial time step.
    """
    if trunc is None:
        trunc = 2.0

    from math import gamma

    tol = abstol + np.abs(y0) * reltol
    eps = np.sqrt(abstol)

    # get a first estimate of the time step
    f0 = f(t0, y0)

    d0 = _hairer_norm(y0 / tol)
    d1 = _hairer_norm(f0 / tol)
    if d0 < abstol or d1 < abstol:
        h0 = abstol / 100
    else:
        h0 = gamma(1 + alpha) * (eps * d0 / d1) ** (1 / alpha)

    # get a second estimate of the time step
    y1 = y0 + h0**alpha / gamma(1 + alpha) * f0
    f1 = f(t0 + h0, y1)

    d2 = _hairer_norm((f1 - f0) / tol) / h0
    dm = max(d1, d2)
    if dm < 1.0e-15:
        h1 = max(abstol / 100, 10 * abstol * h0)
    else:
        h1 = (eps / max(d1, d2)) ** (1 / (trunc + 1))

    return 0.5 * min(10 * h0, h1)


# }}}


# {{{ interface


class StepEstimateError(RuntimeError):
    """An exception raised when a time step estimate has failed."""


@dataclass(frozen=True)
class Controller:
    """A generic time step controller.

    A controller is used to decide if a time step can be accepted and give an
    estimate for the next time step. If the controller is adaptive, it should
    also give guarantees that the next time step will be accepted with high
    probability.
    """

    #: Start of the time interval.
    tstart: float
    #: End of the time interval (leave *None* for infinite time stepping).
    tfinal: float | None
    #: Number of time steps (leave *None* for infinite time stepping).
    nsteps: int | None

    if __debug__:

        def __post_init__(self) -> None:
            if self.tfinal is not None and self.tstart > self.tfinal:
                raise ValueError("Invalid time interval: 'tstart' > 'tfinal'")

            if self.nsteps is not None and self.nsteps <= 0:
                raise ValueError(
                    f"Number of iterations must be positive: {self.nsteps}"
                )

    def finished(self, n: int, t: float) -> bool:
        """Check if the evolution should finish at iteration *n* and time *t*."""
        # fmt: off
        return (
            (self.tfinal is not None and t >= self.tfinal)
            or (self.nsteps is not None and n >= self.nsteps))
        # fmt: on


@singledispatch
def evaluate_error_estimate(
    c: Controller,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    trunc: Array,
    y: Array,
    yprev: Array,
) -> float:
    """Compute a scaled scalar error estimate based on the elementwise error *trunc*.

    For most controllers, *trunc* should be an estimate of the local truncation
    error of the method. This truncation error will be scaled by a relative
    tolerance based on the solution values *y* and *yprev*. In general, this
    should return an error estimate such that :math:`E_{est} < 1.0` means that
    the step is accepted.

    If the step is accepted, then a controller can be generated using
    :func:`evaluate_timestep_factor` and :func:`evaluate_timestep_accept`
    or :func:`evaluate_timestep_reject`.

    :arg trunc: an error estimate for the truncation error at the current time step.
    :returns: a scaled estimate of the error that can be used to construct a
        time step controller.
    """
    raise NotImplementedError(type(c).__name__)


@singledispatch
def evaluate_timestep_factor(
    c: Controller, m: FractionalDifferentialEquationMethod[StateFunctionT], eest: float
) -> float:
    """Compute a factor for the step size control.

    If the step is accepted, this factor can be used to increase or decrease the
    time step, based on the local error estimate.
    """
    raise NotImplementedError(type(c).__name__)


@singledispatch
def evaluate_timestep_accept(
    c: Controller,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    q: float,
    dtprev: float,
    state: dict[str, Any],
) -> float:
    """Evaluate a new time step based on the controller at the current time.

    This function is meant to be called when a step is accepted.

    :arg q: a step size control (see :func:`evaluate_timestep_factor`).
    :arg dtprev: previously used time step.
    :arg state: a dictionary containing a description of the state at the
        current time step. This can vary by solver, but it should contain
        the current time *t*, the current time step *n*, the current state *y*.
        In case the controller is meant to be used with a specific integrator,
        it can contain other necessary information.

    :returns: an estimate for the time step at the next iteration.
    """
    raise NotImplementedError(type(c).__name__)


@singledispatch
def evaluate_timestep_reject(
    c: Controller,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    q: float,
    dtprev: float,
    state: dict[str, Any],
) -> float:
    """Evaluate a new time step based on the controller at the current time.

    This function is meant to be called when a step is rejected and takes the
    same arguments. By default, :func:`evaluate_timestep_accept` is called, but
    some controllers can define different behavior when the step is accepted or
    rejected.
    """
    return evaluate_timestep_accept(c, m, q, dtprev, state)


# }}}


# {{{ non-adaptive controller


@dataclass(frozen=True)
class FixedController(Controller):
    """A fake controller with a fixed time step."""

    #: Fixed time step used by the controller.
    dt: float


@evaluate_error_estimate.register(FixedController)
def _evaluate_error_estimate_fixed(
    c: FixedController,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    trunc: Array,
    y: Array,
    yprev: Array,
) -> float:
    return 0.0


@evaluate_timestep_factor.register(FixedController)
def _evaluate_timestep_factor_fixed(
    c: FixedController,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    eest: float,
) -> float:
    return 1.0


@evaluate_timestep_accept.register(FixedController)
def _evaluate_timestep_accept_fixed(
    c: FixedController,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    q: float,
    dtprev: float,
    state: dict[str, Any],
) -> float:
    dt = q * c.dt
    if c.tfinal is not None:
        eps = 5.0 * np.finfo(m.y0[0].dtype).eps
        dt = min(dt, c.tfinal - state["t"]) + eps

    return dt


@dataclass(frozen=True)
class GradedController(Controller):
    r"""A :class:`Controller` with a variable graded time step.

    This graded grid of time steps is described in [Garrappa2015b]_. It
    essentially takes the form

    .. math::

        \Delta t_n = \frac{t_f - t_s}{N^r} ((n + 1)^r - n^r),

    where the time interval is :math:`[t_s, t_f]` and :math:`N` time steps are
    taken. This graded grid can give full second-order convergence for certain
    methods such as the Predictor-Corrector method (e.g. implemented by
    :class:`~pycaputo.fode.caputo.PECE`).
    """

    #: A grading exponent that controls the clustering of points at
    #: :attr:`~Controller.tstart`.
    r: float

    if __debug__:

        def __post_init__(self) -> None:
            if self.tfinal is None:
                raise ValueError("'tfinal' must be given for the graded estimate.")

            if self.nsteps is None:
                raise ValueError("'nsteps' must be given for the graded estimate")

            if self.r < 1:
                raise ValueError(f"Exponent must be >= 1: {self.r}")

            super().__post_init__()

    def estimate_initial_time_step(
        self,
        m: FractionalDifferentialEquationMethod[StateFunctionT],
        t: float,
        y: Array,
    ) -> float:
        assert self.tfinal is not None
        assert self.nsteps is not None
        return float((self.tfinal - self.tstart) / self.nsteps**self.r)


@evaluate_error_estimate.register(GradedController)
def _evaluate_error_estimate_graded(
    c: GradedController,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    trunc: Array,
    y: Array,
    yprev: Array,
) -> float:
    return 0.0


@evaluate_timestep_factor.register(GradedController)
def _evaluate_timestep_factor_graded(
    c: GradedController,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    eest: float,
) -> float:
    return 1.0


@evaluate_timestep_accept.register(GradedController)
def _evaluate_timestep_accept_graded(
    c: GradedController,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    q: float,
    dtprev: float,
    state: dict[str, Any],
) -> float:
    assert c.tfinal is not None
    assert c.nsteps is not None

    n = state["n"]
    h = (c.tfinal - c.tstart) / (c.nsteps - 1) ** c.r
    dt = q * h * ((n + 1) ** c.r - n**c.r)

    if c.tfinal is not None:
        eps = 5.0 * np.finfo(m.y0[0].dtype).eps
        dt = min(dt, c.tfinal - state["t"]) + eps

    return float(dt)


# }}}


# {{{ adaptive controller


@dataclass(frozen=True)
class AdaptiveController(Controller):
    r"""An adaptive time step controller.

    An adaptive time step controller uses the solution at the current time
    step (and potentially some of its history) to determine a new time step.
    These methods are completely general and can be applied to any evolution
    equation, but better methods may exist for specific cases.

    An adaptive controller takes an error computed by the numerical integrator
    and computes a scaled variant of the form

    .. math::

        \mathcal{E}^{n + 1}_i =
            \frac{E^{n + 1}_i}{
                \epsilon_{\text{abs}}
                + \epsilon_{\text{rel}} \max (|y^n_i|, |y^{n + 1}_i|)}

    for each component. Then, a global norm is computed

    .. math::

        E_{n + 1}^{\text{scaled}} =
        \sqrt{\frac{1}{M} \sum_{i = 0}^{M - 1} \left(\mathcal{E}^{n + 1}_i\right)^2}

    The time step is rejected if :math:`E^{n + 1}_{\text{scaled}} > 1` and
    accepted otherwise. Each controller decides the manner in which the time
    step is updated for each of these cases.
    """

    #: Smallest allowable time step by the controller.
    dtmin: float

    #: Smallest allowable step change factor. If the factor is larger than this,
    #: no change in the time step will be added.
    qmin: float
    #: Largest allowable step change factor. If the factor is smaller than this,
    #: no change in the time step will be added.
    qmax: float

    #: Absolute tolerance used by the controller to compute a scaled norm.
    abstol: float
    #: Relative tolerance used by the controller to compute a scaled norm.
    reltol: float

    nrejects: int = field(default=0, init=False)

    @property
    def max_rejects(self) -> int:
        return 16

    @property
    def safety_factor(self) -> float:
        """A safety factor that multiplies the time step to ensure that the
        next time step is accepted with high probability.
        """
        return 0.9

    @property
    def steady_factor(self) -> float:
        r"""A factor used to check if an accepted time step changed too little.

        This can be used to avoid needless oscillations in the time step during
        adaptation. If the new time step is between a fixed percentage of the
        previous time step, then it is not changed.
        """
        return 0.2


@evaluate_error_estimate.register(AdaptiveController)
def _evaluate_error_estimate_adaptive(
    c: AdaptiveController,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    trunc: Array,
    y: Array,
    yprev: Array,
) -> float:
    if c.nrejects > c.max_rejects:
        return 1.0

    scaled_trunc = trunc / (c.abstol + np.maximum(np.abs(yprev), np.abs(y)) * c.reltol)
    eest = _hairer_norm(scaled_trunc)

    if not np.isfinite(eest):
        raise StepEstimateError("Scaled error estimate is not finite")

    return eest


# }}}

# {{{ integral controller


@dataclass(frozen=True)
class IntegralController(AdaptiveController):
    r"""A standard integral controller.

    This controller is described in Equation 3.12 from [Hairer2008]_ and the
    surrounding discussion. It is essentially given by

    .. math::

        \Delta t^{est}_{n + 1} = \Delta t_n (E^{scaled}_{n + 1})^{-\frac{1}{p + 1}},

    where :math:`p` is the order of the method and the scaled error estimate is
    constructed by means of :func:`evaluate_error_estimate`.
    """


@evaluate_timestep_factor.register(IntegralController)
def _evaluate_timestep_factor_integral(
    c: IntegralController,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    eest: float,
) -> float:
    if eest == 0.0:
        return c.qmax

    order = m.order
    q = c.safety_factor * eest ** (-1.0 / (order + 1))
    q = max(c.qmin, min(c.qmax, q))

    return float(q)


@evaluate_timestep_accept.register(IntegralController)
def _evaluate_timestep_accept_integral(
    c: IntegralController,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    q: float,
    dtprev: float,
    state: dict[str, Any],
) -> float:
    object.__setattr__(c, "nrejects", 0)

    dt = q * dtprev

    # keep the time step if the change is small enough
    if (1.0 - c.steady_factor) * dtprev < dt < (1.0 + c.steady_factor) * dtprev:
        dt = dtprev

    # keep the time step under tfinal
    dt = max(c.dtmin, dt)
    if c.tfinal is not None:
        dt = min(dt, c.tfinal - state["t"]) + 5.0 * np.finfo(m.y0[0].dtype).eps

    return dt


@evaluate_timestep_reject.register(IntegralController)
def _evaluate_timestep_reject_integral(
    c: IntegralController,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    q: float,
    dtprev: float,
    state: dict[str, Any],
) -> float:
    object.__setattr__(c, "nrejects", c.nrejects + 1)
    dt = q * dtprev

    dt = max(c.dtmin, dt)
    if c.tfinal is not None:
        eps = 5.0 * np.finfo(m.y0[0].dtype).eps
        dt = min(dt, c.tfinal - state["t"]) + eps

    return dt


# }}}


# {{{ proportional-integral controller


@dataclass(frozen=True)
class ProportionalIntegralController(AdaptiveController):
    r"""A standard proportional integral (PI) controller.

    This controller is described in Equation 8.25 from [Hairer2010]_ and the
    surrounding discussion. It is essentially given by

    .. math::

        \Delta t^{est}_{n + 1} =
            \Delta t_n
            (E^{scaled}_{n + 1})^{-\frac{2}{p + 1}}
            (E^{scaled}_{n})^{\frac{1}{p + 1}},

    where :math:`p` is the order of the method and the scaled error estimate is
    constructed by means of :func:`evaluate_error_estimate`. This estimate is
    better suited for stiff equations, as it allows for faster adaptation.
    """

    # integral variables
    qi: float = field(default=1.0, init=False)
    eestprev: float = field(default=1.0, init=False)


@evaluate_timestep_factor.register(ProportionalIntegralController)
def _evaluate_timestep_factor_proportional_integral(
    c: ProportionalIntegralController,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    eest: float,
) -> float:
    if eest == 0.0:
        return c.qmax

    # NOTE: the factors used here are from OrdinaryDiffEq.jl
    # beta1, beta2 = 7.0 / 10.0, 2.0 / 5.0
    beta1 = beta2 = 1.0
    order = m.order

    # I controller
    qi = c.safety_factor * eest ** (-beta1 / (order + 1))
    # PI controller
    qp = qi * c.eestprev ** (beta2 / (order + 1))

    qi = max(c.qmin, min(c.qmax, qi))
    qp = max(c.qmin, min(c.qmax, qp))

    object.__setattr__(c, "qi", qi)
    return float(qp)


@evaluate_timestep_accept.register(ProportionalIntegralController)
def _evaluate_timestep_accept_proportional_integral(
    c: ProportionalIntegralController,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    q: float,
    dtprev: float,
    state: dict[str, Any],
) -> float:
    object.__setattr__(c, "nrejects", 0)

    # FIXME: Hairer2010 has an extra `dtprev / dtpreprev` here?
    dt = q * dtprev

    # keep the time step if the change is small enough
    if (1.0 - c.steady_factor) * dtprev < dt < (1.0 + c.steady_factor) * dtprev:
        dt = dtprev

    # keep the time step under tfinal
    dt = max(c.dtmin, dt)
    if c.tfinal is not None:
        dt = min(dt, c.tfinal - state["t"]) + 5.0 * np.finfo(m.y0[0].dtype).eps

    object.__setattr__(c, "eestprev", max(state["eest"], 1.0e-4))
    return dt


@evaluate_timestep_reject.register(ProportionalIntegralController)
def _evaluate_timestep_reject_proportional_integral(
    c: ProportionalIntegralController,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    q: float,
    dtprev: float,
    state: dict[str, Any],
) -> float:
    object.__setattr__(c, "nrejects", c.nrejects + 1)

    dt = max(c.dtmin, c.qi * dtprev)
    if c.tfinal is not None:
        eps = 5.0 * np.finfo(m.y0[0].dtype).eps
        dt = min(dt, c.tfinal - state["t"]) + eps

    return dt


# }}}


# {{{ Jannelli integral controller


def make_jannelli_controller(
    tstart: float = 0.0,
    tfinal: float | None = None,
    nsteps: int | None = None,
    *,
    dtmin: float = 1.0e-6,
    sigma: float = 0.5,
    rho: float = 1.5,
    chimin: float | None = None,
    chimax: float | None = None,
    abstol: float = 1.0e-8,
) -> JannelliIntegralController:
    r"""Construct a :class:`JannelliIntegralController`.

    This functions simply provides some useful defaults

    :arg sigma: factor by which the time step will be decreased in the case the
        step is rejected. A default value of ``0.5`` will half the time step
        each time, but smaller increments are possible.
    :arg rho: factor by which the time step will be increased in the case the
        step is accepted. A default value of ``1.5`` with increase the time
        step by half each time, but smaller increments are possible.
    :arg chimin: a lower bound on the error estimated by the Jannelli controller.
    :arg chimax: an upper bound on the error estimated by the Jannelli controller.
    """
    if chimin is None or chimax is None:
        raise ValueError(
            "Both 'chimin' and 'chimax' must pe provided. These are generally "
            "problem dependent and no good estimation is possible a priori."
        )

    return JannelliIntegralController(
        tstart=tstart,
        tfinal=tfinal,
        nsteps=nsteps,
        dtmin=dtmin,
        qmin=sigma,
        qmax=rho,
        chimin=chimin,
        chimax=chimax,
        abstol=abstol,
        reltol=1.0,  # NOTE: not used by this methof
    )


@dataclass(frozen=True)
class JannelliIntegralController(AdaptiveController):
    r"""A time step controller from [Jannelli2020]_ for fractional equations.

    This controller uses the error estimate

    .. math::

        \chi_{n + 1} \triangleq \Gamma(\alpha + 1)
            \frac{t_{n + 1} - t_n}{t^\alpha_{n + 1} - t^\alpha_{n}}
            \|y_{n + 1} - y_n\|_p.

    This is not a local truncation error as used by other adaptive controllers.
    However, we scale it in the same way as described in :class:`AdaptiveController`
    to obtain :math:`\hat{\chi}_{n + 1}`. Then we set

    .. math::

        E^{scaled}_{n + 1} =
            \frac{\hat{\chi}_{n + 1} - \chi_{min}}{\chi_{max} - \chi_{min}}.

    The time step is then adapted using a fixed factor, based on
    :attr:`~AdaptiveController.qmin` (denoted by :math:`\sigma`) as a decrease
    factor and :attr:`~AdaptiveController.qmax` (denoted by :math:`\rho`) as
    the amplification factor.
    """

    #: Minimum limit of the error estimate.
    chimin: float
    #: Maximum limit of the error estimate.
    chimax: float


@evaluate_error_estimate.register(JannelliIntegralController)
def _evaluate_error_estimate_jannelli(
    c: JannelliIntegralController,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    trunc: Array,
    y: Array,
    yprev: Array,
) -> float:
    if c.nrejects > c.max_rejects:
        return 0.5

    # FIXME: This seems to give very confusing results for the van der Pol
    # oscillator at least. Seems like values where ymax ~ 0 blow up the error
    # even though the region is relatively flat? Probably indicative of a bad
    # normalization in Jannelli2020?

    # ymax = np.abs(yprev)
    # ymax = np.where(ymax < c.abstol, 1.0, ymax)
    chi = _hairer_norm(trunc, p=2)

    # normalize the estimate based on chimin and chimax
    return (chi - c.chimin) / (c.chimax - c.chimin)


@evaluate_timestep_factor.register(JannelliIntegralController)
def _evaluate_timestep_factor_jannelli(
    c: JannelliIntegralController,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    eest: float,
) -> float:
    if eest < 0.0:
        q = c.qmax
    elif eest > 1.0:
        q = c.qmin
    else:
        q = 1.0

    return q


@evaluate_timestep_accept.register(JannelliIntegralController)
def _evaluate_timestep_accept_jannelli(
    c: JannelliIntegralController,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    q: float,
    dtprev: float,
    state: dict[str, Any],
) -> float:
    object.__setattr__(c, "nrejects", 0)

    # keep the time step under tfinal
    dt = max(c.dtmin, q * dtprev)
    if c.tfinal is not None:
        eps = 5.0 * np.finfo(m.y0[0].dtype).eps
        dt = min(dt, c.tfinal - state["t"]) + eps

    return dt


@evaluate_timestep_reject.register(JannelliIntegralController)
def _evaluate_timestep_reject_jannelli(
    c: JannelliIntegralController,
    m: FractionalDifferentialEquationMethod[StateFunctionT],
    q: float,
    dtprev: float,
    state: dict[str, Any],
) -> float:
    dt = q * dtprev
    if dt <= c.dtmin:
        object.__setattr__(c, "nrejects", c.max_rejects + 1)
    else:
        object.__setattr__(c, "nrejects", c.nrejects + 1)

    dt = max(c.dtmin, q * dtprev)
    if c.tfinal is not None:
        eps = 5.0 * np.finfo(m.y0[0].dtype).eps
        dt = min(dt, c.tfinal - state["t"]) + eps

    return dt


# }}}
