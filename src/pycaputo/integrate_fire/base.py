# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Generic, Iterator, NamedTuple, TypeVar

import numpy as np

from pycaputo.derivatives import CaputoDerivative, Side
from pycaputo.fode.base import (
    Event,
    FractionalDifferentialEquationMethod,
    evolve,
    make_initial_condition,
)
from pycaputo.fode.base import (
    StepAccepted as StepAcceptedBase,
)
from pycaputo.fode.base import (
    StepRejected as StepRejectedBase,
)
from pycaputo.history import History, ProductIntegrationHistory
from pycaputo.logging import get_logger
from pycaputo.utils import Array

logger = get_logger(__name__)

# {{{ model interface


class IntegrateFireModel(ABC):
    @abstractmethod
    def source(self, t: float, y: Array) -> Array:
        """Evaluate the right-hand side of the IF model."""

    @abstractmethod
    def source_jac(self, t: float, y: Array) -> Array:
        """Evaluate the Jacobian of the right-hand side of the IF model."""

    @abstractmethod
    def spiked(self, t: float, y: Array) -> float:
        """Check if the neuron has spiked.

        In most cases, this will simply be a delta of :math:`V - V_{peak}`, but
        it is generally left to the model itself to define how the spike is
        recognized.

        :returns: a positive value if the neuron spiked and a negative value
            otherwise.
        """

    @abstractmethod
    def reset(self, t: float, y: Array) -> Array:
        """Evaluate the reset values for the IF model.

        This function assumes that the neuron has spiked, i.e. that :meth:`spiked`
        returns a non-negative value. If this is not the case, the reset should
        not be applied.
        """


ModelT = TypeVar("ModelT", bound=IntegrateFireModel)

# }}}


# {{{ method


@dataclass(frozen=True)
class StepAccepted(StepAcceptedBase):
    #: A flag to denote if the current accepted step was due to a spike.
    spiked: bool


@dataclass(frozen=True)
class StepRejected(StepRejectedBase):
    pass


class AdvanceResult(NamedTuple):
    """Result of :func:`~pycaputo.fode.advance` for
    :class:`CaputoIntegrateFireL1Method` subclasses.
    """

    #: Estimated solution at the next time step.
    y: Array
    #: Estimated truncation error at the next time step.
    trunc: Array
    #: Array to add to the history storage.
    storage: Array

    #: Flag to denote if a spike occurred.
    spiked: Array
    #: Time step taken by the method in case a spike occurred.
    dts: Array


@dataclass(frozen=True)
class CaputoIntegrateFireL1Method(
    FractionalDifferentialEquationMethod, Generic[ModelT]
):
    r"""An implicit discretization of the Caputo derivative for Integrate-and-Fire
    models based on the L1 method.

    A generic leaky Integrate-and-Fire model is given by

    .. math::

        D_C^\alpha[\mathbf{y}](t) = \mathbf{f}(t, \mathbf{y})

    together with a reset condition of the form

    .. math::

        \text{condition}(t, \mathbf{y}) \ge 0
        \quad \text{then} \quad
        \mathbf{y} = \text{reset}(t, \mathbf{y}),

    where the ``condition`` function and the ``reset`` functions are provided
    by the user. For example, if we assume that our model has two variables
    :math:`(V, w)` and we want to model the reset condition

    .. math::

        V \ge V_{peak} \quad \text{then} \quad
        \begin{cases}
        V \gets V_r, \\
        w \gets w + b,
        \end{cases}

    we can set

    .. math::

        \begin{cases}
        \text{condition}(t, [V, w]) = V - V_{peak}, \\
        \text{reset}(t, [V, w]) = (V_r, w + b)
        \end{cases}

    for certain coefficients :math:`V_{peak}, V_r` and :math:`b`. These can
    be very general conditions. The variant of the L1 method implemented here
    takes into account the inherent discontinuity of the solutions to this
    type of equation (see :class:`~pycaputo.differentiation.CaputoL1Method`
    for a smooth implementation).
    """

    #: Integrate-and-Fire model parameters and functions.
    model: ModelT

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()
            if any(a > 1.0 or a < 0.0 for a in self.derivative_order):
                raise ValueError(f"Expected 'alpha' in (0, 1): {self.derivative_order}")

    @cached_property
    def d(self) -> tuple[CaputoDerivative, ...]:
        return tuple([
            CaputoDerivative(order=alpha, side=Side.Left)
            for alpha in self.derivative_order
        ])

    @property
    def order(self) -> float:
        return 1.0

    def solve(self, t: float, y0: Array, c: Array, r: Array) -> Array:
        """Wrapper around :func:`pycaputo.implicit.solve` to solve the implicit
        equation.

        This function should be overwritten for specific applications if better
        solvers are known. For example, many problems can be solved explicitly
        or approximated to a very good degree to provide a better *y0*.
        """
        raise NotImplementedError(type(self).__name__)


@make_initial_condition.register(CaputoIntegrateFireL1Method)
def _make_initial_condition_caputo(m: CaputoIntegrateFireL1Method[ModelT]) -> Array:
    return m.y0[0]


@evolve.register(CaputoIntegrateFireL1Method)
def _evolve_caputo_integrate_fire_l1(
    m: CaputoIntegrateFireL1Method[ModelT],
    *,
    history: History[Any] | None = None,
    dtinit: float | None = None,
) -> Iterator[Event]:
    from pycaputo.controller import (
        StepEstimateError,
        estimate_initial_time_step,
        evaluate_error_estimate,
        evaluate_timestep_accept,
        evaluate_timestep_factor,
        evaluate_timestep_reject,
    )
    from pycaputo.fode.base import StepFailed, advance

    if history is None:
        history = ProductIntegrationHistory.empty_like(np.hstack([m.y0[0], m.y0[0]]))

    # initialize
    c = m.control
    n = 0
    t = c.tstart

    # determine initial condition
    yprev = make_initial_condition(m)
    history.append(t, np.hstack([yprev, yprev]))

    # determine initial time step
    if dtinit is None:
        dtinit = estimate_initial_time_step(
            t, yprev, m.source, m.smallest_derivative_order, trunc=m.order + 1
        )

    dt = dtinit
    yield StepAccepted(
        t=t,
        iteration=n,
        dt=dt,
        y=yprev,
        eest=0.0,
        q=1.0,
        trunc=np.zeros_like(yprev),
        spiked=False,
    )

    while not c.finished(n, t):
        # evolve solution with current estimate of the time step
        result = advance(m, history, yprev, dt)

        assert isinstance(result, AdvanceResult)
        ynext, trunc, storage = result[:3]

        # determine the next time step
        if result.spiked:
            # NOTE: a spike occurred, so we reset everything
            dtnext = dtinit
            accepted = True
            eest = 0.0
            q = 1.0
        else:
            try:
                # estimate error
                eest = evaluate_error_estimate(c, m, trunc, ynext, yprev)
                accepted = eest <= 1.0

                # estimate time step factor using the error
                q = evaluate_timestep_factor(c, m, eest)

                # finally estimate a good dt for the next step
                tmp_state = {"t": t, "n": n, "y": yprev}
                if accepted:
                    dtnext = evaluate_timestep_accept(c, m, q, dt, tmp_state)
                else:
                    dtnext = evaluate_timestep_reject(c, m, q, dt, tmp_state)
            except StepEstimateError as exc:
                accepted = False
                dtnext = 1.0e-6
                logger.error("Failed to estimate timestep.", exc_info=exc)
                yield StepFailed(t=t, iteration=n, reason="Failed to estimate timestep")

        assert dtnext >= 0.0

        # update variables
        assert result.dts >= 0
        if eest <= 1.0:
            dt = float(result.dts)

            n += 1
            t += dt
            yprev = ynext
            history.append(t, storage)

            if result.spiked:
                # NOTE: if we spiked, also yield the value before the
                # discontinuity so that the plots look nice!
                yield StepAccepted(
                    t=t,
                    iteration=n,
                    dt=dt,
                    y=storage[: ynext.size],
                    eest=eest,
                    q=q,
                    trunc=trunc,
                    spiked=bool(result.spiked),
                )

                # NOTE: we need to increment the iteration, but not the time
                # because both values in the spike happen at the same time due
                # to the discontinuity
                n += 1

            yield StepAccepted(
                t=t,
                iteration=n,
                dt=dt,
                y=ynext,
                eest=eest,
                q=q,
                trunc=trunc,
                # NOTE: if we spiked on this step, the previous value will
                # record it so we don't need to do it twice.
                spiked=False,
            )
        else:
            yield StepRejected(
                t=t, iteration=n, dt=dt, y=ynext, eest=eest, q=q, trunc=trunc
            )

        dt = dtnext


def advance_caputo_integrate_fire_l1(
    m: CaputoIntegrateFireL1Method[ModelT],
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> tuple[AdvanceResult, Array]:
    # set next time step
    d = y.size
    n = len(history)
    t = history.ts[n] = history.ts[n - 1] + dt

    # get time history
    ts = history.ts[n] - history.ts[: n + 1]

    # compute convolution coefficients
    alpha = m.alpha.reshape(-1, 1)
    gamma2m = m.gamma2m.reshape(-1, 1)

    omega = (ts[:-1] ** (1 - alpha) - ts[1:] ** (1 - alpha)) / gamma2m
    h = (omega / np.diff(history.ts[: n + 1])).T
    assert h.shape == (n, d)

    # NOTE: we store (t_k, [y^-_k, y^+_k]) at every time step and want to compute
    #   h_{n + 1, k} (y^-_{k + 1} - y^+_k)
    dy = history.storage[1 : n + 1, :d] - history.storage[:n, d:]
    r = np.einsum("ij,ij->j", h[:-1], dy[:-1])

    from pycaputo.fode.caputo import _truncation_error

    ynext = m.solve(t, y, 1 / h[-1], y - r / h[-1])
    trunc = _truncation_error(m.control, m.alpha, t, ynext, t - dt, y)

    result = AdvanceResult(
        ynext, trunc, np.hstack([ynext, ynext]), spiked=np.array(0), dts=np.array(dt)
    )

    return result, r


# }}}


# {{{ spike time


def estimate_spike_time_linear(
    t: float, V: Array, tprev: float, Vprev: Array, Vpeak: Array | float
) -> float:
    """Give a linear estimation of the spike time.

    .. math::

        V(t) = a + b t.

    We assume that the spike occurred between :math:`(t, V)` and
    :math:`(t_{prev}, V_{prev})` at :math:`V_{peak}`. This information can be
    used to provide a simple linear estimation for the spike time.

    :return: an estimation of the spike time.
    """
    assert Vprev <= Vpeak <= V
    ts = (Vpeak - Vprev) / (V - Vprev) * t + (V - Vpeak) / (V - Vprev) * tprev
    assert tprev <= ts <= t

    return float(ts)


def advance_caputo_integrate_fire_spike_linear(
    tprev: float,
    y: Array,
    t: float,
    result: AdvanceResult,
    *,
    v_peak: float,
    v_reset: float,
) -> AdvanceResult:
    # we have spiked, so we need to reconstruct our solution
    yprev = np.array([v_peak], dtype=y.dtype)
    ynext = np.array([v_reset], dtype=y.dtype)

    # estimate the time step used to get here
    ts = estimate_spike_time_linear(t, result.y[0], tprev, y[0], v_peak)
    dt = float(ts - tprev)

    # set the truncation error to zero because we want to reset the next time
    # step to the maximum allowable value.
    trunc = np.zeros_like(y)

    return AdvanceResult(
        ynext, trunc, np.hstack([yprev, ynext]), spiked=np.array(1), dts=np.array(dt)
    )


def estimate_spike_time_exp(
    t: float, V: Array, tprev: float, Vprev: Array, Vpeak: Array | float
) -> float:
    """Give an exponential estimation of the spike time.

    .. math::

        V(t) = a b^t.

    :returns: an estimate of the spike time.
    """
    assert Vprev <= Vpeak <= V
    b = (V / Vprev) ** (1 / (t - tprev))
    ts = tprev + (np.log(Vpeak) - np.log(Vprev)) / np.log(b)

    assert tprev <= ts <= t
    return float(ts)


# }}}
