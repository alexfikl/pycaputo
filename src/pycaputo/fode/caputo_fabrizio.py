# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from pycaputo.controller import FixedController
from pycaputo.derivatives import CaputoFabrizioOperator
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

from .product_integration import AdvanceResult

log = get_logger(__name__)


# {{{ AtanganaSeda


@dataclass(frozen=True)
class AtanganaSeda(
    FractionalDifferentialEquationMethod[CaputoFabrizioOperator, StateFunctionT]
):
    """Discretizations of the :class:`~pycaputo.derivatives.CaputoFabrizioOperator`
    based on the work from [Atangana2021]_.

    .. [Atangana2021] A. Atangana, S. I. Araz,
        *New Numerical Scheme With Newton Polynomial - Theory, Methods, and
        Applications*,
        Elsevier Science & Technology, 2021.
    """

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()

            if not all(isinstance(d, CaputoFabrizioOperator) for d in self.ds):
                raise TypeError(
                    f"Expected 'CaputoFabrizioOperator' operators: {self.ds}"
                )

            if not isinstance(self.control, FixedController):
                raise TypeError(f"Only 'FixedController' is supported: {self.control}")

    @cached_property
    def alpha(self) -> Array:
        return np.array([d.alpha for d in self.ds])

    @cached_property
    def derivative_order(self) -> tuple[float, ...]:
        return tuple([d.alpha for d in self.ds])

    def make_default_history(self) -> ProductIntegrationHistory:
        nsteps = self.control.nsteps
        return ProductIntegrationHistory.empty_like(
            self.y0[0],
            n=512 if nsteps is None else nsteps,
        )


@make_initial_condition.register(AtanganaSeda)
def _make_initial_condition_caputo_fabrizio_atangana_seda(  # type: ignore[misc]
    m: AtanganaSeda[StateFunctionT],
) -> Array:
    return m.y0[0]


@evolve.register(AtanganaSeda)
def _evolve_caputo_fabrizio_atangana_seda(  # type: ignore[misc]
    m: AtanganaSeda[StateFunctionT],
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


# {{{ AtanganaSeda2


@dataclass(frozen=True)
class AtanganaSeda2(AtanganaSeda[StateFunctionT]):
    """Discretization of the :class:`~pycaputo.derivatives.CaputoFabrizioOperator`
    based on Section 2.3 from [Atangana2021]_.
    """

    @property
    def order(self) -> float:
        # FIXME: this should be higher order (see Section 2.3.1 in [Atangana2021]),
        # but there do not seem to be any convergence tests to verify that.
        return 1.0


@advance.register(AtanganaSeda2)
def _advance_caputo_fabrizio_atangana_seda2(  # type: ignore[misc]
    m: AtanganaSeda2[StateFunctionT],
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> AdvanceResult:
    n = len(history)
    t = history.ts[n - 1] + dt

    alpha = m.alpha
    M = np.array([d.normalization() for d in m.ds])
    M1 = (1.0 - alpha) / M
    M2 = alpha / M

    f = history.storage[n - 1]
    if n == 1:
        # NOTE: first iteration: standard Forward Euler
        dy = M1 * f + dt * M2 * f
    else:
        # NOTE: otherwise: Equation 2.60
        fp = history.storage[n - 2]
        dy = M1 * (f - fp) + dt * M2 * (1.5 * f - 0.5 * fp)

    ynext = y + dy
    return AdvanceResult(ynext, np.zeros_like(ynext), m.source(t, ynext))


# }}}


# {{{ AtanganaSeda3


@dataclass(frozen=True)
class AtanganaSeda3(AtanganaSeda[StateFunctionT]):
    """Discretization of the :class:`~pycaputo.derivatives.CaputoFabrizioOperator`
    based on Section 5 from [Atangana2021]_.

    This method is implemented in the ``AS_Method_for_Chaotic_with_CF_Fractional.m``
    code snippet shown in the Appendix. It is a higher-order method compared to
    :class:`AtanganaSeda2`.
    """

    @property
    def order(self) -> float:
        # FIXME: this should be higher order (see Section 5.1 in [Atangana2021])
        return 1.0


@advance.register(AtanganaSeda3)
def _advance_caputo_fabrizio_atangana_seda3(  # type: ignore[misc]
    m: AtanganaSeda3[StateFunctionT],
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> AdvanceResult:
    n = len(history)
    t = history.ts[n - 1] + dt

    alpha = m.alpha
    M = np.array([d.normalization() for d in m.ds])
    M1 = (1.0 - alpha) / M
    M2 = alpha / M

    # NOTE: code in Atangana2021, Appendix A the first two equations are
    #       y1 = y0 + dt * f0
    #       y2 = y1 + dt * (1.5 * f1 - 0.5 * f0)
    # which is not consistent with Section 5, Equation 5.5. We use the correct
    # formulas instead, although some of the results no longer match..

    f = history.storage[n - 1]
    if n == 1:
        # NOTE: first iteration: standard Forward Euler
        dy = M1 * f + dt * M2 * f
    elif n == 2:
        # NOTE: second iteration: AtanganaSeda2
        fp = history.storage[n - 2]
        dy = M1 * (f - fp) + dt * M2 * (1.5 * f - 0.5 * fp)
    else:
        # NOTE: otherwise: Equation 5.12
        fp = history.storage[n - 2]
        fpp = history.storage[n - 3]

        dy = M1 * (f - fp) + dt * M2 * (
            23.0 / 12.0 * f - 4.0 / 3.0 * fp + 5.0 / 12.0 * fpp
        )

    ynext = y + dy
    return AdvanceResult(ynext, np.zeros_like(ynext), m.source(t, ynext))


# }}}
