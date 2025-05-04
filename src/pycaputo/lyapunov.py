# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pycaputo.derivatives import FractionalOperatorT
from pycaputo.logging import get_logger
from pycaputo.stepping import FractionalDifferentialEquationMethod, evolve
from pycaputo.typing import Array, StateFunctionT

log = get_logger(__name__)


# {{{ lyapunov_exponents


def lyapunov_exponents(
    m: FractionalDifferentialEquationMethod[FractionalOperatorT, StateFunctionT],
    *,
    source_jac: StateFunctionT | None = None,
) -> Array:
    """Compute all the Lyapunov exponents using method *m* based on [Li2023]_.

    This method augments the original system with another :math:`n^2` variables
    that correspond to the perturbation matrix. This can be very large if the
    original system is already non-trivial, so it is not recommended to use this
    function in that case.

    .. [Li2023] H. Li, Y. Shen, Y. Han, J. Dong, J. Li,
        *Determining Lyapunov Exponents of Fractional-Order Systems:
        A General Method Based on Memory Principle*,
        Chaos, Solitons & Fractals, Vol. 168, pp. 113167--113167, 2023,
        `DOI <https://doi.org/10.1016/j.chaos.2023.113167>`__.

    :arg source_jac: the Jacobian of the source term provided by *m*. If *m* is
        an implicit method that already defined a Jacobian, this is not required.
    """

    # {{{ combined system

    if not 0 < m.largest_derivative_order < 1:
        raise ValueError(
            f"Only fractional orders in (0, 1) are supported: {m.derivative_order}"
        )

    n = m.y0[0].size
    dtype = m.y0[0].dtype
    if m.y0[0].shape != (n,):
        raise ValueError(
            f"Only 1d array variables are supported: y0 has shape {m.y0[0].shape}"
        )

    if source_jac is None:
        source_jac = getattr(m, "source_jac", None)

    if source_jac is None:
        raise ValueError(
            "Must provide Jacobian in 'source_jac' to compute Lyapunov exponent"
        )

    # NOTE: For the method in [Li2023], we need to construct an augmented system
    # that also contains the combined Jacobian matrix, i.e. Equation (7). This
    # can be written in matrix form for $\Lambda \in \mathbb{R}^{n \times n}$
    #
    #   D^\alpha[\Lambda] = J_f \Lambda
    #   \Lambda(0) = I
    #
    # where `J_f \in \mathbb{R}^{n \times n}` Jacobian of the original system.
    # The initial conditions are set according to [Li2023].

    def lyapunov_source(t: float, y: Array) -> Array:
        return np.hstack([
            m.source(t, y[:n]),
            (source_jac(t, y[:n]) @ y[n:].reshape((n, n))).flatten(),
        ])

    def lyapunov_source_jac(t: float, y: Array) -> Array:
        result = np.empty((n * (n + 1), n * (n + 1)), dtype=y.dtype)
        result[:n, :n] = source_jac(t, y[:n])
        np.fill_diagonal(result[n:, n:], result[:n, :n].flatten())

        return result

    # FIXME: not clear if the initial conditions here make sense for all cases.
    # Might be better to allow the user to overwrite it?
    y0 = tuple([
        np.hstack([m.y0[i], np.full(n, i == 0, dtype=dtype).flatten()])
        for i, yi0 in enumerate(m.y0)
    ])
    ds = tuple([m.ds[i] for i in range(n) for _ in range(n)])

    m_lyap = replace(
        m,
        ds=(*m.ds, *ds),
        source=lyapunov_source,  # type: ignore[arg-type]
        y0=y0,
    )

    if hasattr(m, "source_jac"):
        m_lyap = replace(m_lyap, source_jac=lyapunov_source_jac)  # type: ignore[call-arg]

    # }}}

    # {{{ evolve

    from pycaputo.events import StepAccepted

    tstart = m.control.tstart
    tfinal = tstart

    growth = np.zeros_like(m.y0[0])
    for event in evolve(m_lyap):
        if isinstance(event, StepAccepted):
            tfinal = event.t

    # }}}

    return growth / (tfinal - tstart)


# }}}
