# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
This attempts to reproduce Figure 6 from [Jannelli2020]_ using the fractional
van der Pol oscillator. The setup is not exactly the same as the error indicator
is changed.

This example uses the PECE method to solve the equation, while [Jannelli2020]_
uses the implicit trapezoidal PI method.

.. [Jannelli2020] A. Jannelli,
    *A Novel Adaptive Procedure for Solving Fractional Differential Equations*,
    Journal of Computational Science, Vol. 47, pp. 101220--101220, 2020,
    `DOI <https://doi.org/10.1016/j.jocs.2020.101220>`__.
"""

from __future__ import annotations

import numpy as np

from pycaputo.logging import get_logger
from pycaputo.typing import Array

log = get_logger("tutorial")

# {{{ van der Pol oscillator


# [tutorial-func-start]
def van_der_pol(t: float, y: Array, *, mu: float = 4.0) -> Array:
    return np.array([y[1], mu * (1.0 - y[0] ** 2) * y[1] - y[0]])


def van_der_pol_jac(t: float, y: Array, *, mu: float = 4.0) -> Array:
    return np.array([
        [0.0, 1.0],
        [-mu * 2.0 * y[0] * y[1] - 1.0, mu * (1.0 - y[0] ** 2)],
    ])
    # [tutorial-func-end]


# }}}


# {{{ set up problem

# [tutorial-controller-start]
from pycaputo.controller import make_jannelli_controller
from pycaputo.derivatives import CaputoDerivative as D
from pycaputo.fode import caputo

tstart, tfinal = 0.0, 4.0
c = make_jannelli_controller(
    tstart=tstart,
    tfinal=tfinal,
    nsteps=None,
    dtmin=1.0e-3,
    sigma=0.5,
    rho=1.5,
    chimin=0.03,
    chimax=0.3,
    abstol=1.0e-12,
)
# [tutorial-controller-end]

# [tutorial-method-start]
alpha = 0.8
y0 = np.array([1.0, 0.0])

m = caputo.PECE(
    ds=(D(alpha), D(alpha)),
    control=c,
    source=van_der_pol,
    y0=(y0,),
    corrector_iterations=2,
)
# [tutorial-method-end]

# [tutorial-estimate-start]
from pycaputo.controller import estimate_initial_time_step

dtinit = 1.0e-1
dtest = estimate_initial_time_step(
    tstart,
    y0,
    m.source,
    m.smallest_derivative_order,
    trunc=m.order,
)
# [tutorial-estimate-end]
log.info("Initial time step %.8e estimate %.8e", dtinit, dtest)

# }}}


# {{{ evolve


# [tutorial-evolve-start]
from pycaputo.events import StepAccepted, StepRejected
from pycaputo.stepping import evolve

ts = []
ys = []

truncs = []
eests = []
qs = []

for event in evolve(m, dtinit=dtinit):
    if isinstance(event, StepAccepted):
        ts.append(event.t)
        ys.append(event.y)

        truncs.append(event.trunc)
        eests.append(event.eest)
        qs.append(event.q)
    elif isinstance(event, StepRejected):
        pass
    else:
        raise RuntimeError(event)

    log.info(
        "%s[%06d] t = %.5e dt = %.5e (eest %+.5e q %.5e) energy %.5e",
        "[green][A][/]" if isinstance(event, StepAccepted) else "[red][R][/]",
        event.iteration,
        event.t,
        event.dt,
        event.eest,
        event.q,
        np.linalg.norm(event.y),
    )
    # [tutorial-evolve-end]

# }}}


# {{{ plot

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    log.warning("'matplotlib' is not available.")
    raise SystemExit(0) from exc

from pycaputo import _get_default_dark  # noqa: PLC2701
from pycaputo.utils import figure, set_recommended_matplotlib

for dark, suffix in _get_default_dark():
    set_recommended_matplotlib(dark=dark)

    t = np.array(ts)
    y = np.array(ys).T
    trunc = np.array(truncs).T

    with figure(
        f"tutorial-van-der-pol-adaptive-solution{suffix}", nrows=2, figsize=(8, 8)
    ) as fig:
        ax = fig.axes

        ax[0].plot(t, y[1], "o-", ms=3, fillstyle="none", label="$y$")
        ax[0].plot(t, y[0], "o-", ms=3, fillstyle="none", label="$x$")
        ax[0].legend(loc="lower left", bbox_to_anchor=(0.5, 1.0), ncol=2, mode="expand")

        ax[1].semilogy(t[:-1], np.diff(t), "o-", fillstyle="none", ms=3)
        ax[1].set_xlabel("$t$")
        ax[1].set_ylabel(r"$\Delta t$")
        ax[1].set_ylim([c.dtmin, dtinit])

    with figure(
        f"tutorial-van-der-pol-adaptive-eest{suffix}", nrows=2, figsize=(8, 8)
    ) as fig:
        ax = fig.axes

        ax[0].plot(t, trunc[1], "o-", ms=3, fillstyle="none", label=r"$\tau_y$")
        ax[0].plot(t, trunc[0], "o-", ms=3, fillstyle="none", label=r"$\tau_x$")
        ax[0].legend(loc="lower left", bbox_to_anchor=(0.5, 1.0), ncol=2, mode="expand")

        ax[1].plot(t, eests, "-")
        ax[1].axhline(1.0, ls="--", color="k")
        ax[1].axhline(0.0, ls="--", color="k")
        ax[1].set_xlabel("$t$")
        ax[1].set_ylabel(r"$E_{\text{est}}$")

# }}}
