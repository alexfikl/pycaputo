# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

# This attempts to replicate Figure 4 from [Garrappa2023]
#
# .. [Garrappa2023] R. Garrappa, A. Giusti,
#       *A Computational Approach to Exponential-Type Variable-Order Fractional
#       Differential Equations*, Journal of Scientific Computing, Vol. 96, 2023,
#       `DOI <https://doi.org/10.1007/s10915-023-02283-6>`__.

from __future__ import annotations

import numpy as np

from pycaputo.derivatives import VariableExponentialCaputoDerivative as D
from pycaputo.fode import variable_caputo as caputo
from pycaputo.logging import get_logger

log = get_logger("integrate-and-fire")

# {{{ evolve

tstart, tfinal = 0.0, 2.0
func = caputo.Relaxation(d=D(alpha=(0.6, 0.8), c=2.0), omega=4.0, y0=1.0)

from pycaputo.controller import make_fixed_controller

control = make_fixed_controller(2**-7, tstart=tstart, tfinal=tfinal)
stepper = caputo.VariableExponentialBackwardEuler(
    ds=(func.d,),
    control=control,
    source=func.source,
    source_jac=func.source_jac,
    y0=(func(tstart),),
)

from pycaputo.events import StepCompleted
from pycaputo.stepping import evolve

ts = []
ys = []

for event in evolve(stepper):
    assert isinstance(event, StepCompleted)

    ts.append(event.t)
    ys.append(event.y)

    log.info("%s energy %.5e", event, np.linalg.norm(event.y))

# }}}

# {{{ plot

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise SystemExit(0) from exc

from pycaputo.utils import figure, set_recommended_matplotlib

set_recommended_matplotlib()

with figure("variable-order-caputo-relaxation") as fig:
    ax = fig.gca()

    t = np.array(ts)
    for alpha, c, omega in (
        # FIXME: lambda values are not mentioned in the paper. The ones used in
        # Table 1 do not seem to match the plot at all, so this just guesses
        ((0.6, 0.8), 2.0, 4.0),
        ((0.5, 0.9), 1.0, 4.0),
        ((0.9, 0.6), 1.0, 4.0),
    ):
        d = D(alpha=alpha, c=c)
        func = caputo.Relaxation(d=d, y0=1.0, omega=omega)

        y_ref = np.empty_like(t)
        for i in range(t.size):
            y_ref[i] = func(t[i]).squeeze()

        ax.plot(
            t,
            y_ref,
            label=rf"$\alpha_1 = {alpha[0]:.1f}, \alpha_2 = {alpha[1]:.1f}, c = {c:.1f}$",  # noqa: E501
        )

    ax.plot(t, np.array(ys).squeeze(), "k--", label="Numerical")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y$")
    ax.set_xlim([tstart, tfinal])
    ax.set_ylim([0.0, 1.0])
    ax.set_aspect(0.75)
    ax.legend(prop={"size": 15})

# }}}
