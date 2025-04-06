# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

r"""Solve some example problems using the Caputo-Fabrizio operator from [Atangana2021]_.
Since the textbook does not perform any convergence tests, this is meant to showcase
that the implementation matches.

Note that [Atangana2021]_ does not give many details regarding the simulation.
In particular: the time step :math:`\Delta t` is not provided and the scaling
:math:`M(\alpha)` is not defined.

.. [Atangana2021] A. Atangana, S. I. Araz,
    *New Numerical Scheme With Newton Polynomial - Theory, Methods, and Applications*,
    Elsevier Science & Technology, 2021.
"""

from __future__ import annotations

import csv
import math
import pathlib
from dataclasses import dataclass

import numpy as np

from pycaputo.derivatives import CaputoFabrizioOperator
from pycaputo.logging import get_logger
from pycaputo.typing import Array

log = get_logger("atangana-seda")
dirname = pathlib.Path(__file__).parent


# {{{ right-hand side


@dataclass(frozen=True)
class D(CaputoFabrizioOperator):
    def normalization(self) -> float:
        # NOTE: This is the normalization used in the code snippet from Appendix A
        #   AS_Method_for_Chaotic_with_CF_Fractional.m
        return 1.0 - self.alpha + self.alpha / math.gamma(self.alpha)


def source(t: float, y: Array) -> Array:
    # NOTE: the examples in [Atangana2021] have different signs than below. Using
    # the provided signs, the plots do not seem to match at all and it's unclear
    # where the error is exactly..
    return np.array([
        # Section 5.2, Example 5.1
        -2.0 * t * y[0],
        # Section 5.2, Example 5.2
        np.sin(-5.0 * y[1]),
        # Section 5.2, Example 5.3
        -y[2] - t,
        # Section 5.2, Example 5.4
        -y[3] + 5.0 * t,
    ])


# }}}


# {{{ solve

from pycaputo.controller import make_fixed_controller
from pycaputo.fode import caputo_fabrizio

# NOTE:
#   Example 5.1 and 5.2 go to tfinal=5.0
#   Example 5.3 goes to tfinal=3.0
#   Example 5.4 goes to tfinal=2.0
tfinal = (5.0, 5.0, 3.0, 2.0)
y0 = np.array([1.0, 1.0, 1.0, 0.11])

stepper = caputo_fabrizio.AtanganaSeda3(
    ds=(D(0.9), D(0.99), D(0.78), D(0.82)),
    control=make_fixed_controller(1.0e-2, tstart=0.0, tfinal=max(tfinal)),
    source=source,
    y0=(y0,),
)

from pycaputo.events import StepCompleted
from pycaputo.stepping import evolve

ts = []
ys = []

for event in evolve(stepper):
    assert isinstance(event, StepCompleted)
    ts.append(event.t)
    ys.append(event.y)

    log.info(
        "[%06d] t = %.5e dt = %.5e energy %.5e",
        event.iteration,
        event.t,
        event.dt,
        np.linalg.norm(event.y),
    )

# }}}

# {{{ plot

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    log.warning("'matplotlib' is not available.")
    raise SystemExit(0) from exc

t = np.array(ts)
y = np.array(ys).T

from pycaputo.utils import figure, set_recommended_matplotlib

set_recommended_matplotlib()
for i in range(y.shape[0]):
    with open(
        dirname / "data" / f"atangana-seda-figure5{i + 1}.csv",
        encoding="utf-8",
    ) as fd:
        reader = csv.reader(fd)
        next(reader, None)
        t_ref, y_ref = np.array([(float(row[0]), float(row[1])) for row in reader]).T

    mask = t < tfinal[i]
    with figure(f"caputo-fabrizio-atangana-seda-example{i}") as fig:
        ax = fig.gca()

        ax.plot(t_ref, y_ref, "ko-", ms=3, alpha=0.5, label="Atangana2021")
        ax.plot(t[mask], y[i, mask], label="Us")
        ax.set_xlabel("$t$")
        ax.set_ylabel("$y$")
        ax.set_title(f"Figure 5.{i + 1}")
        ax.legend()

# }}}
