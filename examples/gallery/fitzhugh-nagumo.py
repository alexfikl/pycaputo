# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo import fracevolve, fracplots
from pycaputo.derivatives import CaputoDerivative as D
from pycaputo.fode.gallery import FitzHughNagumo

# setup system (parameters from Figure 4d from [Brandibur2018])
alpha = (0.7, 0.8)
y0 = np.array([0.7, 1.775])
func = FitzHughNagumo(current=1.24567, r=0.08, c=0.7, d=0.8)

print(f"alpha {alpha} y0 {y0} parameters {func}")

# setup controller
from pycaputo.controller import make_fixed_controller

dt = 1.0e-1
control = make_fixed_controller(dt, tstart=0.0, tfinal=1000.0)
print(control)

# setup stepper
from pycaputo.fode import caputo

stepper = caputo.L1(
    ds=tuple(D(alpha_i) for alpha_i in alpha),
    control=control,
    source=func.source,
    source_jac=func.source_jac,
    y0=(y0,),
)

solution = fracevolve(stepper, dtinit=dt)
fracplots(solution, "gallery-fitzhugh-nagumo", legend=["$v$", "$w$"])
