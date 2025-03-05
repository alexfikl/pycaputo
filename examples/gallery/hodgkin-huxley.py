# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pycaputo import fracevolve, fracplots
from pycaputo.derivatives import CaputoDerivative as D
from pycaputo.fode.gallery import HodgkinHuxley, StandardHodgkinHuxleyParameter

# setup system (parameters from Figure 4 from [Nagy2014])
alpha = (0.9,) * 4
param = StandardHodgkinHuxleyParameter.from_name("NagyFigure4")
func = HodgkinHuxley(param)

V0 = -40.0
y0 = np.array([V0, *param.get_steady_state(V0)])

print(f"alpha {alpha} y0 {y0} parameters {param}")

# setup controller
from pycaputo.controller import make_fixed_controller

dt = 1.0e-3
control = make_fixed_controller(dt, tstart=0.0, tfinal=15.0)
print(control)

# setup stepper
from pycaputo.fode import caputo

stepper = caputo.PECE(
    ds=tuple(D(alpha_i) for alpha_i in alpha),
    control=control,
    source=func.source,
    y0=(y0,),
    corrector_iterations=1,
)

solution = fracevolve(stepper, dtinit=dt)
solution = replace(solution, y=solution.y[1:])
fracplots(solution, "gallery-hodgkin-huxley", legend=["$n$", "$m$", "$h$"])
