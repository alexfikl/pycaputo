# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pycaputo import fracevolve, fracplot
from pycaputo.derivatives import CaputoDerivative as D
from pycaputo.fode.gallery import FitzHughRinzel, FitzHughRinzelParameter

# setup system (parameters from Figure 3g from [Mondal2019])
alpha = (0.725, 0.725, 0.725)
y0 = np.array([0.1, 0.1, 0.1])
param = FitzHughRinzelParameter.from_name("MondalSetII")
func = FitzHughRinzel(param)

print(f"alpha {alpha} y0 {y0} parameters {param}")

# setup controller
from pycaputo.controller import make_fixed_controller

dt = 1.0e-1
control = make_fixed_controller(dt, tstart=0.0, tfinal=1500.0)
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
solution = replace(solution, y=solution.y[0])
fracplot(solution, "gallery-fitzhugh-rinzel", ylabel="V")
