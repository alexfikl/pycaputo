# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo import fracevolve, fracplot
from pycaputo.fode.gallery import LotkaVolterra2

# setup system (parameters from Figure 6 from [Ahmed2007])
alpha = (0.995, 0.995)
y0 = np.array([0.43, 1.36])
func = LotkaVolterra2(alpha=2.0, beta=1.25, gamma=1.5, delta=0.75, r=0.0)

print(f"alpha {alpha} y0 {y0} parameters {func}")

# setup controller
from pycaputo.controller import make_fixed_controller

dt = 5.0e-3
control = make_fixed_controller(dt, tstart=0.0, tfinal=100.0)
print(control)

# setup stepper
from pycaputo.fode import caputo

stepper = caputo.PECE(
    derivative_order=alpha,
    control=control,
    source=func.source,
    y0=(y0,),
    corrector_iterations=1,
)

solution = fracevolve(stepper, dtinit=dt)
fracplot(solution, "gallery-lotka-volterra2")
