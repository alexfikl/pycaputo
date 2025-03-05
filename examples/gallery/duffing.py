# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo import fracevolve, fracplot
from pycaputo.derivatives import CaputoDerivative as D
from pycaputo.fode.gallery import Duffing

# setup system (parameters from Figure 5.29 from [Petras2011])
alpha = (0.9, 0.999)
y0 = np.array([0.21, 0.13])
func = Duffing(alpha=0.15, amplitude=0.3, omega=1.0)

print(f"alpha {alpha} y0 {y0} parameters {func}")

# setup controller
from pycaputo.controller import make_fixed_controller

dt = 5.0e-3
control = make_fixed_controller(dt, tstart=0.0, tfinal=200.0)
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
fracplot(solution, "gallery-duffing")
