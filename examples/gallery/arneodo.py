# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo import fracevolve, fracplot
from pycaputo.derivatives import CaputoDerivative as D
from pycaputo.fode.gallery import Arneodo

# setup system (parameters from Figure 5.43 from [Petras2011])
alpha = (0.97, 0.97, 0.96)
y0 = np.array([-0.2, 0.5, 0.2])
func = Arneodo(beta=(-5.5, 3.5, 0.8, -1.0))

print(f"alpha {alpha} y0 {y0} parameters {func}")

# setup controller
from pycaputo.controller import make_fixed_controller

dt = 5.0e-3
control = make_fixed_controller(dt, tstart=0.0, tfinal=100.0)
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
fracplot(solution, "gallery-arneodo")
