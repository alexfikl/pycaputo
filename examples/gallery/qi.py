# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo import fracevolve, fracplot
from pycaputo.derivatives import CaputoDerivative as D
from pycaputo.fode.gallery import Qi

# setup system (parameters from Figure 5 from [Qi2005])
alpha = (0.98, 0.98, 0.98)
y0 = np.array([0.1, 0.2, 0.3])
func = Qi(a=35.0, b=8.0 / 3.0, c=80.0)

print(f"alpha {alpha} y0 {y0} parameters {func}")

# setup controller
from pycaputo.controller import make_fixed_controller

dt = 1.0e-3
control = make_fixed_controller(dt, tstart=0.0, tfinal=50.0)
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
fracplot(solution, "gallery-qi")
