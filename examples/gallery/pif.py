# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pycaputo import fracevolve, fracplot
from pycaputo.derivatives import CaputoDerivative as D
from pycaputo.integrate_fire import pif
from pycaputo.typing import Array


class PIFModel(pif.PIFModel):
    def source(self, t: float, y: Array) -> Array:
        if t < 2.5 or t > 7.5:
            return np.zeros_like(y)
        else:
            return np.array([
                self.param.current * (1.0 - np.cos(2.0 * np.pi * (t - 2.5)))
            ])


# setup system
alpha = 0.85
param = pif.PIFDim(current=24, C=100, v_reset=-65, v_peak=-50.0)
model = PIFModel(param.nondim(alpha, V_ref=1.0, I_ref=1.0))
y0 = np.array([model.param.v_reset - 5.0])

print(f"alpha {alpha} y0 {y0}")
print(model.param)

# setup controller
from pycaputo.controller import make_jannelli_controller

dt = 5.0e-4
control = make_jannelli_controller(
    tstart=0.0,
    tfinal=10.0,
    dtmin=dt,
    chimin=0.001,
    chimax=0.01,
)
print(control)

# setup stepper
stepper = pif.CaputoPerfectIntegrateFireL1Method(
    ds=(D(alpha),),
    control=control,
    y0=(y0,),
    source=model,
)

solution = fracevolve(stepper, dtinit=dt)

ref = model.param.ref
solution = replace(solution, t=ref.time(solution.t), y=ref.var(solution.y))
fracplot(solution, "gallery-pif", ylabel="V")
