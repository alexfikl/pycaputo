# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pycaputo import fracevolve, fracplot
from pycaputo.derivatives import CaputoDerivative as D
from pycaputo.integrate_fire import lif
from pycaputo.typing import Array


class LIFModel(lif.LIFModel):
    def source(self, t: float, y: Array) -> Array:
        if t < 2.5 or t > 7.5:
            current = np.zeros_like(y)
        else:
            current = self.param.current * (1.0 - np.cos(2.0 * np.pi * (t - 2.5)))

        return np.array(current - (y - self.param.e_leak))


# setup system
alpha = 0.85
param = lif.LIFDim(
    current=600.0, C=100.0, gl=25.0, e_leak=-70.0, v_reset=-65.0, v_peak=-50.0
)
model = LIFModel(param.nondim(alpha, V_ref=1.0))
y0 = np.array([model.param.e_leak])

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
stepper = lif.CaputoLeakyIntegrateFireL1Method(
    ds=(D(alpha),),
    control=control,
    y0=(y0,),
    source=model,
)

solution = fracevolve(stepper, dtinit=dt)

ref = model.param.ref
solution = replace(solution, t=ref.time(solution.t), y=ref.var(solution.y))
fracplot(solution, "gallery-lif", ylabel="V")
