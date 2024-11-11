# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo import fracevolve, fracplot
from pycaputo.integrate_fire import lif

# setup system (parameters from Table 1 in [Teka2017])
alpha = 0.2
param = lif.LIFDim(
    current=2000.0, C=500.0, gl=25.0, e_leak=-70.0, v_reset=-70.0, v_peak=-50.0
)
model = lif.LIFModel(param.nondim(alpha, V_ref=1.0))
y0 = np.array([model.param.v_reset])

print(f"alpha {alpha} y0 {y0}")
print(model.param)

# setup controller
from pycaputo.controller import make_jannelli_controller

dt = 5.0e-3
control = make_jannelli_controller(
    tstart=0.0,
    tfinal=100.0,
    dtmin=dt,
    chimin=0.05,
    chimax=0.1,
)
print(control)

# setup stepper
stepper = lif.CaputoLeakyIntegrateFireL1Method(
    derivative_order=(alpha,),
    control=control,
    y0=(y0,),
    source=model,
)

solution = fracevolve(stepper, dtinit=dt)
fracplot(solution, "gallery-lif")
