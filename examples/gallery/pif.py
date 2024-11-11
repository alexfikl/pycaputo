# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo import fracevolve, fracplot
from pycaputo.integrate_fire import pif

# setup system
alpha = 0.85
param = pif.PIFDim(current=160, C=100, v_reset=-48, v_peak=0.0)
model = pif.PIFModel(param.nondim(alpha, V_ref=1.0, I_ref=20.0))
y0 = np.array([(model.param.v_reset + model.param.v_peak) / 2.0])

print(f"alpha {alpha} y0 {y0}")
print(model.param)

# setup controller
from pycaputo.controller import make_jannelli_controller

dt = 5.0e-3
control = make_jannelli_controller(
    tstart=0.0,
    tfinal=32.0,
    dtmin=dt,
    chimin=0.01,
    chimax=0.1,
)
print(control)

# setup stepper
stepper = pif.CaputoPerfectIntegrateFireL1Method(
    derivative_order=(alpha,),
    control=control,
    y0=(y0,),
    source=model,
)

solution = fracevolve(stepper, dtinit=dt)
fracplot(solution, "gallery-pif")
