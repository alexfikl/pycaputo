# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pycaputo import fracevolve, fracplot
from pycaputo.derivatives import CaputoDerivative as D
from pycaputo.integrate_fire import ad_ex

# setup system (parameters from [Naud2008])
alpha = (0.999, 0.999)
param = ad_ex.get_ad_ex_parameters("Naud4d")
model = ad_ex.AdExModel(param.nondim(alpha))
y0 = np.array([model.param.e_leak, 0.0])

print(f"alpha {alpha} y0 {y0}")
print(model.param)

# setup controller
from pycaputo.controller import make_jannelli_controller

dt = 1.0e-4
control = make_jannelli_controller(
    tstart=0.0,
    tfinal=17.0,
    dtmin=dt,
    chimin=0.001,
    chimax=0.01,
)
print(control)

# setup stepper
stepper = ad_ex.CaputoAdExIntegrateFireL1Model(
    ds=tuple(D(alpha_i) for alpha_i in alpha),
    control=control,
    y0=(y0,),
    source=model,
)

ref = model.param.ref
solution = fracevolve(stepper, dtinit=dt)
solution = replace(solution, t=ref.time(solution.t), y=ref.var(solution.y)[0])
fracplot(solution, "gallery-ad-ex", ylabel="V")
