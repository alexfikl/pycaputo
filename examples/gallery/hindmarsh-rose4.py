# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo import fracevolve, fracplots
from pycaputo.derivatives import CaputoDerivative as D
from pycaputo.fode.gallery import HindmarshRose4

# setup system (parameters from Figure 1 from [Giresse2019])
# NOTE: initial conditions are guessed from Figure 1 and Section 4.3
alpha = (0.98, 0.98, 0.98, 0.98)
y0 = np.array([0.3, 0.3, 3.0, 0.3])

a, b, c, d = 1.0, 3.0, 1.0, 5.0
func = HindmarshRose4(
    current=3.2,
    a=a,
    b=b,
    c=c,
    d=d,
    e=1 / 80,
    f=0.88,
    g=0.9,
    h=0.002,
    epsilon=0.005,
    s=4.0,
    p=1.0,
    x0=HindmarshRose4.get_resting_potential(a, b, c, d),
)

print(f"alpha {alpha} y0 {y0} parameters {func}")

# setup controller
from pycaputo.controller import make_fixed_controller

dt = 2.0e-2
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
fracplots(
    solution, "gallery-hindmarsh-rose4", legend=["$x_1$", "$x_2$", "$x_3$", "$x_4$"]
)
