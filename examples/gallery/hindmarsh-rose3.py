# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pycaputo import fracevolve, fracplots
from pycaputo.derivatives import CaputoDerivative as D
from pycaputo.fode.gallery import HindmarshRose3

# parameters from Figure 5a from [Kaslik2017]
alpha = (0.8, 0.8, 0.8)
current = 3.25
a = 1.0
b = 3.0
c = 1.0
d = 5.0
s = 4.0

# set up system
x0 = HindmarshRose3.get_resting_potential(a, b, c, d)
y0 = np.array([x0, c - d * x0**2, 0.0])
func = HindmarshRose3(current=current, a=a, b=b, c=c, d=d, epsilon=0.005, s=s, x0=x0)
print(f"alpha {alpha} y0 {y0} parameters {func}")

# setup controller
from pycaputo.controller import make_fixed_controller

dt = 5.0e-2
control = make_fixed_controller(dt, tstart=0.0, tfinal=2000.0)
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
solution = replace(solution, y=solution.y[0:1, :])
fracplots(solution, "gallery-hindmarsh-rose3", legend=["$x$"])
