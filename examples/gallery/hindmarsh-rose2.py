# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo import fracevolve, fracplots
from pycaputo.fode.gallery import HindmarshRose2

# setup system (parameters from Figure 3b from [Kaslik2017])
alpha = (0.8, 0.8)
y0 = np.array([-1.5, -12])
func = HindmarshRose2(current=3.25, a=1.0, b=3.0, c=1.0, d=5.0)

print(f"alpha {alpha} y0 {y0} parameters {func}")

# setup controller
from pycaputo.controller import make_fixed_controller

dt = 5.0e-3
control = make_fixed_controller(dt, tstart=0.0, tfinal=50.0)
print(control)

# setup stepper
from pycaputo.fode import caputo

stepper = caputo.L1(
    derivative_order=alpha,
    control=control,
    source=func.source,
    source_jac=func.source_jac,
    y0=(y0,),
)

# compute E3 equilibrium point from [Kaslik2017]
r = (func.current + func.c) / func.a
p = (func.b - func.d) / func.a

h = np.polynomial.Polynomial([-r, 0.0, -p, 1.0], symbol="x")
x = h.roots()
x3 = x[np.isreal(x)].real.item()
y3 = func.c - func.d * x3**2
assert x3 > max(0.0, 2.0 / 3.0 * p)

solution = fracevolve(stepper, dtinit=dt)
fracplots(solution, "gallery-hindmarsh-rose2", legend=["$x$", "$y$"])
