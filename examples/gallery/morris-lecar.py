# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pycaputo import fracevolve, fracplot
from pycaputo.derivatives import CaputoDerivative as D
from pycaputo.fode.gallery import MorrisLecar, StandardMorrisLecarParameter

# setup system (parameters from Section 4.2 and Figure 11 from [Shi2014])
alpha = (0.995, 0.995, 0.995)
param = StandardMorrisLecarParameter(
    C=1.0,
    g_L=0.5,
    g_Ca=1.36,
    g_K=2.0,
    v_L=-0.5,
    v_Ca=1.0,
    v_K=-0.725,
    epsilon=0.003,
    v_0=0.1,
    phi=1.0 / 3.0,
    vinf=(-0.01, 0.15, 0.1, 0.16),
)
y0 = np.array([-0.5, param.m_inf(-0.5), -param.v_0])
func = MorrisLecar(param)

print(f"alpha {alpha} y0 {y0} parameters {param}")

# setup controller
from pycaputo.controller import make_fixed_controller

dt = 7.5e-2
control = make_fixed_controller(dt, tstart=0.0, tfinal=1000.0)
print(control)

# setup stepper
from pycaputo.fode import caputo

stepper = caputo.L1(
    ds=tuple(D(alpha_i) for alpha_i in alpha),
    control=control,
    source=func.source,
    source_jac=func.source_jac,
    y0=(y0,),
)

solution = fracevolve(stepper, dtinit=dt)
solution = replace(solution, y=solution.y[0])
fracplot(solution, "gallery-morris-lecar", ylabel="V")
