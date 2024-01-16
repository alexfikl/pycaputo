# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

# This model is taken from [Naud2008] with the same parameters
#
# .. [Naud2008] R. Naud, N. Marcille, C. Clopath, W. Gerstner,
#       *Firing Patterns in the Adaptive Exponential Integrate-and-Fire Model*,
#       Biological Cybernetics, Vol. 99, pp. 335--347, 2008,
#       `DOI <https://doi.org/10.1007/s00422-008-0264-7>`__.

from __future__ import annotations

import numpy as np

from pycaputo.integrate_fire import ad_ex
from pycaputo.logging import get_logger

logger = get_logger("ad-ex")


# {{{ setup

# Fractional derivative order
alpha = (0.9, 0.9)
# simulation time span (non-dimensional)
tstart = 0.0
tfinal = 50.0
# simulation time step (non-dimensional)
dt = 1.0e-2

# NOTE: parameters taken from Table 1, Figure 4h in [Naud2008]
pd = ad_ex.get_ad_ex_parameters("Naud4h")
p = pd.nondim(alpha)
model = ad_ex.AdExModel(p)

# initial condition
rng = np.random.default_rng(seed=None)
y0 = np.array([
    rng.uniform(p.v_reset, p.v_peak),
    rng.uniform(0.0, 1.0),
])

from pycaputo.controller import make_jannelli_controller

c = make_jannelli_controller(
    tstart,
    tfinal,
    dtmin=1.0e-3,
    chimin=0.01,
    chimax=1.0,
    abstol=1.0e-4,
)

m = ad_ex.AdExIntegrateFireL1Method(
    derivative_order=alpha,
    control=c,
    y0=(y0,),
    source=model.source,
    model=model,
)

logger.info("Dimensional variables:\n%s", pd)
logger.info("Non-dimensional variables:\n%s", p)

# }}}


# {{{ evolution

# }}}

# {{{ plotting

# }}}
