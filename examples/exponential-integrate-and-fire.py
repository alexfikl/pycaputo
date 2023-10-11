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

from pycaputo.fode import FixedTimeSpan
from pycaputo.integrate_fire import (
    AdEx,
    AdExIntegrateFireL1Method,
    AdExModel,
    get_ad_ex_parameters,
    get_lambert_time_step,
)
from pycaputo.logging import get_logger

logger = get_logger("e-i-f")


# {{{ setup

# Fractional derivative order
alpha = (0.9, 0.9)
# simulation time span (non-dimensional)
tstart = 0.0
tfinal = 50.0
# simulation time step (non-dimensional)
dt = 1.0e-2

# NOTE: parameters taken from Table 1, Figure 4h in [Naud2008]
pd = get_ad_ex_parameters("Naud4h")
p = AdEx.from_dimensional(pd, alpha)
ad_ex = AdExModel(p)

# initial condition
rng = np.random.default_rng(seed=None)
y0 = np.array(
    [
        rng.uniform(p.v_reset, p.v_peak),
        rng.uniform(0.0, 1.0),
    ]
)

m = AdExIntegrateFireL1Method(
    ad_ex=ad_ex,
    derivative_order=alpha,
    tspan=FixedTimeSpan.from_data(dt, tstart=tstart, tfinal=tfinal),
    source=ad_ex.source,
    source_jac=ad_ex.source_jac,
    y0=(y0,),
    condition=ad_ex.spiked,
    reset=ad_ex.reset,
)

logger.info("Dimensional variables:\n%s", pd)
logger.info("Non-dimensional variables:\n%s", p)
logger.info("Allows Lambert W spikes: %s", get_lambert_time_step(p) is not None)
# }}}


# {{{ evolution

# }}}

# {{{ plotting

# }}}
