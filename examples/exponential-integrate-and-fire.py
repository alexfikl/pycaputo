# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

# This model is taken from [Naud2008] with the same parameters
#
# .. [Naud2008] R. Naud, N. Marcille, C. Clopath, W. Gerstner,
#       *Firing Patterns in the Adaptive Exponential Integrate-and-Fire Model*,
#       Biological Cybernetics, Vol. 99, pp. 335--347, 2008,
#       `DOI <https://doi.org/10.1007/s00422-008-0264-7>`__.

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import numpy as np

from pycaputo.fode import CaputoIntegrateFireL1Method, FixedTimeSpan
from pycaputo.logging import get_logger
from pycaputo.utils import Array, dc_stringify

logger = get_logger("e-i-f")


# {{{ Adaptive Exponential Integrate-and-Fire (AdEx) model from [Naud2008]


class AdExDim(NamedTuple):
    """A dimensional set of parameters for the model from [Naud2008]_."""

    #: Total capacitance :math:`C` (in picofarad *pF*).
    c: float
    #: Added current :math:`I` (in picoamperes *pA*).
    current: float
    #: Total leak conductance :math:`g_L` (in nanosiemens *nS*).
    gl: float
    #: Effective rest potential :math:`E_L` (in microvolts *mV*).
    el: float
    #: Threshold slope factor :math:`\Delta_T` (in microvolts *mV*).
    delta_t: float
    #: Effective threshold potential :math:`V_T` (in microvolts *mV*).
    vt: float

    #: Time constant :math:`\tau_w` (in microseconds *ms*).
    tau_w: float
    #: Conductance :math:`a` (in nanosiemens *nS*).
    a: float

    #: Peak potential :math:`V_{peak}` (in microvolts *mV*).
    v_peak: float
    #: Reset potential :math:`V_r` (in microvolts *mV*).
    v_reset: float
    #: Adaptation current reset offset :math:`b` (in picoamperes *pA*).
    w_b: float

    def __str__(self) -> str:
        return dc_stringify(
            {
                "C      (total capacitance / pF)": self.c,
                "I      (current / pA)": self.current,
                "g_L    (total leak conductance / nS)": self.gl,
                "E_L    (effective rest potential / mV)": self.el,
                "delta_T(threshold slope factor / mV)": self.delta_t,
                "V_T    (effective threshold potential / mV)": self.vt,
                "tau_w  (time scale ratio / ms)": self.tau_w,
                "a      (conductance / nS)": self.a,
                "V_peak (peak potential / mV)": self.v_peak,
                "V_r    (reset potential / mV)": self.v_reset,
                "b      (adaptation current offset / pA)": self.w_b,
            }
        )

    def nondimensional(self) -> AdEx:
        return AdEx(
            t=self.gl / self.c,
            current=self.current / (self.delta_t * self.gl),
            el=(self.el - self.vt) / self.delta_t,
            tau_w=self.c * self.tau_w / self.gl,
            a=self.a / self.gl,
            v_peak=(self.v_peak - self.vt) / self.delta_t,
            v_reset=(self.v_reset - self.vt) / self.delta_t,
            w_b=self.w_b / (self.delta_t * self.gl),
        )


class AdEx(NamedTuple):
    """A version of :class:`AdExDim` that has been non-dimensionalized."""

    #: Time scale.
    t: float

    #: Added current :math:`I`.
    current: float
    #: Effective rest potential :math:`E_L`.
    el: float

    #: Time constant :math:`\tau_w`.
    tau_w: float
    #: Conductance :math:`a`.
    a: float

    #: Peak potential :math:`V_{peak}`.
    v_peak: float
    # Reset potential :math:`V_r`.
    v_reset: float
    #: Adaptation current reset offset :math:`b`.
    w_b: float

    def __str__(self) -> str:
        return dc_stringify(
            {
                "t      (time scale)": self.t,
                "I      (current)": self.current,
                "E_L    (effective rest potential)": self.el,
                "tau_w  (time scale ratio)": self.tau_w,
                "a      (conductance)": self.a,
                "V_peak (peak potential)": self.v_peak,
                "V_r    (reset potential)": self.v_reset,
                "b      (adaptation current offset)": self.w_b,
            }
        )


def ad_ex(t: float, y: Array, *, p: AdEx) -> Array:
    V, w = y
    I, el, tau_w, a, *_ = p  # noqa: E741

    return np.array(
        [
            I - (V - el) + np.exp(V) - w,
            (a * (V - el) - w) / tau_w,
        ]
    )


def ad_ex_jac(t: float, y: Array, *, p: AdEx) -> Array:
    V, w = y
    I, el, tau_w, a, *_ = p  # noqa: E741

    # J_{ij} = d f_i / d y_j
    return np.array(
        [
            [-1.0 + np.exp(V), -1.0],
            [a / tau_w, -1.0 / tau_w],
        ]
    )


def ad_ex_condition(t: float, y: Array, *, p: AdEx) -> float:
    V, _ = y
    return float(V - p.v_peak)


def ad_ex_reset(t: float, y: Array, *, p: AdEx) -> Array:
    V, w = y

    return np.array([p.v_reset, w + p.w_b])


# }}}


# {{{ implementation


@dataclass(frozen=True)
class AdExMethod(CaputoIntegrateFireL1Method):
    #: Parameters for the AdEx model.
    p: AdEx

    def solve(self, t: float, y0: Array, c: Array, r: Array) -> Array:
        # NOTE: small rename to match write-up
        hV, hw = c
        MV, Mw = r
        I, el, tau_w, a, *_ = self.p  # noqa: E741

        # w coefficients: w = c0 V + c1
        c0 = a * hw / (tau_w + hw)
        c1 = (tau_w * Mw - a * hw * el) / (hw + tau_w)

        # V coefficients: d0 V + d1 = d2 exp(V)
        d0 = 1 + hV * (1 + c0)
        d1 = -hV * (I + el - c1) + MV
        d2 = hV

        # solve
        from scipy.special import lambertw

        dstar = -d2 / d0 * np.exp(d1 / d0)
        Vstar = -d1 / d0 - lambertw(dstar)
        wstar = c0 * Vstar + c1

        return np.array([Vstar, wstar])


# }}}


# {{{ setup

# NOTE: parameters taken from Table 1, Figure 4h in [Naud2008]
pd = AdExDim(
    c=100,
    gl=12,
    el=-60,
    vt=-50,
    delta_t=2,
    current=160,
    tau_w=130,
    a=-11,
    v_peak=0.0,
    v_reset=-48,
    w_b=30,
)
p = pd.nondimensional()

# simulation time span (non-dimensional)
tstart = 0.0
tfinal = 50.0
# simulation time step (non-dimensional)
dt = 1.0e-2
# Fractional derivative order
alpha = (0.9, 0.9)

# initial condition
rng = np.random.default_rng(seed=None)
y0 = np.array(
    [
        rng.uniform(p.v_reset, p.v_peak),
        rng.uniform(0.0, 1.0),
    ]
)

m = AdExMethod(
    p=p,
    derivative_order=alpha,
    tspan=FixedTimeSpan.from_data(dt, tstart=tstart, tfinal=tfinal),
    source=partial(ad_ex, p=p),
    source_jac=partial(ad_ex_jac, p=p),
    y0=(y0,),
    condition=partial(ad_ex_condition, p=p),
    reset=partial(ad_ex_reset, p=p),
)

logger.info("Dimensional variables:\n%s", pd)
logger.info("Non-dimensional variables:\n%s", p)

# }}}


# {{{ evolution

# }}}

# {{{ plotting

# }}}
