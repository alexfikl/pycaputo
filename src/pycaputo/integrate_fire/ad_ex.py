# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import NamedTuple, overload

import numpy as np

from pycaputo.history import ProductIntegrationHistory
from pycaputo.integrate_fire.base import (
    AdvanceResult,
    IntegrateFireMethod,
    IntegrateFireModel,
)
from pycaputo.logging import get_logger
from pycaputo.stepping import advance
from pycaputo.typing import Array

log = get_logger(__name__)


# {{{ parameters


class AdExReference(NamedTuple):
    """Reference variables used to non-dimensionalize the AdEx model."""

    alpha: tuple[float, float]
    """Fractional order used to non-dimensionalize."""

    T_ref: float
    """Time scale (in milliseconds: *ms*)."""
    V_off: float
    """Voltage offset (in millivolts: *mV*)."""
    V_ref: float
    """Voltage scale (in millivolts: *mV*)."""
    w_ref: float
    """Adaptation variable scale (in picoamperes: *pA*)."""
    I_ref: float
    """Current scale (in picoamperes: *pA*)."""

    @overload
    def time(self, t: float) -> float: ...

    @overload
    def time(self, t: Array) -> Array: ...

    def time(self, t: float | Array) -> float | Array:
        """Add dimensions to the non-dimensional time *t*."""
        return self.T_ref * t

    def var(self, y: Array) -> Array:
        """Add dimensions to the non-dimensional :math:`(V, w)`."""
        return np.array([self.V_ref * y[0] + self.V_off, self.w_ref * y[1]])


class AdExDim(NamedTuple):
    """Dimensional parameters for the Adaptive Exponential Integrate-and-Fire
    (AdEx) model from [Naud2008]_."""

    current: float
    """Added current :math:`I` (in picoamperes *pA*)."""
    C: float
    """Total capacitance :math:`C` (in picofarad per ms *pF / ms^(alpha - 1)*)."""
    gl: float
    """Total leak conductance :math:`g_L` (in nanosiemens *nS*)."""
    e_leak: float
    """Effective rest potential :math:`E_L` (in microvolts *mV*)."""
    delta_t: float
    r"""Threshold slope factor :math:`\Delta_T` (in microvolts *mV*)."""
    vt: float
    """Effective threshold potential :math:`V_T` (in microvolts *mV*)."""

    tau_w: float
    r"""Time constant :math:`\tau_w` (in microseconds *ms^alpha*)."""
    a: float
    """Conductance :math:`a` (in nanosiemens *nS*)."""

    v_peak: float
    """Peak potential :math:`V_{peak}` (in microvolts *mV*)."""
    v_reset: float
    """Reset potential :math:`V_r` (in microvolts *mV*)."""
    b: float
    """Adaptation current reset offset :math:`b` (in picoamperes *pA*)."""

    def __str__(self) -> str:
        from pycaputo.utils import dc_stringify

        return dc_stringify(
            {
                "I      (current / pA)": self.current,
                "C      (total capacitance / pF/ms^alpha)": self.C,
                "g_L    (total leak conductance / nS)": self.gl,
                "E_L    (effective rest potential / mV)": self.e_leak,
                "delta_T(threshold slope factor / mV)": self.delta_t,
                "V_T    (effective threshold potential / mV)": self.vt,
                "tau_w  (time scale ratio / ms^alpha)": self.tau_w,
                "a      (conductance / nS)": self.a,
                "V_peak (peak potential / mV)": self.v_peak,
                "V_r    (reset potential / mV)": self.v_reset,
                "b      (adaptation current offset / pA)": self.b,
            },
            header=("model", type(self).__name__),
        )

    def ref(self, alpha: float | tuple[float, float]) -> AdExReference:
        r"""Construct reference variables used in non-dimensionalizating the AdEx model.

        The non-dimensionalization is performed using the following rescaling

        .. math::

            \hat{t} = \sqrt[\alpha_1]{\frac{g_L}{C}} t,
            \qquad
            \hat{V} = \frac{V - V_T}{\Delta_T},
            \qquad
            \hat{w} = \frac{w}{g_L \Delta_T}.

        :arg alpha: the order of the fractional derivatives for two model
            components :math:`(V, w)`. These can be the same if the two variables
            use the same order.
        """
        alpha = (alpha, alpha) if isinstance(alpha, float) else alpha
        if not len(alpha) == 2:
            raise ValueError(f"Only 2 orders 'alpha' are required: given {len(alpha)}")

        return AdExReference(
            alpha=alpha,
            T_ref=(self.C / self.gl) ** (1 / alpha[0]),
            V_off=self.vt,
            V_ref=self.delta_t,
            w_ref=self.gl * self.delta_t,
            I_ref=self.gl * self.delta_t,
        )

    def nondim(self, alpha: float | tuple[float, float]) -> AdEx:
        r"""Construct non-dimensional parameters for the AdEx model.

        This uses the reference variables from :meth:`ref` to reduce the parameter
        space to only the non-dimensional threshold values
        :math:`(\hat{V}_{peak}, \hat{V}_r, \hat{b})` and
        :math:`(\hat{I}, \hat{E}_L, \hat{\tau}_w, \hat{a})`.

        """
        ref = self.ref(alpha)
        return AdEx(
            ref=ref,
            current=self.current / ref.I_ref,
            e_leak=(self.e_leak - ref.V_off) / ref.V_ref,
            tau_w=self.tau_w / ref.T_ref ** ref.alpha[1],
            a=self.a / self.gl,
            v_peak=(self.v_peak - ref.V_off) / ref.V_ref,
            v_reset=(self.v_reset - ref.V_off) / ref.V_ref,
            b=self.b / ref.w_ref,
        )


class AdEx(NamedTuple):
    """Non-dimensional parameters for the AdEx model (see :class:`AdExDim`)."""

    ref: AdExReference
    """Reference values used in non-dimensionalization."""

    current: float
    """Added current :math:`I`."""
    e_leak: float
    """Effective rest potential :math:`E_L`."""

    tau_w: float
    r"""Time constant :math:`\tau_w`."""
    a: float
    """Conductance :math:`a`."""

    v_peak: float
    """Peak potential :math:`V_{peak}`."""
    v_reset: float
    """Reset potential :math:`V_r`."""
    b: float
    """Adaptation current reset offset :math:`b`."""

    def __str__(self) -> str:
        from pycaputo.utils import dc_stringify

        return dc_stringify(
            {
                "I      (current)": self.current,
                "E_L    (effective rest potential)": self.e_leak,
                "tau_w  (time scale ratio)": self.tau_w,
                "a      (conductance)": self.a,
                "V_peak (peak potential)": self.v_peak,
                "V_r    (reset potential)": self.v_reset,
                "b      (adaptation current offset)": self.b,
            },
            header=("model", type(self).__name__),
        )


# Parameter values for the integer-order system from [Naud2008]_ Table 1.
AD_EX_PARAMS: dict[str, AdExDim] = {
    "Naud4a": AdExDim(
        C=200,
        gl=10,
        e_leak=-70,
        vt=-50,
        delta_t=2,
        a=2,
        tau_w=30,
        b=0,
        v_reset=-58,
        v_peak=0.0,
        current=500,
    ),
    "Naud4b": AdExDim(
        C=200,
        gl=12,
        e_leak=-70,
        vt=-50,
        delta_t=2,
        a=2,
        tau_w=300,
        b=60,
        v_reset=-58,
        v_peak=0.0,
        current=500,
    ),
    "Naud4c": AdExDim(
        C=130,
        gl=18,
        e_leak=-58,
        vt=-50,
        delta_t=2,
        a=4,
        tau_w=150,
        b=120,
        v_reset=-50,
        v_peak=0.0,
        current=400,
    ),
    "Naud4d": AdExDim(
        C=200,
        gl=10,
        e_leak=-58,
        vt=-50,
        delta_t=2,
        a=2,
        tau_w=120,
        b=100,
        v_reset=-46,
        v_peak=0.0,
        current=210,
    ),
    "Naud4e": AdExDim(
        C=200,
        gl=12,
        e_leak=-70,
        vt=-50,
        delta_t=2,
        a=-10,
        tau_w=300,
        b=0,
        v_reset=-58,
        v_peak=0.0,
        current=300,
    ),
    "Naud4f": AdExDim(
        C=200,
        gl=12,
        e_leak=-70,
        vt=-50,
        delta_t=2,
        a=-6,
        tau_w=300,
        b=0,
        v_reset=-58,
        v_peak=0.0,
        current=110,
    ),
    "Naud4g": AdExDim(
        C=100,
        gl=10,
        e_leak=-65,
        vt=-50,
        delta_t=2,
        a=-10,
        tau_w=90,
        b=30,
        v_reset=-47,
        v_peak=0.0,
        current=350,
    ),
    "Naud4h": AdExDim(
        C=100,
        gl=12,
        e_leak=-60,
        vt=-50,
        delta_t=2,
        a=-11,
        tau_w=130,
        b=30,
        v_reset=-48,
        v_peak=0.0,
        current=160,
    ),
}


@overload
def get_ad_ex_parameters(name: str, alpha: float | tuple[float, float]) -> AdEx: ...


@overload
def get_ad_ex_parameters(name: str, alpha: None = None) -> AdExDim: ...


def get_ad_ex_parameters(
    name: str, alpha: float | tuple[float, float] | None = None
) -> AdEx | AdExDim:
    """Get a set of known parameters for the AdEx model.

    The following parameters are available:

    * From [Naud2008]_ Table 1: sets are named ``NaudXX``, where the suffix
      corresponds to the figure number, e.g. ``Naud4a`` to ``Naud4h``.

    :arg name: a parameter set name.
    :arg alpha: the order of the fractional derivative.

    :returns: a set of non-dimensional parameters if *alpha* is given, otherwise
        the dimensional parameters are returned (see :class:`AdExDim`).
    """
    if name not in AD_EX_PARAMS:
        raise ValueError(
            "Unknown parameter set '{}'. Known values are '{}'".format(
                name, "', '".join(AD_EX_PARAMS)
            )
        )

    if alpha is None:
        return AD_EX_PARAMS[name]

    return AD_EX_PARAMS[name].nondim(alpha)


def get_ad_ex_parameters_latex() -> str:
    from rich.box import Box
    from rich.table import Table

    box = Box("    \n  &\\\n    \n  &\\\n    \n    \n  &\\\n    \n")
    t = Table(
        *[
            "Name",
            "$C$",
            "$E_L$",
            "$V_T$",
            r"$\Delta_T$",
            "$a$",
            r"$\tau_w$",
            "$b$",
            "$V_r$",
            "$I$",
        ],
        box=box,
        header_style=None,
    )

    for name, ad_ex in AD_EX_PARAMS.items():
        t.add_row(*[
            name,
            f"{ad_ex.C:.3f}",
            f"{ad_ex.gl:.3f}",
            f"{ad_ex.e_leak:.3f}",
            f"{ad_ex.vt:.3f}",
            f"{ad_ex.delta_t:.3f}",
            f"{ad_ex.a:.3f}",
            f"{ad_ex.tau_w:.3f}",
            f"{ad_ex.b:.3f}",
            f"{ad_ex.v_reset:.3f}",
            f"{ad_ex.current:.3f}",
        ])

    from rich.console import Console

    output = io.StringIO()
    c = Console(file=output)

    c.print(r"\begin{tabular}{llllll}\toprule")
    c.print(t)
    c.print("")

    return output.getvalue()


# }}}


# {{{ AdEx model


@dataclass(frozen=True)
class AdExModel(IntegrateFireModel):
    r"""Functionals for the AdEx model of [Naud2008]_ with parameters :class:`AdEx`.

    .. math::

        \left\{
        \begin{aligned}
        \frac{\mathrm{d}^{\alpha_1} V}{\mathrm{d} t^{\alpha_1}} & =
        I(t) - (V - E_L) + \exp(V) - w, \\
        \tau_w \frac{\mathrm{d}^{\alpha_2} w}{\mathrm{d} t^{\alpha_2}} & =
            a (V - E_L) - w
        \end{aligned}
        \right.

    where :math:`I` is taken to be constant. The reset condition is given by

    .. math::

        \text{if } V > V_{peak} \qquad \text{then} \qquad
        \begin{cases}
        V \gets V_r, \\
        w \gets w + b.
        \end{cases}
    """

    param: AdEx
    """Non-dimensional parameters for the model."""

    if __debug__:

        def __post_init__(self) -> None:
            if not isinstance(self.param, AdEx):
                raise TypeError(
                    f"Invalid parameter type: '{type(self.param).__name__}'"
                )

    def source(self, t: float, y: Array) -> Array:
        """Evaluate right-hand side of the AdEx model."""
        V, w = y
        p = self.param

        return np.array([
            p.current - (V - p.e_leak) + np.exp(V) - w,
            (p.a * (V - p.e_leak) - w) / p.tau_w,
        ])

    def source_jac(self, t: float, y: Array) -> Array:
        """Evaluate the Jacobian of the right-hand side of the AdEx model."""
        V, _ = y
        p = self.param

        # J_{ij} = d f_i / d y_j
        return np.array([
            [-1.0 + np.exp(V), -1.0],
            [p.a / p.tau_w, -1.0 / p.tau_w],
        ])

    def spiked(self, t: float, y: Array) -> float:
        """Compute a delta from the peak threshold :math:`V_{peak}`.

        :returns: a delta of :math:`V - V_{peak}` that can be used to determine
            if the neuron spiked.
        """
        V, _ = y
        return float(V - self.param.v_peak)

    def reset(self, t: float, y: Array) -> Array:
        r"""Evaluate the reset values at :math:`(t, V, w)`.

        This function assumes that the :math:`V \ge V_{peak}`, so that a spike-reset
        is valid. If this is not the case, the reset is obviously not valid.
        """
        # TODO: should this be a proper check?
        assert self.spiked(t, y) > -10.0 * np.finfo(y.dtype).eps

        _, w = y
        return np.array([self.param.v_reset, w + self.param.b])


# }}}


# {{{ Lambert W solver


def _evaluate_lambert_coefficients(
    ad_ex: AdExModel, t: float, y: Array, h: Array, r: Array
) -> tuple[float, float, float, float, float]:
    # NOTE: small rename to match write-up
    hV, hw = h
    rV, rw = r
    p = ad_ex.param

    # w coefficients: w = c0 V + c1
    c0 = p.a * hw / (p.tau_w + hw)
    c1 = (p.tau_w * rw - p.a * hw * p.e_leak) / (hw + p.tau_w)

    # V coefficients: d0 V + d1 = d2 exp(V)
    dummy = np.zeros_like(y)
    I, _ = ad_ex.source(t, dummy) - 1.0  # noqa: E741
    d0 = 1 + hV * (1 + c0)
    d1 = -hV * (I - c1) - rV
    d2 = hV

    return d0, d1, d2, c0, c1


def _evaluate_lambert_coefficients_time(
    ad_ex: AdExModel, t: float, tprev: float, yprev: Array, r: Array
) -> tuple[float, float, float, float, float]:
    from math import gamma

    alpha = ad_ex.param.ref.alpha
    h = np.array([
        gamma(2 - alpha[0]) * (t - tprev) ** alpha[0],
        gamma(2 - alpha[1]) * (t - tprev) ** alpha[1],
    ])

    return _evaluate_lambert_coefficients(ad_ex, t, yprev, h, yprev - h * r)


def find_maximum_time_step_lambert(
    ad_ex: AdExModel, t: float, tprev: float, yprev: Array, r: Array
) -> float:
    """Find a maximum time step such that the Lambert W function is real.

    This function looks at the argument of the Lambert W function for the AdEx
    model and ensures that it is :math:`> -1/e`. Note that this is not done
    exactly, as we assume that the memory terms *r* are fixed.

    :arg t: initial guess for the spike time.
    :arg tprev: previous time step that was successful, i.e. that resulted in a
        real valued membrane potential.
    :arg yprev: solution at the previous time step.
    :arg r: memory terms, considered fixed for this solution.
    """
    assert t > tprev

    def func(tspike: float) -> float:
        d0, d1, d2, *_ = _evaluate_lambert_coefficients_time(
            ad_ex, tspike, tprev, yprev, r
        )
        return float(d2 / d0 * np.exp(-d1 / d0 + 1.0) - 1.0)

    import scipy.optimize as so

    result = so.root_scalar(f=func, x0=t, bracket=[tprev, t])
    return float(result.root) - tprev


# }}}


# {{{ AdExIntegrateFireL1Method


@dataclass(frozen=True)
class CaputoAdExIntegrateFireL1Model(IntegrateFireMethod[AdExModel]):
    r"""Implementation of the L1 method for the Adaptive Exponential
    Integrate-and-Fire model.

    The model is described by :class:`AdExModel` with parameters :class:`AdEx`.
    """

    @property
    def order(self) -> float:
        # NOTE: this is currently not tested, but it should match the PIF/LIF
        # estimates for the time step even though it does not do the interpolation
        return 1.0

    def solve(self, t: float, y: Array, h: Array, r: Array) -> Array:
        """Solve the implicit equation for the AdEx model.

        In this case, since the right-hand side is nonlinear, but we can solve
        the implicit equation using the Lambert W function, a special function
        that is a solution to

        .. math::

            z = w e^w.
        """
        from scipy.special import lambertw

        d0, d1, d2, c0, c1 = _evaluate_lambert_coefficients(self.source, t, y, h, r)
        dstar = -d2 / d0 * np.exp(-d1 / d0)
        Vstar = -d1 / d0 - lambertw(dstar, tol=1.0e-12)
        Vstar = np.real_if_close(Vstar, tol=100)
        wstar = c0 * Vstar + c1

        ystar = np.array([Vstar, wstar])
        assert np.linalg.norm(ystar - h * self.source(t, ystar) - r) < 1.0e-8

        return ystar


def _ad_ex_spike_reset(
    ad_ex: AdExModel, t: float, tprev: float, yprev: Array, r: Array
) -> tuple[Array, Array]:
    *_, c0, c1 = _evaluate_lambert_coefficients_time(ad_ex, t, tprev, yprev, r)

    p = ad_ex.param
    yprev = np.array([p.v_peak, c0 * p.v_peak + c1], dtype=yprev.dtype)
    ynext = np.array([p.v_reset, yprev[1] + p.b], dtype=yprev.dtype)

    return yprev, ynext


@advance.register(CaputoAdExIntegrateFireL1Model)
def _advance_caputo_ad_ex_l1(  # type: ignore[misc]
    m: CaputoAdExIntegrateFireL1Model,
    history: ProductIntegrationHistory,
    y: Array,
    dt: float,
) -> AdvanceResult:
    from pycaputo.controller import AdaptiveController
    from pycaputo.integrate_fire.base import advance_caputo_integrate_fire_l1
    from pycaputo.integrate_fire.spikes import estimate_spike_time_exp

    tprev = history.current_time
    t = tprev + dt
    result, r = advance_caputo_integrate_fire_l1(m, history, y, dt)

    model = m.source
    p = model.param
    if np.any(np.iscomplex(result.y)):
        # NOTE: if the result is complex, it means the Lambert W function is out
        # of range. We try here to find the maximum time step that would put it
        # back in range and use that to mark the spike.
        try:
            dts = find_maximum_time_step_lambert(model, t, tprev, y, r)
            trunc = np.zeros_like(y)
            spiked = np.array(1)
        except ValueError:
            dts = float(result.dts)
            trunc = np.full_like(y, 1.0e5)

            c = m.control
            if isinstance(c, AdaptiveController):
                # NOTE: if we can't find a maximum time step, just let the
                # adaptive step controller do its thing until it can't anymore
                spiked = np.array(int(c.nrejects > c.max_rejects))
            else:
                # NOTE: otherwise, just hope for the best
                spiked = np.array(1)

        yprev, ynext = _ad_ex_spike_reset(model, t + dts, tprev, y, r)
        result = AdvanceResult(
            y=ynext,
            trunc=trunc,
            storage=np.hstack([yprev, ynext]),
            spiked=spiked,
            dts=np.array(dts),
        )
    elif model.spiked(t, result.y) > 0.0:
        ts = estimate_spike_time_exp(t, result.y[0], tprev, y[0], p.v_peak)
        yprev, ynext = _ad_ex_spike_reset(model, ts, tprev, y, r)

        result = AdvanceResult(
            y=ynext,
            trunc=np.zeros_like(y),
            storage=np.hstack([yprev, ynext]),
            spiked=np.array(1),
            dts=np.array(ts - tprev),
        )

    return result


# }}}
