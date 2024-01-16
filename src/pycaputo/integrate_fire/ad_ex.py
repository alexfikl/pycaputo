# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import NamedTuple, overload

import numpy as np

from pycaputo.integrate_fire.base import CaputoIntegrateFireL1Method, IntegrateFireModel
from pycaputo.logging import get_logger
from pycaputo.utils import Array, dc_stringify

logger = get_logger(__name__)


# {{{ parameters


class AdExDim(NamedTuple):
    """Dimensional parameters for the Adaptive Exponential Integrate-and-Fire
    (AdEx) model from [Naud2008]_."""

    #: Added current :math:`I` (in picoamperes *pA*).
    current: float
    #: Total capacitance :math:`C` (in picofarad per millisecond *pF / ms^(alpha - 1)*).
    c: float
    #: Total leak conductance :math:`g_L` (in nanosiemens *nS*).
    gl: float
    #: Effective rest potential :math:`E_L` (in microvolts *mV*).
    el: float
    #: Threshold slope factor :math:`\Delta_T` (in microvolts *mV*).
    delta_t: float
    #: Effective threshold potential :math:`V_T` (in microvolts *mV*).
    vt: float

    #: Time constant :math:`\tau_w` (in microseconds *ms^alpha*).
    tau_w: float
    #: Conductance :math:`a` (in nanosiemens *nS*).
    a: float

    #: Peak potential :math:`V_{peak}` (in microvolts *mV*).
    v_peak: float
    #: Reset potential :math:`V_r` (in microvolts *mV*).
    v_reset: float
    #: Adaptation current reset offset :math:`b` (in picoamperes *pA*).
    b: float

    def __str__(self) -> str:
        return dc_stringify(
            {
                "I      (current / pA)": self.current,
                "C      (total capacitance / pF/ms^alpha)": self.c,
                "g_L    (total leak conductance / nS)": self.gl,
                "E_L    (effective rest potential / mV)": self.el,
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

    def nondim(self, alpha: float | tuple[float, float]) -> AdEx:
        r"""Construct non-dimensional parameters for the AdEx model.

        The non-dimensionalization is performed using the following rescaling

        .. math::

            \hat{t} = \sqrt[\alpha_1]{\frac{g_L}{C}} t,
            \qquad
            \hat{V} = \frac{V - V_T}{\Delta_T},
            \qquad
            \hat{w} = \frac{w}{g_L \Delta_T}.

        and results in a reduction of the parameter space to the four
        non-dimensional variables :math:`(I, E_L, a, \tau_w)` in the model and
        :math:`(V_{peak}, V_r, b)` in the reset condition.

        :arg alpha: the order of the fractional derivatives for two model
            components :math:`(V, w)`. These can be the same if the two variables
            use the same order.
        """
        alpha = (alpha, alpha) if isinstance(alpha, float) else alpha
        if not len(alpha) == 2:
            raise ValueError(f"Only 2 orders 'alpha' are required: given {len(alpha)}")

        return AdEx(
            alpha=alpha,
            T=(self.gl / self.c) ** (1.0 / alpha[0]),
            current=self.current / (self.delta_t * self.gl),
            el=(self.el - self.vt) / self.delta_t,
            tau_w=(self.gl / self.c) ** (alpha[1] / alpha[0]) * self.tau_w,
            a=self.a / self.gl,
            v_peak=(self.v_peak - self.vt) / self.delta_t,
            v_reset=(self.v_reset - self.vt) / self.delta_t,
            b=self.b / (self.delta_t * self.gl),
        )


class AdEx(NamedTuple):
    """Non-dimensional parameters for the AdEx model (see :class:`AdExDim`)."""

    #: Fractional orders used in the non-dimensionalization.
    alpha: tuple[float, float]
    #: Fractional time scale.
    T: float

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
    #: Reset potential :math:`V_r`.
    v_reset: float
    #: Adaptation current reset offset :math:`b`.
    b: float

    def __str__(self) -> str:
        return dc_stringify(
            {
                "alpha  (fractional order)": self.alpha,
                "T      (fractional time scale)": self.T,
                "I      (current)": self.current,
                "E_L    (effective rest potential)": self.el,
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
        c=200,
        gl=10,
        el=-70,
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
        c=200,
        gl=12,
        el=-70,
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
        c=130,
        gl=18,
        el=-58,
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
        c=200,
        gl=10,
        el=-58,
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
        c=200,
        gl=12,
        el=-70,
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
        c=200,
        gl=12,
        el=-70,
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
        c=100,
        gl=10,
        el=-65,
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
        c=100,
        gl=12,
        el=-60,
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


def get_ad_ex_parameters_latex(alpha: float | tuple[float, float]) -> str:
    from rich.box import Box
    from rich.table import Table

    box = Box("    \n  &\\\n    \n  &\\\n    \n    \n  &\\\n    \n")
    t = Table(
        *[
            "Name",
            "$E_L$",
            "$a$",
            r"$\tau_w$",
            "$b$",
            "$V_r$",
            "$I$",
        ],
        box=box,
        header_style=None,
    )

    for name, param in AD_EX_PARAMS.items():
        ad_ex = param.nondim(alpha)
        t.add_row(*[
            name,
            f"{ad_ex.el:.3f}",
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

    #: Non-dimensional parameters for the model.
    param: AdEx

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
            p.current - (V - p.el) + np.exp(V) - w,
            (p.a * (V - p.el) - w) / p.tau_w,
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


class _PotentialCoefficient(NamedTuple):
    d0: float
    d1: float
    d2: float


class _AdaptationCoefficient(NamedTuple):
    c0: float
    c1: float


def _evaluate_lambert_coefficients(
    ad_ex: AdExModel, h: Array, r: Array
) -> tuple[_PotentialCoefficient, _AdaptationCoefficient]:
    # NOTE: small rename to match write-up
    hV, hw = h
    rV, rw = r
    _, _, I, el, tau_w, a, *_ = ad_ex.param  # noqa: E741

    # w coefficients: w = c0 V + c1
    c0 = a * hw / (tau_w + hw)
    c1 = (tau_w * rw - a * hw * el) / (hw + tau_w)

    # V coefficients: d0 V + d1 = d2 exp(V)
    d0 = 1 + hV * (1 + c0)
    d1 = -hV * (I + el - c1) - rV
    d2 = hV

    return _PotentialCoefficient(d0, d1, d2), _AdaptationCoefficient(c0, c1)


def _find_maximum_time_lambert(ad_ex: AdExModel, t: float, r: Array) -> float:
    from math import gamma

    def func(tspike: float) -> float:
        alpha = ad_ex.param.alpha
        h = np.array([
            gamma(2 - alpha[0]) * (tspike - t) ** alpha[0],
            gamma(2 - alpha[1]) * (tspike - t) ** alpha[1],
        ])

        (d0, d1, d2), _ = _evaluate_lambert_coefficients(ad_ex, h, r)
        return float(d2 / d0 * np.exp(-d1 / d0 + 1.0)) - 1.0

    import scipy.optimize as so

    try:
        result = so.root_scalar(
            f=func,
            x0=1.0e-2,
            bracket=[0.0, 0.5],
        )
        return float(result.root)
    except ValueError:
        # FIXME: what's a good return value here? the calling code should have
        # a `dt = min(dt_min, t_min - t)` to handle any large values
        return 0.5


# }}}


# {{{ AdExIntegrateFireL1Method


def ad_ex_solve(ad_ex: AdExModel, t: float, y0: Array, h: Array, r: Array) -> Array:
    r"""Solve the implicit equation for the AdEx model.

    This solve an implicit nonlinear equation of the form

    .. math::

        \mathbf{y} - \mathbf{h} \odot \mathbf{f}(t, \mathbf{y}) = \mathbf{r}

    where :math:`\mathbf{f}` is given by the source term of the :class:`AdExModel`.
    In that case, we can solve the equation explicitly, i.e. without an iterative
    method, by using the Lambert W function.

    This function can return complex results if the solution is out of range of
    the Lambert W function. This can happen if the simulation becomes unstable
    or it is close to a spike and the time step is too large.

    See :func:`pycaputo.implicit.solve` for an interactive method based on the
    :func:`scipy.optimize.root`.
    """

    from scipy.special import lambertw

    d, c = _evaluate_lambert_coefficients(ad_ex, h, r)
    dstar = -d[2] / d[0] * np.exp(-d[1] / d[0])
    Vstar = -d[1] / d[0] - lambertw(dstar)
    wstar = c[0] * Vstar + c[1]

    return np.array([Vstar, wstar])


@dataclass(frozen=True)
class AdExIntegrateFireL1Method(CaputoIntegrateFireL1Method[AdExModel]):
    #: Parameters for the AdEx model.
    model: AdExModel

    def solve(self, t: float, y0: Array, c: Array, r: Array) -> Array:
        return ad_ex_solve(self.model, t, y0, c, r)


# }}}
