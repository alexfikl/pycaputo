# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from functools import partial

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.logging import get_logger
from pycaputo.utils import Array, set_recommended_matplotlib

logger = get_logger("pycaputo.test_integrate_fire")
dirname = pathlib.Path(__file__).parent
set_recommended_matplotlib()

# {{{ test_ad_ex_parameters


def test_ad_ex_parameters() -> None:
    from pycaputo.integrate_fire.ad_ex import AD_EX_PARAMS

    alpha = (0.77, 0.31)
    for name, param in AD_EX_PARAMS.items():
        ad_ex = param.nondim(alpha)
        assert all(np.all(np.isfinite(np.array(v))) for v in param)
        assert str(param)
        assert str(ad_ex)

        logger.info("[%s] Parameters:\n%s\n%s", name, param, ad_ex)

        assert ad_ex.ref.T_ref > 0
        assert ad_ex.tau_w > 0
        assert ad_ex.v_peak > ad_ex.v_reset


# }}}


# {{{ test_ad_ex_lambert_arg


def ad_ex_lambert_zero_roots(
    a: float, tau_w: float, alpha: float
) -> tuple[float, float]:
    r"""Computes the zeros of :func:`ad_ex_zero` below.

    This is only valid if :math:`h \in [0, 1]`. In other case, we will get
    negative values that do not make any sense.
    """
    from math import sqrt

    a = 1 + a
    b = 1 + tau_w
    c = tau_w

    if a == 0:
        return -c / b, -c / b

    h_min = -b / (2 * a)
    delta = b**2 - 4 * a * c
    if delta <= 0:
        return np.nan, np.nan

    delta = sqrt(delta) / (2 * a)
    return h_min - delta, h_min + delta


def ad_ex_zero(a: float, tau_w: float, h: Array) -> Array:
    return (1 + (1 + a * h / (h + tau_w)) * h) / h


def test_ad_ex_lambert_arg(*, visualize: bool = True) -> None:
    from math import gamma

    from pycaputo.integrate_fire.ad_ex import AD_EX_PARAMS

    dirname = pathlib.Path(__file__).parent
    alpha = 0.9
    a, b = 0, gamma(2 - alpha)

    for name, param in AD_EX_PARAMS.items():
        ad_ex = param.nondim(alpha)
        logger.info("[%s] Parameters: a %.12e tau_w %.12e", name, ad_ex.a, ad_ex.tau_w)

        hm, hp = ad_ex_lambert_zero_roots(ad_ex.a, ad_ex.tau_w, alpha)
        hmin = (hp + hm) / 2.0
        logger.info("[%s] hp %.12e hm %.12e hmin %.12e", name, hp, hm, hmin)
        logger.info("[%s] Works? %s", name, a < hm < b or a < hp < b)

        func = partial(ad_ex_zero, ad_ex.a, ad_ex.tau_w)
        assert np.isnan(hm) or abs(func(hm)) <= 5.0e-14, func(hm)
        assert np.isnan(hp) or abs(func(hp)) <= 5.0e-14, func(hp)

        if visualize:
            from pycaputo.utils import figure

            if np.isnan(hm):
                hm = hp = hmin = 1.0

            h = np.linspace(min(hm, 0), max(hp, 1), 256)
            with figure(dirname / f"test_ad_ex_lambert_arg_{name}") as fig:
                ax = fig.gca()

                ax.plot(h, func(h))
                ax.plot([hm, hmin, hp], [func(hm), func(hmin), func(hp)], "ro")


def test_ad_ex_lambert_limits(*, visualize: bool = True) -> None:
    from math import gamma

    from pycaputo.integrate_fire.ad_ex import (
        AD_EX_PARAMS,
        AdExModel,
        _evaluate_lambert_coefficients,  # noqa: PLC2701
        find_maximum_time_step_lambert,
    )

    alpha = (0.9, 0.4)
    tn = 0.0
    dt = 1.0e-2

    def func(ad_ex: AdExModel, tspike: float, yprev: Array, r: Array) -> float:
        h = np.array([
            gamma(2 - alpha[0]) * (tspike - tn) ** alpha[0],
            gamma(2 - alpha[1]) * (tspike - tn) ** alpha[1],
        ])

        d0, d1, d2, *_ = _evaluate_lambert_coefficients(ad_ex, tspike, yprev, h, r)
        return float(d2 / d0 * np.exp(-d1 / d0 + 1.0))

    rng = np.random.default_rng(seed=42)

    for name, dim in AD_EX_PARAMS.items():
        param = dim.nondim(alpha)
        ad_ex = AdExModel(param)

        y0 = np.array([rng.uniform(param.v_reset, param.v_peak), rng.uniform()])
        r = np.array([dt**a / gamma(1 - a) * yi for a, yi in zip(alpha, y0)])

        tspike = tn + np.logspace(-10.0, -0.3, 256)
        f = np.array([func(ad_ex, t, y0, r) for t in tspike])
        assert np.any(f < 1.0)

        imax = np.argmax(f > 1.0) - 1
        logger.info("max tspike: t %.12e f %.12e", tspike[imax - 1], f[imax - 1])

        try:
            dt_opt = find_maximum_time_step_lambert(ad_ex, tn + dt, tn, y0, r)
        except ValueError:
            dt_opt = 0.5

        tspike_opt = tn + dt_opt
        logger.info("opt tspike: %.12e", tspike_opt)
        assert tspike[imax - 1] <= tspike_opt <= 1.0, tspike_opt

        if visualize:
            from pycaputo.utils import figure

            with figure(dirname / f"test_ad_ex_lambert_limits_{name}") as fig:
                ax = fig.gca()

                ax.plot(tspike, f)
                ax.plot(tspike_opt, func(ad_ex, tspike_opt, y0, r), "ro")
                ax.axhline(0.0, color="k", ls="--")
                ax.axhline(1.0, color="k", ls="--")


# }}}


# {{{ test_ad_ex_solve


def test_ad_ex_solve() -> None:
    from pycaputo.controller import make_jannelli_controller
    from pycaputo.integrate_fire.ad_ex import (
        AD_EX_PARAMS,
        AdExModel,
        CaputoAdExIntegrateFireL1Model,
        get_ad_ex_parameters,
    )

    rng = np.random.default_rng(seed=42)
    dt = 1.0e-2
    alpha = (0.9, 0.71)

    from math import gamma

    for name in AD_EX_PARAMS:
        param = get_ad_ex_parameters(name, alpha)
        ad_ex = AdExModel(param)

        for n in range(16):
            t = dt
            y0 = np.array([rng.uniform(param.v_reset, param.v_peak), rng.uniform()])
            method = CaputoAdExIntegrateFireL1Model(
                derivative_order=alpha,
                control=make_jannelli_controller(chimin=0.1, chimax=1.0),
                y0=(y0,),
                source=ad_ex.source,
                model=ad_ex,
            )

            h = np.array([gamma(2 - a) * dt**a for a in alpha])
            r = np.array([dt**a / gamma(1 - a) * yi for a, yi in zip(alpha, y0)])

            y = method.solve(t, y0, h, r)
            error = y - h * ad_ex.source(t, y) - r
            e_imag = np.linalg.norm(error.imag)
            e_real = np.linalg.norm(error.real)

            assert e_imag < 5.0e-15, (name, n)
            assert e_real < 5.0e-15, (name, n)

        logger.info("[%s] real %.12e imag %.12e", name, e_real, e_imag)


# }}}


# {{{ test_pif_model


@pytest.mark.parametrize(
    ("alpha", "resolutions"),
    [
        (0.50, [(1.0, 2.0), (0.75, 1.5), (0.25, 0.5)]),
        (0.75, [(1.0, 2.0), (0.75, 1.5), (0.25, 0.5)]),
        (0.95, [(0.5, 1.0), (0.25, 0.5), (0.125, 0.25)]),
    ],
)
def test_pif_model(alpha: float, resolutions: list[tuple[float, float]]) -> None:
    from pycaputo.integrate_fire import StepAccepted, pif

    # time interval
    tstart, tfinal = 0.0, 32.0
    # initial time step
    dtinit = 1.0e-1

    param = pif.PIFDim(current=160, C=100, v_reset=-48, v_peak=0.0)
    model = pif.PIFModel(param.nondim(alpha, V_ref=1.0, I_ref=20.0))

    # initial condition
    rng = np.random.default_rng(seed=42)
    y0 = np.array([rng.uniform(model.param.v_reset, model.param.v_peak)])

    from pycaputo.controller import make_jannelli_controller
    from pycaputo.fode import evolve
    from pycaputo.utils import EOCRecorder

    tspikes_ref = model.param.constant_spike_times(tfinal, V0=y0[0])

    if tspikes_ref.size < 1:
        raise ValueError(
            "This test expects at least one spike to check the order. "
            "Try increasing 'alpha' or 'tfinal'."
        )

    logger.info("Found %d spikes", tspikes_ref.size)

    eoct = EOCRecorder(order=1.0)
    eocy = EOCRecorder(order=2.0 - alpha)
    for chimin, chimax in resolutions:
        c = make_jannelli_controller(
            tstart,
            tfinal,
            dtmin=1.0e-5,
            chimin=chimin,
            chimax=chimax,
            abstol=1.0e-4,
        )

        stepper = pif.CaputoPerfectIntegrateFireL1Method(
            derivative_order=(alpha,),
            control=c,
            y0=(y0,),
            source=model.source,
            model=model,
        )

        ts = []
        ys = []
        tspikes = []
        for event in evolve(stepper, dtinit=dtinit):
            if isinstance(event, StepAccepted):
                ts.append(event.t)
                ys.append(event.y)
                if event.spiked:
                    tspikes.append(event.t)
            else:
                pass

        dtmax = np.max(np.diff(ts))
        err = la.norm(tspikes - tspikes_ref) / la.norm(tspikes_ref)

        logger.info("dt %.12e error %.12e", dtmax, err)
        eoct.add_data_point(dtmax, err)

        ts = []
        ys = []
        for event in evolve(stepper, dtinit=dtinit):
            if isinstance(event, StepAccepted):
                ts.append(event.t)
                ys.append(event.y)
                if event.spiked:
                    break
            else:
                pass

        t = np.array(ts[:-2])
        y = np.array(ys).squeeze()[:-2]
        y_ref = y0 + model.param.current * t**alpha / stepper.gamma1p

        dtmax = np.max(np.diff(t))
        err = la.norm(y - y_ref) / la.norm(y_ref)
        logger.info("dt %.12e error %.12e", dtmax, err)
        eocy.add_data_point(dtmax, err)

    logger.info("\n%s\n%s", eoct, eocy)

    assert eoct.order is not None
    assert eoct.order - 0.25 < eoct.estimated_order < eoct.order + 0.25

    assert eocy.order is not None
    assert eocy.estimated_order > 0.7


# }}}

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
