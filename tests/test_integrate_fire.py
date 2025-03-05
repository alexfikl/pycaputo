# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from functools import partial

import numpy as np
import numpy.linalg as la
import pytest

from pycaputo.logging import get_logger
from pycaputo.typing import Array
from pycaputo.utils import get_environ_bool, set_recommended_matplotlib

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent
ENABLE_VISUAL = get_environ_bool("ENABLE_VISUAL")

log = get_logger(f"pycaputo.{TEST_FILENAME.stem}")
set_recommended_matplotlib()

# {{{ test_ad_ex_parameters


def test_ad_ex_parameters() -> None:
    """
    Check the consistency of the AdEx parameters.
    Check that the non-dimensionalization is performed according to expectations.
    """

    from pycaputo.integrate_fire.ad_ex import AD_EX_PARAMS, AdEx, AdExDim

    def ad_ex_func(p: AdEx, y: Array) -> Array:
        V, w = y

        return np.array([
            p.current - (V - p.e_leak) + np.exp(V) - w,
            (p.a * (V - p.e_leak) - w) / p.tau_w,
        ])

    def ad_ex_dim_func(p: AdExDim, y: Array) -> Array:
        V, w = y

        return np.array([
            (
                p.current
                - p.gl * (V - p.e_leak)
                + p.gl * p.delta_t * np.exp((V - p.vt) / p.delta_t)
                - w
            ),
            (p.a * (V - p.e_leak) - w) / p.tau_w,
        ])

    alpha = (0.77, 0.31)
    rng = np.random.default_rng(seed=42)

    for name, param in AD_EX_PARAMS.items():
        ad_ex = param.nondim(alpha)
        assert all(np.all(np.isfinite(np.array(v))) for v in param)
        assert str(param)
        assert str(ad_ex)

        log.info("[%s] Parameters:\n%s\n%s", name, param, ad_ex)

        assert ad_ex.ref.T_ref > 0
        assert ad_ex.tau_w > 0
        assert ad_ex.v_peak > ad_ex.v_reset

        ref = ad_ex.ref
        for _ in range(16):
            y = np.array([
                rng.uniform(ad_ex.v_reset, ad_ex.v_peak),
                rng.uniform(0.0, 1.0),
            ])

            # compute non-dimensional and dimensional right-hand sides
            f = ad_ex_func(ad_ex, y)
            f_ref = ad_ex_dim_func(param, ad_ex.ref.var(y))

            # re-dimensionalize right-hand side and check
            f_dim = np.array([
                f[0] * ref.w_ref,
                f[1] * ref.w_ref / ref.T_ref ** alpha[1],
            ])

            err = la.norm(f_dim - f_ref) / la.norm(f_ref)
            assert err < 3.0e-15


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


def ad_ex_zero(a: float, tau_w: float, h: float) -> float:
    return (1 + (1 + a * h / (h + tau_w)) * h) / h


def test_ad_ex_lambert_arg() -> None:
    """
    Check that the zeros of the Lambert functions are found.

    This checks that all the parameter bundles that we have allow solving the
    equation at least in the case of equal fractional orders.
    """

    from math import gamma

    from pycaputo.integrate_fire.ad_ex import AD_EX_PARAMS

    alpha = 0.9
    a, b = 0, gamma(2 - alpha)

    for name, param in AD_EX_PARAMS.items():
        ad_ex = param.nondim(alpha)
        log.info("[%s] Parameters: a %.12e tau_w %.12e", name, ad_ex.a, ad_ex.tau_w)

        hm, hp = ad_ex_lambert_zero_roots(ad_ex.a, ad_ex.tau_w, alpha)
        hmin = (hp + hm) / 2.0
        log.info("[%s] hp %.12e hm %.12e hmin %.12e", name, hp, hm, hmin)
        log.info("[%s] Works? %s", name, a < hm < b or a < hp < b)

        func = partial(ad_ex_zero, ad_ex.a, ad_ex.tau_w)
        assert np.isnan(hm) or abs(func(hm)) <= 5.0e-14, func(hm)
        assert np.isnan(hp) or abs(func(hp)) <= 5.0e-14, func(hp)

        if ENABLE_VISUAL:
            from pycaputo.utils import figure

            if np.isnan(hm):
                hm = hp = hmin = 1.0

            filename = f"test_ad_ex_lambert_arg_{name}"
            with figure(TEST_DIRECTORY / filename, normalize=True) as fig:
                ax = fig.gca()

                h = np.linspace(min(hm, 0), max(hp, 1), 256)
                ax.plot(h, np.vectorize(func)(h))
                ax.plot([hm, hmin, hp], [func(hm), func(hmin), func(hp)], "ro")


# }}}


# {{{ test_ad_ex_lambert_limits


def test_ad_ex_lambert_limits() -> None:
    """
    Check that the max time step can be found from the Lambert function and it
    is between the expected limits.
    """

    from math import gamma

    from pycaputo.controller import make_jannelli_controller
    from pycaputo.derivatives import CaputoDerivative as D
    from pycaputo.integrate_fire.ad_ex import (
        AD_EX_PARAMS,
        AdExModel,
        CaputoAdExIntegrateFireL1Model,
        _evaluate_lambert_coefficients_time,  # noqa: PLC2701
        find_maximum_time_step_lambert,
    )

    alpha = (0.9, 0.4)
    tn = 0.0
    dt = 1.0e1

    def ad_ex_coeff(t: float, tprev: float) -> Array:
        return np.array([
            gamma(2 - alpha[0]) * (t - tprev) ** alpha[0],
            gamma(2 - alpha[1]) * (t - tprev) ** alpha[1],
        ])

    def func(ad_ex: AdExModel, tspike: float, yprev: Array, r: Array) -> float:
        d0, d1, d2, *_ = _evaluate_lambert_coefficients_time(
            ad_ex, tspike, tn, yprev, r
        )
        return float(d2 / d0 * np.exp(-d1 / d0 + 1.0))

    rng = np.random.default_rng(seed=42)

    for name, dim in AD_EX_PARAMS.items():
        param = dim.nondim(alpha)
        ad_ex = AdExModel(param)

        y0 = np.array([rng.uniform(param.v_reset, param.v_peak) - 10, rng.uniform()])
        r = np.array([
            dt**a / gamma(1 - a) * yi for a, yi in zip(alpha, y0, strict=True)
        ])

        method = CaputoAdExIntegrateFireL1Model(
            ds=(D(alpha[0]), D(alpha[1])),
            control=make_jannelli_controller(chimin=0.1, chimax=1.0),
            y0=(y0,),
            source=ad_ex,
        )

        # evaluate the function and check that it has a root
        tspike = tn + np.logspace(-10.0, -0.3, 128)
        f = np.array([func(ad_ex, t, y0, r) for t in tspike])
        assert np.any(f < 1.0)

        # check that the solution is actually complex
        h = ad_ex_coeff(tn + dt, tn)
        ynext = method.solve(tn + dt, y0, h, y0 - h * r)
        log.info("%s: %s", name, np.iscomplex(ynext))
        assert np.any(np.iscomplex(ynext))

        imax = np.argmax(f > 1.0) - 1
        log.info("max tspike: t %.12e f %.12e", tspike[imax - 1], f[imax - 1])

        # find the maximum time step
        dt_opt = 0.9 * find_maximum_time_step_lambert(ad_ex, tn + dt, tn, y0, r)

        # check that it's larger than the function root estimate
        tspike_opt = tn + dt_opt
        log.info("opt tspike: %.12e", tspike_opt)
        assert tspike[imax - 1] <= tspike_opt, tspike_opt

        # check that the solution at the new time step is not complex anymore
        h = ad_ex_coeff(tspike_opt, tn)
        ynext = method.solve(tspike_opt, y0, h, y0 - h * r)
        assert not np.any(np.iscomplex(ynext)), ynext

        if ENABLE_VISUAL:
            from pycaputo.utils import figure

            filename = f"test_ad_ex_lambert_limits_{name}"
            with figure(TEST_DIRECTORY / filename, normalize=True) as fig:
                ax = fig.gca()

                ax.plot(tspike, f)
                ax.plot(tspike_opt, func(ad_ex, tspike_opt, y0, r), "ro")
                ax.axhline(0.0, color="k", ls="--")
                ax.axhline(1.0, color="k", ls="--")


# }}}


# {{{ test_ad_ex_solve


def test_ad_ex_solve() -> None:
    """
    Check that the implicit equation for the AdEx model can be solved.
    """

    from pycaputo.controller import make_jannelli_controller
    from pycaputo.derivatives import CaputoDerivative as D
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
                ds=(D(alpha[0]), D(alpha[1])),
                control=make_jannelli_controller(chimin=0.1, chimax=1.0),
                y0=(y0,),
                source=ad_ex,
            )

            h = np.array([gamma(2 - a) * dt**a for a in alpha])
            r = np.array([
                dt**a / gamma(1 - a) * yi for a, yi in zip(alpha, y0, strict=True)
            ])

            y = method.solve(t, y0, h, r)
            error: Array = y - h * ad_ex.source(t, y) - r
            e_imag = np.linalg.norm(error.imag)
            e_real = np.linalg.norm(error.real)

            assert e_imag < 5.0e-15, (name, n)
            assert e_real < 5.0e-15, (name, n)

        log.info("[%s] real %.12e imag %.12e", name, e_real, e_imag)


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
    """
    Check that the PIF model converges for known solutions.
    """

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
    from pycaputo.derivatives import CaputoDerivative as D
    from pycaputo.stepping import evolve
    from pycaputo.utils import EOCRecorder, stringify_eoc

    tspikes_ref = model.param.constant_spike_times(tfinal, V0=y0[0])

    if tspikes_ref.size < 1:
        raise ValueError(
            "This test expects at least one spike to check the order. "
            "Try increasing 'alpha' or 'tfinal'."
        )

    log.info("Found %d spikes", tspikes_ref.size)

    from scipy.special import gamma

    eoct = EOCRecorder(order=1.0, name="tspikes")
    eocy = EOCRecorder(order=2.0 - alpha, name="y")
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
            ds=(D(alpha),),
            control=c,
            y0=(y0,),
            source=model,
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

        dtmax = np.max(np.diff(ts))
        err = la.norm(tspikes - tspikes_ref) / la.norm(tspikes_ref)

        log.info("dt %.12e error %.12e", dtmax, err)
        eoct.add_data_point(dtmax, err)

        ts = []
        ys = []
        for event in evolve(stepper, dtinit=dtinit):
            if isinstance(event, StepAccepted):
                ts.append(event.t)
                ys.append(event.y)
                if event.spiked:
                    break

        t = np.array(ts[:-2])
        y = np.array(ys).squeeze()[:-2]
        y_ref = y0 + model.param.current * t**alpha / gamma(1 + stepper.alpha)

        dtmax = np.max(np.diff(t))
        err = la.norm(y - y_ref) / la.norm(y_ref)
        log.info("dt %.12e error %.12e", dtmax, err)
        eocy.add_data_point(dtmax, err)

    log.info("\n%s", stringify_eoc(eoct, eocy))

    assert eoct.order is not None
    assert eoct.order - 0.25 < eoct.estimated_order < eoct.order + 0.25

    assert eocy.order is not None
    assert eocy.estimated_order > 0.7


# }}}


# {{{ test spike time estimates


def test_spike_time_estimate_linear() -> None:
    """
    Check the linear time step estimate.
    """

    from pycaputo.integrate_fire.spikes import estimate_spike_time_linear

    rng = np.random.default_rng(seed=42)
    for _ in range(16):
        tp, t = np.sort(rng.uniform(0.0, 0.1, size=2))
        Vp, Vpeak, V = np.sort(-np.sqrt(rng.uniform(1.0, 20, size=3)))

        ts = estimate_spike_time_linear(t, V, tp, Vp, Vpeak)
        assert tp <= ts <= t

        Vpeak_est = (ts - tp) / (t - tp) * V + (t - ts) / (t - tp) * Vp
        error = abs(Vpeak - Vpeak_est)

        log.info("Error: %.8e", error)
        assert error < 1.0e-14


def test_spike_time_estimate_quadratic() -> None:
    """
    Check the quadratic time step estimate.
    """

    from pycaputo.integrate_fire.spikes import estimate_spike_time_quadratic

    rng = np.random.default_rng(seed=42)
    for _ in range(16):
        tpp, tp, t = np.sort(rng.uniform(0.0, 0.1, size=3))
        Vpp, Vp, Vpeak, V = np.sort(-np.sqrt(rng.uniform(1.0, 20, size=4)))

        ts = estimate_spike_time_quadratic(t, V, tp, Vp, tpp, Vpp, Vpeak)
        assert tpp <= tp <= ts <= t

        Vpeak_est = (
            (ts - tp) * (ts - tpp) / ((t - tp) * (t - tpp)) * V
            + (ts - t) * (ts - tpp) / ((tp - t) * (tp - tpp)) * Vp
            + (ts - t) * (ts - tp) / ((tpp - t) * (tpp - tp)) * Vpp
        )
        error = abs(Vpeak - Vpeak_est)

        log.info("Error: %.8e", error)
        assert error < 1.0e-13


def test_spike_time_estimate_exponential() -> None:
    """
    Check the exponential time step estimate.
    """

    from pycaputo.integrate_fire.spikes import estimate_spike_time_exp

    rng = np.random.default_rng(seed=42)
    for _ in range(16):
        tp, t = np.sort(rng.uniform(0.0, 0.1, size=2))
        Vp, Vpeak, V = np.sort(-np.sqrt(rng.uniform(1.0, 20, size=3)))

        ts = estimate_spike_time_exp(t, V, tp, Vp, Vpeak)
        assert tp <= ts <= t

        a = (V - Vp) / (np.exp(t) - np.exp(tp))
        Vpeak_est = a * np.exp(ts) + (Vp - a * np.exp(tp))
        error = abs(Vpeak - Vpeak_est)

        log.info("Error: %.8e", error)
        assert error < 1.0e-12


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
