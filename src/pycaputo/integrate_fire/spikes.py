# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo.typing import Array

# {{{ linear spike time estimate


def estimate_spike_time_linear(
    t: float, V: Array, tprev: float, Vprev: Array, Vpeak: Array | float
) -> float:
    """Give a linear estimation of the spike time.

    .. math::

        V(t) = a t + b.

    We assume that the spike occurred between :math:`(t, V)` and
    :math:`(t_{prev}, V_{prev})` at :math:`V_{peak}`. This information can be
    used to provide a simple linear estimation for the spike time.

    :return: an estimation of the spike time.
    """
    assert Vprev <= Vpeak <= V
    ts = (Vpeak - Vprev) / (V - Vprev) * t + (V - Vpeak) / (V - Vprev) * tprev
    assert tprev <= ts <= t

    return float(ts)


# }}}


# {{{ quadratic spike time estimate


def estimate_spike_time_quadratic(
    t: float,
    V: Array,
    tprev: float,
    Vprev: Array,
    tpprev: float,
    Vpprev: Array,
    Vpeak: Array | float,
) -> float:
    """Give a quadratic estimation of the spike time.

    .. math::

        V(t) = a t^2 + b t + c.

    We assume that the spike occurred between :math:`(t, V)` and
    :math:`(t_{prev}, V_{prev})` at :math:`V_{peak}`. This information can be
    used to provide a simple quadratic estimation for the spike time.

    :return: an estimation of the spike time.
    """
    assert Vpprev <= Vprev <= Vpeak <= V
    assert tpprev < tprev < t

    a = ((V - Vprev) / (t - tprev) - (Vprev - Vpprev) / (tprev - tpprev)) / (t - tpprev)
    b = (
        tpprev**2 * (V - Vprev) - tprev**2 * (V - Vpprev) + t**2 * (Vprev - Vpprev)
    ) / ((tprev - tpprev) * (t - tpprev) * (t - tprev))
    c = Vpprev - (Vprev - Vpprev) / (tprev - tpprev) * tpprev + tpprev * tprev * a
    d = b**2 - 4 * a * (c - Vpeak)
    assert d >= 0

    ts = (-b + np.sqrt(d)) / (2 * a)

    assert tprev < ts < t
    return float(ts)


# }}}


# {{{ exponential spike time estimate


def estimate_spike_time_exp(
    t: float, V: Array, tprev: float, Vprev: Array, Vpeak: Array | float
) -> float:
    """Give an exponential estimation of the spike time.

    .. math::

        V(t) = a e^t + b.

    :returns: an estimate of the spike time.
    """
    assert Vprev <= Vpeak <= V
    assert tprev < t

    ts = np.log(
        (Vpeak - Vprev) / (V - Vprev) * np.exp(t)
        + (V - Vpeak) / (V - Vprev) * np.exp(tprev)
    )

    assert tprev <= ts <= t
    return float(ts)


# }}}
