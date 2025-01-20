# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from math import gamma

import numpy as np

from pycaputo.derivatives import GrunwaldLetnikovDerivative, Side
from pycaputo.grid import Points, UniformPoints
from pycaputo.logging import get_logger
from pycaputo.typing import Array, ArrayOrScalarFunction

from .base import DerivativeMethod, diff

log = get_logger(__name__)


@dataclass(frozen=True)
class GrunwaldLetnikovMethod(DerivativeMethod):
    """A method used to evaluate a
    :class:`~pycaputo.derivatives.GrunwaldLetnikovDerivative`.

    Note that for sufficiently smooth functions, the Grünwald-Letnikov derivative
    coincides with the Riemann-Liouville derivative. Therefore, these methods
    can also be used to approximate the Riemann-Liouville derivative.
    """

    alpha: float
    """Order of the Grünwald-Letnikov derivative that is being discretized."""

    if __debug__:

        def __post_init__(self) -> None:
            if self.alpha < 0:
                raise ValueError(f"Negative orders are not supported: {self.alpha}")

    @property
    def d(self) -> GrunwaldLetnikovDerivative:
        return GrunwaldLetnikovDerivative(self.alpha, side=Side.Left)


# {{{ GrunwaldLetnikov


@dataclass(frozen=True)
class GrunwaldLetnikov(GrunwaldLetnikovMethod):
    """Standard method of evaluating the Grünwald-Letnikov derivative by
    truncating the limit.

    This method is defined in Section 5.3.1 from [Li2020]_ for uniform grids.
    The method is first-order for smooth functions.
    """


@diff.register(GrunwaldLetnikov)
def _diff_grunwald_letnikov_method(
    m: GrunwaldLetnikov, f: ArrayOrScalarFunction, p: Points
) -> Array:
    if not isinstance(p, UniformPoints):
        raise TypeError(f"{type(m).__name__!r} only supports uniform grids")

    alpha = m.alpha
    fx = f(p.x) if callable(f) else f

    # NOTE: subtract f(a) to ensure that the function is always zero at 0
    fa = fx[0]
    fx = fx - fa

    df = np.empty(fx.shape, dtype=fx.dtype)
    df[0] = np.nan

    from scipy.special import binom

    h = p.dx[0]
    k = np.arange(df.size - 1, -1, -1)
    omega = (-1) ** k * binom(alpha, k) / h**alpha

    for n in range(1, df.size):
        df[n] = np.sum(omega[-n - 1 :] * fx[: n + 1])

    # NOTE: add back correction for subtracting f(a)
    df[1:] = df[1:] + (p.x[1:] - p.a) ** (-alpha) / gamma(1 - alpha) * fa

    return df


# }}}


# {{{ ShiftedGrunwaldLetnikov


@dataclass(frozen=True)
class ShiftedGrunwaldLetnikov(GrunwaldLetnikovMethod):
    r"""Approximate the Grünwald-Letnikov derivative using a shifted method.

    Note that the shifted method requires knowledge of a point outside the
    interval :math:`[a, b]` at :math:`b + s h`, where :math:`s` is the
    :attr:`shift`. If the function value is not available, this value is
    computed by extrapolation.

    This method is described in Section 5.3.2 from [Li2020]_ for uniform grids.
    The method is second-order for smooth functions with the shift
    :math:`\alpha / 2` (see :meth:`recommended_shift_for_alpha`).
    """

    shift: float
    """Desired shift in the formula."""

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()

            if not 0.0 <= self.shift <= 1.0:
                raise ValueError(f"Shift must be in 0 <= shift <= 1: {self.shift}")

    @classmethod
    def recommended_shift_for_alpha(cls, alpha: float) -> float | None:
        r"""Provide a recommended shift for a given order :math:`\alpha`.

        :returns: a recommended shift or *None*, if no shift is known for that
            range of :math:`\alpha`. If no shift is known, the Grünwald-Letnikov
            methods may not work very well.
        """

        return alpha / 2.0 if 0.0 < alpha < 2.0 else None


@diff.register(ShiftedGrunwaldLetnikov)
def _diff_shifted_grunwald_letnikov_method(
    m: ShiftedGrunwaldLetnikov, f: ArrayOrScalarFunction, p: Points
) -> Array:
    if not isinstance(p, UniformPoints):
        raise TypeError(f"{type(m).__name__!r} only supports uniform grids")

    h = p.dx[0]
    if callable(f):
        fa = f(p.x[0])
        fx: Array = f(p.x + m.shift * h) - fa
    else:
        raise NotImplementedError

    alpha = m.alpha
    df = np.empty(fx.shape, dtype=fx.dtype)
    df[0] = np.nan

    from scipy.special import binom

    k = np.arange(df.size - 1, -1, -1)
    omega = (-1) ** k * binom(alpha, k) / h**alpha

    for n in range(1, df.size):
        df[n] = np.sum(omega[-n - 1 :] * fx[: n + 1])

    # NOTE: add back correction for subtracting f(a)
    df[1:] = df[1:] + (p.x[1:] - p.a) ** (-alpha) / gamma(1 - alpha) * fa

    return df


# }}}


# {{{ TianZhouDeng2


@dataclass(frozen=True)
class TianZhouDeng2(GrunwaldLetnikovMethod):
    r"""Approximate the Grünwald-Letnikov derivative using a weighted method.

    Note that the shifted method requires knowledge of a point outside the
    interval :math:`[a, b]` at :math:`b + s h`, where :math:`s` is the
    :attr:`shift`. If the function value is not available, this value is
    computed by extrapolation.

    This method is described in Section 5.3.3 from [Li2020]_ for uniform grids.
    A more detailed analysis can be found in [Tian2015]_.

    The method is second-order for smooth functions with appropriate shifts
    (see :meth:`recommended_shift_for_alpha`).
    """

    shift: tuple[float, float]
    """Desired shifts in the formula, referred to as :math:`(p, q)` in [Li2020]_."""

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()

            if not all(-1.0 <= s <= 1.0 for s in self.shift):
                raise ValueError(f"Shift must be in 0 <= shift <= 1: {self.shift}")

            if abs(self.shift[0] - self.shift[1]) < 1.0e-14:
                raise ValueError(f"Shifts cannot be equal: {self.shift}")

    @classmethod
    def recommended_shift_for_alpha(cls, alpha: float) -> tuple[float, float] | None:
        r"""Provide a recommended shift for a given order :math:`\alpha`.

        :returns: a recommended shift or *None*, if no shift is known for that
            range of :math:`\alpha`. If no shift is known, the Grünwald-Letnikov
            methods may not work very well.
        """
        if 0.0 < alpha < 1.0:
            # NOTE: [Tian2015] doesn't really talk about 0 < alpha < 1? They just
            # say that taking this (p, q) pair gives the centered difference
            # scheme when alpha -> 1^+.
            return (1.0, 0.0)
        elif 1.0 < alpha < 2.0:
            return (1.0, -1.0)
        else:
            return None


@diff.register(TianZhouDeng2)
def _diff_tian_zhou_deng_2(
    m: TianZhouDeng2, f: ArrayOrScalarFunction, p: Points
) -> Array:
    if not isinstance(p, UniformPoints):
        raise TypeError(f"{type(m).__name__!r} only supports uniform grids")

    h = p.dx[0]
    s_p, s_q = m.shift

    if callable(f):
        fa = f(p.x[0])
        fx_p: Array = f(p.x + s_p * h) - fa
        fx_q: Array = f(p.x + s_q * h) - fa
    else:
        raise NotImplementedError

    alpha = m.alpha
    df = np.empty(fx_p.shape, dtype=fx_p.dtype)
    df[0] = np.nan

    from scipy.special import binom

    w_p = 0.5 * (alpha - 2 * s_q) / (s_p - s_q) / h**alpha
    w_q = 0.5 * (2 * s_p - alpha) / (s_p - s_q) / h**alpha

    k = np.arange(df.size - 1, -1, -1)
    omega = (-1) ** k * binom(alpha, k)

    for n in range(1, df.size):
        # FIXME: the indices here don't match [Li2020] Equation 5.94
        # fmt: off
        df[n] = (
            w_p * np.sum(omega[-n - 1 :] * fx_p[: n + 1])
            + w_q * np.sum(omega[-n - 1 :] * fx_q[: n + 1])
        )
        # fmt: on

    # NOTE: add back correction for subtracting f(a)
    df[1:] = df[1:] + (p.x[1:] - p.a) ** (-alpha) / gamma(1 - alpha) * fa

    return df


# }}}


# {{{ TianZhouDeng3


@dataclass(frozen=True)
class TianZhouDeng3(GrunwaldLetnikovMethod):
    r"""Approximate the Grünwald-Letnikov derivative using a weighted method.

    Note that the shifted method requires knowledge of a point outside the
    interval :math:`[a, b]` at :math:`b + s h`, where :math:`s` is the
    :attr:`shift`. If the function value is not available, this value is
    computed by extrapolation.

    This method is described in Section 5.3.3 from [Li2020]_ for uniform grids.
    A more detailed analysis can be found in [Tian2015]_.

    The method is third-order for smooth functions with appropriate shifts
    (see :meth:`recommended_shift_for_alpha`).
    """

    shift: tuple[float, float, float]
    """Desired shifts in the formula, referred to as :math:`(p, q, r)` in [Li2020]_."""

    if __debug__:

        def __post_init__(self) -> None:
            super().__post_init__()

            if not all(-1.0 <= s <= 1.0 for s in self.shift):
                raise ValueError(f"Shift must be in 0 <= shift <= 1: {self.shift}")

            p, q, r = self.shift
            if abs(p - q) < 1.0e-14 and abs(p - r) < 1.0e-14:
                raise ValueError(f"Shifts cannot be equal: {self.shift}")

    @classmethod
    def recommended_shift_for_alpha(
        cls, alpha: float
    ) -> tuple[float, float, float] | None:
        r"""Provide a recommended shift for a given order :math:`\alpha`.

        :returns: a recommended shift or *None*, if no shift is known for that
            range of :math:`\alpha`. If no shift is known, the Grünwald-Letnikov
            methods may not work very well.
        """
        if 0.0 < alpha < 2.0:
            # NOTE: [Tian2015] doesn't really talk about 0 < alpha < 1?
            return (1.0, 0.0, -1.0)
        else:
            return None


@diff.register(TianZhouDeng3)
def _diff_tian_zhou_deng_3(
    m: TianZhouDeng3, f: ArrayOrScalarFunction, p: Points
) -> Array:
    if not isinstance(p, UniformPoints):
        raise TypeError(f"{type(m).__name__!r} only supports uniform grids")

    h = p.dx[0]
    s_p, s_q, s_r = m.shift

    if callable(f):
        fa = f(p.x[0])
        fx_p: Array = f(p.x + s_p * h) - fa
        fx_q: Array = f(p.x + s_q * h) - fa
        fx_r: Array = f(p.x + s_r * h) - fa
    else:
        raise NotImplementedError

    alpha = m.alpha
    df = np.empty(fx_p.shape, dtype=fx_p.dtype)
    df[0] = np.nan

    from scipy.special import binom

    w_p = (
        (12 * s_q * s_r - (6 * s_q + 6 * s_r + 1) * alpha + 3 * alpha**2)
        / (s_q * s_r - s_p * s_q - s_p * s_r + s_p**2)
        / (12 * h**alpha)
    )
    w_q = (
        (12 * s_p * s_r - (6 * s_p + 6 * s_r + 1) * alpha + 3 * alpha**2)
        / (s_p * s_r - s_p * s_q - s_q * s_r + s_q**2)
        / (12 * h**alpha)
    )
    w_r = (
        (12 * s_p * s_q - (6 * s_p + 6 * s_q + 1) * alpha + 3 * alpha**2)
        / (s_p * s_q - s_p * s_r - s_q * s_r + s_r**2)
        / (12 * h**alpha)
    )

    k = np.arange(df.size - 1, -1, -1)
    omega = (-1) ** k * binom(alpha, k)

    for n in range(1, df.size):
        # FIXME: the indices here don't match [Li2020] Equation 5.105
        df[n] = (
            w_p * np.sum(omega[-n - 1 :] * fx_p[: n + 1])
            + w_q * np.sum(omega[-n - 1 :] * fx_q[: n + 1])
            + w_r * np.sum(omega[-n - 1 :] * fx_r[: n + 1])
        )

    # NOTE: add back correction for subtracting f(a)
    df[1:] = df[1:] + (p.x[1:] - p.a) ** (-alpha) / gamma(1 - alpha) * fa

    return df


# }}}
