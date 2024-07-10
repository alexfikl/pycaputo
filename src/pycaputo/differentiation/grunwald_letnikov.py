# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycaputo.derivatives import GrunwaldLetnikovDerivative, Side
from pycaputo.grid import Points, UniformPoints
from pycaputo.logging import get_logger
from pycaputo.utils import Array, ArrayOrScalarFunction

from .base import DerivativeMethod, diff

logger = get_logger(__name__)


@dataclass(frozen=True)
class GrunwaldLetnikovMethod(DerivativeMethod):
    """A method used to evaluate a
    :class:`~pycaptuo.derivative.GrunwaldLetnikovDerivative`.
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
    """


@diff.register(GrunwaldLetnikov)
def _diff_grunwald_letnikov_method(
    m: GrunwaldLetnikov, f: ArrayOrScalarFunction, p: Points
) -> Array:
    if not isinstance(p, UniformPoints):
        raise TypeError(f"{type(m).__name__!r} only supports uniform grids")

    alpha = m.alpha
    fx = f(p.x) if callable(f) else f

    df = np.empty(fx.shape, dtype=fx.dtype)
    df[0] = np.nan

    from scipy.special import binom

    h = p.dx[0]
    k = np.arange(df.size - 1, -1, -1)
    omega = (-1) ** k * binom(alpha, k) / h**alpha

    for n in range(1, df.size):
        df[n] = np.sum(omega[-n - 1 :] * fx[: n + 1])

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
    """

    shift: float
    """Desired shift in the formula."""

    @classmethod
    def optimal_shift_for_alpha(cls, alpha: float) -> float | None:
        r"""Compute the optimal shift for a given fractional order :math:`\alpha`.

        :returns: the optimal shift or *None*, if no shift is known for that
            range of :math:`\alpha`. In that case, using
            :class:`GrunwaldLetnikov` is recommended.
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
        fx = f(p.x + m.shift * h)
    else:
        fx = f
        raise NotImplementedError

    alpha = m.alpha
    df = np.empty(fx.shape, dtype=fx.dtype)
    df[0] = np.nan

    from scipy.special import binom

    k = np.arange(df.size - 1, -1, -1)
    omega = (-1) ** k * binom(alpha, k) / h**alpha

    for n in range(1, df.size):
        df[n] = np.sum(omega[-n - 1 :] * fx[: n + 1])

    return df


# }}}
