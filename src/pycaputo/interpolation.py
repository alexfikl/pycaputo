# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterator

import numpy as np

from pycaputo.utils import Array


@dataclass(frozen=True)
class InterpStencil:
    r"""Approximation of a function by Lagrange poylnomials on a uniform grid.

    .. math::

        f(x_m) = \sum_{k \in \text{offsets}} f_{m + k} \ell_k(x_m)

    where :math:`\ell_k` is the :math:`k`-th Lagrange polynomial. The approximation
    is of the :attr:`order` of the Lagrange polynomials.

    Note that Lagrange polynomials are notoriously ill-conditioned on uniform
    grids (Runge phenomenon), so they should not be used with high-order.
    """

    #: Coefficients used in the stencil.
    coeffs: Array
    #: Offsets around the centered :math:`0` used in the stencil.
    offsets: Array
    #: Point at which the interpolation was evaluated.
    x: float

    @property
    def order(self) -> int:
        """Order of the Lagrange polynomial."""
        return self.coeffs.size

    @cached_property
    def padded_coeffs(self) -> Array:
        """Padded coefficients that are symmetric around the :math:`0`
        index and can be easily applied as a convolution.
        """
        n = np.max(np.abs(self.offsets))
        coeffs = np.zeros(2 * n + 1, dtype=self.coeffs.dtype)
        coeffs[n + self.offsets] = self.coeffs

        return coeffs

    @cached_property
    def trunc(self) -> float:
        """Truncation error of the interpolation."""
        return determine_truncation_error(self.offsets, self.x)


def apply_interpolation(s: InterpStencil, f: Array) -> Array:
    """Apply the stencil to a function *f*.

    Note that only interior points are correctly computed. Any boundary
    points will contain invalid values.

    :returns: the stencil applied to the function *f*.
    """
    a = s.padded_coeffs.astype(f.dtype)
    return np.convolve(f, a, mode="same")


def determine_truncation_error(offsets: Array, x: float, h: float = 1.0) -> float:
    r"""Approximate the truncation error of the Lagrange interpolation.

    .. math::

        f(x) - \sum_{k \in \text{offsets}} \ell_k(x) f_{m + k} =
        c \frac{\mathrm{d}^n f}{\mathrm{d} x^n}(\xi),

    where the constant :math:`c` is approximated as the truncation error. The
    Error itself also depends on the derivatives of :math:`f` at a undetermined
    point :math:`\xi` in the interpolation interval.

    :arg h: grid size on the physical grid, which is necessary for an accurate
        estimate.
    """

    c = h**offsets.size / math.factorial(offsets.size) * np.prod(x - offsets)
    return float(c)


def wandering(
    n: int,
    wanderer: int | bool | float = 1.0,  # noqa: FBT001
    landscape: int | bool | float = 0.0,  # noqa: FBT001
) -> Iterator[Array]:
    for i in range(n):
        yield np.array([landscape] * i + [wanderer] + [landscape] * (n - i - 1))


def make_lagrange_approximation(
    bounds: tuple[int, int],
    x: int | float = 0.5,
    *,
    dtype: np.dtype[Any] | None = None,
) -> InterpStencil:
    r"""Construct interpolation coefficients at *x* on a uniform grid.

    :arg bounds: inclusive left and right bounds on the stencil around a point
        :math:`x_i`. For example, ``(-1, 2)`` defines the 4 point stencil
        :math:`\{x_{i - 1}, x_i, x_{i + 1}, x_{i + 2}\}`.
    :arg x: point (in index space) at which to evaluate the function, i.e. to
        evaluate at :math:`x_{i + 1/2}` this should be :math:`1/2`, to evaluate at
        :math:`x_{i - 3/2}` this should be :math:`-3/2`, etc.
    """
    if len(bounds) != 2:
        raise ValueError(f"Stencil bounds are invalid: {bounds}")

    if bounds[0] > bounds[1]:
        bounds = (bounds[1], bounds[0])

    if bounds[0] > 0 or bounds[1] < 0:
        raise ValueError(f"Bounds must be (smaller <= 0, bigger >= 0): {bounds}")

    if dtype is None:
        dtype = np.dtype(np.float64)
    dtype = np.dtype(dtype)

    # evaluate Lagrange polynomials
    offsets = np.arange(bounds[0], bounds[1] + 1)

    x = float(x)
    if x.is_integer() and bounds[0] <= x <= bounds[1]:
        coeffs = np.zeros(offsets.size, dtype=dtype)
        coeffs[abs(bounds[0]) + int(x)] = 1.0
    else:
        # NOTE: this evaluates l_i(x) = prod((x - x_m) / (x_i - x_m), i != m)
        coeffs = np.array(
            [
                np.prod((x - offsets[not_n]) / (offsets[n] - offsets[not_n]))
                for n, not_n in enumerate(
                    wandering(offsets.size, wanderer=False, landscape=True)
                )
            ],
            dtype=dtype,
        )

    return InterpStencil(coeffs=coeffs, offsets=offsets, x=x)
