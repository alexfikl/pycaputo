# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any, NamedTuple

import numpy as np

from pycaputo.utils import Array


class Truncation(NamedTuple):
    """A representation of the truncation error of a :class:`Stencil`."""

    #: Order of the approximation.
    order: int
    #: Truncation error coefficient (without the function derivative).
    error: float


@dataclass(frozen=True)
class Stencil:
    r"""Approximation of a derivative by finite difference on a uniform grid.

    .. math::

        \frac{\mathrm{d}^n\, f}{\mathrm{d}\, x^n}
        \approx \frac{1}{h^n} \sum_{i \in \text{indices}} a_i f_i

    where :math:`a_i` are the given coefficients :attr:`coeffs` and :math:`f_i`
    are the point function evaluations. The approximation is to an order and
    truncation error can be retrieved from :attr:`trunc`.
    """

    #: Order of the derivative approximated by the stencil.
    derivative: int
    #: Coefficients used in the stencil.
    coeffs: Array
    #: Indices around the centered :math:`0` used in the stencil.
    indices: Array

    @cached_property
    def padded_coeffs(self) -> Array:
        """Padded coefficients that are symmetric around the :math:`0`
        index and can be easily applied as a convolution.
        """
        n = np.max(np.abs(self.indices))
        coeffs = np.zeros(2 * n + 1, dtype=self.coeffs.dtype)
        coeffs[n + self.indices] = self.coeffs

        return coeffs

    @cached_property
    def trunc(self) -> Truncation:
        """Truncation error and order of the stencil approximation."""
        return determine_stencil_truncation_error(self)


def apply_stencil(s: Stencil, f: Array, h: float = 1.0) -> Array:
    """Apply the stencil to a function *f* and a step size *h*.

    Note that only interior points are correctly computed. Any boundary
    points will contain invalid values.

    :returns: the stencil applies to the function *f*.
    """
    a = s.padded_coeffs.astype(f.dtype)
    return np.convolve(f, a, mode="same") / h**s.derivative


def determine_stencil_truncation_error(
    s: Stencil,
    *,
    atol: float | None = None,
) -> Truncation:
    r"""Determine the order and truncation error for the stencil *s*.

    .. math::

        \frac{\mathrm{d}^n\, f}{\mathrm{d}\, x^n}
        - \sum_{i \in \text{indices}} \frac{a_i}{h^n} f_i
        = c \frac{\mathrm{d}^p\, f}{\mathrm{d}\, x^p},

    where :math:`c` is the expected truncation error coefficient and :math:`p`
    is the order of the approximation.

    :arg atol: absolute tolerance to check whether the coefficient :math:`c`
        is sufficiently close to zero.
    """
    if atol is None:
        atol = float(100.0 * np.finfo(s.coeffs.dtype).eps)

    c = 0.0
    i = s.derivative
    indices = s.indices.astype(s.coeffs.dtype)
    while i < 64 and np.allclose(c, 0.0, atol=atol, rtol=0.0):
        i += 1
        c = s.coeffs @ indices**i / math.factorial(i)

    return Truncation(i - s.derivative, c)


def make_taylor_approximation(
    derivative: int,
    bounds: tuple[int, int],
    *,
    dtype: np.dtype[Any] | None = None,
) -> Stencil:
    r"""Determine a finite difference stencil by solving a linear system from the
    Taylor expansion.

    :arg derivative: integer order of the approximated derivative, e.g. :math:`3`
        for the third derivative.
    :arg bounds: inclusive left and right bounds on the stencil around a point
        :math:`x_i`. For example, ``(-1, 2)`` defines the 4 point stencil
        :math:`\{x_{i - 1}, x_i, x_{i + 1}, x_{i + 2}\}`.
    """

    if derivative <= 0:
        raise ValueError(f"Negative derivatives are invalid: {derivative}")

    if len(bounds) != 2:
        raise ValueError(f"Stencil bounds are invalid: {bounds}")

    if bounds[0] > bounds[1]:
        bounds = (bounds[1], bounds[0])

    if bounds[0] >= 0 or bounds[1] <= 0:
        raise ValueError(f"Bounds must be (smaller, bigger): {bounds}")

    if dtype is None:
        dtype = np.dtype(np.float64)
    dtype = np.dtype(dtype)

    # construct the system
    indices = np.arange(bounds[0], bounds[1] + 1)
    A = np.array(
        [indices**i / math.factorial(i) for i in range(indices.size)], dtype=dtype
    )
    b = np.zeros(indices.shape, dtype=dtype)
    b[derivative] = 1.0

    # determine coefficients
    x = np.linalg.solve(A, b)
    assert np.allclose(np.sum(x), 0.0)

    return Stencil(
        derivative=derivative,
        coeffs=x,
        indices=indices,
    )
