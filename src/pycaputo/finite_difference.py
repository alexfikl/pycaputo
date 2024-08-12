# q SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any, NamedTuple

import numpy as np

from pycaputo.typing import Array, ScalarFunction


class Truncation(NamedTuple):
    """A representation of the truncation error of a :class:`DiffStencil`."""

    order: int
    """Order of the approximation."""
    error: float
    """Truncation error coefficient (without the function derivative)."""


@dataclass(frozen=True)
class DiffStencil:
    r"""Approximation of a derivative by finite difference on a uniform grid.

    .. math::

        \frac{\mathrm{d}^n\, f}{\mathrm{d}\, x^n}(x_m)
        \approx \frac{1}{h^n} \sum_{k \in \text{offsets}} a_k f_{m + k}

    where :math:`a_k` are the given coefficients :attr:`coeffs` and :math:`f_m`
    are the point function evaluations. The approximation is to an order and
    truncation error can be retrieved from :attr:`trunc`.
    """

    """Order of the derivative approximated by the stencil."""
    derivative: int
    coeffs: Array
    """Coefficients used in the stencil."""
    offsets: Array
    """Offsets around the centered :math:`0` used in the stencil."""

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
    def trunc(self) -> Truncation:
        """Truncation error and order of the stencil approximation."""
        return determine_stencil_truncation_error(self)


def apply_derivative(s: DiffStencil, f: Array, h: float = 1.0) -> Array:
    """Apply the stencil to a function *f* and a step size *h*.

    Note that only interior points are correctly computed. Any boundary
    points will contain invalid values.

    :returns: the stencil applied to the function *f*.
    """
    a = s.padded_coeffs.astype(f.dtype)
    return np.convolve(f, a, mode="same") / h**s.derivative


def apply_function_derivative(
    s: DiffStencil, f: ScalarFunction, x: Array, h: float = 1.0
) -> float:
    """Compute derivative of a function *f* at *x* with a step size *h*.

    :returns: derivative of *f*.
    """
    xi = x + s.offsets * h
    fx = f(xi)

    return float(s.coeffs * fx / h**s.derivative)


def determine_stencil_truncation_error(
    s: DiffStencil,
    *,
    atol: float | None = None,
) -> Truncation:
    r"""Determine the order and truncation error for the stencil *s*.

    .. math::

        \frac{\mathrm{d}^n\, f}{\mathrm{d}\, x^n}(x_m)
        - \frac{1}{h^n} \sum_{k \in \text{offsets}} a_k f_{m + k}
        = c \frac{\mathrm{d}^p\, f}{\mathrm{d}\, x^p}(x_m),

    where :math:`c` is the expected truncation error coefficient and :math:`p`
    is the order of the approximation. Note that we just find the first :math:`c`
    that is sufficiently close to zero, but the truncation error also depends on
    the function :math:`f`, whose higher order derivatives can cancel.

    :arg atol: absolute tolerance to check whether the coefficient :math:`c`
        is sufficiently close to zero.
    """
    if atol is None:
        atol = float(100.0 * np.finfo(s.coeffs.dtype).eps)

    c = 0.0
    i = s.derivative
    offsets = s.offsets.astype(s.coeffs.dtype)
    while i < 64 and np.allclose(c, 0.0, atol=atol, rtol=0.0):
        i += 1
        c = s.coeffs @ offsets**i / math.factorial(i)

    return Truncation(i - s.derivative, c)


def modified_wavenumber(s: DiffStencil, k: Array) -> Array:
    """Compute the modified wavenumber of the stencil *s* at each number *k*.

    :arg k: wavenumber at which to compute the derivative.
    """
    km = np.empty(k.shape, dtype=np.complex128)

    for n in range(k.shape[0]):
        exp = np.exp(1.0j * s.offsets * k[n])
        km[n] = np.sum(s.coeffs * exp)

    return km


def make_taylor_approximation(
    derivative: int,
    bounds: tuple[int, int],
    *,
    dtype: np.dtype[Any] | None = None,
) -> DiffStencil:
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

    if bounds[0] > 0 or bounds[1] < 0:
        raise ValueError(f"Bounds must be (smaller <= 0, bigger >= 0): {bounds}")

    if dtype is None:
        dtype = np.dtype(np.float64)
    dtype = np.dtype(dtype)

    # construct the system
    offsets = np.arange(bounds[0], bounds[1] + 1)
    A = np.array(
        [offsets**i / math.factorial(i) for i in range(offsets.size)], dtype=dtype
    )
    b = np.zeros(offsets.shape, dtype=dtype)
    b[derivative] = 1.0

    # determine coefficients
    x = np.linalg.solve(A, b)
    assert np.allclose(np.sum(x), 0.0)

    return DiffStencil(
        derivative=derivative,
        coeffs=x,
        offsets=offsets,
    )


def make_central_taylor_approximation(
    derivative: int,
    order: int,
    *,
    dtype: np.dtype[Any] | None = None,
) -> DiffStencil:
    r"""Construct a standard central stencil for the *derivative*-th derivative
    of order *order*.

    In general, for the :math:`m`-th derivative with accuracy order :math:`n`,
    we have

    .. math::

        2 p + 1 = 2 \left\lfloor \frac{m + 1}{2} \right\rfloor + n - 1

    Note that centered stencils can only have even orders of accuracy.
    """

    if order % 2:
        raise ValueError(f"Order is not even: '{order}'")

    nterms = 2 * math.floor((derivative + 1) / 2) - 1 + order
    p = (nterms - 1) // 2

    return make_taylor_approximation(derivative, (-p, p), dtype=dtype)
