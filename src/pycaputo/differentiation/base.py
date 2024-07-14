# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import singledispatch

from pycaputo.derivatives import FractionalOperator
from pycaputo.grid import Points
from pycaputo.typing import Array, ArrayOrScalarFunction, Scalar


@dataclass(frozen=True)
class DerivativeMethod(ABC):
    """A generic method used to evaluate a fractional derivative at a set of points."""

    @property
    def name(self) -> str:
        """An (non-unique) identifier for the differentiation method."""
        cls = self.__class__
        name = cls.__name__
        module = cls.__module__.split(".")[-1]
        return f"diff_{module}_{name}".lower()

    @property
    @abstractmethod
    def d(self) -> FractionalOperator:
        """The fractional operator that is being discretized by the method."""


@singledispatch
def quadrature_weights(m: DerivativeMethod, p: Points, n: int) -> Array:
    r"""Evaluate the quadrature weights for the method *m* with points *p*.

    In general, a fractional operator is an integral operator that we can write
    as

    .. math::

        D^*[f](x_n) \approx \sum_{k = 0}^n w_{n, k} f_k

    where :math:`w_{n, k}` is a row of the weight matrix that is computed by this
    function. In some cases of interest, e.g. uniform, it is sufficient to
    compute the :math:`w_{N, k}` row and perform the convolution by FFT.

    :arg p: a grid on which to compute the quadrature weights at the point
        ``p.x[n]``.
    """
    raise NotImplementedError(
        f"Cannot evaluate quadrature weights for method '{type(m).__name__}'"
    )


@singledispatch
def diffs(m: DerivativeMethod, f: ArrayOrScalarFunction, p: Points, n: int) -> Scalar:
    r"""Evaluate the fractional derivative of *f* using method *m* at the point
    *p.a* with the underlying grid.

    This function evaluates :math:`D^*[f](b)`, where :math:`f: [a, b] \to \mathbb{R}`,
    i.e. at the end of the interval. This is in contract to :func:`diff`, which
    evaluates the fractional derivative at all the points in the grid *p*.

    .. note::

        The function *f* can be provided as a callable of as an array of values
        evaluated on the grid *p*. However, not all methods support both variants
        (e.g. if they require evaluating the function at additional points).

    :arg m: method used to evaluate the derivative.
    :arg f: a simple function for which to evaluate the derivative.
    :arg p: an array of points at which to evaluate the derivative.
    """
    raise NotImplementedError(
        f"Cannot evaluate fractional derivative with method '{type(m).__name__}'"
    )


@singledispatch
def diff(m: DerivativeMethod, f: ArrayOrScalarFunction, p: Points) -> Array:
    """Evaluate the fractional derivative of *f* at *p* using method *m*.

    This function uses :func:`diffs` to evaluate the fractional derivative at
    all the points in *p*. In most cases it is a trivial wrapper.

    :arg m: method used to evaluate the derivative.
    :arg f: a simple function for which to evaluate the derivative.
    :arg p: an array of points at which to evaluate the derivative.
    """
    raise NotImplementedError(
        f"Cannot evaluate fractional derivative with method '{type(m).__name__}'"
    )
