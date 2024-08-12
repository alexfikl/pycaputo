# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import singledispatch

import numpy as np

from pycaputo.derivatives import FractionalOperator
from pycaputo.grid import Points
from pycaputo.typing import Array, ArrayOrScalarFunction, Scalar


class FunctionCallableError(TypeError):
    """An error raised by :func:`diffs` or :func:`diff` when the argument provided
    for evaluation must be a callable
    (satisfy :class:`~pycaputo.typing.ScalarFunction`).
    """


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

    .. note::

        Not all methods can compute the weights for a single evaluation in
        this fashion. This function should be implemented for methods where the
        *nth* row of :math:`w` can be computed efficiently. Other methods may
        implement :func:`differentiation_matrix` instead.

    The weights can also be constructed using :func:`differentiation_matrix`, if
    implemented. For a safe function with this fallback, see
    :func:`~pycaputo.differentiation.quadrature_weights_fallback`.

    :arg p: a grid on which to compute the quadrature weights for the point ``p.x[n]``.
    :arg n: the index of the point within *p*.
    :returns: an array of quadrature weights of shape ``(n,)``.
    """

    raise NotImplementedError(
        f"Cannot evaluate quadrature weights for method '{type(m).__name__}' "
        "('quadrature_weights' is not implemented)"
    )


@singledispatch
def differentiation_matrix(m: DerivativeMethod, p: Points) -> Array:
    """Evaluate the differentiation matrix for the method *m* at points *p*.

    Without additional knowledge, the value of the derivative at *p.a* is not
    known. Therefore, the first row of the matrix should not be used in computation.
    By default, this function is implemented in terms of :func:`quadrature_weights`

    :returns: a two-dimensional array of shape ``(n, n)``, where *n* is the
        number of points in *p*.
    """
    n = p.size
    W = np.zeros((n, n), dtype=p.dtype)

    # NOTE: make it very obvious that the first row is garbage
    W[0, :] = np.nan

    try:
        for i in range(1, W.shape[0]):
            w = quadrature_weights(m, p, i)
            W[i, : w.size] = w
    except NotImplementedError:
        raise NotImplementedError(
            f"Cannot evaluate differentiation matrix for method '{type(m).__name__}' "
            "('differentiation_matrix' is not implemented)"
        ) from None

    return W


@singledispatch
def diffs(m: DerivativeMethod, f: ArrayOrScalarFunction, p: Points, n: int) -> Scalar:
    r"""Evaluate the fractional derivative of *f* using method *m* at the point
    *p.x[n]* with the underlying grid.

    This function evaluates :math:`D^*[f](x_n)`, where :math:`f: [a, b] \to \mathbb{R}`.
    This is in contract to :func:`diff`, which evaluates the fractional derivative
    at all the points in the grid *p*.

    The function *f* can be provided as a callable of as an array of values
    evaluated on the grid *p*. However, not all methods support both variants
    (e.g. if they require evaluating the function at additional points).

    The function values can also be obtained from :func:`diff` directly, if
    implemented. For a safe function with this fallback, see
    :func:`~pycaputo.differentiation.diffs_fallback`.

    :arg m: method used to evaluate the derivative.
    :arg f: a simple function for which to evaluate the derivative.
    :arg p: an array of points at which to evaluate the derivative.
    :returns: a scalar representing the fractional derivative approximation at
        the *nth* point in *p*.
    """
    raise NotImplementedError(
        "Cannot evaluate pointwise fractional derivative with method "
        f"'{type(m).__name__}' ('diffs' is not implemented)"
    )


@singledispatch
def diff(m: DerivativeMethod, f: ArrayOrScalarFunction, p: Points) -> Array:
    """Evaluate the fractional derivative of *f* at *p* using method *m*.

    By default, this function is implemented in terms of :func:`diffs`.

    :arg f: an array or function for which to evaluate the derivative.
    :arg p: a set points at which to evaluate the derivative.
    :returns: an array of shape ``(n,)``, where *n* is the number of points in *p*,
        that contains the evaluation of the fractional derivative at each point.
    """
    n = 1

    try:
        df1 = np.array(diffs(m, f, p, n + 1))
    except NotImplementedError:
        raise NotImplementedError(
            f"Cannot evaluate fractional derivative with method '{type(m).__name__}' "
            "('diff' is not implemented)"
        ) from None

    df = np.empty((p.size, *df1.shape), dtype=df1.dtype)
    df[0] = np.nan
    df[1] = df1

    for n in range(2, df.size):
        df[n] = diffs(m, f, p, n + 1)

    return df
