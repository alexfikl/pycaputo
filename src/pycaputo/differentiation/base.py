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
        this fashion. This function is mainly implemented for methods where the
        *nth* row can be computed efficiently.

    :arg p: a grid on which to compute the quadrature weights for the point ``p.x[n]``.
    :arg n: the index of the point within *p*.
    """
    raise NotImplementedError(
        f"Cannot evaluate quadrature weights for method '{type(m).__name__}' "
        "('quadrature_weights' is not implemented)"
    )


@singledispatch
def diffs(m: DerivativeMethod, f: ArrayOrScalarFunction, p: Points, n: int) -> Scalar:
    r"""Evaluate the fractional derivative of *f* using method *m* at the point
    *p.x[n]* with the underlying grid.

    This function evaluates :math:`D^*[f](x_n)`, where :math:`f: [a, b] \to \mathbb{R}`.
    This is in contract to :func:`diff`, which evaluates the fractional derivative
    at all the points in the grid *p*.

    .. note::

        The function *f* can be provided as a callable of as an array of values
        evaluated on the grid *p*. However, not all methods support both variants
        (e.g. if they require evaluating the function at additional points).

    :arg m: method used to evaluate the derivative.
    :arg f: a simple function for which to evaluate the derivative.
    :arg p: an array of points at which to evaluate the derivative.
    """
    try:
        result = diff(m, f, p)
    except NotImplementedError:
        raise NotImplementedError(
            "Cannot evaluate pointwise fractional derivative with method "
            f"'{type(m).__name__}' ('diffs' is not implemented)"
        ) from None

    from warnings import warn

    warn(
        f"'diffs' is not implemented for '{type(m).__name__}' (aka '{m.name}'). "
        "Falling back to 'diff', which is likely significantly slower! Use "
        "'diff' directly if this is acceptable.",
        stacklevel=2,
    )

    if not 0 <= n <= p.size:
        raise IndexError(f"Index 'n' out of range: 0 <= {n} < {p.size}")

    return np.array(result[n])


@singledispatch
def diff(m: DerivativeMethod, f: ArrayOrScalarFunction, p: Points) -> Array:
    """Evaluate the fractional derivative of *f* at *p* using method *m*.

    This function uses :func:`diffs` to evaluate the fractional derivative at
    all the points in *p*. In most cases it is a trivial wrapper.

    .. note::

        All methods are expected to implement this method, while
        :func:`quadrature_weights` and :func:`diffs` are optional and not
        supported by all methods.

    :arg m: method used to evaluate the derivative.
    :arg f: a simple function for which to evaluate the derivative.
    :arg p: an array of points at which to evaluate the derivative.
    """
    raise NotImplementedError(
        f"Cannot evaluate fractional derivative with method '{type(m).__name__}' "
        "('diff' is not implemented)"
    )


def quadrature_matrix(m: DerivativeMethod, p: Points, w00: float = 0.0) -> Array:
    """Evaluate the quadrature matrix for a given method *m* at points *p*.

    This requires that the method implements the :func:`quadrature_weights`. As
    fractional operator are usually convolutional, this results in a lower
    triangular matrix.

    The value of the derivative at ``p.a`` is not defined. The value of the
    matrix at ``W[0, 0]`` can be controlled using *w00*, which is set to 0 by
    default. This default choice results in a singular operator and may not
    always be desired.

    Using this matrix operator, we can evaluate the derivative equivalently as

    .. code:: python

        mat = quadrature_matrix(m, p)

        df_mat = mat @ f
        df_fun = diff(m, f, p)
        assert np.allclose(df_mat, df_fun)

    :returns: a two-dimensional array of shape ``(n, n)``, where *n* is the
        number of points in *p*.
    """

    n = p.size
    W = np.zeros((n, n))
    W[0, 0] = w00

    for i in range(1, n):
        W[i, : i + 1] = quadrature_weights(m, p, i + 1)

    return W
