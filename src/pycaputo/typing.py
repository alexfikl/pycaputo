# SPDX-FileCopyrightText: 2023-2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    ParamSpec,
    Protocol,
    TypeAlias,
    runtime_checkable,
)

import numpy as np
from typing_extensions import TypeAliasType, TypeIs, TypeVar

if TYPE_CHECKING:
    from dataclasses import Field

# {{{ TypeVars

# NOTE: sphinx doesn't seem to render this correctly at the moment, so it's
# written explicitly in `misc_others.rst`
P = ParamSpec("P")

T = TypeVar("T")
"""A generic invariant :class:`typing.TypeVar`."""
R = TypeVar("R")
"""A generic invariant :class:`typing.TypeVar`."""

PathLike = os.PathLike[str] | str
"""A union of types supported as paths."""


# }}}


# {{{ numbers

Integer: TypeAlias = int | np.integer[Any]
"""An alias for supported integer types."""
Float: TypeAlias = int | float | np.integer[Any] | np.floating[Any]
"""An alias for supported floating point types."""

# }}}

# {{{ numpy


# TODO: Should probably just depend on `optype`. We already pull it in through
# `scipy-stubs`. For now this is a very minimalist copy-paste from there. It
# contains parts of
#   optype/numpy/_array.py
#   optype/numpy/_scalar.py

ShapeT = TypeVar("ShapeT", bound=tuple[int, ...], default=tuple[Any, ...])
"""An invariant type alias for ``tuple[int, ...]``."""
ScalarTypeT = TypeVar("ScalarTypeT", bound=np.generic, default=Any)
"""An invariant type alias for :mod:`numpy` scalars."""

ArrayND = TypeAliasType(
    "ArrayND",
    np.ndarray[ShapeT, np.dtype[ScalarTypeT]],
    type_params=(ScalarTypeT, ShapeT),
)
"""A type alias for a shape and type generic :class:`numpy.ndarray`."""

Array0D = TypeAliasType(
    "Array0D",
    np.ndarray[tuple[()], np.dtype[ScalarTypeT]],
    type_params=(ScalarTypeT,),
)
"""A type alias for a 0-dimensional :class:`ArrayND`."""

Array1D = TypeAliasType(
    "Array1D",
    np.ndarray[tuple[int], np.dtype[ScalarTypeT]],
    type_params=(ScalarTypeT,),
)
"""A type alias for a 1-dimensional :class:`ArrayND`."""

Array2D = TypeAliasType(
    "Array2D",
    np.ndarray[tuple[int, int], np.dtype[ScalarTypeT]],
    type_params=(ScalarTypeT,),
)
"""A type alias for a 2-dimensional :class:`ArrayND`."""

Array3D = TypeAliasType(
    "Array3D",
    np.ndarray[tuple[int, int, int], np.dtype[ScalarTypeT]],
    type_params=(ScalarTypeT,),
)

# TODO: we probably want to also support complex "scalars" down the road

Scalar: TypeAlias = int | float | np.floating[Any]
"""Scalar type alias (generally a value convertible to a :class:`float`)."""
ScalarLike: TypeAlias = Scalar | Array0D[np.number[Any]]
"""A scalar-like value, which may include array of shape ``()``."""

# NOTE: these are very deprecated and should not be used
Array: TypeAlias = ArrayND[np.floating[Any]]
"""Array type alias for a floating point :class:`numpy.ndarray`."""
IntegerArray: TypeAlias = ArrayND[np.integer[Any]]
"""Array type alias for an integer :class:`numpy.ndarray`."""

# }}}


# {{{ dataclass


class DataclassInstance(Protocol):
    """Dataclass protocol from
    `typeshed <https://github.com/python/typeshed/blob/770724013de34af6f75fa444cdbb76d187b41875/stdlib/_typeshed/__init__.pyi#L329-L334>`__."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


# }}}


# {{{ callable protocols


@runtime_checkable
class ScalarFunction(Protocol):
    """A generic callable that can be evaluated at :math:`x`.

    .. automethod:: __call__
    """

    def __call__(
        self, x: ArrayND[ScalarTypeT, ShapeT], /
    ) -> ArrayND[ScalarTypeT, ShapeT]:
        """
        :arg x: a scalar or array at which to evaluate the function.
        """


@runtime_checkable
class DifferentiableScalarFunction(Protocol):
    """A :class:`ScalarFunction` that can also compute its integer order derivatives.

    .. automethod:: __call__
    """

    def __call__(
        self, x: ArrayND[ScalarTypeT, ShapeT], /, d: int = 0
    ) -> ArrayND[ScalarTypeT, ShapeT]:
        """Evaluate the function or any of its derivatives.

        :arg x: a scalar or array at which to evaluate the function.
        :arg d: order of the derivative.
        """


ArrayOrScalarFunction = (
    ArrayND[np.floating[Any]] | ScalarFunction | DifferentiableScalarFunction
)
"""A union of scalar functions."""


def is_scalar_function(f: object) -> TypeIs[ScalarFunction]:
    """A type guard for scalar functions."""
    # NOTE: do not feel inclined to use `isinstance(f, ScalarFunction)` here, since
    # it does nothing -- it only checks that the `__call__` method exists.
    return callable(f)


def is_differentiable_function(f: object) -> TypeIs[DifferentiableScalarFunction]:
    """A type guard for scalar functions."""
    return callable(f)


@runtime_checkable
class StateFunction(Protocol):
    r"""A generic callable for right-hand side functions
    :math:`\mathbf{f}(t, \mathbf{y})`.

    .. automethod:: __call__
    """

    def __call__(
        self, t: float, y: ArrayND[ScalarTypeT, ShapeT], /
    ) -> ArrayND[ScalarTypeT, ShapeT]:
        """
        :arg t: time at which to evaluate the function.
        :arg y: state vector value at which to evaluate the function.
        """


@runtime_checkable
class ScalarStateFunction(Protocol):
    """A generic callable similar to :class:`StateFunction` that returns a
    scalar.

    .. automethod:: __call__
    """

    def __call__(self, t: float, y: ArrayND[ScalarTypeT, ShapeT], /) -> float:
        """
        :arg t: time at which to evaluate the function.
        :arg y: state vector value at which to evaluate the function.
        """


StateFunctionT = TypeVar("StateFunctionT", bound=StateFunction)
"""An invariant :class:`~typing.TypeVar` bound to :class:`StateFunction`."""


# }}}
