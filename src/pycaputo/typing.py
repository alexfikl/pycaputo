# SPDX-FileCopyrightText: 2023-2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from dataclasses import Field
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

import numpy as np

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


if TYPE_CHECKING:
    Array = np.ndarray[Any, Any]
    Scalar = np.number[Any] | Array
else:
    Array = np.ndarray
    """Array type alias for :class:`numpy.ndarray`."""
    Scalar = np.number | Array
    """Scalar type alias (generally a value convertible to a :class:`float`)."""


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

    def __call__(self, x: Array, /) -> Array:
        """
        :arg x: a scalar or array at which to evaluate the function.
        """


@runtime_checkable
class DifferentiableScalarFunction(Protocol):
    """A :class:`ScalarFunction` that can also compute its integer order derivatives.

    .. automethod:: __call__
    """

    def __call__(self, x: Array, /, d: int = 0) -> Array:
        """Evaluate the function or any of its derivatives.

        :arg x: a scalar or array at which to evaluate the function.
        :arg d: order of the derivative.
        """


ArrayOrScalarFunction = Array | ScalarFunction | DifferentiableScalarFunction
"""A union of scalar functions."""


@runtime_checkable
class StateFunction(Protocol):
    r"""A generic callable for right-hand side functions
    :math:`\mathbf{f}(t, \mathbf{y})`.

    .. automethod:: __call__
    """

    def __call__(self, t: float, y: Array, /) -> Array:
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

    def __call__(self, t: float, y: Array, /) -> float:
        """
        :arg t: time at which to evaluate the function.
        :arg y: state vector value at which to evaluate the function.
        """


StateFunctionT = TypeVar("StateFunctionT", bound=StateFunction)
"""An invariant :class:`~typing.TypeVar` bound to :class:`StateFunction`."""


# }}}
