# SPDX-FileCopyrightText: 2023-2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
import sys
from dataclasses import Field
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec


# {{{ TypeVars

P = ParamSpec("P")

T = TypeVar("T")
"""A generic invariant :class:`typing.TypeVar`."""
R = TypeVar("R")
"""A generic invariant :class:`typing.TypeVar`."""
PathLike = Union[pathlib.Path, str]
"""A union of types supported as paths."""

# }}}

# {{{ numpy


if TYPE_CHECKING:
    Array = np.ndarray[Any, Any]
    Scalar = Union[np.number[Any], Array]
else:
    Array = np.ndarray
    """Array type alias for :class:`numpy.ndarray`."""
    Scalar = Union[np.number, Array]
    """Scalar type alias (generally a value convertible to a :class:`float`)."""


# }}}

# {{{ dataclass


class DataclassInstance(Protocol):
    """Dataclass protocol from
    `typeshed <https://github.com/python/typeshed/blob/770724013de34af6f75fa444cdbb76d187b41875/stdlib/_typeshed/__init__.pyi#L329-L334>`__."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


# }}}


# {{{ callable protocols


class ScalarFunction(Protocol):
    """A generic callable that can be evaluated at :math:`x`.

    .. automethod:: __call__
    """

    def __call__(self, x: Array, /) -> Array:
        """
        :arg x: a scalar or array at which to evaluate the function.
        """


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


@runtime_checkable
class DifferentiableScalarFunction(Protocol):
    """A :class:`ScalarFunction` that can also compute its integer order derivatives.

    By default no derivatives are implemented, so subclasses can handle any such
    cases.
    """

    def __call__(self, x: Array, /, d: int = 0) -> Array:
        """Evaluate the function or any of its derivatives.

        :arg x: a scalar or array at which to evaluate the function.
        :arg d: order of the derivative.
        """


StateFunctionT = TypeVar("StateFunctionT", bound=StateFunction)
"""An invariant :class:`~typing.TypeVar` bound to :class:`StateFunction`."""

ArrayOrScalarFunction = Union[Array, ScalarFunction, DifferentiableScalarFunction]
"""A union of scalar functions."""


# }}}
