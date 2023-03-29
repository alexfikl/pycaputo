# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass


@dataclass(frozen=True)
class FractionalDerivative:
    """Generic type of fractional derivative.

    .. attribute:: order

        The order of the derivative.
    """

    order: float


@dataclass(frozen=True)
class RiemannLiouvilleDerivative(FractionalDerivative):
    pass


@dataclass(frozen=True)
class CaputoDerivative(FractionalDerivative):
    pass
