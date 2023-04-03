# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from pycaputo.utils import Array


@dataclass(frozen=True)
class Points:
    a: float
    b: float
    x: Array

    @property
    def n(self) -> int:
        return self.x.size

    @cached_property
    def dx(self) -> Array:
        return np.diff(self.x)


@dataclass(frozen=True)
class UniformPoints(Points):
    pass


def make_uniform_points(n: int, a: float = 0.0, b: float = 1.0) -> UniformPoints:
    return UniformPoints(a=a, b=b, x=np.linspace(a, b, n))
