# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

from pycaputo.logging import get_logger
from pycaputo.typing import Array

log = get_logger(__name__)


@dataclass(frozen=True)
class Event:
    """Event issued in :func:`~pycaputo.stepping.evolve` describing the state
    of the evolution.
    """


@dataclass(frozen=True)
class StepFailed(Event):
    """Result of a failed update to time :attr:`t`."""

    t: float
    """Current time."""
    iteration: int
    """Current iteration."""
    reason: str
    """A reason on why the step failed (if available)."""

    def __str__(self) -> str:
        return f"Event failed at iteration {self.iteration}: {self.reason}"


@dataclass(frozen=True)
class StepCompleted(Event):
    """Result of a successful update to time :attr:`t`."""

    t: float
    """Current time."""
    iteration: int
    """Current iteration."""
    dt: float
    """Final time of the simulation."""
    y: Array
    """State at the time :attr:`t`."""

    eest: float
    """Relative error estimate (useful when using adaptive step size)."""
    q: float
    """Time step adaptation factor (useful when using adaptive step size)."""
    trunc: Array
    """Estimated truncation error (useful when using adaptive step size)."""

    def __str__(self) -> str:
        return f"[{self.iteration:06d}] t = {self.t:.5e} dt {self.dt:.5e}"


@dataclass(frozen=True)
class StepAccepted(StepCompleted):
    """Result of a successful update where the time step was accepted."""


@dataclass(frozen=True)
class StepRejected(StepCompleted):
    """Result of a successful update where the time step was rejected."""
