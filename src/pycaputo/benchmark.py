# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pycaputo.logging import get_logger

log = get_logger(__name__)


# {{{ get_basic_machine_info


@dataclass(frozen=True)
class BenchmarkMachine:
    """A short description of the machine the benchmark ran on."""

    system: str
    """The (versioned) system that the benchmark ran on."""
    python: str
    """The (versioned) Python implementation the benchmark ran on."""
    cpu: str
    """The CPU family the benchmark ran on."""
    arch: str
    """The architecture of the CPU."""

    def __str__(self) -> str:
        return (
            f"System: {self.system}\n"
            f"Python: {self.python}\n"
            f"CPU:    {self.cpu} ({self.arch})"
        )


def get_basic_machine_info(stats: dict[str, Any]) -> BenchmarkMachine:
    """Gets basic machine information from a ``pytest-benchmark`` result file.

    :arg stats: the contents of the JSON file.
    """

    machine = stats["machine_info"]
    cpu = machine["cpu"]

    return BenchmarkMachine(
        system="{} ({})".format(machine["system"], machine["release"]),
        python="{} v{}".format(
            machine["python_implementation"], machine["python_version"]
        ),
        cpu="{} {}".format(cpu["vendor_id_raw"], cpu["brand_raw"]),
        arch=cpu["arch_string_raw"],
    )


# }}}


# {{{ get benchmark results


@dataclass(frozen=True)
class BenchmarkResult:
    """Minimal information for the benchmark results.

    .. automethod:: asbxp
    """

    name: str
    """A representative name for the benchmark."""
    shortname: str
    """A short identifier for the benchmark."""

    min: float
    """The shortest time (in seconds) taken to run a round of the benchmark."""
    max: float
    """The longer time (in seconds) taken to run a round of the benchmark."""
    mean: float
    """The mean time (in seconds) over all benchmark rounds."""
    median: float
    """The median time (in seconds) over all benchmark rounds."""
    stddev: float
    """The standard deviation (in seconds) over the benchmark rounds."""
    iqr: float
    """The interquartile range (in seconds) over all benchmark rounds."""

    q1: float
    """First quartile (mainly used for plotting)."""
    q3: float
    """Third quartile (mainly used for plotting)."""

    def __str__(self) -> str:
        return f"{self.name}: {self.mean:.4f} ± {self.stddev:.4f} (seconds)"

    def asbxp(self, *, shortname: bool = False) -> dict[str, Any]:
        """
        :returns: a dictionary of keys used by :meth:`~matplotlib.axes.Axes.bxp`
            plots.
        """
        return {
            # required
            "med": self.median,
            "q1": self.q1,
            "q3": self.q3,
            "whislo": self.q1 - 1.5 * self.iqr,
            "whishi": self.q3 + 1.5 * self.iqr,
            # optional
            "mean": self.mean,
            "fliers": None,
            "cilo": None,
            "cihi": None,
            "label": self.shortname if shortname else self.name,
        }


def get_benchmark_results(stats: dict[str, Any]) -> list[BenchmarkResult]:
    """Gets basic machine information from a ``pytest-benchmark`` result file.

    :arg stats: the contents of the JSON file.
    """

    return [
        BenchmarkResult(
            name=result["param"],
            shortname="{}{:03d}".format(result["group"], i),
            min=result["stats"]["min"],
            max=result["stats"]["max"],
            mean=result["stats"]["mean"],
            median=result["stats"]["median"],
            stddev=result["stats"]["stddev"],
            iqr=result["stats"]["iqr"],
            q1=result["stats"]["q1"],
            q3=result["stats"]["q3"],
        )
        for i, result in enumerate(stats["benchmarks"])
    ]


# }}}
