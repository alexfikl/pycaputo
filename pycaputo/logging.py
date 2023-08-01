# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import os

import rich


def stringify_table(table: rich.table.Table) -> str:
    """Stringify a rich table."""
    import io

    from rich.console import Console

    file = io.StringIO()
    console = Console(file=file)
    console.print(table)

    return str(file.getvalue())


def get_logger(
    module: str,
    level: int | str | None = None,
) -> logging.Logger:
    """Create a new logging for the module *module*.

    The logger is created using a :class:`rich.logging.RichHandler` for fancy
    highlighting. The ``NO_COLOR`` environment variable can be used to
    disable colors.

    :arg module: a name for the module to create a logger for.
    :arg level: if *None*, the default value is taken to from the
        ``PYCAPUTO_LOGGING_LEVEL`` environment variable and falls back to the
        ``INFO`` level if it does not exist (see :mod:`logging`).
    """
    if level is None:
        level = os.environ.get("PYCAPUTO_LOGGING_LEVEL", "INFO").upper()

    if isinstance(level, str):
        level = getattr(logging, level.upper())

    assert isinstance(level, int)

    logger = logging.getLogger(module)
    logger.setLevel(level)

    from rich.highlighter import NullHighlighter
    from rich.logging import RichHandler

    no_color = "NO_COLOR" in os.environ
    handler = RichHandler(
        level,
        show_time=True,
        omit_repeated_times=False,
        show_level=True,
        show_path=True,
        highlighter=NullHighlighter() if no_color else None,
        markup=True,
    )

    logger.addHandler(handler)

    return logger
