# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import logging
import os
from typing import Optional, Union


def get_logger(
    module: str,
    level: Optional[Union[int, str]] = None,
) -> logging.Logger:
    if level is None:
        level = os.environ.get("PYCAPUTO_LOGGING_LEVEL", "INFO").upper()

    if isinstance(level, str):
        level = getattr(logging, level.upper())

    assert isinstance(level, int)

    from rich.logging import RichHandler

    logger = logging.getLogger(module)
    logger.propagate = False
    logger.setLevel(level)
    logger.addHandler(RichHandler())

    return logger
