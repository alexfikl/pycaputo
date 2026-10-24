# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from pycaputo.utils import set_up_logging

# {{{ logging


@pytest.fixture(scope="session", autouse=True)
def set_up_test_logging() -> None:
    set_up_logging("INFO")


# }}}
