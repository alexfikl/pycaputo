# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _confmatplotlib() -> None:
    from pycaputo.utils import set_recommended_matplotlib

    set_recommended_matplotlib()
