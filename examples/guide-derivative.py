# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""This example showcases how to define a new fractional operator."""

from __future__ import annotations

from dataclasses import dataclass

from pycaputo.derivatives import FractionalOperator


# [class-definition-start]
@dataclass(frozen=True)
class PrabhakarIntegral(FractionalOperator):
    alpha: float
    r"""Parameter in the Prabhakar function :math:`E^\gamma_{\alpha, \beta}`."""
    beta: float
    r"""Parameter in the Prabhakar function :math:`E^\gamma_{\alpha, \beta}`."""
    gamma: float
    r"""Parameter in the Prabhakar function :math:`E^\gamma_{\alpha, \beta}`."""

    mu: float
    """Scaling used in the Prabhakar function argument."""
    # [class-definition-end]


d = PrabhakarIntegral(alpha=0.5, beta=0.5, gamma=0.9, mu=1.0)
