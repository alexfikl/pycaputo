# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from pycaputo.fode.gallery import HodgkinHuxleyParameter
from pycaputo.utils import figure, set_recommended_matplotlib

try:
    import matplotlib  # noqa: F401

    set_recommended_matplotlib()
except ImportError as exc:
    raise SystemExit(0) from exc

param = HodgkinHuxleyParameter.from_name("HodgkinHuxley")

V = np.linspace(-110, 50, 1024)
alpha_n, alpha_m, alpha_h = param.alpha(V)
beta_n, beta_m, beta_h = param.beta(V)

# NOTE: Reproduces Figure 4 in [HodgkinHuxley1952]
with figure("hodgkin-huxley-rate-functions-n") as fig:
    ax = fig.gca()

    ax.plot(V, alpha_n, label=r"$\alpha_n$")
    ax.plot(V, beta_n, label=r"$\beta_n$")
    ax.set_xlabel("$V$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4)

# NOTE: Reproduces Figure 7 in [HodgkinHuxley1952]
mask = V < 10
with figure("hodgkin-huxley-rate-functions-m") as fig:
    ax = fig.gca()

    ax.plot(V[mask], alpha_m[mask], label=r"$\alpha_m$")
    ax.plot(V[mask], beta_m[mask], label=r"$\beta_m$")
    ax.set_xlabel("$V$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4)

# NOTE: Reproduces Figure 9 in [HodgkinHuxley1952]
mask = V < 30
with figure("hodgkin-huxley-rate-functions-h") as fig:
    ax = fig.gca()

    ax.plot(V[mask], alpha_h[mask], label=r"$\alpha_h$")
    ax.plot(V[mask], beta_h[mask], label=r"$\beta_h$")
    ax.set_xlabel("$V$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4)

# }}}
