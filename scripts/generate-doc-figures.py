# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import subprocess  # noqa: S404
import sys
import tempfile

import rich.logging

SCRIPT_FILENAME = pathlib.Path(__file__)

log = logging.getLogger(SCRIPT_FILENAME.stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())


EXAMPLES_DIR = pathlib.Path("examples").resolve()
ARTIFACTS = {
    # tutorial examples
    "tutorial-van-der-pol-adaptive.py": {
        "tutorial-van-der-pol-adaptive-eest.svg",
        "tutorial-van-der-pol-adaptive-solution.svg",
    },
    "tutorial-brusselator.py": {
        "tutorial-brusselator.svg",
        "tutorial-brusselator-cycle.svg",
    },
    "tutorial-caputo-l1.py": {
        "tutorial-caputo-l1.svg",
    },
    # chaos gallery examples
    "gallery/arneodo.py": {
        "gallery-arneodo.svg",
    },
    "gallery/brusselator.py": {
        "gallery-brusselator.svg",
    },
    "gallery/chen.py": {
        "gallery-chen.svg",
    },
    "gallery/chua.py": {
        "gallery-chua.svg",
    },
    "gallery/cnn.py": {
        "gallery-cnn.svg",
    },
    "gallery/duffing.py": {
        "gallery-duffing.svg",
    },
    "gallery/genesio-tesi.py": {
        "gallery-genesio-tesi.svg",
    },
    "gallery/labyrinth.py": {
        "gallery-labyrinth.svg",
    },
    "gallery/lorenz.py": {
        "gallery-lorenz.svg",
    },
    "gallery/lorenz84.py": {
        "gallery-lorenz84.svg",
    },
    "gallery/lotka-volterra.py": {
        "gallery-lotka-volterra2.svg",
    },
    "gallery/lotka-volterra3.py": {
        "gallery-lotka-volterra3.svg",
    },
    "gallery/liu.py": {
        "gallery-liu.svg",
    },
    "gallery/lu.py": {
        "gallery-lu.svg",
    },
    "gallery/ma-chen.py": {
        "gallery-ma-chen.svg",
    },
    "gallery/newton-leipnik.py": {
        "gallery-newton-leipnik.svg",
    },
    "gallery/qi.py": {
        "gallery-qi.svg",
    },
    "gallery/rossler.py": {
        "gallery-rossler.svg",
    },
    "gallery/van-der-pol.py": {
        "gallery-van-der-pol.svg",
    },
    "gallery/volta.py": {
        "gallery-volta.svg",
    },
    # neural gallery examples
    "gallery/pif.py": {
        "gallery-pif.svg",
    },
    "gallery/lif.py": {
        "gallery-lif.svg",
    },
    "gallery/adex.py": {
        "gallery-ad-ex.svg",
    },
    "gallery/fitzhugh-nagumo.py": {
        "gallery-fitzhugh-nagumo.svg",
    },
    "gallery/fitzhugh-rinzel.py": {
        "gallery-fitzhugh-rinzel.svg",
    },
    "gallery/morris-lecar.py": {
        "gallery-morris-lecar.svg",
    },
    "gallery/hindmarsh-rose2.py": {
        "gallery-hindmarsh-rose2.svg",
    },
    "gallery/hindmarsh-rose3.py": {
        "gallery-hindmarsh-rose3.svg",
    },
    "gallery/hindmarsh-rose4.py": {
        "gallery-hindmarsh-rose4.svg",
    },
    "gallery/hodgkin-huxley.py": {
        "gallery-hodgkin-huxley.svg",
    },
}


def main(outdir: pathlib.Path, scripts: list[str] | None = None) -> int:
    outdir = outdir.resolve()
    if not outdir.exists():
        log.error("Directory does not exist: '%s'", outdir)
        return 1

    unique_scripts = set(scripts) if scripts is not None else set()
    for script in unique_scripts:
        if script not in ARTIFACTS:
            log.error("Unknown script: '%s'.", script)
            return 1

    env = os.environ.copy()
    env.update({
        "PYCAPUTO_LOGGING_LEVEL": "ERROR",
        "PYCAPUTO_SAVEFIG": "SVG",
        "PYCAPUTO_DARK": "BOTH",
        "PYTHONPATH": str(pathlib.Path.cwd()),
    })

    for name, artifacts in ARTIFACTS.items():
        if scripts is not None and name not in unique_scripts:
            continue

        script_path = EXAMPLES_DIR / name

        with tempfile.TemporaryDirectory() as cwd:
            try:
                subprocess.run(  # noqa: S603
                    [sys.executable, "-O", str(script_path)],
                    cwd=cwd,
                    env=env,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                log.error("Failed to run script: '%s'", script_path, exc_info=exc)
                return 1

            for artifact in artifacts:
                path = pathlib.Path(cwd) / artifact

                for color in ("light", "dark"):
                    infile = path.parent / f"{path.stem}-{color}{path.suffix}"
                    outfile = outdir / infile.name

                    shutil.copyfile(infile, outfile)
                    log.info("Generated '%s' from '%s'.", outfile.name, infile)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir",
        default="docs",
        type=pathlib.Path,
        help="Root folder for the docs",
    )
    parser.add_argument(
        "-s",
        "--script",
        default=None,
        action="append",
        help="A script to run and get figures from (accepts multiple)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="only show error messages"
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    scripts = args.script
    if len(scripts) == 1 and scripts[0] == "all":
        scripts = list(ARTIFACTS)

    raise SystemExit(main(args.dir, scripts=scripts))
