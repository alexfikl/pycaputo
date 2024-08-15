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

logger = logging.getLogger(pathlib.Path(__file__).stem)
logger.setLevel(logging.ERROR)
logger.addHandler(rich.logging.RichHandler())


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
    # gallery examples
    "gallery/arneodo.py": {
        "gallery-arneodo.svg",
    },
    "gallery/brusselator.py": {
        "gallery-brusselator.svg",
    },
    "gallery/chen.py": {
        "gallery-chen.svg",
    },
    "gallery/duffing.py": {
        "gallery-duffing.svg",
    },
    "gallery/genesio_tesi.py": {
        "gallery-genesio-tesi.svg",
    },
    "gallery/lorenz.py": {
        "gallery-lorenz.svg",
    },
    "gallery/liu.py": {
        "gallery-liu.svg",
    },
    "gallery/lu.py": {
        "gallery-lu.svg",
    },
    "gallery/newton-leipnik.py": {
        "gallery-newton-leipnik.svg",
    },
    "gallery/rossler.py": {
        "gallery-rossler.svg",
    },
    "gallery/van-der-pol.py": {
        "gallery-van-der-pol.svg",
    },
}


def main(outdir: pathlib.Path, scripts: list[str] | None = None) -> int:
    outdir = outdir.resolve()
    if not outdir.exists():
        logger.error("Directory does not exist: '%s'", outdir)
        return 1

    unique_scripts = set(scripts) if scripts is not None else set()
    for script in unique_scripts:
        if script not in ARTIFACTS:
            logger.error("Unknown script: '%s'.", script)
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
                logger.error("Failed to run script: '%s'", script_path, exc_info=exc)
                return 1

            for artifact in artifacts:
                path = pathlib.Path(cwd) / artifact

                for color in ("light", "dark"):
                    infile = path.parent / f"{path.stem}-{color}{path.suffix}"
                    outfile = outdir / infile.name

                    shutil.copyfile(infile, outfile)
                    logger.info("Generated '%s' from '%s'.", outfile.name, infile)

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
        logger.setLevel(logging.INFO)

    raise SystemExit(main(args.dir, scripts=args.script))
