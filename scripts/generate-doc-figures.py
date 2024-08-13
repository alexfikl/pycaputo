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
    "gallery/brusselator.py": {
        "gallery-brusselator.svg",
    },
    "gallery/duffing.py": {
        "gallery-duffing.svg",
    },
    "gallery/van-der-pol.py": {
        "gallery-van-der-pol.svg",
    },
}


def main(outdir: pathlib.Path) -> int:
    outdir = outdir.resolve()
    if not outdir.exists():
        logger.error("Directory does not exist: '%s'", outdir)
        return 1

    env = os.environ.copy()
    env.update({
        "PYCAPUTO_LOGGING_LEVEL": "ERROR",
        "PYCAPUTO_SAVEFIG": "SVG",
        "PYCAPUTO_DARK": "BOTH",
        "PYTHONPATH": str(pathlib.Path.cwd()),
    })

    for name, artifacts in ARTIFACTS.items():
        script = EXAMPLES_DIR / name

        with tempfile.TemporaryDirectory() as cwd:
            try:
                subprocess.run(  # noqa: S603
                    [sys.executable, "-O", str(script)],
                    cwd=cwd,
                    env=env,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                logger.error("Failed to run script: '%s'", script, exc_info=exc)
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
        "-q", "--quiet", action="store_true", help="only show error messages"
    )
    args = parser.parse_args()

    if not args.quiet:
        logger.setLevel(logging.INFO)

    raise SystemExit(main(args.dir))
