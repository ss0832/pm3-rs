#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fetch the MOPAC v23.2.5 oracle binary used by the pm3-rs validation harness.

MOPAC is not redistributed with pm3-rs. This downloads the official portable
archive from the openmopac/mopac GitHub release, verifies its SHA-256, and
unpacks it into the gitignored ``tools/oracle/mopac/`` directory — which is where
``run_mopac.py`` looks by default. Nothing is installed system-wide: the archives
are self-contained.

    python tools/oracle/fetch_mopac.py

Set ``MOPAC_EXE`` instead if you already have MOPAC v23.2.5 somewhere.

Only the **Windows** archive's SHA-256 is recorded here, because that is the one
the validation record in ``tools/oracle/PM3_VALIDATION.md`` was actually produced
against. On any other platform the download is refused unless you pass
``--sha256`` with a digest you have checked yourself against the release page.
Recording a hash nobody verified would be worse than having none: it would look
like a check while being a guess.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

VERSION = "23.2.5"
RELEASE = f"https://github.com/openmopac/mopac/releases/download/v{VERSION}"

# Asset name and the relative path of the executable inside it, per platform.
ASSETS = {
    "Windows": (f"mopac-{VERSION}-win.zip", f"mopac-{VERSION}-win/bin/mopac.exe"),
    "Linux": (f"mopac-{VERSION}-linux.tar.gz", f"mopac-{VERSION}-linux/bin/mopac"),
    "Darwin": (f"mopac-{VERSION}-mac.tar.gz", f"mopac-{VERSION}-mac/bin/mopac"),
}

# A mismatch means the download is not the archive this validation record was
# built against, so the oracle numbers would not be reproducible — refuse rather
# than proceed.
KNOWN_SHA256 = {
    f"mopac-{VERSION}-win.zip":
        "d55b19edb043b7b467863ddeaae0c7eabf6b90b7c223598b246626d180930771",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(archive: Path, destination: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
        return
    # `filter="data"` refuses absolute paths and traversal outside the target;
    # it is the default from Python 3.14 but has to be asked for before that.
    with tarfile.open(archive) as bundle:
        try:
            bundle.extractall(destination, filter="data")
        except TypeError:  # pragma: no cover - Python < 3.12
            bundle.extractall(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing extraction instead of leaving it alone",
    )
    parser.add_argument(
        "--sha256",
        help="expected SHA-256 of the archive, for a platform whose digest is not "
        "recorded here. Check it against the GitHub release page first.",
    )
    parser.add_argument(
        "--asset",
        help="override the archive filename, if the release names it differently",
    )
    arguments = parser.parse_args()

    system = platform.system()
    if system not in ASSETS:
        print(f"unsupported platform {system!r}; set MOPAC_EXE by hand", file=sys.stderr)
        return 1
    default_asset, executable_path = ASSETS[system]
    asset = arguments.asset or default_asset

    here = Path(__file__).resolve().parent
    destination = here / "mopac"
    executable = destination / executable_path

    if executable.exists() and not arguments.force:
        print(f"MOPAC {VERSION} already present at {executable}")
        print("Re-run with --force to replace it.")
        return 0

    expected = arguments.sha256 or KNOWN_SHA256.get(asset)
    if expected is None:
        print(
            f"no recorded SHA-256 for {asset}.\n"
            f"Check the digest on {RELEASE} and re-run with:\n"
            f"    python {Path(__file__).name} --sha256 <digest>\n"
            "Only the Windows archive's digest is recorded, because that is the one "
            "the validation record was produced against.",
            file=sys.stderr,
        )
        return 2

    url = f"{RELEASE}/{asset}"
    print(f"Downloading MOPAC {VERSION} from {url} ...")
    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / asset
        try:
            with urllib.request.urlopen(url, timeout=600) as response:  # noqa: S310
                with archive.open("wb") as out:
                    shutil.copyfileobj(response, out)
        except OSError as error:
            print(f"download failed: {error}", file=sys.stderr)
            return 1

        actual = sha256(archive)
        if actual.lower() != expected.lower():
            print(
                f"SHA-256 mismatch for {asset}\n"
                f"  expected {expected.lower()}\n"
                f"  actual   {actual}",
                file=sys.stderr,
            )
            return 3
        print(f"SHA-256 verified: {actual}")

        if arguments.force and destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        extract(archive, destination)

    if not executable.exists():
        print(
            f"extraction finished but {executable} is missing; the release may name "
            "its contents differently on this platform",
            file=sys.stderr,
        )
        return 4
    # The tarballs do not always carry the executable bit through every extractor.
    if system != "Windows":
        executable.chmod(executable.stat().st_mode | 0o111)

    print(f"MOPAC {VERSION} ready at {executable}")
    print("The pm3-rs oracle harness (tools/oracle/run_mopac.py) will find it automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
