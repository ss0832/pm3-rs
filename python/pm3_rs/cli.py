# SPDX-License-Identifier: GPL-3.0-or-later
"""Console-script entry point for the ``pm3-rs`` command.

The command a ``pip install`` puts on your path is the same one the standalone Rust executable
provides: this module hands ``sys.argv`` straight to the compiled implementation rather than
reimplementing the argument parsing in Python, so the two cannot drift apart.

Run ``pm3-rs --help`` for the usage text.
"""

from __future__ import annotations

import sys

from . import _native


def main(argv: list[str] | None = None) -> int:
    """Run the command line and return its exit code.

    Parameters
    ----------
    argv:
        Argument vector including the program name, as in :data:`sys.argv`. Defaults to
        :data:`sys.argv` itself.
    """
    return _native.cli_main(list(sys.argv if argv is None else argv))


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())