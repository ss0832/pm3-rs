# SPDX-License-Identifier: GPL-3.0-or-later
"""Re-verify the Sparkle constants in ``tests/molecules.rs`` against a live MOPAC run.

``all_element_validation.py`` covers all sixty atom codes and writes
``ALL_ELEMENTS_RESULTS.json``. This is the narrow version: the fifteen La-Lu trivalent
Sparkles only, run against MOPAC directly, and compared with the numbers actually written
into the Rust test.

The point is the direction of the check. A constant in a test file agrees with a JSON
file that agrees with a MOPAC run that happened once; nothing in that chain notices if
the JSON is regenerated from a different fixture, or if the constant is transcribed with
a digit dropped. This closes it by going back to the executable.

Usage::

    set MOPAC_EXE=...\\mopac.exe
    python tools/oracle/sparkle_oracle.py

Exits non-zero if any constant is off by more than ``--tolerance`` from MOPAC, or if
pm3-rs is off by more than the tolerance ``tests/molecules.rs`` asserts.
"""

import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import run_mopac  # noqa: E402

# The constants as written in tests/molecules.rs. Kept here as a literal copy on purpose:
# if the two ever disagree, that is the transcription error this script exists to catch.
BAKED = {
    "La": (57, 308.8165760141),
    "Ce": (58, 455.7459808922),
    "Pr": (59, 432.5533772791),
    "Nd": (60, 165.9978745878),
    "Pm": (61, 174.2617752293),
    "Sm": (62, 69.7982119301),
    "Eu": (63, 152.6294139239),
    "Gd": (64, 61.2555949727),
    "Tb": (65, 170.2111131818),
    "Dy": (66, 162.1081459730),
    "Ho": (67, 83.6642295858),
    "Er": (68, 13.5285286309),
    "Tm": (69, 149.6781631169),
    "Yb": (70, 139.9077771803),
    "Lu": (71, 98.8177592430),
}

# all_element_validation.py's own LnF3 fixture: neutral, trigonal planar, r = 2.10 Angstrom.
# The full-precision sin(60) matters -- a rounded copy moves the answer by 1.7e-4 kcal/mol,
# which looks like a regression and is a different molecule.
R = 2.10
SIN60 = 0.8660254038


def fixture(symbol):
    y = SIN60 * R
    return [
        (symbol, 0.0, 0.0, 0.0),
        ("F", R, 0.0, 0.0),
        ("F", -0.5 * R, y, 0.0),
        ("F", -0.5 * R, -y, 0.0),
    ]


def pm3_rs_heat(cli, xyz):
    """The crate's own heat of formation, through the CLI so the shipped path is what is tested."""
    out = subprocess.run(
        [cli, "energy", xyz, "--method", "PM3", "--charge", "0", "--multiplicity", "1"],
        check=True, capture_output=True, text=True, timeout=180,
    ).stdout
    for line in out.splitlines():
        if line.startswith("Heat of formation:"):
            return float(line.split(":", 1)[1].split()[0])
    raise ValueError(f"no heat of formation in:\n{out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", default=os.path.join(ROOT, "target", "release", "pm3_rs_cli.exe"),
                        help="pm3-rs CLI binary")
    parser.add_argument("--tolerance", type=float, default=1.0e-5,
                        help="kcal/mol; the bound tests/molecules.rs asserts")
    args = parser.parse_args()

    have_cli = os.path.exists(args.cli)
    if not have_cli:
        print(f"note: {args.cli} not found; checking the constants against MOPAC only")

    print(f"{'sym':>5}{'MOPAC':>20}{'in tests/molecules.rs':>24}{'delta':>12}"
          f"{'pm3-rs':>20}{'delta':>12}")
    worst_constant = 0.0
    worst_pm3 = 0.0
    failures = []

    with tempfile.TemporaryDirectory() as tmp:
        for symbol, (_z, baked) in BAKED.items():
            xyz = os.path.join(tmp, f"{symbol}F3.xyz")
            with open(xyz, "w", encoding="ascii") as handle:
                handle.write(f"4\n{symbol}F3 sparkle\n")
                for element, x, y, z in fixture(symbol):
                    handle.write(f"{element} {x:.10f} {y:.10f} {z:.10f}\n")

            mopac = run_mopac.run(xyz, method="PM3", charge=0, mult=1, mode="gradient",
                                  extra_keywords=("NOREOR",))["heat_of_formation_kcal"]
            if mopac is None:
                failures.append(f"{symbol}: MOPAC reported no heat of formation")
                continue

            d_constant = abs(mopac - baked)
            worst_constant = max(worst_constant, d_constant)
            got = pm3_rs_heat(args.cli, xyz) if have_cli else float("nan")
            d_pm3 = abs(got - mopac) if have_cli else float("nan")
            if have_cli:
                worst_pm3 = max(worst_pm3, d_pm3)

            print(f"{symbol:>5}{mopac:>20.10f}{baked:>24.10f}{d_constant:>12.2e}"
                  f"{got:>20.10f}{d_pm3:>12.2e}")

            if d_constant > args.tolerance:
                failures.append(
                    f"{symbol}: tests/molecules.rs has {baked:.10f}, MOPAC says {mopac:.10f}"
                )
            if have_cli and d_pm3 > args.tolerance:
                failures.append(
                    f"{symbol}: pm3-rs {got:.10f} vs MOPAC {mopac:.10f}, off by {d_pm3:.3e}"
                )

    print()
    print(f"worst |MOPAC - baked constant| = {worst_constant:.3e} kcal/mol")
    if have_cli:
        print(f"worst |pm3-rs - MOPAC|         = {worst_pm3:.3e} kcal/mol")
    if failures:
        print()
        print("FAILURES:")
        for line in failures:
            print(f"  {line}")
        return 1
    print(f"all fifteen Sparkles agree with MOPAC within {args.tolerance:g} kcal/mol")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
