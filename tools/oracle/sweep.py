# SPDX-License-Identifier: GPL-3.0-or-later
"""Broad PM3 validation sweep: compare pm3-rs vs MOPAC across element classes.

Generates simple hydrides / halides for a wide range of elements, runs both
codes at plain PM3 (1SCF), and flags any heat-of-formation mismatch. Finds
core bugs that molecule-specific tests miss.

Usage: python sweep.py            (uses pm3_rs_cli in target/release or debug)
"""
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CLI = None
for cand in (
    os.path.join(ROOT, "target", "release", "pm3_rs_cli.exe"),
    os.path.join(ROOT, "target", "debug", "pm3_rs_cli.exe"),
):
    if os.path.exists(cand):
        CLI = cand
        break
sys.path.insert(0, HERE)
import run_mopac  # noqa: E402

# (name, atoms[(sym,x,y,z)], charge, mult). Simple geometries covering many
# elements: hydrides (X-H ~ covalent) and a few halides / oxides.
def hydride(sym, n, rxh, tag=""):
    """Central atom + n hydrogens (linear/planar/tetra approx)."""
    atoms = [(sym, 0.0, 0.0, 0.0)]
    if n == 1:
        atoms.append(("H", 0.0, 0.0, rxh))
    elif n == 2:
        a = math.radians(52)
        atoms += [("H", rxh * math.sin(a), 0, rxh * math.cos(a)),
                  ("H", -rxh * math.sin(a), 0, rxh * math.cos(a))]
    elif n == 3:
        for k in range(3):
            th = math.radians(120 * k)
            atoms.append(("H", rxh * 0.94 * math.cos(th), rxh * 0.94 * math.sin(th), -0.33 * rxh))
    elif n == 4:
        d = rxh / math.sqrt(3)
        atoms += [("H", d, d, d), ("H", -d, -d, d), ("H", -d, d, -d), ("H", d, -d, -d)]
    return (f"{sym}H{n}{tag}", atoms, 0, 1)

def diatomic(a, b, r):
    return (f"{a}{b}", [(a, 0, 0, 0), (b, 0, 0, r)], 0, 1)

CASES = [
    hydride("B", 3, 1.19), hydride("Be", 2, 1.34), hydride("C", 4, 1.09),
    hydride("N", 3, 1.02), hydride("O", 2, 0.96), diatomic("F", "H", 0.92),
    hydride("Al", 3, 1.59), hydride("Si", 4, 1.48), hydride("P", 3, 1.42),
    hydride("S", 2, 1.34), diatomic("Cl", "H", 1.27),
    hydride("Ga", 3, 1.55), hydride("Ge", 4, 1.53), hydride("As", 3, 1.52),
    hydride("Se", 2, 1.46), diatomic("Br", "H", 1.41),
    hydride("Sn", 4, 1.70), hydride("Sb", 3, 1.70), hydride("Te", 2, 1.66),
    diatomic("I", "H", 1.61), hydride("In", 3, 1.75),
    hydride("Pb", 4, 1.87), hydride("Bi", 3, 1.81),
    # d-block transition metals as tetrahalides / hydrides
    diatomic("Zn", "Cl", 2.07) if False else ("ZnCl2", [("Zn", 0, 0, 0), ("Cl", 0, 0, 2.07), ("Cl", 0, 0, -2.07)], 0, 1),
    ("TiCl4", [("Ti", 0, 0, 0), ("Cl", 1.25, 1.25, 1.25), ("Cl", -1.25, -1.25, 1.25), ("Cl", -1.25, 1.25, -1.25), ("Cl", 1.25, -1.25, -1.25)], 0, 1),
    ("ScCl3", [("Sc", 0, 0, 0), ("Cl", 2.3, 0, 0), ("Cl", -1.15, 1.99, 0), ("Cl", -1.15, -1.99, 0)], 0, 1),
    ("VCl4", [("V", 0, 0, 0), ("Cl", 1.28, 1.28, 1.28), ("Cl", -1.28, -1.28, 1.28), ("Cl", -1.28, 1.28, -1.28), ("Cl", 1.28, -1.28, -1.28)], 0, 2),
    ("CrCl2", [("Cr", 0, 0, 0), ("Cl", 0, 0, 2.2), ("Cl", 0, 0, -2.2)], 0, 5),
    ("MnCl2", [("Mn", 0, 0, 0), ("Cl", 0, 0, 2.2), ("Cl", 0, 0, -2.2)], 0, 6),
    ("FeCl2", [("Fe", 0, 0, 0), ("Cl", 0, 0, 2.15), ("Cl", 0, 0, -2.15)], 0, 5),
    ("NiCl2", [("Ni", 0, 0, 0), ("Cl", 0, 0, 2.1), ("Cl", 0, 0, -2.1)], 0, 3),
    ("CuCl", [("Cu", 0, 0, 0), ("Cl", 0, 0, 2.05)], 0, 1),
    ("MoCl2", [("Mo", 0, 0, 0), ("Cl", 0, 0, 2.3), ("Cl", 0, 0, -2.3)], 0, 5),
    ("WCl2", [("W", 0, 0, 0), ("Cl", 0, 0, 2.3), ("Cl", 0, 0, -2.3)], 0, 5),
    ("AgCl", [("Ag", 0, 0, 0), ("Cl", 0, 0, 2.28)], 0, 1),
    ("HgCl2", [("Hg", 0, 0, 0), ("Cl", 0, 0, 2.25), ("Cl", 0, 0, -2.25)], 0, 1),
    # charged / radical
    ("NH4+", [("N", 0, 0, 0), ("H", 0.63, 0.63, 0.63), ("H", -0.63, -0.63, 0.63), ("H", -0.63, 0.63, -0.63), ("H", 0.63, -0.63, -0.63)], 1, 1),
    ("OH-", [("O", 0, 0, 0), ("H", 0, 0, 0.96)], -1, 1),
    ("NO", [("N", 0, 0, 0), ("O", 0, 0, 1.15)], 0, 2),
    ("O2", [("O", 0, 0, 0), ("O", 0, 0, 1.21)], 0, 3),
]


def write_xyz(path, atoms):
    with open(path, "w") as fh:
        fh.write(f"{len(atoms)}\nsweep\n")
        for s, x, y, z in atoms:
            fh.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")


def main():
    tmpdir = os.path.join(HERE, "_sweep")
    os.makedirs(tmpdir, exist_ok=True)
    worst = []
    for name, atoms, charge, mult in CASES:
        xyz = os.path.join(tmpdir, f"{name.replace('+', 'p').replace('-', 'm')}.xyz")
        write_xyz(xyz, atoms)
        try:
            mo = run_mopac.run(xyz, "PM3", charge, mult, "1scf")
            mo_hof = mo["heat_of_formation_kcal"]
        except Exception as e:
            print(f"{name:10s}  MOPAC FAILED: {e}")
            continue
        try:
            out = subprocess.run(
                [CLI, "energy", xyz, "--charge", str(charge), "--multiplicity", str(mult)],
                capture_output=True, text=True, timeout=120,
            )
            rs_hof = None
            for line in out.stdout.splitlines():
                if "Heat of formation" in line:
                    rs_hof = float(line.split(":")[1].split("kcal")[0])
            if rs_hof is None:
                print(f"{name:10s}  pm3-rs FAILED: {out.stdout.strip()[:80]} {out.stderr.strip()[:80]}")
                worst.append((name, None, mo_hof, None))
                continue
        except Exception as e:
            print(f"{name:10s}  pm3-rs ERROR: {e}")
            continue
        if mo_hof is None:
            print(f"{name:10s}  MOPAC HoF missing")
            continue
        diff = rs_hof - mo_hof
        flag = "  <-- MISMATCH" if abs(diff) > 0.1 else ""
        print(f"{name:10s} MOPAC={mo_hof:12.4f}  pm3-rs={rs_hof:12.4f}  d={diff:+9.4f}{flag}")
        worst.append((name, rs_hof, mo_hof, diff))
    print("\n=== MISMATCHES (|d| > 0.1 kcal/mol or failure) ===")
    for name, rs, mo, diff in worst:
        if rs is None or (diff is not None and abs(diff) > 0.1):
            print(f"  {name}: pm3-rs={rs} MOPAC={mo} d={diff}")


if __name__ == "__main__":
    main()
