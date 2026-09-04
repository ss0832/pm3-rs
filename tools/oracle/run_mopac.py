# SPDX-License-Identifier: GPL-3.0-or-later
"""Run MOPAC v23.2.5 on a geometry and parse the .aux/.out results.

Oracle harness for pm3-rs validation. MOPAC binary location (downloaded by the
setup step, gitignored): tools/oracle/mopac/mopac-23.2.5-win/bin/mopac.exe

Usage:
    python run_mopac.py <xyz-file> [--method PM3]
        [--charge Q] [--mult M] [--mode 1scf|gradient|force] [--keep]

Prints a JSON dict with heat_of_formation_kcal, energies, charges, dipole,
gradients (kcal/mol/Å) and frequencies (cm-1) when available.

MOPAC v23.2.5 is used for plain PM3. Its executable does not expose the legacy
PM3-D3/D3H4 keywords; those correction constants are source-verified separately.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MOPAC_EXE = os.environ.get(
    "MOPAC_EXE",
    os.path.join(HERE, "mopac", "mopac-23.2.5-win", "bin", "mopac.exe"),
)


def read_xyz(path):
    with open(path) as fh:
        lines = fh.read().splitlines()
    nat = int(lines[0].split()[0])
    atoms = []
    for line in lines[2 : 2 + nat]:
        parts = line.split()
        atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms


def write_mop(path, atoms, method="PM3", charge=0, mult=1, mode="1scf", extra_keywords=()):
    keywords = [method, "PRECISE", f"CHARGE={charge}", "AUX(PRECISION=9)"]
    if mode == "1scf":
        keywords.append("1SCF")
    elif mode == "gradient":
        keywords += ["1SCF", "GRADIENTS"]
    elif mode == "force":
        keywords += ["FORCE", "LET"]
    elif mode != "optimize":
        raise ValueError("mode must be 1scf, gradient, force, or optimize")
    if mult > 1:
        keywords.append({2: "DOUBLET", 3: "TRIPLET", 4: "QUARTET", 5: "QUINTET"}.get(mult, f"MS={(mult-1)/2}"))
        keywords.append("UHF")
    keywords.extend(extra_keywords)
    with open(path, "w") as fh:
        fh.write(" ".join(keywords) + "\n")
        fh.write("pm3-rs oracle reference\n\n")
        for sym, x, y, z in atoms:
            fh.write(f"{sym:3s} {x:15.8f} 1 {y:15.8f} 1 {z:15.8f} 1\n")


AUX_VEC = re.compile(r"([A-Z_.0-9]+):([A-Z0-9/().-]*)\[(\d+)\]=\s*(.*)", re.S)


def parse_aux(path):
    """Parse MOPAC .aux into {key: float|list-of-floats}."""
    with open(path) as fh:
        text = fh.read()
    out = {}
    # Scalar entries: KEY:UNITS=value  or KEY=value
    for m in re.finditer(r"^ ([A-Z_.0-9]+)(?::([A-Z0-9/().^-]*))?=(.+)$", text, re.M):
        key, _units, val = m.group(1), m.group(2), m.group(3).strip()
        try:
            out[key] = float(val.replace("D", "E"))
        except ValueError:
            out[key] = val
    # Vector entries: KEY:UNITS[n]= v1 v2 ... (block until next KEY)
    for m in re.finditer(
        r"^ ([A-Z_.0-9]+)(?::([A-Z0-9/().^-]*))?\[(\d+)\]=\s*\n?((?:[^\n=]*\n)*?)(?=^ [A-Z#])",
        text,
        re.M,
    ):
        key, _units, _n, block = m.groups()
        vals = []
        ok = True
        for tok in block.split():
            try:
                vals.append(float(tok.replace("D", "E")))
            except ValueError:
                ok = False
                break
        if ok and vals:
            out[key] = vals
    return out


def parse_cartesian_hessian(path, aux):
    """Return the MOPAC FORCE Cartesian Hessian in eV/Bohr^2.

    MOPAC writes the lower triangle after a comment line, so the generic AUX
    vector parser intentionally does not consume it. The stored matrix is
    mass-weighted and expressed in millidynes/Angstrom.
    """
    with open(path) as fh:
        text = fh.read()
    match = re.search(r"^ HESSIAN_MATRIX:[^\n=]+\[(\d+)\]=\s*$", text, re.M)
    if match is None:
        return None
    expected = int(match.group(1))
    values = []
    for line in text[match.end() :].splitlines():
        if line.startswith(" #") or not line.strip():
            continue
        if re.match(r"^ [A-Z][A-Z_.0-9]*(?::[^=]*)?(?:\[\d+\])?=", line):
            break
        for token in line.split():
            values.append(float(token.replace("D", "E")))
    if len(values) != expected:
        raise ValueError(f"expected {expected} packed Hessian values, found {len(values)}")

    masses = aux.get("ISOTOPIC_MASSES")
    if not isinstance(masses, list) or not masses:
        raise ValueError("MOPAC FORCE output did not contain ISOTOPIC_MASSES")
    ndof = 3 * len(masses)
    if expected != ndof * (ndof + 1) // 2:
        raise ValueError("packed Hessian size does not match the atom count")

    # 1 millidyne/Angstrom = 100 N/m. Convert to eV/Angstrom^2, then to
    # eV/Bohr^2. Undo MOPAC's sqrt(m_i*m_j) mass weighting at the same time.
    bohr_to_angstrom = 0.529177210903
    mdyn_per_angstrom_to_ev_per_bohr2 = (100.0 / 16.02176634) * bohr_to_angstrom**2
    dof_masses = [mass for mass in masses for _ in range(3)]
    matrix = [[0.0] * ndof for _ in range(ndof)]
    k = 0
    for i in range(ndof):
        for j in range(i + 1):
            value = (
                values[k]
                * (dof_masses[i] * dof_masses[j]) ** 0.5
                * mdyn_per_angstrom_to_ev_per_bohr2
            )
            matrix[i][j] = value
            matrix[j][i] = value
            k += 1
    return matrix


def run(
    xyz,
    method="PM3",
    charge=0,
    mult=1,
    mode="1scf",
    keep=False,
    workdir=None,
    extra_keywords=(),
    displacements=(),
):
    atoms = read_xyz(xyz)
    for atom_index, axis, delta in displacements:
        if not 0 <= atom_index < len(atoms):
            raise ValueError(f"atom index {atom_index + 1} is outside the XYZ geometry")
        if axis not in (0, 1, 2):
            raise ValueError(f"axis {axis} is invalid")
        sym, x, y, z = atoms[atom_index]
        coord = [x, y, z]
        coord[axis] += delta
        atoms[atom_index] = (sym, *coord)
    tmp = workdir or tempfile.mkdtemp(prefix="pm3rs_oracle_")
    os.makedirs(tmp, exist_ok=True)
    name = os.path.splitext(os.path.basename(xyz))[0]
    mop = os.path.join(tmp, name + ".mop")
    write_mop(mop, atoms, method, charge, mult, mode, extra_keywords)
    subprocess.run([MOPAC_EXE, mop], check=True, capture_output=True, cwd=tmp)
    aux_path = os.path.join(tmp, name + ".aux")
    aux = parse_aux(aux_path)
    result = {
        "method": method,
        "mode": mode,
        "heat_of_formation_kcal": aux.get("HEAT_OF_FORMATION"),
        "energy_electronic_ev": aux.get("ENERGY_ELECTRONIC"),
        "energy_nuclear_ev": aux.get("ENERGY_NUCLEAR"),
        "total_energy_ev": aux.get("TOTAL_ENERGY"),
        "dipole_debye": aux.get("DIP_VEC"),
        "dipole_magnitude_debye": aux.get("DIPOLE"),
        "charges": aux.get("ATOM_CHARGES"),
        # Density-derived quantities. Energies, gradients and Hessians can all
        # agree while the density itself is subtly wrong (a misplaced one-center
        # term cancels in the energy expression but not in the populations), so
        # the reproduction audit compares these too.
        "eigenvalues_ev": aux.get("EIGENVALUES"),
        # MOPAC does not necessarily print every eigenvalue: SET_OF_MOS gives the
        # inclusive 1-based index window that EIGENVALUES actually covers (for a
        # Sparkle complex it drops the deepest levels). Comparing the raw list
        # against a full set of MOs silently skips those cases, so the window
        # travels with the eigenvalues.
        "set_of_mos": aux.get("SET_OF_MOS"),
        "mo_occupancies": aux.get("MOLECULAR_ORBITAL_OCCUPANCIES"),
        "num_electrons": aux.get("NUM_ELECTRONS"),
        "ionization_potential_ev": aux.get("IONIZATION_POTENTIAL"),
        "scf_cycles": aux.get("NUMBER_SCF_CYCLES"),
        "gradients_kcal_mol_ang": aux.get("GRADIENTS"),
        "frequencies_cm": aux.get("VIB._FREQ"),
        "hessian_ev_per_bohr2": parse_cartesian_hessian(aux_path, aux),
        "spin_sz": aux.get("SPIN_COMPONENT"),
        "aux_keys": sorted(aux.keys()),
    }
    if not keep and workdir is None:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        result["workdir"] = tmp
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("--method", default="PM3")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--mode", default="1scf", choices=["1scf", "gradient", "force", "optimize"])
    ap.add_argument("--keyword", action="append", default=[], help="additional MOPAC keyword (repeatable)")
    ap.add_argument(
        "--displace",
        action="append",
        nargs=3,
        metavar=("ATOM", "AXIS", "DELTA_ANG"),
        default=[],
        help="displace one XYZ atom before the run (atom is 1-based; axis is x/y/z)",
    )
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    axis = {"x": 0, "y": 1, "z": 2}
    displacements = []
    for atom, direction, delta in args.displace:
        direction = direction.lower()
        if direction not in axis:
            ap.error(f"invalid displacement axis {direction!r}; use x, y, or z")
        if int(atom) < 1:
            ap.error("displacement atom indices are 1-based")
        displacements.append((int(atom) - 1, axis[direction], float(delta)))
    result = run(
        args.xyz,
        args.method,
        args.charge,
        args.mult,
        args.mode,
        args.keep,
        extra_keywords=args.keyword,
        displacements=displacements,
    )
    json.dump(result, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
