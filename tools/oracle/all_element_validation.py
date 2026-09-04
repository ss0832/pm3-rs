# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare pm3-rs with MOPAC for every supported PM3 atom code.

Each regular PM3 element is placed in a deterministic closed-shell hydride or
noble-gas dimer so that two-center integrals, core-core repulsion, gradients,
and Hessians are exercised. La-Lu use neutral LnF3 Sparkle systems. Cb and the
+/- point atoms use dedicated fixtures. By default the reference Cartesian
Hessian is a central difference of MOPAC analytic gradients at exactly the same
input geometry; ``--hessian-source force`` also audits MOPAC FORCE output.

The MOPAC executable is selected through the MOPAC_EXE environment variable.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import run_mopac  # noqa: E402

BOHR_TO_ANGSTROM = 0.529177210903
KCAL_TO_EV = 0.0433641153087705

# Cordero/MOPAC-style covalent radii, used only to select a reproducible test
# distance. The validation compares the two programs at the identical geometry.
COVALENT_RADII = {
    1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76,
    7: 0.71, 8: 0.66, 9: 0.57, 10: 0.58, 11: 1.66, 12: 1.41,
    13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 18: 1.06,
    19: 2.03, 20: 1.76, 30: 1.22, 31: 1.22, 32: 1.20, 33: 1.19,
    34: 1.20, 35: 1.20, 36: 1.16, 37: 2.20, 38: 1.95, 48: 1.44,
    49: 1.42, 50: 1.39, 51: 1.39, 52: 1.38, 53: 1.39, 54: 1.40,
    55: 2.44, 56: 2.15, 80: 1.32, 81: 1.45, 82: 1.46, 83: 1.48,
}


def csv_rows(path):
    with open(path, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def find_cli(explicit=None):
    candidates = [explicit] if explicit else []
    candidates += [
        os.path.join(ROOT, "target-pm3", "release", "pm3_rs_cli.exe"),
        os.path.join(ROOT, "target-pm3", "debug", "pm3_rs_cli.exe"),
        os.path.join(ROOT, "target", "release", "pm3_rs_cli.exe"),
        os.path.join(ROOT, "target", "debug", "pm3_rs_cli.exe"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError("build pm3_rs_cli first or pass --cli")


def write_xyz(path, atoms, title):
    with open(path, "w", encoding="ascii") as handle:
        handle.write(f"{len(atoms)}\n{title}\n")
        for symbol, x, y, z in atoms:
            handle.write(f"{symbol:3s} {x: .10f} {y: .10f} {z: .10f}\n")


def closed_shell_hydride(symbol, valence, distance):
    """Return a compact closed-shell hydride and its charge."""
    if valence in (1, 7, 8):
        atoms = [(symbol, 0.0, 0.0, 0.0), ("H", distance, 0.0, 0.0)]
        return atoms, 1 if valence == 8 else 0
    if valence in (2, 6):
        atoms = [
            (symbol, 0.0, 0.0, 0.0),
            ("H", distance, 0.0, 0.0),
            ("H", -0.25 * distance, 0.9682458366 * distance, 0.0),
        ]
        return atoms, 0
    if valence == 3:
        atoms = [(symbol, 0.0, 0.0, 0.0)]
        for x, y in ((1.0, 0.0), (-0.5, 0.8660254038), (-0.5, -0.8660254038)):
            atoms.append(("H", x * distance, y * distance, 0.0))
        return atoms, 0
    if valence == 5:
        # A pyramidal group-15 hydride avoids the distinct planar SCF basin
        # encountered for SbH3 while remaining a deterministic closed-shell
        # two-center validation fixture.
        radial = 0.9539392014
        height = 0.3
        atoms = [(symbol, 0.0, 0.0, 0.0)]
        for x, y in ((1.0, 0.0), (-0.5, 0.8660254038), (-0.5, -0.8660254038)):
            atoms.append(
                ("H", x * radial * distance, y * radial * distance, height * distance)
            )
        return atoms, 0
    if valence == 4:
        d = distance / 3.0**0.5
        atoms = [
            (symbol, 0.0, 0.0, 0.0),
            ("H", d, d, d),
            ("H", -d, -d, d),
            ("H", -d, d, -d),
            ("H", d, -d, -d),
        ]
        return atoms, 0
    raise ValueError(f"unsupported PM3 valence count {valence} for {symbol}")


def build_cases():
    element_rows = csv_rows(os.path.join(ROOT, "src", "data", "element_data.csv"))
    element_data = {int(row["z"]): row for row in element_rows}
    pm3_rows = csv_rows(os.path.join(ROOT, "src", "data", "pm3_parameters.csv"))
    cases = []
    for row in pm3_rows:
        z = int(row["z"])
        if z > 100 or z == 87:
            continue
        symbol = row["sym"]
        radius = COVALENT_RADII.get(z, 1.45)
        distance = max(0.90, 1.05 * (radius + COVALENT_RADII[1]))
        valence = int(round(float(element_data[z]["tore"])))
        if z == 2:
            atoms = [(symbol, 0.0, 0.0, 0.0), ("H", 0.90, 0.0, 0.0)]
            charge = 1
        elif z in (10, 18, 36, 54):
            # Closed-shell noble-gas dimer. Artificial noble-gas hydrides have
            # several near-degenerate charge-transfer SCF solutions.
            atoms = [(symbol, 0.0, 0.0, 0.0), (symbol, 3.0, 0.0, 0.0)]
            charge = 0
        else:
            atoms, charge = closed_shell_hydride(symbol, valence, distance)
        cases.append(
            {
                "atom_code": z,
                "symbol": symbol,
                "kind": "element",
                "name": f"{symbol}{'2' if z in (10, 18, 36, 54) else f'H{len(atoms) - 1}'}_z{z}",
                "atoms": atoms,
                "charge": charge,
                "multiplicity": 1,
            }
        )

    for row in csv_rows(os.path.join(ROOT, "src", "data", "pm3_sparkles.csv")):
        z = int(row["z"])
        symbol = row["sym"]
        r = 2.10
        cases.append(
            {
                "atom_code": z,
                "symbol": symbol,
                "kind": "sparkle",
                "name": f"{symbol}F3_sparkle",
                "atoms": [
                    (symbol, 0.0, 0.0, 0.0),
                    ("F", r, 0.0, 0.0),
                    ("F", -0.5 * r, 0.8660254038 * r, 0.0),
                    ("F", -0.5 * r, -0.8660254038 * r, 0.0),
                ],
                "charge": 0,
                "multiplicity": 1,
            }
        )

    cases.append(
        {
            "atom_code": 102,
            "symbol": "Cb",
            "kind": "special",
            "name": "CbH_capped_bond",
            "atoms": [
                ("C", 0.0, 0.0, 0.0),
                ("Cb", 0.6276, 0.6276, 0.6276),
                ("H", -0.6276, -0.6276, 0.6276),
                ("H", -0.6276, 0.6276, -0.6276),
                ("H", 0.6276, -0.6276, -0.6276),
            ],
            "charge": 0,
            "multiplicity": 1,
        }
    )
    water = [
        ("O", 0.0, 0.0, 0.0),
        ("H", 0.9584, 0.0, 0.0),
        ("H", -0.24, 0.9278, 0.0),
    ]
    for z, symbol, charge in ((104, "+", 1), (106, "-", -1)):
        cases.append(
            {
                "atom_code": z,
                "symbol": symbol,
                "kind": "special",
                "name": f"water_point_{'plus' if charge > 0 else 'minus'}",
                "atoms": water + [(symbol, 0.0, 0.0, 3.0)],
                "charge": charge,
                "multiplicity": 1,
            }
        )
    return cases


def run_cli(cli, operation, xyz, charge, multiplicity):
    command = [
        cli,
        operation,
        xyz,
        "--charge",
        str(charge),
        "--multiplicity",
        str(multiplicity),
        "--method",
        "PM3",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
    return completed.stdout


def parse_heat(text):
    match = re.search(r"^Heat of formation:\s*([-+0-9.eE]+)", text, re.M)
    if match is None:
        raise ValueError("pm3_rs_cli did not print a heat of formation")
    return float(match.group(1))


def parse_gradient(text, natoms):
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[0].isdigit():
            rows.append([float(value) for value in fields[1:]])
    if len(rows) != natoms:
        raise ValueError(f"expected {natoms} gradient rows, found {len(rows)}")
    return rows


def parse_hessian(text, ndof):
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("# Cartesian Hessian")) + 1
    except StopIteration as error:
        raise ValueError("pm3_rs_cli did not print a Hessian") from error
    matrix = []
    for line in lines[start : start + ndof]:
        row = [float(value) for value in line.split()]
        if len(row) != ndof:
            raise ValueError("invalid pm3_rs_cli Hessian row")
        matrix.append(row)
    if len(matrix) != ndof:
        raise ValueError("incomplete pm3_rs_cli Hessian")
    return matrix


def parse_charges_and_dipole(text):
    """Mulliken charges (e) and the dipole vector (Debye) from `pm3_rs_cli charges`."""
    charges = []
    dipole = None
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0].isdigit():
            charges.append(float(fields[2]))
        elif line.startswith("Dipole:"):
            # "Dipole: x y z D; |mu| = m D"
            head = line.split(";")[0].replace("Dipole:", "").replace("D", "").split()
            dipole = [float(value) for value in head[:3]]
    if not charges or dipole is None:
        raise ValueError("pm3_rs_cli did not print charges and a dipole")
    return charges, dipole


def parse_mo_energies(text):
    """Ascending molecular-orbital energies (eV) from `pm3_rs_cli energy`."""
    match = re.search(r"^MO energies \(eV\):\s*\[(.*)\]\s*$", text, re.M)
    if match is None:
        raise ValueError("pm3_rs_cli did not print MO energies")
    body = match.group(1).strip()
    if not body:
        return []
    return [float(value) for value in body.split(",")]


def max_abs_difference(left, right):
    """Largest absolute elementwise difference of two flat sequences."""
    return max(abs(a - b) for a, b in zip(left, right))


# Orbital energies beyond this magnitude come from MOPAC's -9,999,999 eV capped-bond
# resonance sentinel rather than from anything physical.
SENTINEL_EV = 1.0e5

# MOPAC writes a 1e-12 placeholder in place of each orbital it declines to report.
MO_PADDING_EV = 1.0e-9


def align_mo_window(rust_mo, mopac_mo, set_of_mos):
    """Line up pm3-rs's MO list with the orbitals MOPAC actually reports.

    Two MOPAC conventions get in the way of a naive elementwise comparison, and both
    hide exactly the cases with the most unusual electronic structure:

    * `EIGENVALUES` is not always the complete spectrum. `SET_OF_MOS` gives the
      inclusive 1-based index range it covers, and for a Sparkle complex it starts
      above 1 — so pm3-rs's list has to be sliced to the same window.
    * For the capped bond `Cb`, MOPAC drops the two orbitals dominated by its
      -9,999,999 eV resonance sentinel and pads the list with exact zeros, while
      pm3-rs reports them at about +-2.8e6 eV. Comparing the padded entries against
      real orbitals produces meaningless multi-eV "errors" that mask the fact that
      the nine physical orbitals agree to 1e-8 eV.

    Returns `(rust_window, mopac_window)`, both ascending and of equal length, or
    `None` when the two lists cannot be aligned at all.
    """
    ordered = sorted(mopac_mo)
    rust_physical = [value for value in rust_mo if abs(value) < SENTINEL_EV]
    mopac_physical = [value for value in ordered if abs(value) < SENTINEL_EV]
    if len(mopac_physical) > len(rust_physical):
        # MOPAC pads the sentinel-dominated orbitals it dropped with a 1e-12
        # placeholder, not an exact zero, so the padding is matched by magnitude.
        trimmed = [value for value in mopac_physical if abs(value) > MO_PADDING_EV]
        if len(trimmed) == len(rust_physical):
            mopac_physical = trimmed
    if len(rust_physical) == len(mopac_physical) and rust_physical:
        return rust_physical, mopac_physical
    if len(ordered) == len(rust_mo):
        return rust_mo, ordered
    if isinstance(set_of_mos, str):
        fields = set_of_mos.split()
        if len(fields) == 2:
            try:
                first, last = int(fields[0]), int(fields[1])
            except ValueError:
                return None
            if 1 <= first <= last <= len(rust_mo) and last - first + 1 == len(ordered):
                return rust_mo[first - 1 : last], ordered
    return None


def max_vector_difference(left, right):
    return max(abs(a - b) for row_a, row_b in zip(left, right) for a, b in zip(row_a, row_b))


def mopac_fd_hessian(xyz, case, step_angstrom):
    """Central difference the MOPAC analytic gradient in the input frame."""
    natoms = len(case["atoms"])
    ndof = 3 * natoms
    matrix = [[0.0] * ndof for _ in range(ndof)]
    displacement_bohr = step_angstrom / BOHR_TO_ANGSTROM
    gradient_factor = KCAL_TO_EV * BOHR_TO_ANGSTROM
    for column in range(ndof):
        atom, axis = divmod(column, 3)
        gradients = []
        for sign in (1.0, -1.0):
            displaced = run_mopac.run(
                xyz,
                method="PM3",
                charge=case["charge"],
                mult=case["multiplicity"],
                mode="gradient",
                extra_keywords=("NOREOR",),
                displacements=((atom, axis, sign * step_angstrom),),
            )["gradients_kcal_mol_ang"]
            if displaced is None:
                raise ValueError("MOPAC displaced calculation did not report a gradient")
            gradients.append([value * gradient_factor for value in displaced])
        plus, minus = gradients
        for row in range(ndof):
            matrix[row][column] = (plus[row] - minus[row]) / (2.0 * displacement_bohr)
    # Remove the small antisymmetric component from independent SCF solves.
    for i in range(ndof):
        for j in range(i):
            value = 0.5 * (matrix[i][j] + matrix[j][i])
            matrix[i][j] = value
            matrix[j][i] = value
    return matrix


def validate_case(case, cli, directory, hessian_source, fd_step, mopac_keywords=()):
    xyz = os.path.join(directory, case["name"] + ".xyz")
    write_xyz(xyz, case["atoms"], case["name"])
    mopac = run_mopac.run(
        xyz,
        method="PM3",
        charge=case["charge"],
        mult=case["multiplicity"],
        mode="force" if hessian_source == "force" else "gradient",
        extra_keywords=("NOREOR",) + tuple(mopac_keywords),
    )
    if mopac["heat_of_formation_kcal"] is None:
        raise ValueError("MOPAC did not report a heat of formation")
    if mopac["gradients_kcal_mol_ang"] is None:
        raise ValueError("MOPAC did not report a gradient")
    if hessian_source == "force":
        mopac_hessian = mopac["hessian_ev_per_bohr2"]
        if mopac_hessian is None:
            raise ValueError("MOPAC did not report a Cartesian Hessian")
    else:
        mopac_hessian = mopac_fd_hessian(xyz, case, fd_step)

    energy_text = run_cli(cli, "energy", xyz, case["charge"], case["multiplicity"])
    gradient_text = run_cli(cli, "gradient", xyz, case["charge"], case["multiplicity"])
    hessian_text = run_cli(cli, "hessian", xyz, case["charge"], case["multiplicity"])
    charges_text = run_cli(cli, "charges", xyz, case["charge"], case["multiplicity"])
    natoms = len(case["atoms"])
    rust_gradient = parse_gradient(gradient_text, natoms)
    rust_hessian = parse_hessian(hessian_text, 3 * natoms)
    flat_mopac_gradient = mopac["gradients_kcal_mol_ang"]
    mopac_gradient = [
        [
            flat_mopac_gradient[3 * atom + axis] * KCAL_TO_EV * BOHR_TO_ANGSTROM
            for axis in range(3)
        ]
        for atom in range(natoms)
    ]

    # Density-derived comparisons. The energy expression is stationary in the
    # density, so an error in a one-center term can cancel out of the energy and
    # its derivatives while still leaving the density wrong; Mulliken charges,
    # the dipole (which adds the s-p hybrid term) and the orbital eigenvalues
    # probe the density and the Fock matrix directly.
    rust_charges, rust_dipole = parse_charges_and_dipole(charges_text)
    rust_mo = parse_mo_energies(energy_text)
    charge_error = None
    if mopac["charges"] is not None and len(mopac["charges"]) == len(rust_charges):
        charge_error = max_abs_difference(rust_charges, mopac["charges"])
    dipole_error = None
    # MOPAC is not self-consistent about the dipole origin of a charged system: its
    # 1SCF/GRADIENTS path reports the dipole about the centre of mass, its FORCE path
    # about the coordinate origin. (HeH+ at the same geometry: 2.62260 D from 1SCF,
    # 3.49218 D from FORCE — a difference of exactly 4.80320 D/(e*A) times the 0.18104 A
    # centre-of-mass offset.) pm3-rs follows the documented centre-of-mass convention, so
    # comparing against the FORCE path would compare two different definitions. The
    # finite-difference sweep is the dipole oracle; the FORCE sweep records why it abstains.
    dipole_origin_note = None
    if hessian_source == "force" and abs(case["charge"]) > 0:
        dipole_origin_note = "skipped: MOPAC FORCE reports a charged system's dipole about the coordinate origin, not the centre of mass"
    elif mopac["dipole_debye"] is not None and len(mopac["dipole_debye"]) >= 3:
        dipole_error = max_abs_difference(rust_dipole, mopac["dipole_debye"][:3])
    mo_error = None
    mo_relative_error = None
    mo_error_finite = None
    mopac_mo = mopac["eigenvalues_ev"]
    if isinstance(mopac_mo, list) and rust_mo:
        window = align_mo_window(rust_mo, mopac_mo, mopac.get("set_of_mos"))
        if window is not None:
            rust_window, mopac_window = window
            mo_error = max_abs_difference(rust_window, mopac_window)
            mo_relative_error = max(
                abs(a - b) / max(abs(a), abs(b), 1.0)
                for a, b in zip(rust_window, mopac_window)
            )
            # MOPAC's capped bond `Cb` carries a -9,999,999 eV resonance sentinel,
            # which puts one orbital eight orders of magnitude above the rest. Its
            # absolute difference is meaningless as a model-agreement measure (the
            # relative one is ~1e-9), so the physically comparable orbitals are also
            # reported on their own.
            finite = [
                (a, b)
                for a, b in zip(rust_window, mopac_window)
                if abs(a) < SENTINEL_EV and abs(b) < SENTINEL_EV
            ]
            if finite:
                mo_error_finite = max(abs(a - b) for a, b in finite)

    result = {key: value for key, value in case.items() if key != "atoms"}
    result.update(
        {
            "natoms": natoms,
            "energy_mopac_kcal": mopac["heat_of_formation_kcal"],
            "energy_pm3_rs_kcal": parse_heat(energy_text),
            "energy_abs_error_kcal": abs(parse_heat(energy_text) - mopac["heat_of_formation_kcal"]),
            "gradient_max_abs_error_ev_per_bohr": max_vector_difference(rust_gradient, mopac_gradient),
            "hessian_max_abs_error_ev_per_bohr2": max_vector_difference(
                rust_hessian, mopac_hessian
            ),
            "charge_max_abs_error_e": charge_error,
            "dipole_max_abs_error_debye": dipole_error,
            "dipole_origin_note": dipole_origin_note,
            "mo_energy_max_abs_error_ev": mo_error,
            "mo_energy_max_abs_error_ev_finite": mo_error_finite,
            "mo_energy_max_rel_error": mo_relative_error,
            "energy_rel_error": (
                abs(parse_heat(energy_text) - mopac["heat_of_formation_kcal"])
                / max(abs(mopac["heat_of_formation_kcal"]), 1.0)
            ),
            "n_mo_compared": len(mopac["eigenvalues_ev"]) if mo_error is not None else 0,
            "mopac_hessian_source": hessian_source,
            "mopac_fd_step_angstrom": fd_step if hessian_source == "fd" else None,
            "mopac_extra_keywords": list(mopac_keywords),
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli")
    parser.add_argument("--output", default=os.path.join(HERE, "ALL_ELEMENTS_RESULTS.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--atom-code", type=int, action="append", default=[])
    parser.add_argument("--hessian-source", choices=("fd", "force"), default="fd")
    parser.add_argument("--fd-step", type=float, default=1.0e-3)
    parser.add_argument(
        "--mopac-keyword",
        action="append",
        default=[],
        metavar="KEYWORD",
        help=(
            "extra MOPAC keyword (repeatable). Use CAMP to run MOPAC's Camp-King "
            "converger, which is needed wherever MOPAC's default SCF path lands in a "
            "higher fixed point than the one pm3-rs converges to."
        ),
    )
    args = parser.parse_args()
    cli = find_cli(args.cli)
    cases = build_cases()
    if args.atom_code:
        requested = set(args.atom_code)
        cases = [case for case in cases if case["atom_code"] in requested]
    if args.limit:
        cases = cases[: args.limit]

    results = []
    with tempfile.TemporaryDirectory(prefix="pm3rs_all_elements_") as directory:
        for index, case in enumerate(cases, 1):
            try:
                result = validate_case(
                    case,
                    cli,
                    directory,
                    args.hessian_source,
                    args.fd_step,
                    args.mopac_keyword,
                )
                result["status"] = "ok"
                optional = lambda value, fmt: format(value, fmt) if value is not None else "n/a"
                print(
                    f"[{index:02d}/{len(cases):02d}] {case['name']:<22s} "
                    f"dE={result['energy_abs_error_kcal']:.3e} kcal/mol "
                    f"dG={result['gradient_max_abs_error_ev_per_bohr']:.3e} eV/Bohr "
                    f"dH={result['hessian_max_abs_error_ev_per_bohr2']:.3e} eV/Bohr^2 "
                    f"dq={optional(result['charge_max_abs_error_e'], '.3e')} e "
                    f"dmu={optional(result['dipole_max_abs_error_debye'], '.3e')} D "
                    f"dEmo={optional(result['mo_energy_max_abs_error_ev'], '.3e')} eV"
                )
            except Exception as error:  # keep the exhaustive audit running
                result = {key: value for key, value in case.items() if key != "atoms"}
                result.update({"status": "error", "error": str(error)})
                print(f"[{index:02d}/{len(cases):02d}] {case['name']:<22s} ERROR: {error}")
            results.append(result)

    successful = [result for result in results if result["status"] == "ok"]
    summary = {
        "mopac_version": "23.2.5",
        "hessian_source": args.hessian_source,
        "fd_step_angstrom": args.fd_step if args.hessian_source == "fd" else None,
        "case_count": len(results),
        "successful_count": len(successful),
        "error_count": len(results) - len(successful),
        "max_energy_abs_error_kcal": max(
            (result["energy_abs_error_kcal"] for result in successful), default=None
        ),
        "max_gradient_abs_error_ev_per_bohr": max(
            (result["gradient_max_abs_error_ev_per_bohr"] for result in successful), default=None
        ),
        "max_hessian_abs_error_ev_per_bohr2": max(
            (result["hessian_max_abs_error_ev_per_bohr2"] for result in successful), default=None
        ),
        "max_charge_abs_error_e": max(
            (
                result["charge_max_abs_error_e"]
                for result in successful
                if result.get("charge_max_abs_error_e") is not None
            ),
            default=None,
        ),
        "max_dipole_abs_error_debye": max(
            (
                result["dipole_max_abs_error_debye"]
                for result in successful
                if result.get("dipole_max_abs_error_debye") is not None
            ),
            default=None,
        ),
        "max_mo_energy_abs_error_ev": max(
            (
                result["mo_energy_max_abs_error_ev"]
                for result in successful
                if result.get("mo_energy_max_abs_error_ev") is not None
            ),
            default=None,
        ),
        # Same maximum with MOPAC's -9,999,999 eV capped-bond sentinel orbitals removed:
        # this is the number that actually measures model agreement.
        "max_mo_energy_abs_error_ev_finite": max(
            (
                result["mo_energy_max_abs_error_ev_finite"]
                for result in successful
                if result.get("mo_energy_max_abs_error_ev_finite") is not None
            ),
            default=None,
        ),
        "max_energy_rel_error": max(
            (
                result["energy_rel_error"]
                for result in successful
                if result.get("energy_rel_error") is not None
            ),
            default=None,
        ),
        "mo_comparison_count": sum(
            1 for result in successful if result.get("mo_energy_max_abs_error_ev") is not None
        ),
        "mopac_extra_keywords": list(args.mopac_keyword),
    }
    payload = {"summary": summary, "results": results}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    if summary["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
