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


def validate_case(case, cli, directory, hessian_source, fd_step):
    xyz = os.path.join(directory, case["name"] + ".xyz")
    write_xyz(xyz, case["atoms"], case["name"])
    mopac = run_mopac.run(
        xyz,
        method="PM3",
        charge=case["charge"],
        mult=case["multiplicity"],
        mode="force" if hessian_source == "force" else "gradient",
        extra_keywords=("NOREOR",),
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
            "mopac_hessian_source": hessian_source,
            "mopac_fd_step_angstrom": fd_step if hessian_source == "fd" else None,
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
                result = validate_case(case, cli, directory, args.hessian_source, args.fd_step)
                result["status"] = "ok"
                print(
                    f"[{index:02d}/{len(cases):02d}] {case['name']:<22s} "
                    f"dE={result['energy_abs_error_kcal']:.3e} kcal/mol "
                    f"dG={result['gradient_max_abs_error_ev_per_bohr']:.3e} eV/Bohr "
                    f"dH={result['hessian_max_abs_error_ev_per_bohr2']:.3e} eV/Bohr^2"
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
