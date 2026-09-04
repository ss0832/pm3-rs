# SPDX-License-Identifier: GPL-3.0-or-later
"""Oracle-free consistency audit of the pm3-rs derivative and density machinery.

The MOPAC sweep in `all_element_validation.py` answers "is this the same model?".
This script answers the complementary question — "is the code self-consistent?" —
using only pm3-rs itself, so it runs without MOPAC and catches the class of bugs
an external oracle can hide:

* an analytic derivative that disagrees with a finite difference of the very
  energy it claims to differentiate (an AD chain-rule slip);
* a gradient whose components do not sum to zero (a broken action/reaction pair
  in the two-center loop — invisible in the energy);
* a net torque about the centre of mass (a rotationally non-invariant term, the
  failure mode the `frame` module exists to prevent);
* an asymmetric Hessian;
* Mulliken charges that do not sum to the total charge;
* a reported dipole that is not the field derivative of the energy (`mu = dE/dF`,
  neutral systems only -- the two are about different origins for an ion).

Every check is a physical identity that must hold exactly for *any* correct
implementation, whatever the parameters are.

    python tools/oracle/internal_consistency.py [--cli PATH] [--atom-code Z]...
                                                [--limit N] [--output JSON]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import all_element_validation as av  # noqa: E402  (path set above)

BOHR_TO_ANGSTROM = av.BOHR_TO_ANGSTROM


def parse_energy_ev(text):
    match = re.search(r"^Total energy:\s*([-+0-9.eE]+)", text, re.M)
    if match is None:
        raise ValueError("pm3_rs_cli did not print a total energy")
    return float(match.group(1))


def displaced(atoms, index, axis, delta_angstrom):
    out = [list(atom) for atom in atoms]
    out[index][1 + axis] += delta_angstrom
    return [tuple(atom) for atom in out]


def energy_at(cli, directory, case, atoms, tag):
    xyz = os.path.join(directory, f"{case['name']}_{tag}.xyz")
    av.write_xyz(xyz, atoms, case["name"])
    text = av.run_cli(cli, "energy", xyz, case["charge"], case["multiplicity"])
    return parse_energy_ev(text)


def gradient_at(cli, directory, case, atoms, tag):
    xyz = os.path.join(directory, f"{case['name']}_{tag}.xyz")
    av.write_xyz(xyz, atoms, case["name"])
    text = av.run_cli(cli, "gradient", xyz, case["charge"], case["multiplicity"])
    return av.parse_gradient(text, len(atoms))


def audit_case(case, cli, directory, step_angstrom):
    atoms = case["atoms"]
    natoms = len(atoms)
    ndof = 3 * natoms

    base_energy_text = av.run_cli(
        cli,
        "energy",
        _write(directory, case, atoms, "base"),
        case["charge"],
        case["multiplicity"],
    )
    charges_text = av.run_cli(
        cli,
        "charges",
        _write(directory, case, atoms, "base"),
        case["charge"],
        case["multiplicity"],
    )
    gradient_text = av.run_cli(
        cli,
        "gradient",
        _write(directory, case, atoms, "base"),
        case["charge"],
        case["multiplicity"],
    )
    hessian_text = av.run_cli(
        cli,
        "hessian",
        _write(directory, case, atoms, "base"),
        case["charge"],
        case["multiplicity"],
    )

    analytic = av.parse_gradient(gradient_text, natoms)
    hessian = av.parse_hessian(hessian_text, ndof)
    charges, _dipole = av.parse_charges_and_dipole(charges_text)

    # --- 1. analytic gradient vs a central difference of the energy it differentiates.
    step_bohr = step_angstrom / BOHR_TO_ANGSTROM
    numeric = [[0.0] * 3 for _ in range(natoms)]
    for atom in range(natoms):
        for axis in range(3):
            plus = energy_at(
                cli, directory, case, displaced(atoms, atom, axis, step_angstrom), "p"
            )
            minus = energy_at(
                cli, directory, case, displaced(atoms, atom, axis, -step_angstrom), "m"
            )
            numeric[atom][axis] = (plus - minus) / (2.0 * step_bohr)
    gradient_error = av.max_vector_difference(analytic, numeric)

    # --- 2. analytic Hessian vs a central difference of the analytic gradient.
    hessian_numeric = [[0.0] * ndof for _ in range(ndof)]
    for column in range(ndof):
        atom, axis = divmod(column, 3)
        plus = gradient_at(
            cli, directory, case, displaced(atoms, atom, axis, step_angstrom), "hp"
        )
        minus = gradient_at(
            cli, directory, case, displaced(atoms, atom, axis, -step_angstrom), "hm"
        )
        flat_plus = [value for row in plus for value in row]
        flat_minus = [value for row in minus for value in row]
        for row in range(ndof):
            hessian_numeric[row][column] = (flat_plus[row] - flat_minus[row]) / (
                2.0 * step_bohr
            )
    hessian_error = av.max_vector_difference(hessian, hessian_numeric)

    # --- 3. translational invariance: the gradient components must sum to zero.
    translation_residual = max(
        abs(sum(analytic[atom][axis] for atom in range(natoms))) for axis in range(3)
    )

    # --- 4. rotational invariance: no net torque about the centre of geometry.
    centre = [sum(atom[1 + axis] for atom in atoms) / natoms for axis in range(3)]
    torque = [0.0, 0.0, 0.0]
    for atom_index, atom in enumerate(atoms):
        # Positions in Bohr so the torque matches the eV/Bohr gradient units.
        r = [(atom[1 + axis] - centre[axis]) / BOHR_TO_ANGSTROM for axis in range(3)]
        g = analytic[atom_index]
        torque[0] += r[1] * g[2] - r[2] * g[1]
        torque[1] += r[2] * g[0] - r[0] * g[2]
        torque[2] += r[0] * g[1] - r[1] * g[0]
    rotation_residual = max(abs(value) for value in torque)

    # --- 5. Hessian symmetry.
    symmetry_residual = max(
        abs(hessian[i][j] - hessian[j][i]) for i in range(ndof) for j in range(ndof)
    )

    # --- 6. Mulliken charges must sum to the total charge.
    charge_residual = abs(sum(charges) - case["charge"])

    # --- 7. mu = dE/dF: the reported dipole must be the field derivative of the energy.
    #
    # The sign is plus, not the minus the physics convention would give. MOPAC's FIELD= keyword
    # -- and this code, which reproduces it to eight digits -- takes E = E0 + mu.F, because the
    # reported dipole points from negative to positive, the chemistry convention, and that is
    # the direction the energy rises in. Measured on water: dE/dF_z = -1.76190 D against a
    # reported -1.76187 D.
    #
    # Restricted to neutral systems on purpose. dE/dF is the dipole about the *field's* origin,
    # which is the coordinate origin; the reported dipole is about the centre of mass, as MOPAC
    # reports it. For a neutral molecule those are the same vector and for an ion they differ by
    # Q times the separation, so a charged case would fail this for a reason that is not a defect.
    dipole_residual = None
    if abs(case["charge"]) < 1.0e-12:
        # Volts per Angstrom. Small enough that the quadratic term (the polarizability) stays
        # under the tolerance, large enough to clear the SCF's own convergence noise.
        strength = 0.002
        derivative = []
        for axis in range(3):
            energies = []
            for sign in (1.0, -1.0):
                components = [0.0, 0.0, 0.0]
                components[axis] = sign * strength
                energies.append(
                    parse_energy_ev(
                        run_cli_with_field(
                            cli,
                            "energy",
                            _write(directory, case, atoms, "field"),
                            case["charge"],
                            case["multiplicity"],
                            components,
                        )
                    )
                )
            # E is in eV and the field in V/A, so dE/dF is in e.A; the reported dipole is in
            # Debye, and 1 e.A = 4.803205 D.
            derivative.append((energies[0] - energies[1]) / (2.0 * strength) * 4.8032047)
        dipole_residual = max(abs(a - b) for a, b in zip(derivative, _dipole))

    return {
        "name": case["name"],
        "symbol": case["symbol"],
        "atom_code": case["atom_code"],
        "natoms": natoms,
        "energy_ev": parse_energy_ev(base_energy_text),
        "gradient_vs_fd_energy_ev_per_bohr": gradient_error,
        "hessian_vs_fd_gradient_ev_per_bohr2": hessian_error,
        "translation_residual_ev_per_bohr": translation_residual,
        "rotation_residual_ev": rotation_residual,
        "hessian_asymmetry_ev_per_bohr2": symmetry_residual,
        "charge_sum_residual_e": charge_residual,
        "dipole_vs_field_derivative_debye": dipole_residual,
        "fd_step_angstrom": step_angstrom,
    }


def run_cli_with_field(cli, operation, xyz, charge, multiplicity, field):
    """`av.run_cli` with a uniform external field, in volts per Angstrom."""
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
        "--field",
        ",".join(repr(value) for value in field),
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=180
    )
    return completed.stdout


def _write(directory, case, atoms, tag):
    xyz = os.path.join(directory, f"{case['name']}_{tag}.xyz")
    av.write_xyz(xyz, atoms, case["name"])
    return xyz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli")
    parser.add_argument("--output", default=os.path.join(HERE, "INTERNAL_CONSISTENCY.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--atom-code", type=int, action="append", default=[])
    parser.add_argument("--fd-step", type=float, default=1.0e-3)
    args = parser.parse_args()

    cli = av.find_cli(args.cli)
    cases = av.build_cases()
    if args.atom_code:
        requested = set(args.atom_code)
        cases = [case for case in cases if case["atom_code"] in requested]
    if args.limit:
        cases = cases[: args.limit]

    results = []
    with tempfile.TemporaryDirectory(prefix="pm3rs_consistency_") as directory:
        for index, case in enumerate(cases, 1):
            try:
                result = audit_case(case, cli, directory, args.fd_step)
                result["status"] = "ok"
                print(
                    f"[{index:02d}/{len(cases):02d}] {case['name']:<22s} "
                    f"dG={result['gradient_vs_fd_energy_ev_per_bohr']:.3e} "
                    f"dH={result['hessian_vs_fd_gradient_ev_per_bohr2']:.3e} "
                    f"trans={result['translation_residual_ev_per_bohr']:.3e} "
                    f"rot={result['rotation_residual_ev']:.3e} "
                    f"sym={result['hessian_asymmetry_ev_per_bohr2']:.3e} "
                    f"dq={result['charge_sum_residual_e']:.3e} "
                    + (
                        "mu=n/a"
                        if result['dipole_vs_field_derivative_debye'] is None
                        else f"mu={result['dipole_vs_field_derivative_debye']:.3e}"
                    )
                )
            except Exception as error:  # keep the exhaustive audit running
                result = {
                    "name": case["name"],
                    "symbol": case["symbol"],
                    "atom_code": case["atom_code"],
                    "status": "error",
                    "error": str(error),
                }
                print(f"[{index:02d}/{len(cases):02d}] {case['name']:<22s} ERROR: {error}")
            results.append(result)

    ok = [r for r in results if r["status"] == "ok"]
    summary = {
        "case_count": len(results),
        "successful_count": len(ok),
        "error_count": len(results) - len(ok),
        "fd_step_angstrom": args.fd_step,
        "max_gradient_vs_fd_energy_ev_per_bohr": max(
            (r["gradient_vs_fd_energy_ev_per_bohr"] for r in ok), default=None
        ),
        "max_hessian_vs_fd_gradient_ev_per_bohr2": max(
            (r["hessian_vs_fd_gradient_ev_per_bohr2"] for r in ok), default=None
        ),
        "max_translation_residual_ev_per_bohr": max(
            (r["translation_residual_ev_per_bohr"] for r in ok), default=None
        ),
        "max_rotation_residual_ev": max((r["rotation_residual_ev"] for r in ok), default=None),
        "max_hessian_asymmetry_ev_per_bohr2": max(
            (r["hessian_asymmetry_ev_per_bohr2"] for r in ok), default=None
        ),
        "max_dipole_vs_field_derivative_debye": max(
            (
                r["dipole_vs_field_derivative_debye"]
                for r in ok
                if r.get("dipole_vs_field_derivative_debye") is not None
            ),
            default=None,
        ),
        "max_charge_sum_residual_e": max((r["charge_sum_residual_e"] for r in ok), default=None),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "results": results}, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
