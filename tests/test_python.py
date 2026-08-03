# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke tests for the pm3-rs Python and ASE APIs.

Run inside the maturin venv:  .venv\\Scripts\\python -m pytest tests/test_python.py
"""

import numpy as np
import pm3_rs

WATER_Z = [8, 1, 1]
WATER_XYZ = np.array([[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.24, 0.9278, 0.0]])
CH3_Z = [6, 1, 1, 1]
CH3_XYZ = np.array(
    [[0.0, 0.0, 0.0], [0.0, 1.078, 0.0], [0.9336, -0.539, 0.0], [-0.9336, -0.539, 0.0]]
)


def test_single_point_matches_mopac():
    out = pm3_rs.single_point(WATER_Z, WATER_XYZ)
    assert out["converged"]
    assert abs(out["heat_of_formation_kcal"] - (-53.2301514568744)) < 5e-3
    assert abs(sum(out["charges"])) < 1e-6
    assert out["charges"][0] < -0.3
    assert out["unrestricted"] is False


def test_charge_and_multiplicity_received():
    # An explicit +1 charge changes the electron count and the energy.
    neutral = pm3_rs.single_point(WATER_Z, WATER_XYZ, charge=0.0)
    cation = pm3_rs.single_point(WATER_Z, WATER_XYZ, charge=1.0, multiplicity=2)
    assert abs(neutral["heat_of_formation_kcal"] - cation["heat_of_formation_kcal"]) > 50.0
    # Multiplicity 2 with an odd electron count forces UHF.
    assert cation["unrestricted"] is True


def test_reference_rhf_uhf_selection():
    # Forcing UHF on a closed-shell singlet must still work and match RHF energy.
    rhf = pm3_rs.single_point(WATER_Z, WATER_XYZ, reference="rhf")
    uhf = pm3_rs.single_point(WATER_Z, WATER_XYZ, reference="uhf")
    assert rhf["unrestricted"] is False
    assert uhf["unrestricted"] is True
    assert abs(rhf["heat_of_formation_kcal"] - uhf["heat_of_formation_kcal"]) < 1e-4
    # RHF on an open shell must raise.
    try:
        pm3_rs.single_point(CH3_Z, CH3_XYZ, multiplicity=2, reference="rhf")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_gradient_and_forces_independent():
    g = pm3_rs.gradient(WATER_Z, WATER_XYZ)
    f = pm3_rs.forces(WATER_Z, WATER_XYZ)
    grad = np.asarray(g["gradient_ev_per_angstrom"])
    force = np.asarray(f["forces_ev_per_angstrom"])
    assert grad.shape == (3, 3) and force.shape == (3, 3)
    # forces == -gradient.
    assert np.allclose(force, -grad, atol=1e-9)
    # atomic-unit variants present in both.
    assert np.asarray(g["gradient_hartree_per_bohr"]).shape == (3, 3)
    assert np.asarray(f["forces_hartree_per_bohr"]).shape == (3, 3)


def test_hessian_native():
    h = pm3_rs.hessian(WATER_Z, WATER_XYZ)
    hess = np.asarray(h["hessian_hartree_per_bohr2"])
    assert hess.shape == (9, 9)
    assert h["ndof"] == 9
    # Hessian is symmetric.
    assert np.allclose(hess, hess.T, atol=1e-6)


def test_pm3_d3_family_is_available():
    base = pm3_rs.single_point(WATER_Z, WATER_XYZ, method="pm3")
    for method in ("pm3-d3", "pm3-d3h4", "pm3-d3h4x"):
        corrected = pm3_rs.single_point(WATER_Z, WATER_XYZ, method=method)
        assert corrected["converged"]
        assert np.isfinite(corrected["energy_ev"])
        assert abs(corrected["energy_ev"] - base["energy_ev"]) > 1e-10


def test_mopac_special_point_charge():
    numbers = [8, 1, 1, 104]
    positions = np.vstack([WATER_XYZ, [0.0, 0.0, 3.0]])
    out = pm3_rs.single_point(numbers, positions, charge=1.0)
    assert abs(out["heat_of_formation_kcal"] - (-47.2642956996679)) < 5e-3
    assert abs(out["charges"][3] - 1.0) < 1e-12


def test_ase_calculator_and_methods():
    from ase import Atoms
    from pm3_rs.ase import PM3

    atoms = Atoms(numbers=WATER_Z, positions=WATER_XYZ)
    atoms.calc = PM3(charge=0.0, multiplicity=1, reference="auto")
    e = atoms.get_potential_energy()  # eV
    f = atoms.get_forces()            # eV/Angstrom
    assert np.isfinite(e) and f.shape == (3, 3)
    sp = pm3_rs.single_point(WATER_Z, WATER_XYZ)
    assert abs(e - sp["energy_ev"]) < 1e-6
    # get_gradient == -forces.
    grad = atoms.calc.get_gradient(atoms)
    assert np.allclose(grad, -f, atol=1e-6)
    # get_hessian in eV/Angstrom^2, symmetric.
    hess = atoms.calc.get_hessian(atoms)
    assert hess.shape == (9, 9)
    assert np.allclose(hess, hess.T, atol=1e-4)


def test_ase_reference_and_charge():
    from ase import Atoms
    from pm3_rs.ase import PM3

    atoms = Atoms(numbers=CH3_Z, positions=CH3_XYZ)
    atoms.calc = PM3(charge=0.0, multiplicity=2, reference="uhf")
    e = atoms.get_potential_energy()
    assert np.isfinite(e)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: OK")
