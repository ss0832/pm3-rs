# SPDX-License-Identifier: GPL-3.0-or-later
"""**Documented-API conformance tests** for `docs/python-api.md` and `README.md`.

Each test transcribes a documented example or a stated guarantee and checks it
holds: the function names, the returned dict keys, the units and their
conversions, the argument names, and the ASE calculator surface. If a doc
example stops working, or a documented key is renamed, a test here fails.

Numerical *values* are covered by the MOPAC oracle regressions (Rust
`tests/molecules.rs`, plus the two oracle checks kept here); this file covers
the *surface*.

Run inside the maturin venv::

    maturin develop --release --features python
    python -m pytest tests/test_python_api.py
"""

import math

import numpy as np
import pytest

import pm3_rs
from pm3_rs import native

WATER_Z = [8, 1, 1]
# The frozen geometry used by the MOPAC v23.2.5 oracle (tools/oracle/PM3_VALIDATION.md).
WATER_R = np.array([[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.24, 0.9278, 0.0]])
DISTORTED_R = np.array([[0.0, 0.0, 0.0], [1.02, 0.05, 0.0], [-0.28, 0.96, 0.10]])
ORACLE_HOF = -53.2301514568744
ORACLE_HOF_OPTIMIZED = -53.4330121104622
# MOPAC FORCE at the pm3-rs optimized minimum.
ORACLE_FREQUENCIES = (1743.46, 3868.68, 3989.81)

HARTREE_EV = 27.211386245988
BOHR_ANG = 0.529177210903
DEBYE_TO_E_ANGSTROM = 0.2081943

ase = pytest.importorskip("ase", reason="the ASE calculator tests need ASE")


# ---------------------------------------------------------------------------
# docs/python-api.md — top-level re-exports
# ---------------------------------------------------------------------------


def test_documented_top_level_reexports():
    """"The top-level module re-exports the native functions for convenience."""
    for name in ("single_point", "gradient", "forces", "optimize", "frequencies", "hessian"):
        assert name in pm3_rs.__all__
        assert getattr(pm3_rs, name) is getattr(native, name)


# ---------------------------------------------------------------------------
# docs/python-api.md — pm3_rs.native
# ---------------------------------------------------------------------------


def test_single_point_returns_documented_keys_and_units():
    sp = native.single_point(WATER_Z, WATER_R)
    documented = {
        "energy_hartree", "energy_ev", "heat_of_formation_kcal", "electronic_ev",
        "core_ev", "charges", "dipole_debye", "homo_ev", "lumo_ev",
        "converged", "unrestricted",
    }
    assert documented <= set(sp), f"missing: {documented - set(sp)}"
    # "energies are returned in Hartree (atomic units), with convenience eV …"
    assert abs(sp["energy_hartree"] * HARTREE_EV - sp["energy_ev"]) < 1e-9
    assert abs(sp["energy_ev"] - (sp["electronic_ev"] + sp["core_ev"])) < 1e-8
    assert abs(sp["heat_of_formation_kcal"] - ORACLE_HOF) < 5e-3
    # "charges (Mulliken, e)" sum to the total charge; "dipole_debye ([x, y, z])".
    assert len(sp["charges"]) == 3 and abs(sum(sp["charges"])) < 1e-8
    assert sp["charges"][0] < 0 < sp["charges"][1]
    assert len(sp["dipole_debye"]) == 3
    assert sp["converged"] is True and sp["unrestricted"] is False
    assert sp["homo_ev"] < sp["lumo_ev"]


def test_documented_common_arguments():
    """The `charge` / `multiplicity` / `reference` / `method` table."""
    neutral = native.single_point(WATER_Z, WATER_R, charge=0, multiplicity=1)
    cation = native.single_point(WATER_Z, WATER_R, charge=1, multiplicity=2)
    assert abs(sum(cation["charges"]) - 1.0) < 1e-8
    assert cation["unrestricted"] is True and neutral["unrestricted"] is False

    # reference: "auto" (default) | "rhf" | "uhf"
    assert native.single_point(WATER_Z, WATER_R, reference="rhf")["unrestricted"] is False
    forced = native.single_point(WATER_Z, WATER_R, reference="uhf")
    assert forced["unrestricted"] is True
    assert abs(forced["energy_ev"] - neutral["energy_ev"]) < 1e-5

    # method: the four documented variants.
    energy = {
        m: native.single_point(WATER_Z, WATER_R, method=m)["energy_ev"]
        for m in ("pm3", "pm3-d3", "pm3-d3h4", "pm3-d3h4x")
    }
    assert energy["pm3-d3"] < energy["pm3"], "D3 dispersion must be attractive"
    assert energy["pm3-d3h4"] != energy["pm3-d3"]
    # No halogens in water, so the X term must contribute exactly zero.
    assert energy["pm3-d3h4x"] == energy["pm3-d3h4"]


def test_inputs_accept_lists_and_arrays():
    a = native.single_point(WATER_Z, WATER_R)
    b = native.single_point(np.asarray(WATER_Z), WATER_R.tolist())
    assert abs(a["energy_ev"] - b["energy_ev"]) < 1e-12


def test_gradient_documented_keys_and_unit_conversion():
    g = native.gradient(WATER_Z, DISTORTED_R)
    for key in ("energy_hartree", "energy_ev", "heat_of_formation_kcal",
                "gradient_hartree_per_bohr", "gradient_ev_per_angstrom"):
        assert key in g
    au = np.asarray(g["gradient_hartree_per_bohr"])
    ev = np.asarray(g["gradient_ev_per_angstrom"])
    assert au.shape == (3, 3) and ev.shape == (3, 3)
    assert np.allclose(ev, au * (HARTREE_EV / BOHR_ANG), atol=1e-9)
    # No net force on the molecule (translational invariance).
    assert np.allclose(au.sum(axis=0), 0.0, atol=1e-8)
    assert np.abs(au).max() > 1e-4, "distorted water must have a real gradient"


def test_forces_are_minus_gradient():
    """"Forces = −dE/dR"."""
    g = native.gradient(WATER_Z, DISTORTED_R)
    f = native.forces(WATER_Z, DISTORTED_R)
    for key in ("energy_hartree", "energy_ev", "heat_of_formation_kcal",
                "forces_hartree_per_bohr", "forces_ev_per_angstrom"):
        assert key in f
    assert np.allclose(np.asarray(f["forces_hartree_per_bohr"]),
                       -np.asarray(g["gradient_hartree_per_bohr"]), atol=1e-14)
    assert np.allclose(np.asarray(f["forces_ev_per_angstrom"]),
                       -np.asarray(g["gradient_ev_per_angstrom"]), atol=1e-12)


def test_gradient_is_analytic():
    """"Analytic Hellmann–Feynman nuclear gradient" — check it against a finite
    difference of the energy the same API reports."""
    au = np.asarray(native.gradient(WATER_Z, DISTORTED_R)["gradient_hartree_per_bohr"])
    step = 1e-4  # Å
    worst = 0.0
    for atom in range(3):
        for k in range(3):
            plus, minus = DISTORTED_R.copy(), DISTORTED_R.copy()
            plus[atom, k] += step
            minus[atom, k] -= step
            difference = (
                native.single_point(WATER_Z, plus)["energy_hartree"]
                - native.single_point(WATER_Z, minus)["energy_hartree"]
            ) / (2 * step) * BOHR_ANG
            worst = max(worst, abs(difference - au[atom, k]))
    assert worst < 1e-6, f"max |analytic - numeric| = {worst:.3e}"


def test_optimize_documented_keys():
    start = native.single_point(WATER_Z, WATER_R)
    opt = native.optimize(WATER_Z, WATER_R)
    for key in ("positions_angstrom", "energy_hartree", "heat_of_formation_kcal",
                "converged", "iterations"):
        assert key in opt
    positions = np.asarray(opt["positions_angstrom"])
    assert positions.shape == (3, 3)
    assert opt["converged"] is True and opt["iterations"] > 0
    assert opt["energy_hartree"] < start["energy_hartree"]
    assert abs(opt["heat_of_formation_kcal"] - ORACLE_HOF_OPTIMIZED) < 5e-3
    residual = np.asarray(native.gradient(WATER_Z, positions)["gradient_hartree_per_bohr"])
    assert np.abs(residual).max() < 1e-4


def test_frequencies_documented_keys_and_ordering():
    opt = native.optimize(WATER_Z, WATER_R)
    result = native.frequencies(WATER_Z, opt["positions_angstrom"])
    assert {"frequencies_cm", "eigenvalues"} <= set(result)
    frequencies = np.asarray(result["frequencies_cm"])
    assert frequencies.shape == (9,)
    assert np.asarray(result["eigenvalues"]).shape == (9,)
    # "ascending; negatives are imaginary"
    assert np.all(np.diff(frequencies) >= -1e-9)
    # 6 translations/rotations near zero, then the 3 real vibrations.
    assert np.all(np.abs(frequencies[:6]) < 50.0), frequencies[:6]
    for got, want in zip(frequencies[6:], ORACLE_FREQUENCIES):
        assert abs(got - want) < 3.0, f"{got} vs MOPAC {want}"


def test_hessian_documented_keys_and_symmetry():
    h = native.hessian(WATER_Z, WATER_R)
    assert {"hessian_hartree_per_bohr2", "ndof"} <= set(h)
    assert h["ndof"] == 9
    matrix = np.asarray(h["hessian_hartree_per_bohr2"])
    assert matrix.shape == (9, 9)
    assert np.allclose(matrix, matrix.T, atol=1e-8)
    # Cross-check against a finite difference of the analytic gradient.
    step = 1e-3
    fd = np.zeros((9, 9))
    for j in range(9):
        plus, minus = WATER_R.copy(), WATER_R.copy()
        plus[j // 3, j % 3] += step
        minus[j // 3, j % 3] -= step
        gp = np.asarray(native.gradient(WATER_Z, plus)["gradient_hartree_per_bohr"]).reshape(-1)
        gm = np.asarray(native.gradient(WATER_Z, minus)["gradient_hartree_per_bohr"]).reshape(-1)
        fd[:, j] = (gp - gm) / (2 * step) * BOHR_ANG
    assert np.abs(matrix - 0.5 * (fd + fd.T)).max() < 5e-4


def test_documented_native_example_block():
    """The `### Example` block of docs/python-api.md, transcribed."""
    numbers = [8, 1, 1]  # water: O, H, H
    positions = np.array([[0.00, 0.00, 0.00],
                          [0.96, 0.00, 0.00],
                          [-0.24, 0.93, 0.00]])

    sp = native.single_point(numbers, positions)
    assert math.isfinite(sp["heat_of_formation_kcal"])

    # PM3-D3H4 optimization, then frequencies at the minimum
    opt = native.optimize(numbers, positions, method="pm3-d3h4")
    freqs = native.frequencies(numbers, opt["positions_angstrom"], method="pm3-d3h4")
    assert len(freqs["frequencies_cm"]) == 9

    # A cation: NH4+ is a closed-shell singlet.
    nh4_positions = np.array([[0.0, 0.0, 0.0],
                              [0.63, 0.63, 0.63],
                              [-0.63, -0.63, 0.63],
                              [-0.63, 0.63, -0.63],
                              [0.63, -0.63, -0.63]])
    g = native.gradient([7, 1, 1, 1, 1], nh4_positions, charge=1, multiplicity=1)
    assert np.asarray(g["gradient_hartree_per_bohr"]).shape == (5, 3)


def test_readme_python_quickstart():
    numbers = [8, 1, 1]
    positions = np.array([[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.24, 0.9278, 0.0]])
    result = pm3_rs.single_point(numbers, positions, method="pm3")
    gradient = pm3_rs.gradient(numbers, positions, method="pm3-d3h4")
    assert abs(result["heat_of_formation_kcal"] - ORACLE_HOF) < 5e-3
    assert np.asarray(gradient["gradient_ev_per_angstrom"]).shape == (3, 3)


def test_invalid_input_raises_rather_than_aborting():
    from ase import Atoms  # noqa: F401  (kept local; this test needs no ASE object)

    cases = [
        lambda: native.single_point([22, 17], [[0, 0, 0], [0, 0, 2.2]]),          # no PM3 Ti
        lambda: native.single_point(WATER_Z, WATER_R, method="pm6"),              # unknown method
        lambda: native.single_point(WATER_Z, WATER_R, reference="rohf"),          # unknown reference
        lambda: native.single_point(WATER_Z, WATER_R, charge=1, multiplicity=1),  # parity mismatch
    ]
    for case in cases:
        with pytest.raises(Exception) as excinfo:
            case()
        assert str(excinfo.value), "error must carry a message"


# ---------------------------------------------------------------------------
# docs/python-api.md — pm3_rs.ase.PM3
# ---------------------------------------------------------------------------


def test_ase_calculator_documented_surface():
    from ase import Atoms
    from pm3_rs.ase import PM3

    assert PM3.implemented_properties == [
        "energy", "forces", "charges", "dipole", "heat_of_formation_kcal", "hessian"
    ]
    calc = PM3(charge=0, multiplicity=1, reference="auto", method="pm3")
    assert (calc.charge, calc.multiplicity, calc.reference, calc.method) == (0, 1, "auto", "pm3")

    atoms = Atoms(numbers=WATER_Z, positions=WATER_R)
    atoms.calc = calc
    sp = native.single_point(WATER_Z, WATER_R)
    f_ref = native.forces(WATER_Z, WATER_R)

    # ASE units throughout: eV, eV/Å, e, e·Å.
    assert abs(atoms.get_potential_energy() - sp["energy_ev"]) < 1e-9
    assert np.allclose(atoms.get_forces(), np.asarray(f_ref["forces_ev_per_angstrom"]), atol=1e-9)
    assert np.allclose(atoms.get_charges(), np.asarray(sp["charges"]), atol=1e-12)
    assert np.allclose(atoms.get_dipole_moment(),
                       np.asarray(sp["dipole_debye"]) * DEBYE_TO_E_ANGSTROM, atol=1e-9)
    assert abs(calc.get_property("heat_of_formation_kcal", atoms) - ORACLE_HOF) < 5e-3
    # "get_gradient … the negative of forces"
    assert np.allclose(calc.get_gradient(atoms), -atoms.get_forces(), atol=1e-12)


def test_ase_hessian_is_lazy_then_cached():
    """"The Hessian is lazy: … never as part of an energy/forces cycle, and then
    cached like any other ASE property"."""
    from ase import Atoms
    from pm3_rs.ase import PM3

    atoms = Atoms(numbers=WATER_Z, positions=WATER_R)
    atoms.calc = PM3()
    atoms.get_potential_energy()
    atoms.get_forces()
    assert "hessian" not in atoms.calc.results, "Hessian must not be built eagerly"

    hessian = atoms.calc.get_hessian()
    assert hessian.shape == (9, 9)
    assert np.allclose(hessian, hessian.T, atol=1e-6)
    assert "hessian" in atoms.calc.results, "Hessian must be cached after a request"
    # Hartree/Bohr² → eV/Å².
    au = np.asarray(native.hessian(WATER_Z, WATER_R)["hessian_hartree_per_bohr2"])
    assert np.allclose(hessian, au * (HARTREE_EV / BOHR_ANG**2), atol=1e-8)


def test_ase_get_frequencies():
    from ase import Atoms
    from pm3_rs.ase import PM3

    opt = native.optimize(WATER_Z, WATER_R)
    atoms = Atoms(numbers=WATER_Z, positions=np.asarray(opt["positions_angstrom"]))
    atoms.calc = PM3()
    # Documented signature is `get_frequencies(atoms=None)`; both spellings work
    # once the calculator is bound, and passing `atoms` works with no prior run.
    explicit = atoms.calc.get_frequencies(atoms)
    atoms.get_potential_energy()
    implicit = atoms.calc.get_frequencies()
    assert explicit.shape == (9,)
    assert np.allclose(explicit, implicit, atol=1e-9)
    for got, want in zip(explicit[6:], ORACLE_FREQUENCIES):
        assert abs(got - want) < 3.0, f"{got} vs MOPAC {want}"


def test_ase_accessor_without_atoms_reports_actionable_error():
    """An accessor called with no argument before any calculation has no
    structure to work with; it must say so rather than raise an
    `AttributeError` from deep inside the conversion layer."""
    from ase import Atoms
    from pm3_rs.ase import PM3

    for accessor in ("get_potential_energy", "get_forces", "get_gradient",
                     "get_hessian", "get_frequencies"):
        atoms = Atoms(numbers=WATER_Z, positions=WATER_R)
        calc = PM3()
        atoms.calc = calc
        with pytest.raises(RuntimeError, match="no Atoms object"):
            getattr(calc, accessor)()
        # …and the same accessor works as soon as a structure is supplied.
        assert getattr(calc, accessor)(atoms) is not None


def test_documented_ase_example_block():
    """The `### Example` block of docs/python-api.md and the README's ASE block."""
    from ase.build import molecule
    from ase.optimize import BFGS
    from pm3_rs.ase import PM3

    atoms = molecule("H2O")
    atoms.calc = PM3(method="pm3-d3h4")

    BFGS(atoms, logfile=None).run(fmax=0.02)   # optimize (uses get_forces)
    energy = atoms.get_potential_energy()      # eV
    frequencies = atoms.calc.get_frequencies() # cm^-1 (Hessian built lazily)
    hessian = atoms.calc.get_hessian()         # (9, 9) eV/Å²

    assert math.isfinite(energy)
    assert hessian.shape == (9, 9)
    assert np.all(np.abs(frequencies[:6]) < 100.0), frequencies[:6]
    assert np.all((frequencies[6:] > 1000.0) & (frequencies[6:] < 4200.0)), frequencies[6:]

    # README's block.
    readme = molecule("H2O")
    readme.calc = PM3(method="pm3-d3h4")
    assert math.isfinite(readme.get_potential_energy())
    assert readme.get_forces().shape == (3, 3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
