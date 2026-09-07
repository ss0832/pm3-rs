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

import ast
import inspect
import math
import os
from pathlib import Path

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


# Acetamide at its PM3 minimum: the geometry from tests/data/mopac_oracle.tsv.
ACETAMIDE_Z = [8, 6, 7, 6, 1, 1, 1, 1, 1]
ACETAMIDE_R = np.array(
    [
        [0.443779398, 1.296449633, -0.035323927],  # O
        [0.052004373, 0.138591678, -0.005705918],  # C
        [1.033082758, -0.888429398, -0.066016666],  # N
        [-1.395111112, -0.272769730, -0.000022728],  # C
        [0.760253342, -1.735205329, 0.378344955],  # H (amide)
        [-2.061780078, 0.599727626, 0.013338598],  # H
        [-1.639979626, -0.861723040, -0.894240325],  # H
        [-1.632894254, -0.884590928, 0.880357946],  # H
        [1.959783051, -0.608911632, 0.167787223],  # H (amide)
    ]
)


def test_mmok_rides_on_the_method_string_and_is_off_by_default():
    """``+mmok`` selects MOPAC's amide correction; plain ``pm3`` must not include it.

    MOPAC applies MMOK by default and pm3-rs does not, which is the one place the two
    codes disagree about what "PM3" means. The correction is post-SCF, so it moves the
    heat of formation and leaves the density alone -- that asymmetry is the assertion.
    """
    plain = native.single_point(ACETAMIDE_Z, ACETAMIDE_R, method="pm3")
    corrected = native.single_point(ACETAMIDE_Z, ACETAMIDE_R, method="pm3+mmok")

    # MOPAC v23.2.5 at this geometry: -51.032295890 with NOMM, -48.705646220 by default.
    assert abs(plain["heat_of_formation_kcal"] - (-51.032_295_890)) < 2e-3
    assert abs(corrected["heat_of_formation_kcal"] - (-48.705_646_220)) < 2e-3

    # Post-SCF: same wavefunction, different reported heat of formation.
    assert plain["electronic_ev"] == corrected["electronic_ev"]
    assert plain["homo_ev"] == corrected["homo_ev"]
    assert plain["charges"] == corrected["charges"]

    # It composes with the correction variants rather than replacing them.
    assert native.single_point(ACETAMIDE_Z, ACETAMIDE_R, method="pm3-d3h4+mmok")[
        "energy_ev"
    ] != native.single_point(ACETAMIDE_Z, ACETAMIDE_R, method="pm3-d3h4")["energy_ev"]

    # And a molecule with no amide is untouched by it.
    assert (
        native.single_point(WATER_Z, WATER_R, method="pm3+mmok")["energy_ev"]
        == native.single_point(WATER_Z, WATER_R, method="pm3")["energy_ev"]
    )


def test_mmok_reaches_the_gradient_and_not_only_the_energy():
    """The term is differentiated, so an optimization with it on minimizes what it reports.

    Written over the crate's dual numbers, so this is really a check that the flag reached
    the gradient path at all -- a correction that moved the energy and not the forces would
    make ``optimize`` walk downhill on one surface and report another.
    """
    plain = native.gradient(ACETAMIDE_Z, ACETAMIDE_R, method="pm3")
    corrected = native.gradient(ACETAMIDE_Z, ACETAMIDE_R, method="pm3+mmok")
    difference = np.abs(
        np.asarray(corrected["gradient_ev_per_angstrom"])
        - np.asarray(plain["gradient_ev_per_angstrom"])
    ).max()
    assert difference > 1e-3, "the amide torsion term never reached the gradient"


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
    assert {"frequencies_cm", "eigenvalues", "n_rigid", "rigid_residual_cm"} <= set(result)
    frequencies = np.asarray(result["frequencies_cm"])
    assert frequencies.shape == (9,)
    assert np.asarray(result["eigenvalues"]).shape == (9,)
    # "ascending; negatives are imaginary"
    assert np.all(np.diff(frequencies) >= -1e-9)
    # Six translations/rotations at **exactly** zero -- projected out, not recognised for being
    # small -- then the three real vibrations. `n_rigid` is the rank of the rigid-body subspace,
    # found from the geometry, so a caller no longer has to know that a bent triatomic has six.
    #
    # `== 0.0`, not `< 1e-6`: those eigenvalues are *assigned* zero after the projection
    # identifies them by their overlap with the rigid-body subspace, so exactness is by
    # construction rather than by luck, and a tolerance would assert less than is true. Counted
    # rather than sliced, because the zeros lead the list only at a minimum -- an imaginary mode
    # sorts below them.
    assert result["n_rigid"] == 6
    assert int(np.sum(frequencies == 0.0)) == result["n_rigid"], frequencies
    # What those directions carried before projection is still reported, as the measure of the
    # Hessian's quality that the old `< 50.0` bound was really testing.
    assert 0.0 < result["rigid_residual_cm"] < 50.0, result["rigid_residual_cm"]
    for got, want in zip(frequencies[result["n_rigid"] :], ORACLE_FREQUENCIES):
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
        "energy",
        # ASE's own name for the Mermin electronic free energy `E - TS`, which is what
        # `get_potential_energy(force_consistent=True)` returns and what the forces are the
        # gradient of once the occupations are fractional. Not a Gibbs free energy: no
        # zero-point energy, no vibrational partition function, no `pV`, no nuclear entropy.
        "free_energy",
        "electronic_entropy_ts_ev",
        "forces", "charges", "dipole", "heat_of_formation_kcal", "hessian",
        "stress",
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
    # Exactly zero, not "< 100.0" -- the rigid-body subspace is projected out of the mass-weighted
    # Hessian and those eigenvalues are then assigned zero, so this is exact by construction.
    assert int(np.sum(frequencies == 0.0)) == 6, frequencies
    assert np.all((frequencies[6:] > 1000.0) & (frequencies[6:] < 4200.0)), frequencies[6:]

    # README's block.
    readme = molecule("H2O")
    readme.calc = PM3(method="pm3-d3h4")
    assert math.isfinite(readme.get_potential_energy())
    assert readme.get_forces().shape == (3, 3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# --- the console script -------------------------------------------------------------------
#
# `pip install` puts a `pm3-rs` command on the path, and it runs the *compiled* CLI rather than
# a Python reimplementation. That means the Python half and the Rust half have to stay wired
# together across a rebuild, and nothing else in this suite would notice if they came apart:
# every other test goes through the native functions directly.
#
# They did come apart once, silently -- the module built, imported and computed correctly while
# the entry point raised `AttributeError` on the first character typed at a shell. These tests
# exist because that is invisible from inside Python.


def test_the_console_script_entry_point_is_wired_to_the_compiled_cli():
    from pm3_rs import cli

    assert hasattr(native._native, "cli_main"), (
        "the compiled module is missing `cli_main`, so the installed `pm3-rs` command would "
        "fail with AttributeError at startup"
    )
    assert cli.main(["pm3-rs", "--version"]) == 0


def test_the_console_script_reports_failure_through_its_exit_code(tmp_path):
    from pm3_rs import cli

    missing = tmp_path / "not-a-molecule.xyz"
    assert cli.main(["pm3-rs", "energy", str(missing)]) != 0


def test_the_console_script_computes_the_same_energy_as_the_api(tmp_path, capfd):
    # `capfd`, not `capsys`: the command prints from Rust, straight to the process's stdout
    # file descriptor, and never touches `sys.stdout` for pytest to swap out.
    from pm3_rs import cli

    xyz = tmp_path / "water.xyz"
    xyz.write_text("3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n")
    assert cli.main(["pm3-rs", "energy", str(xyz)]) == 0
    printed = capfd.readouterr().out

    reference = pm3_rs.single_point(
        [8, 1, 1], [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.24, 0.9278, 0.0]]
    )["energy_ev"]

    # Pull the number out rather than matching the layout: what has to agree is the physics,
    # not how the command chose to print it.
    line = next(l for l in printed.splitlines() if l.startswith("Total energy:"))
    printed_energy = float(line.split(":", 1)[1].strip().split()[0])
    assert abs(printed_energy - reference) < 1.0e-6, (
        f"the command printed {printed_energy} where the library gives {reference}"
    )


# ---------------------------------------------------------------------------
# Every command, through the installed console script
#
# The Rust side crosses every command against every flag (`src/cli.rs`, the
# `every_command_and_flag_combination_parses_or_explains_itself` matrix), but it calls
# `parse_args` and `run` directly. That is the parser and the dispatch; it is not the
# path a user takes. `pip install` puts a `pm3-rs` command on the path that goes
# Python -> `_native.cli_main` -> Rust, and until now exactly one command -- `energy` --
# was ever exercised across that boundary.
#
# What this layer catches that the Rust one cannot: a command reachable in Rust and not
# through the entry point, an exit code that does not survive the crossing, and a
# promised output file that never appears.
# ---------------------------------------------------------------------------

WATER_XYZ = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n"

# 18 Bohr. Every periodic width must exceed the 14 Bohr short-range cutoff or the
# Gamma-point answer is wrong regardless of how cleanly it converged; see
# tests/test_periodic_python_api.py, which explains why 14 Bohr is a trap and not a
# boundary.
CELL_ANGSTROM = "9.525"

# command -> (extra argv, the file it documents itself as writing)
MOLECULAR_COMMANDS = {
    "energy": ([], None),
    "gradient": ([], None),
    "charges": ([], None),
    "orbitals": ([], None),
    "orbitals --coefficients": (["--coefficients"], None),
    "frequencies": ([], None),
    "hessian": ([], None),
    "ir": ([], None),
    "optimize": ([], "water.pm3opt.xyz"),
    "molden": ([], "water.molden"),
    "molden --sto": (["--sto"], "water.molden"),
}

PERIODIC_COMMANDS = {
    "energy": [],
    "charges": [],
    "stress": [],
    "phonons": [],
    "bands": [],
    "born": [],
    "dielectric": [],
    "berry": [],
    "phonon-bands": ["--supercell", "2,1,1"],
    # The one command with required flags of its own: a field to apply, and the k-mesh
    # whose strings the Berry phase is taken along.
    "finite-field": ["--field", "0.001,0,0", "--kpts", "3,1,1"],
}


@pytest.mark.parametrize("case", sorted(MOLECULAR_COMMANDS))
def test_every_molecular_command_runs_through_the_console_script(case, tmp_path, capfd):
    """Exit code 0, some output, and the artifact the usage text promises."""
    from pm3_rs import cli

    extra, artifact = MOLECULAR_COMMANDS[case]
    command = case.split()[0]
    xyz = tmp_path / "water.xyz"
    xyz.write_text(WATER_XYZ)
    if artifact:
        target = tmp_path / artifact
        target.unlink(missing_ok=True)

    code = cli.main(["pm3-rs", command, str(xyz), *extra])
    printed = capfd.readouterr().out
    assert code == 0, f"`{case}` exited {code} through the console script"
    assert printed.strip(), f"`{case}` printed nothing"

    if artifact:
        written = tmp_path / artifact
        assert written.exists(), (
            f"`{case}` documents itself as writing {artifact} and did not. Through the "
            f"entry point that is indistinguishable from success."
        )
        assert written.stat().st_size > 0, f"`{case}` wrote an empty {artifact}"


@pytest.mark.slow
@pytest.mark.parametrize("command", sorted(PERIODIC_COMMANDS))
def test_every_periodic_command_runs_through_the_console_script(command, tmp_path, capfd):
    """The same, for the commands that need a lattice.

    Marked slow: each is a lattice sum, and several are response solves on top -- seconds
    each rather than milliseconds. Run without them using ``-m "not slow"``.
    """
    from pm3_rs import cli

    xyz = tmp_path / "water.xyz"
    xyz.write_text(WATER_XYZ)

    code = cli.main(
        ["pm3-rs", command, str(xyz), "--cell", CELL_ANGSTROM, *PERIODIC_COMMANDS[command]]
    )
    printed = capfd.readouterr().out
    assert code == 0, f"periodic `{command}` exited {code} through the console script"
    assert printed.strip(), f"periodic `{command}` printed nothing"


@pytest.mark.slow
def test_a_periodic_optimization_writes_the_cell_into_the_geometry_it_saves(tmp_path, capfd):
    """The relaxed structure has to carry its lattice, or the answer is half-thrown-away.

    Plain XYZ has nowhere to put a cell. While the CLI could only hold the cell fixed that
    was merely incomplete -- the caller still had the cell they passed in. With
    ``--relax-cell`` the lattice *is* the result, so dropping it loses the part of the
    answer that took the work. The extended-XYZ ``Lattice="..."`` comment is how it
    survives, and this reads it back the way the next tool would.
    """
    from pm3_rs import cli

    xyz = tmp_path / "water.xyz"
    xyz.write_text(WATER_XYZ)
    code = cli.main(
        ["pm3-rs", "optimize", str(xyz), "--cell", CELL_ANGSTROM, "--max-steps", "2"]
    )
    capfd.readouterr()
    assert code == 0

    written = tmp_path / "water.pm3opt.xyz"
    assert written.exists(), "the periodic optimizer wrote no geometry"
    comment = written.read_text().splitlines()[1]
    assert "Lattice=" in comment, (
        f"the saved geometry has no lattice: {comment!r}. Everything downstream would read "
        f"it as an isolated molecule."
    )
    assert "pbc=" in comment, f"the saved geometry does not say which axes are periodic: {comment!r}"

    # Nine numbers, and they are the cell that went in -- this run holds it fixed.
    vectors = comment.split('Lattice="', 1)[1].split('"', 1)[0].split()
    assert len(vectors) == 9, f"expected nine lattice components, got {len(vectors)}: {vectors}"
    edge = float(CELL_ANGSTROM)
    assert float(vectors[0]) == pytest.approx(edge, abs=1e-6)
    assert float(vectors[4]) == pytest.approx(edge, abs=1e-6)
    assert float(vectors[8]) == pytest.approx(edge, abs=1e-6)


@pytest.mark.parametrize("command", sorted(PERIODIC_COMMANDS))
def test_a_command_that_needs_a_lattice_refuses_rather_than_inventing_one(
    command, tmp_path, capfd
):
    """A periodic-only command without ``--cell`` must fail, and say why.

    ``energy`` and ``charges`` are the two that legitimately run either way, so they are
    the control: the point of the test is that the rest do not quietly produce a
    molecular answer to a question that was about a crystal.
    """
    from pm3_rs import cli

    xyz = tmp_path / "water.xyz"
    xyz.write_text(WATER_XYZ)

    code = cli.main(["pm3-rs", command, str(xyz)])
    captured = capfd.readouterr()

    if command in ("energy", "charges"):
        assert code == 0, f"`{command}` runs on a molecule and must keep doing so"
        return

    assert code != 0, (
        f"`{command}` needs a lattice and exited 0 without one. Either it invented a cell "
        f"or it answered a different question than the one asked."
    )
    message = (captured.out + captured.err).lower()
    assert "cell" in message, (
        f"`{command}` failed without mentioning --cell; the message a user sees is "
        f"{captured.out + captured.err!r}"
    )


# ---------------------------------------------------------------------------
# The three Python-facing layers must agree on what arguments exist
#
# `pm3_rs.native` calls `_native` positionally, and `_native.pyi` is what a type
# checker reads. Nothing connected the three, so they drifted: the stub declared
# `optimize(max_iter, gtol)` arguments that never existed, and `native.py` quietly
# dropped `long_range_cutoff`, which put the whole linear-scaling path out of
# reach from `import pm3_rs`. PyO3 publishes the real signature, so both drifts
# are mechanically checkable.
# ---------------------------------------------------------------------------


def _native_signatures():
    """Every `_native` function's parameter names, in declaration order."""
    from pm3_rs import _native

    out = {}
    for name in dir(_native):
        if name.startswith("_"):
            continue
        fn = getattr(_native, name)
        text = getattr(fn, "__text_signature__", None)
        if text is None:
            continue
        inner = text.strip()[1:-1]
        params = []
        for piece in inner.split(","):
            piece = piece.strip()
            if not piece or piece in ("/", "*"):
                continue
            params.append(piece.split("=", 1)[0].strip())
        out[name] = params
    return out


def _stub_signatures():
    """Every `def` in `_native.pyi`, as parameter-name lists."""
    stub = Path(__file__).resolve().parents[1] / "python" / "pm3_rs" / "_native.pyi"
    source = stub.read_text(encoding="utf-8")
    tree = ast.parse(source)
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        out[node.name] = [a.arg for a in node.args.args]
    return out


def test_the_type_stub_matches_the_extension():
    native_sigs = _native_signatures()
    stub_sigs = _stub_signatures()

    assert native_sigs, "PyO3 stopped publishing __text_signature__; this test needs rewriting"
    assert set(stub_sigs) == set(native_sigs), (
        f"stub declares {sorted(set(stub_sigs) - set(native_sigs))} that do not exist, "
        f"and omits {sorted(set(native_sigs) - set(stub_sigs))}"
    )
    for name, expected in sorted(native_sigs.items()):
        assert stub_sigs[name] == expected, (
            f"_native.pyi declares {name}{tuple(stub_sigs[name])} "
            f"but the extension takes {tuple(expected)}"
        )


def test_every_native_argument_is_reachable_from_the_python_layer():
    """`pm3_rs.native` must expose every argument the extension accepts.

    Dropping one is invisible -- the call still works, the default is still
    applied -- and it takes a whole capability out of reach. That is exactly how
    `long_range_cutoff` became unreachable.
    """
    from pm3_rs import native as native_module

    native_sigs = _native_signatures()
    for name, expected in sorted(native_sigs.items()):
        wrapper = getattr(native_module, name, None)
        if wrapper is None:
            # `cli_main` is reached through `pm3_rs.cli`, not the native wrapper layer.
            continue
        accepted = set(inspect.signature(wrapper).parameters)
        missing = [p for p in expected if p not in accepted]
        assert not missing, (
            f"pm3_rs.native.{name} cannot pass {missing}, which _native.{name} accepts"
        )


def _native_call_sites():
    """Every ``_native.<name>`` the wrapper layer reaches for, by source inspection.

    Read out of the source rather than by calling anything: a wrapper is only
    executed when its own arguments are valid, so a missing extension function
    hides behind whatever argument checking runs before it.
    """
    wrapper = Path(__file__).resolve().parents[1] / "python" / "pm3_rs" / "native.py"
    tree = ast.parse(wrapper.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "_native"
        ):
            names.add(node.attr)
    return names


def test_every_native_call_site_exists_in_the_extension():
    """`native.py` must not call an `_native` function the extension does not have.

    This is the stale-build guard. The extension is compiled from `src/python.rs`
    and the wrapper layer is plain Python shipped beside it, so the two travel
    separately: an editable install whose `.pyd` predates a new `#[pyfunction]`
    imports cleanly, passes anything that does not touch the new function, and
    raises `AttributeError` only at the call. That is what made `dynamical_matrix`
    look like a Rust-only feature when it had been wired to Python all along --
    the wiring was right and the compiled artefact was six commits old.

    The failure names the remedy, because the remedy is not obvious from an
    `AttributeError` deep inside a wrapper.
    """
    from pm3_rs import _native

    sites = _native_call_sites()
    # Non-vacuity. This test is a source scan, so a change to how `native.py` reaches the
    # extension -- an alias, a `getattr`, a re-import under another name -- would leave the scan
    # finding nothing and the assertion below passing on an empty set. Naming one function that
    # must always be there turns that silent hole into a failure.
    assert "single_point" in sites, (
        f"the scan found {sorted(sites)}, which does not look like the wrapper layer; "
        f"`native.py` no longer calls the extension as `_native.<name>` and this test "
        f"needs rewriting rather than deleting"
    )

    missing = sorted(name for name in sites if not hasattr(_native, name))
    assert not missing, (
        f"pm3_rs.native calls _native.{{{','.join(missing)}}}, which this build of the "
        f"extension does not export. Either src/python.rs never registered them in "
        f"`#[pymodule] fn _native`, or the compiled extension is older than the source. "
        f"Rebuild with `python -m maturin develop` and re-run."
    )


def test_the_imported_package_is_the_one_in_this_repository():
    """The suite must be testing this working tree, not a copy installed beside it.

    The other half of the stale-build guard, and the half that was missing when it
    mattered. `test_every_native_call_site_exists_in_the_extension` compares the
    wrapper's source against the *imported* extension, so it catches a stale `.pyd`
    -- but only because the extension carries compiled symbols to interrogate. A
    stale `ase.py` or `native.py` has nothing to interrogate: it imports cleanly,
    every symbol resolves, and the suite reports green while exercising code the
    working tree no longer contains.

    That happened here. A non-editable `site-packages/pm3_rs/` directory shadowed
    the source tree; `maturin develop` reported "Setting installed package as
    editable" and "Installed", `pip show` then found no such package, and the
    freshly compiled `.pyd` sat unused in `python/pm3_rs/` while Python imported a
    day-old copy. Directories in site-packages precede anything a `.pth` appends,
    so even a correct editable install would have lost to it.

    Checked by resolved path rather than by version string, because a stale copy of
    a version that was never bumped reports the version you expect.

    Set `PM3_ALLOW_INSTALLED_PACKAGE=1` to run this suite against an installed
    wheel on purpose -- packaging checks legitimately want that.
    """
    import pm3_rs

    if os.environ.get("PM3_ALLOW_INSTALLED_PACKAGE") == "1":
        pytest.skip("PM3_ALLOW_INSTALLED_PACKAGE=1: testing an installed wheel on purpose")

    expected = (Path(__file__).resolve().parents[1] / "python" / "pm3_rs").resolve()
    actual = Path(pm3_rs.__file__).resolve().parent
    assert actual == expected, (
        f"`import pm3_rs` resolves to {actual}, not the {expected} in this repository, so "
        f"this suite is reporting on code that is not the code under test. Remove the "
        f"shadowing install with `pip uninstall pm3-rs-python`, then put the source tree on "
        f"the path -- a `.pth` file in site-packages containing a single line, "
        f"{expected.parent}, does it and survives a `maturin develop` that reports an editable "
        f"install without performing one."
    )


def test_every_documented_name_is_actually_callable():
    """Every name in `pm3_rs.__all__` must resolve to something callable.

    `__all__` is the package's own statement of what it offers, and it is written
    by hand. A name that is exported but unreachable is worse than one that was
    never exported: it survives `dir()`, it survives an import, and it fails only
    when someone believes the documentation and calls it.
    """
    import pm3_rs

    broken = []
    for name in pm3_rs.__all__:
        member = getattr(pm3_rs, name, None)
        if member is None:
            broken.append(f"{name} (absent)")
        elif name != "native" and not callable(member):
            broken.append(f"{name} (not callable: {type(member).__name__})")
    assert not broken, f"pm3_rs.__all__ promises {broken}"


def test_the_shipped_package_is_ascii_only():
    """Every ``.py`` under ``pm3_rs`` must be pure ASCII.

    Docstrings are printed -- by ``help()``, by IDEs, by ``print(fn.__doc__)`` --
    and they are printed to whatever encoding the user's console has. On a
    Japanese, Chinese or Korean Windows console that is a legacy codepage, and
    ``cp932`` cannot represent ``A``-with-ring, ``mu``, superscripts, en/em
    dashes or the minus sign. Sixteen docstrings in this package used to raise
    ``UnicodeEncodeError`` there: the package imported, computed and returned
    correct numbers, and ``help()`` on it crashed.

    So the rule is mechanical rather than tasteful: the *shipped Python* is
    ASCII. Rust sources, Markdown and commit messages keep their typography --
    nothing prints those through a console codec, and the CLI's own output is
    already ASCII.
    """
    package = Path(__file__).resolve().parents[1] / "python" / "pm3_rs"
    offenders = []
    for path in sorted(package.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            bad = [c for c in line if ord(c) > 127]
            if bad:
                offenders.append(f"{path.name}:{number}: {''.join(sorted(set(bad)))}")
    assert not offenders, "non-ASCII in the shipped package:\n  " + "\n  ".join(offenders)


def test_every_docstring_survives_a_legacy_console():
    """The same rule, stated as the symptom rather than as the cause.

    This is what actually breaks, so it is worth asserting directly: encode every
    docstring the package exposes into ``cp932`` and require it to survive. It
    would still pass if the ASCII rule above were replaced by something subtler,
    and it fails for exactly the reason a user would report.
    """
    import pm3_rs
    from pm3_rs import native as native_module

    subjects = [("pm3_rs", pm3_rs.__doc__), ("pm3_rs.native", native_module.__doc__)]
    for name in dir(native_module):
        if name.startswith("_"):
            continue
        member = getattr(native_module, name)
        if callable(member) and member.__doc__:
            subjects.append((f"native.{name}", member.__doc__))

    ase_module = pytest.importorskip("pm3_rs.ase")
    subjects.append(("pm3_rs.ase", ase_module.__doc__))
    for name in dir(ase_module.PM3):
        if name.startswith("_"):
            continue
        member = getattr(ase_module.PM3, name, None)
        doc = getattr(member, "__doc__", None)
        if doc:
            subjects.append((f"PM3.{name}", doc))

    broken = []
    for name, doc in subjects:
        if doc is None:
            continue
        try:
            doc.encode("cp932")
        except UnicodeEncodeError as error:
            broken.append(f"{name}: {doc[error.start:error.end]!r}")
    assert not broken, "docstrings a cp932 console cannot print:\n  " + "\n  ".join(broken)


# ---------------------------------------------------------------------------
# Molden export
# ---------------------------------------------------------------------------


def test_molden_export_is_reachable_from_every_layer(tmp_path):
    """The document is the same whichever layer asks for it.

    Three surfaces exist for a reason -- Rust, the native wrapper, ASE -- and the
    point of having all three is that they agree. `long_range_cutoff` was
    unreachable from `pm3_rs` for a whole release because nobody checked.
    """
    from ase import Atoms

    from pm3_rs.ase import PM3

    text = pm3_rs.molden(WATER_Z, WATER_R)
    assert text == native.molden(WATER_Z, WATER_R), "the re-export must not diverge"
    for section in ("[Molden Format]", "[Atoms] AU", "[GTO]", "[MO]"):
        assert section in text, f"missing {section}"

    atoms = Atoms(numbers=WATER_Z, positions=WATER_R)
    atoms.calc = PM3()
    written = tmp_path / "water.molden"
    atoms.calc.write_molden(written, atoms)
    assert written.read_text(encoding="utf-8") == text


def test_molden_occupations_account_for_every_electron():
    """A viewer reads the occupations; if they do not sum to the electron count the
    file is describing a different molecule."""
    text = pm3_rs.molden(WATER_Z, WATER_R)
    total = sum(
        float(line.split("=", 1)[1])
        for line in text.splitlines()
        if line.startswith(" Occup=")
    )
    assert abs(total - 8.0) < 1e-9, f"water has eight valence electrons, the file says {total}"


def test_molden_is_refused_for_a_periodic_structure():
    from ase import Atoms

    from pm3_rs.ase import PM3

    atoms = Atoms(numbers=WATER_Z, positions=WATER_R, cell=[10.0, 10.0, 10.0], pbc=True)
    atoms.calc = PM3()
    with pytest.raises(RuntimeError, match="molecular"):
        atoms.calc.write_molden("unused.molden", atoms)


def test_an_unbound_calculator_says_so_rather_than_crashing():
    """Every accessor that reaches past `get_property` goes through the same guard.

    Before this, `write_molden` on a fresh calculator raised
    `AttributeError: 'NoneType' object has no attribute 'get_atomic_numbers'`,
    which says nothing about what the caller should do differently.
    """
    from pm3_rs.ase import PM3

    calc = PM3()
    for call in (
        lambda: calc.get_frequencies(),
        lambda: calc.get_phonons(),
        lambda: calc.write_molden("unused.molden"),
    ):
        with pytest.raises(RuntimeError, match="no Atoms object yet"):
            call()


WATER_POSITIONS = [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]
CHAIN_CELL = [[6.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]


def test_the_dipole_module_is_reachable_from_both_python_layers():
    """`pm3_rs::dipole` is one matrix used twice, and both uses are now callable.

    The reported dipole and the external field's coupling are built from the same operator, so
    exposing the operator exposes what they agree on rather than two numbers that happen to.
    """
    import numpy as np
    import pm3_rs

    result = pm3_rs.dipole([8, 1, 1], WATER_POSITIONS, operator=True)
    debye = np.asarray(result["dipole_debye"])
    assert np.linalg.norm(debye) == pytest.approx(1.7619, abs=1e-3)
    # 1 D = 0.2081943 e.A, the same vector in the other unit.
    assert np.allclose(np.asarray(result["dipole_e_angstrom"]), debye * 0.2081943)

    tensor = np.asarray(result["derivatives_e"])
    assert tensor.shape == (3, 9)
    # The translational sum rule: moving the whole molecule moves its dipole by its charge times
    # the displacement, which for a neutral molecule is not at all.
    for axis in range(3):
        assert abs(tensor[axis].reshape(3, 3)[:, axis].sum()) < 1e-6

    operator = np.asarray(result["operator_bohr"])
    assert operator.shape[0] == 3 and operator.shape[1] == operator.shape[2]
    for axis in range(3):
        assert np.allclose(operator[axis], operator[axis].T), "the operator must be symmetric"

    from ase import Atoms
    from pm3_rs.ase import PM3

    atoms = Atoms("OH2", positions=WATER_POSITIONS)
    atoms.calc = PM3()
    assert np.allclose(atoms.calc.get_dipole_derivatives(atoms), tensor)
    assert np.allclose(atoms.calc.get_dipole_operator(atoms), operator)


def test_the_dynamical_matrix_is_reachable_from_both_python_layers():
    """`pbc::dfpt` gives the matrix, not only what it diagonalizes to."""
    import numpy as np
    import pm3_rs

    q = [0.25, 0.0, 0.0]
    full = pm3_rs.dynamical_matrix(
        [8, 1, 1], WATER_POSITIONS, CHAIN_CELL, q, pbc=[True, False, False]
    )
    assert np.asarray(full["real"]).shape == (9, 9)
    assert np.asarray(full["imag"]).shape == (9, 9)
    # Reported so a wrong assembly is visible rather than averaged away.
    assert full["hermitian_defect"] < 1e-6

    rigid = pm3_rs.dynamical_matrix(
        [8, 1, 1], WATER_POSITIONS, CHAIN_CELL, q, pbc=[True, False, False], rigid_ion=True
    )
    moved = np.abs(np.asarray(full["real"]) - np.asarray(rigid["real"])).max()
    assert moved > 1.0, "the electronic response should be most of the answer, not a correction"

    # The masses travel with the matrix, and they have to: mass weighting `D(q)` needs the
    # isotope-averaged values this crate carries, which a Python caller cannot obtain from
    # anywhere else in the package. Returning the matrix without them left the frequencies at a
    # wavevector reachable only from Rust -- the matrix was there and could not be used.
    masses = np.asarray(full["masses"], dtype=float)
    assert masses.shape == (3,), "one mass per atom, in the order the matrix indexes them"
    assert masses[0] > masses[1], "oxygen is heavier than hydrogen"

    # And the frequencies the crate derives from them are the ones `phonons` reports, so the two
    # entry points cannot drift into describing different dispersions.
    from_matrix = np.asarray(full["frequencies_cm"], dtype=float)
    from_phonons = np.asarray(
        pm3_rs.phonons(
            [8, 1, 1], WATER_POSITIONS, CHAIN_CELL, pbc=[True, False, False], q=q
        )["frequencies_cm"],
        dtype=float,
    )
    assert np.allclose(from_matrix, from_phonons, atol=1e-6), (
        f"dynamical_matrix says {from_matrix} and phonons says {from_phonons}"
    )

    from ase import Atoms
    from pm3_rs.ase import PM3

    atoms = Atoms("OH2", positions=WATER_POSITIONS, cell=CHAIN_CELL, pbc=[True, False, False])
    atoms.calc = PM3()
    real, imaginary = atoms.calc.get_dynamical_matrix(q, atoms)
    assert np.allclose(real, np.asarray(full["real"]))
    assert np.allclose(imaginary, np.asarray(full["imag"]))


def test_the_heavy_ase_results_are_computed_once_per_geometry():
    """Asking twice must not calculate twice, and moving an atom must not reuse the old answer.

    These are not ASE properties -- `ase.vibrations` and `ase.phonons` build their own force
    constants out of `get_forces` -- so ASE's own `results` cache neither holds them nor clears
    them. Without the geometry check beside each one, a caller who asked for frequencies and
    then for an infrared spectrum paid for two Hessians.
    """
    import numpy as np
    from ase import Atoms
    from pm3_rs.ase import PM3

    atoms = Atoms("OH2", positions=WATER_POSITIONS)
    atoms.calc = PM3()

    first = atoms.calc.get_ir_spectrum(atoms)
    again = atoms.calc.get_ir_spectrum(atoms)
    assert again is first, "the second call rebuilt a Hessian it already had"

    moved = atoms.copy()
    moved.positions[1, 1] += 0.05
    moved.calc = atoms.calc
    after = atoms.calc.get_ir_spectrum(moved)
    assert after is not first, "a moved atom must invalidate the cache"
    assert not np.allclose(after["frequencies_cm"], first["frequencies_cm"])


def test_a_phonon_cache_is_keyed_on_the_wavevector():
    """One wavevector's answer must not be handed back for another's."""
    import numpy as np
    from ase import Atoms
    from pm3_rs.ase import PM3

    atoms = Atoms("OH2", positions=WATER_POSITIONS, cell=CHAIN_CELL, pbc=[True, False, False])
    atoms.calc = PM3()

    quarter = atoms.calc.get_phonons(atoms, q=[0.25, 0.0, 0.0])
    assert atoms.calc.get_phonons(atoms, q=[0.25, 0.0, 0.0]) is quarter
    boundary = atoms.calc.get_phonons(atoms, q=[0.5, 0.0, 0.0])
    assert boundary is not quarter
    assert not np.allclose(boundary["frequencies_cm"], quarter["frequencies_cm"])