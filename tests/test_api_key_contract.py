# SPDX-License-Identifier: GPL-3.0-or-later
"""**Every native function returns the keys it promises.**

Through 0.2.3 the two geometry optimizers returned disjoint result dicts:
``optimize`` had ``positions_angstrom`` / ``energy_hartree`` / ``iterations`` and no
``energy_ev``; ``relax`` had ``positions`` / ``energy_ev`` / ``steps`` and no heat of
formation. Both are documented, so reading the docs for one function and applying it to
the other -- which is what anyone does after a molecular run and a periodic one -- ended
a structure optimization with a ``KeyError`` from the call that had done the most work.

Nothing caught it, because the key checks that existed were written per function, by
hand, for the functions someone happened to think about. ``optimize`` had none at all.

So this file is a **table**, not a set of hand-written assertions. Every public function
in :mod:`pm3_rs.native` must appear in ``CONTRACTS`` with the keys it guarantees, and
:func:`test_the_table_covers_every_native_function` fails if a new function is added
without declaring one. Adding a function and forgetting its contract is then a test
failure rather than a user's traceback.

The keys here are the *guarantee*, not a snapshot: a function may return more. Removing
or renaming one is a breaking change and must fail here.

Run inside the maturin venv::

    python -m pytest tests/test_api_key_contract.py
"""

import inspect

import numpy as np
import pytest

from pm3_rs import native

# --- systems -------------------------------------------------------------------------

WATER_Z = [8, 1, 1]
WATER_R = [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.24, 0.9278, 0.0]]

BOHR = 0.52917721
# 18 Bohr. The Gamma-point condition is that every periodic width exceed the 14 Bohr
# short-range cutoff, so this is comfortably inside it -- see tests/test_periodic_python_api.py,
# which explains why 14.0 * BOHR is a trap rather than a boundary.
CUBE = 18.0 * BOHR
# A one-dimensional chain: periodic along x, vacuum across.
CHAIN_CELL = [[6.0, 0.0, 0.0], [0.0, 30.0 * BOHR, 0.0], [0.0, 0.0, 30.0 * BOHR]]
CHAIN_PBC = [True, False, False]

# --- shared key groups ---------------------------------------------------------------

# Every function that converges an SCF and reports its energy reports it in both units.
ENERGY = {"energy_ev", "energy_hartree"}
# The three geometry optimizers. Both names for each concept, because both are documented
# and neither is wrong; see the module docstring.
OPTIMIZER = ENERGY | {
    "positions",
    "positions_angstrom",
    "heat_of_formation_kcal",
    "converged",
    "steps",
    "iterations",
}

# --- the table -----------------------------------------------------------------------
#
# name -> (call, required keys). `None` for the keys means the function does not return a
# dict at all and is checked by its own test instead.

CONTRACTS = {
    # -- molecular ---------------------------------------------------------------------
    "single_point": (
        lambda: native.single_point(WATER_Z, WATER_R),
        ENERGY
        | {
            "heat_of_formation_kcal",
            "electronic_ev",
            "core_ev",
            "charges",
            "dipole_debye",
            "homo_ev",
            "lumo_ev",
            "converged",
            "unrestricted",
        },
    ),
    "gradient": (
        lambda: native.gradient(WATER_Z, WATER_R),
        ENERGY
        | {"heat_of_formation_kcal", "gradient_hartree_per_bohr", "gradient_ev_per_angstrom"},
    ),
    "forces": (
        lambda: native.forces(WATER_Z, WATER_R),
        ENERGY | {"heat_of_formation_kcal", "forces_hartree_per_bohr", "forces_ev_per_angstrom"},
    ),
    "optimize": (lambda: native.optimize(WATER_Z, WATER_R), OPTIMIZER),
    "frequencies": (
        lambda: native.frequencies(WATER_Z, WATER_R),
        {"frequencies_cm", "eigenvalues", "modes", "masses"},
    ),
    "orbitals": (
        lambda: native.orbitals(WATER_Z, WATER_R),
        {
            "mo_energies_ev", "mo_energies_hartree", "mo_coefficients", "occupations",
            "n_occupied", "homo_index", "lumo_index", "homo_ev", "lumo_ev", "gap_ev",
            "ao_labels", "unrestricted", "n_beta",
            # Present and None for a restricted run, so the key set does not depend on the shell.
            "mo_energies_beta_ev", "mo_energies_beta_hartree", "mo_coefficients_beta",
            "occupations_beta",
        },
    ),
    "hessian": (
        lambda: native.hessian(WATER_Z, WATER_R),
        {"hessian_hartree_per_bohr2", "ndof"},
    ),
    "ir_spectrum": (
        lambda: native.ir_spectrum(WATER_Z, WATER_R),
        {"frequencies_cm", "intensities_km_per_mol", "dipole_derivatives_e", "ndof"},
    ),
    "dipole": (lambda: native.dipole(WATER_Z, WATER_R), {"dipole_debye"}),
    # `molden` returns the document as a string, not a dict.
    "molden": (lambda: native.molden(WATER_Z, WATER_R), None),
    # -- divide and conquer ------------------------------------------------------------
    "divide_and_conquer": (
        lambda: native.divide_and_conquer(WATER_Z, WATER_R),
        ENERGY | {"heat_of_formation_kcal", "charges", "converged"},
    ),
    "divide_and_conquer_forces": (
        lambda: native.divide_and_conquer_forces(WATER_Z, WATER_R),
        ENERGY | {"forces_ev_per_angstrom"},
    ),
    # The partitioned optimizer answers to the same names as the two whole-system ones --
    # that symmetry is what the KeyError this suite exists for came from.
    "divide_and_conquer_optimize": (
        lambda: native.divide_and_conquer_optimize(WATER_Z, WATER_R, max_steps=2),
        OPTIMIZER | {"n_subsystems"},
    ),
    # -- periodic ----------------------------------------------------------------------
    "periodic_single_point": (
        lambda: native.periodic_single_point(WATER_Z, WATER_R, CUBE),
        ENERGY
        | {
            "electronic_ev",
            "core_ev",
            "correction_ev",
            "ewald_ev",
            "heat_of_formation_kcal",
            "charges",
            "converged",
            "n_kpoints",
            "homo_ev",
            "lumo_ev",
            # Both branches return every key, differing only in which are None. Which keys
            # existed used to depend on which branch ran -- the bug this file generalizes.
            "band_gap_ev",
            "fermi_ev",
            "gamma_margin_bohr",
            "entropy_ts_ev",
            "free_energy_ev",
            # `None` unless a retry rescued the SCF, and how far the density sloshed on the way.
            # Present on both branches so the key set does not depend on which one ran -- the
            # asymmetry this whole file exists to prevent.
            "rescued_by",
            "charge_swing",
        },
    ),
    "periodic_forces": (
        lambda: native.periodic_forces(WATER_Z, WATER_R, CUBE),
        ENERGY
        | {
            "forces_ev_per_angstrom",
            "stress_ev_per_angstrom3",
            "charges",
            "heat_of_formation_kcal",
            "gamma_margin_bohr",
            "entropy_ts_ev",
            "free_energy_ev",
            "converged",
        },
    ),
    "relax": (
        lambda: native.relax(WATER_Z, WATER_R, CUBE, max_steps=1),
        OPTIMIZER | {"cell"},
    ),
    # `modes` and `masses` are the polarization vectors, which the Γ path computed and discarded
    # until 0.2.4. A frequency without them says how fast a mode vibrates and not what moves.
    "phonons": (
        lambda: native.phonons(WATER_Z, WATER_R, CUBE),
        {"frequencies_cm", "modes", "masses", "acoustic_residual_cm", "energy_ev"},
    ),
    "dynamical_matrix": (
        lambda: native.dynamical_matrix(WATER_Z, WATER_R, CHAIN_CELL, [0.25, 0.0, 0.0],
                                        pbc=CHAIN_PBC),
        {"real", "imag", "masses", "frequencies_cm", "hermitian_defect", "q"},
    ),
    "bands": (
        lambda: native.bands(WATER_Z, WATER_R, CUBE, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        {"bands_ev", "bands_beta_ev", "distances_per_bohr", "fermi_ev", "kpoints"},
    ),
    "phonon_bands": (
        lambda: native.phonon_bands(WATER_Z, WATER_R, CHAIN_CELL, [2, 1, 1],
                                    [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], points=2,
                                    pbc=CHAIN_PBC),
        {"q", "frequencies_cm", "supercell", "acoustic_sum_rule_residual", "gamma_margin_bohr"},
    ),
    "born_charges": (
        lambda: native.born_charges(WATER_Z, WATER_R, CUBE),
        {"born_charges", "sum_rule_residual", "gamma_margin_bohr"},
    ),
    "dielectric": (
        lambda: native.dielectric(WATER_Z, WATER_R, CUBE),
        {"polarizability", "epsilon", "gamma_margin_bohr"},
    ),
    "berry_polarization": (
        lambda: native.berry_polarization(WATER_Z, WATER_R, CUBE, strings=3),
        {"total", "electronic", "ionic", "phase", "quantum", "string_length",
         "gamma_margin_bohr"},
    ),
    "finite_field": (
        lambda: native.finite_field(WATER_Z, WATER_R, CUBE, [0.0005, 0.0, 0.0], [3, 1, 1]),
        # `energy` is this function's own name for the converged total; `energy_ev` is what
        # the other twenty-one call it. Both, for the reason `optimize` carries both.
        {"energy", "energy_ev", "enthalpy_ev", "polarization", "electronic_polarization",
         "ionic_polarization", "resolved", "phase", "field", "converged", "iterations",
         "gamma_margin_bohr"},
    ),
}


def test_the_table_covers_every_native_function():
    """A new native function must declare its key contract here.

    This is the part that makes the file a guarantee rather than a list. Without it,
    the next function added is the next `optimize`: documented, uncovered, and wrong.
    """
    public = {
        name
        for name in dir(native)
        if not name.startswith("_")
        and callable(getattr(native, name))
        and getattr(getattr(native, name), "__module__", None) == "pm3_rs.native"
    }
    missing = public - set(CONTRACTS)
    stale = set(CONTRACTS) - public
    assert not missing, f"native functions with no declared key contract: {sorted(missing)}"
    assert not stale, f"contracts for functions that no longer exist: {sorted(stale)}"


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_the_documented_keys_are_all_present(name):
    """Call the function and check every promised key is there.

    `KeyError` is the failure mode this replaces, so the assertion reports the whole
    missing set rather than the first one: which keys are absent is the diagnosis.
    """
    call, required = CONTRACTS[name]
    result = call()
    if required is None:
        assert isinstance(result, str) and result, f"{name} returns a document"
        return
    assert isinstance(result, dict), f"{name} returns a dict"
    missing = required - set(result)
    assert not missing, (
        f"pm3_rs.native.{name}() promises {sorted(required)} but does not return "
        f"{sorted(missing)}; it returned {sorted(result)}"
    )


def test_every_optimizer_agrees_on_every_shared_name():
    """The regression itself: every relaxation entry point, key for key.

    `optimize(...)["energy_ev"]` and `relax(...)["iterations"]` were both `KeyError` in
    0.2.3, in opposite directions, which is what an asymmetry costs: neither caller was
    wrong and both were broken. `divide_and_conquer_optimize` is here from its first
    release so it never gets to be the third spelling.
    """
    optimizers = (
        ("optimize", native.optimize(WATER_Z, WATER_R)),
        ("relax", native.relax(WATER_Z, WATER_R, CUBE, max_steps=1)),
        (
            "divide_and_conquer_optimize",
            native.divide_and_conquer_optimize(WATER_Z, WATER_R, max_steps=2),
        ),
    )

    for label, result in optimizers:
        for key in OPTIMIZER:
            assert key in result, f"{label}() is missing {key!r}"
        # And the aliases have to be aliases, not two different numbers.
        assert result["positions"] == result["positions_angstrom"], label
        assert result["steps"] == result["iterations"], label
        assert result["energy_ev"] == pytest.approx(
            result["energy_hartree"] * 27.211386245988, abs=1e-9
        ), label


@pytest.mark.parametrize(
    "optimizer, kwargs",
    [
        ("optimize", {}),
        ("divide_and_conquer_optimize", {"buffer_radius": 6.0}),
    ],
)
def test_a_converged_optimizer_met_the_tolerance_it_was_given(optimizer, kwargs):
    """`converged: True` means the forces are under `force_tol`, in the documented unit.

    `force_tol` is eV/Angstrom in every Python signature and eV/Bohr inside the
    optimizer, and a gradient is per unit length -- so it scales the *opposite* way to a
    length. Multiplying by the wrong one of the two conversion factors leaves the
    tolerance 3.6x looser than asked for, and nothing says so: the run still stops, still
    reports `converged: True`, and on a soft coordinate stops a third of an Angstrom
    short. The same slip is on record for `relax`'s `stress_tol`, which was 6.75x loose
    for the same reason, so it is checked here rather than trusted.

    A dimer, because the intermolecular coordinate is soft enough for a loose tolerance
    to show; a single water's stiff bonds hide it.
    """
    z = [8, 1, 1, 8, 1, 1]
    r = [
        [0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469],
        [3.0, 0.0, 0.117], [3.0, 0.757, -0.469], [3.0, -0.757, -0.469],
    ]
    force_tol = 0.02

    result = getattr(native, optimizer)(z, r, force_tol=force_tol, **kwargs)
    assert result["converged"], f"{optimizer} did not converge; the tolerance is untested"

    # Re-measure at the geometry it stopped at, through the matching force accessor.
    forces_fn = native.divide_and_conquer_forces if "conquer" in optimizer else native.forces
    forces = forces_fn(z, result["positions_angstrom"], **kwargs)["forces_ev_per_angstrom"]
    largest = np.abs(np.asarray(forces)).max()
    assert largest <= force_tol, (
        f"{optimizer} reported converged at force_tol={force_tol} eV/A but the largest "
        f"force there is {largest:.4f} eV/A -- {largest / force_tol:.1f}x the tolerance"
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda n: native.frequencies(WATER_Z, WATER_R, cphf_max_iter=n),
        lambda n: native.hessian(WATER_Z, WATER_R, cphf_max_iter=n),
        lambda n: native.ir_spectrum(WATER_Z, WATER_R, cphf_max_iter=n),
        lambda n: native.phonons(WATER_Z, WATER_R, CUBE, cphf_max_iter=n),
        lambda n: native.born_charges(WATER_Z, WATER_R, CUBE, cphf_max_iter=n),
        lambda n: native.dielectric(WATER_Z, WATER_R, CUBE, cphf_max_iter=n),
    ],
    ids=["frequencies", "hessian", "ir_spectrum", "phonons", "born_charges", "dielectric"],
)
def test_the_cphf_cap_reaches_the_solver(call):
    """`cphf_max_iter` has to be honoured, not accepted and dropped.

    It was a pair of hard-coded 400s inside the crate, so a stiff response could only be
    rescued by editing and rebuilding. An argument that is accepted and ignored is worse
    than no argument, so this asserts it in the only way that cannot be faked: a cap of
    one is too few passes for any real response, and must **fail**.

    A generous cap must still succeed, which is what says the failure came from the cap
    rather than from the call being broken.
    """
    call(2000)

    with pytest.raises(Exception) as caught:
        call(1)
    message = str(caught.value).lower()
    assert "converge" in message, (
        f"a one-iteration CPHF failed for a reason other than not converging: {caught.value}"
    )


def test_the_phonon_eigenvectors_are_a_basis_and_not_just_present():
    """`modes` has to be the eigenvectors, not an array of the right shape.

    A key that exists and holds nothing useful is the failure this file is for, so this
    checks the two properties only real eigenvectors have: the set is orthonormal, and
    the acoustic modes -- the ones the projection set to exactly zero -- are the uniform
    translations, which is what identifies them physically rather than by their index.
    """
    result = native.phonons(WATER_Z, WATER_R, CUBE)
    modes = np.asarray(result["modes"])
    masses = np.asarray(result["masses"])
    ndof = 3 * len(WATER_Z)
    assert modes.shape == (ndof, ndof), f"expected {ndof}x{ndof} modes, got {modes.shape}"

    # Orthonormal: `M Mᵀ = I` for a real symmetric eigenproblem, mode-major as returned.
    gram = modes @ modes.T
    assert np.allclose(gram, np.eye(ndof), atol=1e-8), (
        "the modes are not orthonormal, so they are not an eigenbasis"
    )

    # The exactly-zero frequencies are the acoustic branch, and an acoustic mode at Γ moves
    # every atom the same way. De-weighting by sqrt(mass) is what turns it back into that.
    frequencies = np.asarray(result["frequencies_cm"])
    acoustic = np.flatnonzero(frequencies == 0.0)
    assert len(acoustic) == 3, f"expected three exactly-zero modes, got {len(acoustic)}"
    for index in acoustic:
        cartesian = (modes[index].reshape(-1, 3) / np.sqrt(masses)[:, None])
        # Every atom's displacement parallel to the first one's, to a part in 1e-6.
        first = cartesian[0]
        assert np.linalg.norm(first) > 0.0
        for row in cartesian[1:]:
            assert np.allclose(row, first, rtol=1e-6, atol=1e-9), (
                f"acoustic mode {index} is not a uniform translation: {cartesian}"
            )


def test_the_orbital_spectrum_agrees_with_what_single_point_reports():
    """`orbitals` and `single_point` must name the same frontier.

    They reach it by different routes -- `single_point` reads `Pm3Result`'s resolved
    `homo_ev`/`lumo_ev`, `orbitals` hands out the whole spectrum and the indices into it -- and
    two answers to "where is the HOMO" is exactly the kind of thing this file exists to prevent.
    """
    sp = native.single_point(WATER_Z, WATER_R)
    orb = native.orbitals(WATER_Z, WATER_R)

    assert orb["homo_ev"] == pytest.approx(sp["homo_ev"], abs=1e-12)
    assert orb["lumo_ev"] == pytest.approx(sp["lumo_ev"], abs=1e-12)
    # And the indices point at those energies in the spectrum.
    energies = orb["mo_energies_ev"]
    assert energies[orb["homo_index"]] == pytest.approx(sp["homo_ev"], abs=1e-12)
    assert energies[orb["lumo_index"]] == pytest.approx(sp["lumo_ev"], abs=1e-12)
    assert orb["gap_ev"] == pytest.approx(sp["lumo_ev"] - sp["homo_ev"], abs=1e-12)

    # Ascending, aufbau-filled, and shaped as documented.
    assert all(a <= b for a, b in zip(energies, energies[1:]))
    assert orb["occupations"] == [2.0] * orb["n_occupied"] + [0.0] * (
        len(energies) - orb["n_occupied"]
    )
    coefficients = np.asarray(orb["mo_coefficients"])
    assert coefficients.shape == (len(energies), len(energies))
    assert len(orb["ao_labels"]) == len(energies), "one label per coefficient row"
    # Water: four orbitals on oxygen, one on each hydrogen, in that order.
    assert [tuple(label) for label in orb["ao_labels"]] == [
        (0, "O", "s"), (0, "O", "px"), (0, "O", "py"), (0, "O", "pz"),
        (1, "H", "s"), (2, "H", "s"),
    ]


def test_an_open_shell_orbital_set_carries_both_spins():
    """A radical has two genuinely different orbital sets, and the beta one is not optional."""
    methyl_z = [6, 1, 1, 1]
    methyl_r = [[0.0, 0.0, 0.0], [0.0, 1.078, 0.0],
                [0.9336, -0.539, 0.0], [-0.9336, -0.539, 0.0]]
    orb = native.orbitals(methyl_z, methyl_r, multiplicity=2)

    assert orb["unrestricted"] is True
    assert orb["n_occupied"] == orb["n_beta"] + 1, "a doublet has one unpaired electron"
    assert orb["mo_energies_beta_ev"] is not None
    assert orb["occupations"] == [1.0] * orb["n_occupied"] + [0.0] * (
        len(orb["mo_energies_ev"]) - orb["n_occupied"]
    )
    # The frontier is taken across both channels: the beta LUMO of a radical sits below the
    # alpha one, so reading the alpha spectrum alone gives the wrong answer.
    beta = orb["mo_energies_beta_ev"]
    assert orb["lumo_ev"] == pytest.approx(
        min(orb["mo_energies_ev"][orb["n_occupied"]], beta[orb["n_beta"]]), abs=1e-12
    )


def test_every_native_function_documents_its_keys():
    """Each wrapper's docstring names the keys it returns.

    A contract that lives only in a test is a contract the caller cannot read. The
    check is deliberately loose -- it looks for the key name anywhere in the docstring --
    because the prose is written for people, not parsed.
    """
    unexplained = {}
    for name, (_call, required) in sorted(CONTRACTS.items()):
        if required is None:
            continue
        doc = inspect.getdoc(getattr(native, name)) or ""
        absent = sorted(key for key in required if key not in doc)
        if absent:
            unexplained[name] = absent
    assert not unexplained, (
        "these functions return keys their docstring never mentions: "
        + "; ".join(f"{n}: {ks}" for n, ks in unexplained.items())
    )
