# SPDX-License-Identifier: GPL-3.0-or-later
"""The periodic and divide-and-conquer Python surface.

Every claim the docstrings make is executed here. The point is not to re-derive
the physics — the Rust tests do that — but to pin the *boundary*: units, shapes,
Voigt ordering, and the conventions the ASE calculator promises.
"""

from __future__ import annotations

import numpy as np
import pytest

import pm3_rs

WATER = {
    "numbers": [8, 1, 1],
    "positions": [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.24, 0.9278, 0.0]],
}
# 18 Bohr, which is inside the Gamma-point validity condition.
EDGE_ANGSTROM = 18.0 * 0.52917721
BOHR = 0.52917721

# The cell the response tests use, in Angstrom.
#
# Not ``14.0 * BOHR``, which is where these started and which is a trap. The Gamma-point condition
# is that every periodic width exceed ``short_range_cutoff``, and that default is *exactly* 14
# Bohr -- so a 14 Bohr cell sits on the boundary, and ``14.0 * 0.52917721`` converts back to
# 13.999999982 Bohr and lands on the invalid side of it. Nothing about that fails loudly: the SCF
# converges cleanly to a well-defined wrong answer, 35 eV away from the one just across the
# boundary, carrying a spurious imaginary mode that made ``eps_0`` incomplete. Every test below
# asserts a positive margin rather than trusting this constant to stay right.
RESPONSE_CELL = 18.0 * BOHR


def water_chain(n, spacing=3.2):
    numbers, positions = [], []
    for i in range(n):
        x = spacing * i
        numbers += [8, 1, 1]
        positions += [[x, 0.0, 0.0], [x + 0.9584, 0.0, 0.0], [x - 0.24, 0.9278, 0.0]]
    return numbers, positions


class TestPeriodicNative:
    def test_gamma_point_single_point(self):
        r = pm3_rs.periodic_single_point(
            WATER["numbers"], WATER["positions"], EDGE_ANGSTROM
        )
        assert r["converged"]
        assert r["n_kpoints"] == 1
        assert np.isfinite(r["energy_ev"])
        # A cube of that edge holds one water nearly in isolation. "Nearly" is the
        # operative word: the remaining few meV is the real dipole-dipole
        # interaction of the water lattice, which falls off as 1/L^3 and is a
        # result rather than an error.
        molecular = pm3_rs.single_point(WATER["numbers"], WATER["positions"])
        assert abs(r["energy_ev"] - molecular["energy_ev"]) < 1.0e-2

    def test_gamma_margin_is_reported_and_positive_here(self):
        r = pm3_rs.periodic_single_point(
            WATER["numbers"], WATER["positions"], EDGE_ANGSTROM
        )
        # 18 Bohr against a 14 Bohr exchange cutoff leaves 4 Bohr of margin.
        assert r["gamma_margin_bohr"] == pytest.approx(4.0, abs=1e-6)

    def test_gamma_margin_goes_negative_for_a_narrow_cell(self):
        """The diagnostic has to actually diagnose. A 12 Bohr cell is too narrow."""
        r = pm3_rs.periodic_single_point(
            WATER["numbers"], WATER["positions"], 12.0 * BOHR
        )
        assert r["gamma_margin_bohr"] < 0.0

    def test_a_k_mesh_runs_and_reports_its_points(self):
        r = pm3_rs.periodic_single_point(
            WATER["numbers"], WATER["positions"], 12.0 * BOHR, kpts=(2, 2, 2)
        )
        assert r["converged"]
        # A Gamma-centred 2x2x2 mesh is entirely self-paired, so nothing reduces.
        assert r["n_kpoints"] == 8
        assert np.isfinite(r["band_gap_ev"])

    def test_cell_accepts_the_shapes_ase_users_write(self):
        by_scalar = pm3_rs.periodic_single_point(
            WATER["numbers"], WATER["positions"], EDGE_ANGSTROM
        )["energy_ev"]
        by_lengths = pm3_rs.periodic_single_point(
            WATER["numbers"], WATER["positions"], [EDGE_ANGSTROM] * 3
        )["energy_ev"]
        by_matrix = pm3_rs.periodic_single_point(
            WATER["numbers"], WATER["positions"], np.eye(3) * EDGE_ANGSTROM
        )["energy_ev"]
        assert by_scalar == pytest.approx(by_lengths)
        assert by_scalar == pytest.approx(by_matrix)

    def test_forces_and_stress_units_and_shapes(self):
        r = pm3_rs.periodic_forces(WATER["numbers"], WATER["positions"], EDGE_ANGSTROM)
        forces = np.asarray(r["forces_ev_per_angstrom"])
        assert forces.shape == (3, 3)
        # Translational invariance survives the unit conversion.
        assert np.allclose(forces.sum(axis=0), 0.0, atol=1e-8)
        stress = np.asarray(r["stress_ev_per_angstrom3"])
        assert stress.shape == (6,)
        assert np.all(np.isfinite(stress))

    def test_a_slab_reports_an_in_plane_stress_and_exact_zeros_elsewhere(self):
        """A slab strains in its plane and nowhere else.

        The out-of-plane components must be exactly zero rather than small: a
        variable-cell optimizer reads a small number as a direction to move in, and
        would stretch the vacuum.
        """
        cell = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 30.0]]
        r = pm3_rs.periodic_forces(
            WATER["numbers"], WATER["positions"], cell, pbc=(True, True, False)
        )
        stress = np.asarray(r["stress_ev_per_angstrom3"])
        assert stress.shape == (6,)
        assert np.all(np.isfinite(stress))
        # Voigt order is (xx, yy, zz, yz, xz, xy); z is the non-periodic direction.
        for index, label in ((2, "zz"), (3, "yz"), (4, "xz")):
            assert stress[index] == 0.0, f"slab stress {label} should be absent, got {stress[index]}"
        assert abs(stress[0]) + abs(stress[1]) > 0.0, "the in-plane stress should be live"
        assert np.asarray(r["forces_ev_per_angstrom"]).shape == (3, 3)

    def test_pbc_without_a_cell_is_refused(self):
        with pytest.raises(Exception):
            pm3_rs.native._native.periodic_single_point(
                WATER["numbers"], WATER["positions"], None, [True, True, True]
            )

    def test_phonons_have_three_acoustic_modes(self):
        r = pm3_rs.phonons(WATER["numbers"], WATER["positions"], EDGE_ANGSTROM)
        frequencies = np.asarray(r["frequencies_cm"])
        assert frequencies.shape == (9,)
        # Exactly three zeros: the acoustic branch is projected out of the mass-weighted matrix,
        # so it is empty rather than small. Only translations -- a crystal is not invariant under
        # rotating its contents inside a fixed lattice, so the librations are left alone.
        assert int(np.sum(frequencies == 0.0)) == 3, frequencies
        # The three internal modes of water survive whatever the cell does.
        assert np.sort(np.abs(frequencies))[-3:].min() > 1000.0
        # Now the *pre*-projection residual, so it measures the lattice sums rather than restating
        # what the projection did. Reading it off the projected spectrum could only return zero.
        assert 0.0 < r["acoustic_residual_cm"] < 20.0, r["acoustic_residual_cm"]


class TestDivideAndConquer:
    def test_a_reaching_buffer_reproduces_the_full_result(self):
        numbers, positions = water_chain(3)
        full = pm3_rs.single_point(numbers, positions)
        dc = pm3_rs.divide_and_conquer(
            numbers, positions, buffer_radius=200.0, smearing_ev=1e-4
        )
        assert dc["converged"]
        assert dc["dropped_pairs"] == 0
        assert dc["energy_ev"] == pytest.approx(full["energy_ev"], abs=1e-5)

    def test_widening_the_buffer_converges(self):
        numbers, positions = water_chain(5)
        full = pm3_rs.single_point(numbers, positions)["energy_ev"]
        narrow = abs(
            pm3_rs.divide_and_conquer(
                numbers, positions, core_radius=1.6, buffer_radius=2.1
            )["energy_ev"]
            - full
        )
        wide = abs(
            pm3_rs.divide_and_conquer(
                numbers, positions, core_radius=1.6, buffer_radius=7.4
            )["energy_ev"]
            - full
        )
        assert wide < narrow
        assert wide < 1.0e-3

    def test_the_partitioning_is_reported(self):
        numbers, positions = water_chain(6)
        r = pm3_rs.divide_and_conquer(
            numbers, positions, core_radius=1.6, buffer_radius=2.1
        )
        # Six waters cut into pieces: more than one subsystem, none of them the
        # whole system.
        assert r["n_subsystems"] > 1
        assert 0 < r["largest_subsystem"] < 6 * 6

    def test_the_periodic_path_runs(self):
        numbers, positions = water_chain(3)
        r = pm3_rs.divide_and_conquer(
            numbers, positions, cell=30.0 * BOHR, buffer_radius=200.0, smearing_ev=1e-4
        )
        assert r["converged"]
        assert "gamma_margin_bohr" in r


ase = pytest.importorskip("ase")


class TestAseCalculator:
    def _cell(self, atoms):
        atoms.set_cell([EDGE_ANGSTROM] * 3)
        atoms.set_pbc(True)
        return atoms

    def test_pbc_switches_the_calculator_to_the_periodic_path(self):
        from ase import Atoms

        from pm3_rs.ase import PM3

        molecular = Atoms(numbers=WATER["numbers"], positions=WATER["positions"])
        molecular.calc = PM3()
        molecular_energy = molecular.get_potential_energy()

        periodic = self._cell(
            Atoms(numbers=WATER["numbers"], positions=WATER["positions"])
        )
        periodic.calc = PM3()
        periodic_energy = periodic.get_potential_energy()
        # Same physics in a large cell, so the two agree to within the lattice's
        # own dipole-dipole interaction — which is what says the periodic branch
        # was taken and produced something sane rather than that it was skipped.
        assert periodic_energy == pytest.approx(molecular_energy, abs=1e-2)
        assert periodic_energy != molecular_energy

    def test_stress_is_a_voigt_vector_in_ase_units(self):
        from ase import Atoms

        from pm3_rs.ase import PM3

        atoms = self._cell(Atoms(numbers=WATER["numbers"], positions=WATER["positions"]))
        atoms.calc = PM3()
        stress = atoms.get_stress()
        assert stress.shape == (6,)
        assert np.all(np.isfinite(stress))

    def test_stress_on_a_molecule_is_refused(self):
        from ase import Atoms

        from pm3_rs.ase import PM3

        atoms = Atoms(numbers=WATER["numbers"], positions=WATER["positions"])
        atoms.calc = PM3()
        with pytest.raises(Exception):
            atoms.get_stress()

    def test_get_phonons(self):
        from ase import Atoms

        from pm3_rs.ase import PM3

        atoms = self._cell(Atoms(numbers=WATER["numbers"], positions=WATER["positions"]))
        atoms.calc = PM3()
        modes = atoms.calc.get_phonons(atoms)
        assert len(modes["frequencies_cm"]) == 9

    def test_a_k_mesh_can_be_requested_at_construction(self):
        from ase import Atoms

        from pm3_rs.ase import PM3

        atoms = Atoms(numbers=WATER["numbers"], positions=WATER["positions"])
        atoms.set_cell([12.0 * BOHR] * 3)
        atoms.set_pbc(True)
        atoms.calc = PM3(kpts=(2, 2, 2))
        assert np.isfinite(atoms.get_potential_energy())

    def test_one_evaluation_converges_one_scf(self, monkeypatch):
        """A `calculate` cycle must call the extension once, not twice.

        The comment in `_calculate_periodic` says producing the gradient unasked "turns three
        SCFs per step into one". It did not: it called `periodic_forces` and then
        `periodic_single_point`, and every field it read off the second is already returned by
        the first. That is one wasted self-consistent solve per force evaluation, which in a
        molecular-dynamics run is half the total cost.
        """
        from ase import Atoms

        from pm3_rs import native
        from pm3_rs.ase import PM3

        calls = []
        for name in ("periodic_forces", "periodic_single_point", "forces", "single_point"):
            original = getattr(native, name)

            def counted(*args, _name=name, _original=original, **kwargs):
                calls.append(_name)
                return _original(*args, **kwargs)

            monkeypatch.setattr(native, name, counted)

        atoms = self._cell(Atoms(numbers=WATER["numbers"], positions=WATER["positions"]))
        atoms.calc = PM3()
        atoms.get_potential_energy()
        atoms.get_forces()
        atoms.get_charges()
        assert calls == ["periodic_forces"], (
            f"one geometry, one SCF -- the extension was called {calls}"
        )

        calls.clear()
        molecular = Atoms(numbers=WATER["numbers"], positions=WATER["positions"])
        molecular.calc = PM3()
        molecular.get_potential_energy()
        molecular.get_charges()
        molecular.get_dipole_moment()
        assert calls == ["forces"], f"one geometry, one SCF -- the extension was called {calls}"

    def test_smearing_reaches_the_energy_and_the_charges_alike(self):
        """Everything in one `results` dict must come from the same SCF.

        `periodic_forces` had no `smearing_ev` argument at all, so the energy and forces came
        from a strictly-filled solution while the charges came from a smeared one. On an
        insulator the two agree and nothing shows; on a metal -- the only reason to set a
        smearing -- they are different self-consistent states reported as one result.

        Checked against the wrapper layer directly, because that is the layer that has to agree.
        """
        from ase import Atoms

        from pm3_rs import native
        from pm3_rs.ase import PM3

        atoms = Atoms(numbers=WATER["numbers"], positions=WATER["positions"])
        atoms.set_cell([12.0 * BOHR] * 3)
        atoms.set_pbc(True)
        atoms.calc = PM3(kpts=(2, 2, 2), smearing_ev=0.5)

        direct = native.periodic_forces(
            atoms.get_atomic_numbers(),
            atoms.get_positions(),
            np.asarray(atoms.get_cell(), dtype=float),
            pbc=[True, True, True],
            kpts=(2, 2, 2),
            smearing_ev=0.5,
        )
        assert atoms.get_potential_energy() == pytest.approx(direct["energy_ev"], abs=1e-9)
        assert np.allclose(atoms.get_charges(), direct["charges"], atol=1e-9)

    def test_a_field_under_periodic_boundary_conditions_is_refused_not_dropped(self):
        """`-F.r` is not lattice-periodic, and the periodic paths say so.

        The calculator stored a `field` and then never passed it to any periodic entry point --
        none of which takes one -- so `pbc::refuse_field` was never reached and the caller got
        field-free numbers with nothing to indicate the field had been ignored.
        """
        from ase import Atoms

        from pm3_rs.ase import PM3

        atoms = self._cell(Atoms(numbers=WATER["numbers"], positions=WATER["positions"]))
        atoms.calc = PM3(field=[0.0, 0.0, 0.5])
        for call in (
            lambda: atoms.get_potential_energy(),
            lambda: atoms.calc.get_phonons(atoms),
            lambda: atoms.calc.get_dynamical_matrix([0.25, 0.0, 0.0], atoms),
            lambda: atoms.calc.get_bands([[0.0, 0.0, 0.0]], atoms),
            lambda: atoms.calc.relax(atoms),
        ):
            with pytest.raises(RuntimeError, match="lattice-periodic"):
                call()

        # And the molecular path still carries one, which is the half that always worked.
        molecular = Atoms(numbers=WATER["numbers"], positions=WATER["positions"])
        molecular.calc = PM3(field=[0.0, 0.0, 0.5])
        with_field = molecular.get_potential_energy()
        molecular.calc = PM3()
        assert with_field != molecular.get_potential_energy()

    def test_the_calculator_mesh_reaches_the_dynamical_matrix(self):
        """`get_dynamical_matrix` honours `PM3(kpts=...)`, as `get_phonons` does.

        It read only the call-site `kpts`, so the two accessors described different
        calculations for the same calculator, with no argument at either call site saying so.
        """
        from ase import Atoms

        from pm3_rs.ase import PM3

        atoms = Atoms(numbers=WATER["numbers"], positions=WATER["positions"])
        atoms.set_cell([12.0 * BOHR] * 3)
        atoms.set_pbc(True)
        q = [0.5, 0.0, 0.0]

        meshed = PM3(kpts=(2, 1, 1)).get_dynamical_matrix(q, atoms)[0]
        explicit = PM3().get_dynamical_matrix(q, atoms, kpts=(2, 1, 1))[0]
        gamma = PM3().get_dynamical_matrix(q, atoms)[0]

        assert np.allclose(meshed, explicit), "the calculator's mesh was not used"
        assert not np.allclose(meshed, gamma), (
            "the 2x1x1 response equals the Gamma-only one, so this test cannot tell them apart"
        )

    def test_a_periodic_cell_refuses_the_molecular_near_field_split(self):
        """`long_range_cutoff` is molecular; the periodic path would have ignored it.

        `run_dc_gamma` never reads it -- the periodic long range comes from the lattice sum --
        so setting it and passing a cell ran the unscreened O(N^2) route while the caller
        believed they had asked for the linear-scaling one.
        """
        numbers, positions = water_chain(3)
        with pytest.raises(ValueError, match="long_range_cutoff"):
            pm3_rs.divide_and_conquer(
                numbers, positions, cell=30.0 * BOHR, long_range_cutoff=12.0
            )
        # Molecular, it is honoured rather than refused.
        assert np.isfinite(
            pm3_rs.divide_and_conquer(numbers, positions, long_range_cutoff=12.0)["energy_ev"]
        )

    def test_the_free_energy_is_the_mermin_one_and_says_so(self):
        """`free_energy` is `E - TS` of the electrons, and cannot be read as a Gibbs energy.

        Three quantities live near each other and are easy to confuse:

        * ``energy`` -- the electronic energy `E`.
        * ``free_energy`` -- the **Mermin electronic** free energy `E - TS`, whose gradient the
          forces are once the occupations are fractional. This is ASE's meaning of the name.
        * ``heat_of_formation_kcal`` -- MOPAC's parameterized dH_f at 298 K, fitted rather than
          computed, and not a thermodynamic potential at all.

        None of them is a Gibbs free energy: there is no zero-point energy, no vibrational
        partition function and no nuclear entropy anywhere in this package.
        """
        from ase import Atoms

        from pm3_rs.ase import PM3

        atoms = Atoms(numbers=WATER["numbers"], positions=WATER["positions"])
        atoms.set_cell([12.0 * BOHR] * 3)
        atoms.set_pbc(True)

        # Without smearing the occupations are a step, so `TS` is exactly zero and the two
        # energies must coincide -- not merely agree closely.
        cold = PM3(kpts=(2, 2, 2))
        atoms.calc = cold
        energy = atoms.get_potential_energy()
        assert cold.results["electronic_entropy_ts_ev"] == 0.0
        assert cold.results["free_energy"] == energy
        assert atoms.get_potential_energy(force_consistent=True) == energy

        # With smearing they separate, and the free energy is the lower of the two: `S >= 0`.
        warm = PM3(kpts=(2, 2, 2), smearing_ev=1.0)
        atoms.calc = warm
        warm_energy = atoms.get_potential_energy()
        ts = warm.results["electronic_entropy_ts_ev"]
        free = atoms.get_potential_energy(force_consistent=True)
        assert ts > 0.0, "a smeared metal-like filling has entropy"
        assert free == pytest.approx(warm_energy - ts, abs=1e-12)
        assert free < warm_energy

        # And it is none of the other two named quantities.
        assert free != warm.results["heat_of_formation_kcal"]

    def test_changing_a_model_selector_invalidates_the_cached_result(self):
        """Changing the charge on the calculator must change the energy it reports.

        ASE clears `results` when `check_state` reports a change, and the base implementation
        looks only at the `Atoms`. These selectors live as plain attributes, so nothing told ASE
        the model had moved and the previous answer came straight back out of the cache. The
        `_heavy` memo beside it fingerprinted all of them from the start -- so within one file
        the lazy accessors were right and the ASE properties were not.
        """
        from ase import Atoms

        from pm3_rs.ase import PM3

        atoms = Atoms(numbers=WATER["numbers"], positions=WATER["positions"])
        atoms.calc = PM3()
        neutral = atoms.get_potential_energy()

        atoms.calc.charge = 1
        atoms.calc.multiplicity = 2
        cation = atoms.get_potential_energy()
        assert cation != neutral, "the cached neutral energy came back for the cation"

        # And it really is the cation's energy, not merely a different number.
        direct = pm3_rs.single_point(
            WATER["numbers"], WATER["positions"], charge=1.0, multiplicity=2
        )
        assert cation == pytest.approx(direct["energy_ev"], abs=1e-9)

        # Setting them back returns the original answer, so the invalidation is on the value and
        # not merely on having been touched.
        atoms.calc.charge = 0
        atoms.calc.multiplicity = 1
        assert atoms.get_potential_energy() == pytest.approx(neutral, abs=1e-9)

    def test_born_charges_and_dielectric_reach_python(self):
        """The v0.2.2 response properties are callable from every layer, not only from Rust."""
        from ase import Atoms

        from pm3_rs.ase import PM3

        cell = RESPONSE_CELL
        result = pm3_rs.born_charges(WATER["numbers"], WATER["positions"], cell)
        assert result["gamma_margin_bohr"] > 0.0, (
            f"the margin is {result['gamma_margin_bohr']:.3g} Bohr, so one k-point is not enough "
            f"and these charges are a response of the wrong ground state"
        )
        born = np.asarray(result["born_charges"], dtype=float)
        assert born.shape == (3, 3, 3), "one 3x3 tensor per atom"
        # Translating the crystal produces no dipole. Reported rather than enforced, so the
        # number is there to be looked at.
        assert result["sum_rule_residual"] < 1e-6
        assert np.allclose(born.sum(axis=0), 0.0, atol=1e-6)

        # Oxygen carries the negative charge, and the tensors are anisotropic -- which is the
        # part a rigid point-charge model could not produce.
        assert born[0][0][0] < 0.0 < born[1][0][0]
        assert abs(born[0][0][0] - born[0][2][2]) > 1e-3, "an isotropic Z* means no response"

        d = pm3_rs.dielectric(WATER["numbers"], WATER["positions"], cell)
        alpha = np.asarray(d["polarizability"], dtype=float)
        epsilon = np.asarray(d["epsilon"], dtype=float)
        assert alpha.shape == (3, 3)
        assert np.allclose(alpha, alpha.T, atol=1e-6), "alpha must be symmetric"
        assert np.all(np.diag(alpha) > 0.0)
        assert np.all(np.diag(epsilon) > 1.0), "eps_inf exceeds one"

        # A chain has a polarizability but no dielectric constant: no volume to divide by.
        numbers, positions = water_chain(1)
        chain = pm3_rs.dielectric(
            numbers, positions, [6.0 * BOHR, 30.0 * BOHR, 30.0 * BOHR], pbc=[True, False, False]
        )
        assert chain["epsilon"] is None
        assert np.asarray(chain["polarizability"], dtype=float)[0][0] > 0.0

        atoms = self._cell(Atoms(numbers=WATER["numbers"], positions=WATER["positions"]))
        atoms.calc = PM3()
        assert np.allclose(
            np.asarray(atoms.calc.get_born_charges(atoms)["born_charges"], dtype=float).shape,
            (3, 3, 3),
        )
        assert atoms.calc.get_dielectric(atoms)["epsilon"] is not None

    def test_the_static_dielectric_tensor_reaches_python(self):
        """``eps_0`` is opt-in, and is ``eps_inf`` plus a positive ionic term.

        Opt-in because it costs a Gamma-point phonon run and a set of Born charges on top of the
        field response. The keys only appear when it is asked for, which is what the first half
        of this checks -- a caller who did not pay for it must not find a stale or default one.
        """
        from ase import Atoms

        from pm3_rs.ase import PM3

        cell = RESPONSE_CELL
        plain = pm3_rs.dielectric(WATER["numbers"], WATER["positions"], cell)
        assert plain["gamma_margin_bohr"] > 0.0, (
            f"the margin is {plain['gamma_margin_bohr']:.3g} Bohr, so this tensor is the response "
            f"of a ground state one k-point cannot represent"
        )
        assert "epsilon_static" not in plain, "the ionic half must not appear unasked"

        full = pm3_rs.dielectric(
            WATER["numbers"], WATER["positions"], cell, include_ionic=True
        )
        static = np.asarray(full["epsilon_static"], dtype=float)
        electronic = np.asarray(full["epsilon_electronic"], dtype=float)
        ionic = np.asarray(full["epsilon_ionic"], dtype=float)

        # The electronic half is the same tensor the cheap call returns, not a second calculation
        # that happens to be close: the same response solved the same way.
        assert np.allclose(electronic, np.asarray(plain["epsilon"], dtype=float), atol=1e-9)
        assert np.allclose(static, electronic + ionic, atol=1e-12)

        # Letting the nuclei relax can only add polarizability, so eps_0 >= eps_inf along every
        # direction. Checked on the diagonal, where the inequality is a statement about a single
        # direction rather than about the ordering of two matrices.
        assert np.all(np.diag(ionic) > 0.0), f"the ionic term is not positive: {np.diag(ionic)}"
        assert np.all(np.diag(static) > np.diag(electronic))

        # Three skipped modes are the acoustic branch. More would mean this geometry is not a
        # minimum, and then the ionic term is missing whatever those modes carried -- so the
        # count is asserted rather than the tensor being trusted on its own.
        assert full["skipped_modes"] == 3, (
            f"{full['skipped_modes']} modes had omega^2 <= 0; above the three acoustic ones this "
            f"structure is not a minimum and eps_0 is incomplete"
        )

        atoms = self._cell(Atoms(numbers=WATER["numbers"], positions=WATER["positions"]))
        atoms.calc = PM3()
        # The memo is keyed on `include_ionic`, so the cheap call must not answer the rich one.
        assert "epsilon_static" not in atoms.calc.get_dielectric(atoms)
        rich = atoms.calc.get_dielectric(atoms, include_ionic=True)
        assert np.allclose(
            np.asarray(rich["epsilon_static"], dtype=float),
            np.asarray(rich["epsilon_electronic"], dtype=float)
            + np.asarray(rich["epsilon_ionic"], dtype=float),
            atol=1e-12,
        )

    def test_berry_polarization_and_phonon_bands_reach_python(self):
        """Two features that existed only in Rust until the wiring audit found them."""
        from ase import Atoms

        from pm3_rs.ase import PM3

        numbers = [9, 1]
        positions = [[0.0, 0.0, 0.0], [0.93, 0.0, 0.0]]
        cell = 16.0 * BOHR

        p = pm3_rs.berry_polarization(numbers, positions, cell, strings=12)
        assert p["gamma_margin_bohr"] > 0.0
        assert p["string_length"] == 12
        total = np.asarray(p["total"], dtype=float)
        assert np.allclose(
            total,
            np.asarray(p["electronic"], dtype=float)
            + np.asarray(p["ionic"], dtype=float),
            atol=1e-12,
        )
        # The quantum is `a/V` per lattice vector, and the total is only defined against it.
        #
        # Asserted as a relationship rather than against a literal. `16.0 * BOHR` is an Angstrom
        # number, and which Bohr it converts back to depends on whose conversion constant is
        # used -- pinning a digit string would be testing that constant, not this formula.
        #
        # For a cubic cell of edge `a` the quantum is `a/a^3 = 1/a^2`, so the edge it implies is
        # recoverable and can be checked against the one asked for.
        quantum = np.asarray(p["quantum"], dtype=float)
        assert quantum.shape == (3, 3)
        assert np.allclose(quantum, np.diag(np.diag(quantum)), atol=1e-15), (
            "a cubic cell's quanta lie along the axes"
        )
        implied_edge = 1.0 / np.sqrt(quantum[0][0])
        assert implied_edge == pytest.approx(16.0, rel=1e-6), (
            f"the quantum implies a cell edge of {implied_edge:.6f} Bohr, not the 16 asked for"
        )
        assert np.allclose(np.diag(quantum), quantum[0][0], rtol=1e-12), "cubic is isotropic"
        # Non-vacuity: a non-polar cell would make every assertion above pass on zeros.
        assert abs(np.asarray(p["ionic"], dtype=float)[0]) > 1e-6

        # Refining the string must not move a converged answer much.
        coarse = pm3_rs.berry_polarization(numbers, positions, cell, strings=6)
        assert abs(coarse["phase"][0] - p["phase"][0]) < 1e-4

        bands = pm3_rs.phonon_bands(
            numbers,
            positions,
            cell,
            supercell=[2, 1, 1],
            path=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            points=4,
        )
        assert bands["supercell"] == [2, 1, 1]
        assert len(bands["q"]) == 5, "four per segment plus the final corner"
        assert all(len(row) == 3 * len(numbers) for row in bands["frequencies_cm"])
        assert np.isfinite(bands["acoustic_sum_rule_residual"])

        atoms = self._cell(Atoms(numbers=numbers, positions=positions))
        atoms.calc = PM3()
        assert atoms.calc.get_berry_polarization(atoms)["string_length"] == 12
        assert (
            atoms.calc.get_phonon_bands(
                [2, 1, 1], [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], atoms=atoms, points=2
            )["supercell"]
            == [2, 1, 1]
        )

    def test_the_finite_field_reaches_python(self):
        """The Berry-phase finite field, and the polarizability it reproduces.

        Hydrogen, because it has no ``p`` orbitals: the intra-atomic ``dd`` moment the Berry
        phase cannot carry is then identically zero, and the two routes share the position
        operator by construction rather than by approximation. On a cell containing ``dd`` they
        legitimately differ -- on HF by a factor of 1.46 -- and that is not a defect in either.
        """
        numbers = [1, 1]
        positions = [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]
        cell = 20.0 * BOHR
        strength = 2.0e-4
        hartree_to_ev = 27.211386245988
        volume = (20.0) ** 3  # Bohr^3, the cell this uses

        runs = {}
        for sign in (-1.0, 1.0):
            runs[sign] = pm3_rs.finite_field(
                numbers, positions, cell, [sign * strength, 0.0, 0.0], [6, 1, 1]
            )
        for run in runs.values():
            assert run["converged"] and run["iterations"] > 0
            assert run["gamma_margin_bohr"] > 0.0
            # The reported enthalpy is the quantity it claims to be.
            expected = run["energy"] - volume * float(
                np.dot(run["field"], run["polarization"])
            )
            assert run["enthalpy_ev"] == pytest.approx(expected, abs=1e-9)

        dp = (
            np.asarray(runs[1.0]["polarization"], dtype=float)
            - np.asarray(runs[-1.0]["polarization"], dtype=float)
        ) / (2.0 * strength)
        alpha_xx = dp[0] * volume * hartree_to_ev

        cphf = np.asarray(
            pm3_rs.dielectric(numbers, positions, cell)["polarizability"], dtype=float
        )
        assert cphf[0][0] > 1.0, "no response here to compare"
        ratio = alpha_xx / cphf[0][0]
        assert abs(ratio - 1.0) < 5.0e-3, (
            f"the finite-field alpha_xx is {alpha_xx:.5f} against the CPHF {cphf[0][0]:.5f} "
            f"(ratio {ratio:.5f}); these share the SCF and nothing else"
        )

    def test_the_lo_to_term_splits_the_optical_branches(self):
        """`lo_to_direction` gives the longitudinal `q -> 0` limit instead of the transverse one.

        Without it a polar crystal's LO and TO branches stay degenerate at Gamma. The term is a
        rank-one positive semi-definite update, so it can only raise the spectrum, and the limit
        it produces depends on the direction -- which is the whole content of the effect.
        """
        cell = 10.0 * BOHR
        gamma = pm3_rs.phonons(WATER["numbers"], WATER["positions"], cell, q=[0.0, 0.0, 0.0])
        along_x = pm3_rs.phonons(
            WATER["numbers"],
            WATER["positions"],
            cell,
            q=[0.0, 0.0, 0.0],
            lo_to_direction=[1.0, 0.0, 0.0],
        )
        along_y = pm3_rs.phonons(
            WATER["numbers"],
            WATER["positions"],
            cell,
            q=[0.0, 0.0, 0.0],
            lo_to_direction=[0.0, 1.0, 0.0],
        )

        bare = np.asarray(gamma["frequencies_cm"], dtype=float)
        x = np.asarray(along_x["frequencies_cm"], dtype=float)
        y = np.asarray(along_y["frequencies_cm"], dtype=float)

        # Raised, never lowered -- to within the acoustic floor, where the square root of a
        # near-zero eigenvalue flips sign freely.
        assert np.all(x >= bare - 1e-3)
        assert np.max(x - bare) > 1.0, "the term changed nothing"
        # And direction dependent, which `D(0)` alone cannot be.
        assert np.max(np.abs(x - y)) > 1.0

        # It needs a `q` to be the limit of.
        with pytest.raises(ValueError, match="lo_to_direction"):
            pm3_rs.phonons(
                WATER["numbers"], WATER["positions"], cell, lo_to_direction=[1.0, 0.0, 0.0]
            )

    def test_divide_and_conquer_forces_reach_python(self):
        """`dc_gradient` and `dc_periodic_gradient` had no Python route at all.

        Both existed in Rust and were reachable from nowhere else, which left the one thing a
        large-system run actually needs -- forces -- out of reach from the layer such runs are
        driven from.

        A buffer wide enough to reach the whole molecule reproduces the full gradient exactly,
        which is what separates the partitioning from the truncation: a mismatch there is wiring
        rather than approximation.
        """
        numbers, positions = water_chain(3)

        reaching = pm3_rs.divide_and_conquer_forces(
            numbers, positions, buffer_radius=200.0, smearing_ev=1e-4
        )
        full = pm3_rs.forces(numbers, positions)
        assert reaching["stress_ev_per_angstrom3"] is None, "a molecule has no strain"
        assert np.allclose(
            reaching["forces_ev_per_angstrom"], full["forces_ev_per_angstrom"], atol=1e-4
        ), "a buffer reaching the whole system should reproduce the full gradient"

        # And the periodic branch reports a stress, in the Voigt shape ASE expects.
        periodic = pm3_rs.divide_and_conquer_forces(
            numbers, positions, cell=30.0 * BOHR, buffer_radius=200.0, smearing_ev=1e-4
        )
        stress = np.asarray(periodic["stress_ev_per_angstrom3"], dtype=float)
        assert stress.shape == (6,)
        assert np.all(np.isfinite(stress))
        assert "gamma_margin_bohr" in periodic

    def test_the_ase_calculator_exposes_divide_and_conquer_forces(self):
        from ase import Atoms

        from pm3_rs.ase import PM3

        numbers, positions = water_chain(3)
        atoms = Atoms(numbers=numbers, positions=positions)
        atoms.calc = PM3()
        result = atoms.calc.divide_and_conquer_forces(atoms, buffer_radius=200.0, smearing_ev=1e-4)
        forces = np.asarray(result["forces_ev_per_angstrom"], dtype=float)
        assert forces.shape == (len(numbers), 3)
        assert np.allclose(forces, atoms.get_forces(), atol=1e-4)
