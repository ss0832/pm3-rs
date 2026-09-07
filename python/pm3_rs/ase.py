# SPDX-License-Identifier: GPL-3.0-or-later
"""ASE calculator for PM3.

Uses ASE's convention throughout: energies in **eV**, forces in **eV/A**,
positions in **A**, and the Hessian in **eV/A^2**. Internally it calls the
native (atomic-unit) API and converts at this boundary. PM3 is natively an eV
method, so energies and forces pass through without a Hartree round-trip.

``charge``, ``multiplicity``, ``reference`` and ``method`` are set at construction
and used for every property evaluation.

- ``reference`` selects the SCF reference: ``"auto"`` (default; RHF closed shell,
  UHF open shell), ``"rhf"`` (force restricted) or ``"uhf"`` (force unrestricted).
- ``method`` selects the correction variant: ``"pm3"`` (default), ``"pm3-d3"``,
  ``"pm3-d3h4"`` or ``"pm3-d3h4x"``.

Energy, forces, charges and dipole are computed eagerly in :meth:`calculate` and
served through ASE's standard accessors (:meth:`get_potential_energy`,
:meth:`get_forces`, ...). The Hessian and vibrational frequencies are **not**
standard ASE properties and are computed lazily, on demand, via
:meth:`get_hessian` / :meth:`get_frequencies` (they are never part of a
``calculate`` cycle).
"""

from __future__ import annotations

import numpy as np

try:
    from ase.calculators.calculator import (
        Calculator,
        PropertyNotImplementedError,
        all_changes,
    )
    from ase.units import Bohr, Hartree
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The PM3 ASE calculator requires ASE. Install with `pip install pm3-rs-python[ase]`."
    ) from exc

from . import native

# Debye -> e.A (ASE dipole unit).
_DEBYE_TO_E_ANGSTROM = 0.2081943


class PM3(Calculator):
    """PM3 semiempirical calculator (ASE units: eV, eV/A, A, eV/A^2).

    Parameters
    ----------
    charge : int
        Total (formal) molecular charge.
    multiplicity : int
        Spin multiplicity (2S+1); 1 = singlet, 2 = doublet, ...
    reference : {"auto", "rhf", "uhf"}
        Restricted/unrestricted SCF reference (default "auto").
    method : {"pm3", "pm3-d3", "pm3-d3h4", "pm3-d3h4x"}
        PM3 correction variant (default "pm3").

    ``results`` is populated with ``energy`` (eV), ``forces`` (eV/A),
    ``charges`` (Mulliken, e), ``dipole`` (e.A) and ``heat_of_formation_kcal``
    (kcal/mol) -- each declared in :attr:`implemented_properties` so it is reachable
    through ASE's ``get_property`` / ``get_charges`` / ``get_dipole_moment`` API.
    """

    # Every entry here is populated by `calculate`, so `get_property(name)` works for each.
    # `hessian` (eV/A^2) is declared so it is reachable through the standard ASE property API,
    # but it is computed **lazily** -- only when a caller actually requests it (via
    # `get_hessian`/`get_property("hessian")`); a plain energy/forces cycle never builds it.
    implemented_properties = [
        "energy",
        # The **Mermin electronic** free energy `E - TS`, which is ASE's meaning of the name and
        # what `get_potential_energy(force_consistent=True)` returns. With Fermi-Dirac smearing
        # the variational functional is `E - TS`, so that -- not `E` -- is the potential whose
        # nuclear gradient `forces` is. Equal to `energy` whenever `smearing_ev` is zero, which
        # includes every molecular calculation.
        #
        # NOT a Gibbs free energy. `G = H - T*S_total` needs zero-point energy, a vibrational
        # partition function and a `pV` term, none of which exist in this crate -- and the
        # entropy here is the electronic one of fractional band occupations at a fictitious
        # electronic temperature, not a thermodynamic entropy of the nuclei. The related
        # `heat_of_formation_kcal` is a third thing again: MOPAC's parameterized dH_f at 298 K,
        # fitted rather than computed. `electronic_entropy_ts_ev` carries the `TS` itself so the
        # difference is visible rather than inferred.
        "free_energy",
        "electronic_entropy_ts_ev",
        "forces",
        "charges",
        "dipole",
        "heat_of_formation_kcal",
        "hessian",
        # Only produced for a periodic structure. ASE expects the 6-component Voigt vector in
        # eV/A^3, which is what `periodic_forces` returns. Every periodic dimensionality has one;
        # only an isolated cell reports no stress at all, having no strain to differentiate.
        "stress",
    ]

    def __init__(
        self,
        charge: int = 0,
        multiplicity: int = 1,
        reference: str = "auto",
        method: str = "pm3",
        kpts=None,
        smearing_ev: float = 0.0,
        magnetization: str = "fixed",
        field=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self.reference = str(reference)
        self.method = str(method)
        self.kpts = None if kpts is None else [int(k) for k in np.asarray(kpts).reshape(-1)]
        self.smearing_ev = float(smearing_ev)
        # How the two spin channels share a Fermi level: "fixed" holds the multiplicity's moment,
        # "free" lets it come out where the electronic structure puts it. See
        # `pm3_rs.native.periodic_single_point`.
        self.magnetization = str(magnetization)
        # Volts per Angstrom, matching the eV/A this class reports everywhere. Molecular only --
        # and the periodic accessors raise rather than compute a field-free answer, because
        # nothing under `pbc` carries a field and passing one through would drop it silently.
        self.field = None if field is None else [float(v) for v in np.asarray(field).reshape(-1)]
        # The heavy results that are not ASE properties, each with the geometry it was computed
        # at. See `_memo`.
        self._heavy = {}
        # What the model selectors were at the last `calculate`. See `check_state`.
        self._selectors_at_calculate = None

    def _selectors(self):
        """Everything other than the structure that changes the answer."""
        return (
            self.charge,
            self.multiplicity,
            self.reference,
            self.method,
            None if self.kpts is None else tuple(self.kpts),
            self.smearing_ev,
            self.magnetization,
            None if self.field is None else tuple(self.field),
        )

    def check_state(self, atoms, tol=1e-15):
        """Whether `results` is still valid: for the model as well as for the structure.

        ASE clears `results` when `check_state` reports a change, and the base implementation
        compares only the `Atoms`. These selectors are plain attributes rather than entries in
        `self.parameters`, and `Calculator.set()` -- the mechanism that would invalidate on a
        change -- is never involved, so

            atoms.get_potential_energy()
            atoms.calc.charge = 1
            atoms.get_potential_energy()   # the neutral energy, again

        returned the cached neutral result. The `_heavy` memo beside it has fingerprinted all of
        this from the start, so the lazy accessors were right and the ASE properties were not.
        """
        changes = list(super().check_state(atoms, tol))
        if self._selectors_at_calculate != self._selectors():
            changes.append("model")
        return changes

    def _fingerprint(self, atoms, extra=()):
        """What a cached heavy result was computed at.

        Everything a result could depend on: the structure, the cell and its periodicity, and
        every model selector this calculator carries. `extra` adds whatever the particular
        accessor takes as arguments -- a wavevector, a k-mesh, a path.

        "Every model selector" has to mean every one. The calculator-level `kpts` and
        `smearing_ev` were left out, so an accessor that reads them off `self` -- rather than
        taking them as an argument and putting them in `extra` -- would hand back a result
        computed under a different Brillouin-zone sampling than the one currently set.
        """
        return (
            atoms.get_atomic_numbers().tobytes(),
            atoms.get_positions().tobytes(),
            np.asarray(atoms.get_cell(), dtype=float).tobytes(),
            tuple(bool(v) for v in atoms.get_pbc()),
            self.charge,
            self.multiplicity,
            self.reference,
            self.method,
            None if self.kpts is None else tuple(self.kpts),
            self.smearing_ev,
            self.magnetization,
            None if self.field is None else tuple(self.field),
        ) + tuple(extra)

    def _memo(self, name, atoms, extra, compute):
        """Compute a heavy result once per geometry, then hand back the same one.

        These are not ASE properties -- ``ase.phonons`` and ``ase.vibrations`` build their own
        force-constant matrices out of ``get_forces`` -- so ASE's ``results`` cache neither holds
        them nor clears them. The geometry they were computed at is stored beside each and
        compared, which is the same test ``Calculator.check_state`` makes and gives the same
        answer without pretending to be a property.

        Only the most recent result per accessor is kept. A caller sweeping a phonon dispersion
        moves through wavevectors and would otherwise accumulate one dynamical matrix per point.
        """
        key = self._fingerprint(atoms, extra)
        cached = self._heavy.get(name)
        if cached is not None and cached[0] == key:
            return cached[1]
        value = compute()
        self._heavy[name] = (key, value)
        return value

    @staticmethod
    def _is_periodic(atoms):
        """Whether this structure should go through the periodic path.

        Taken from the Atoms object rather than from a constructor flag: ASE users
        set ``atoms.pbc`` and expect it to be honoured, and a calculator that
        silently ignored it would give molecular numbers for a crystal.
        """
        return atoms is not None and bool(np.any(atoms.get_pbc()))

    def _refuse_periodic_field(self, what):
        """A field under periodic boundary conditions is refused, not dropped.

        `-F.r` is not lattice-periodic, so there is no periodic Hamiltonian to add it to and the
        energy per cell would depend on which cell was chosen. The Rust layer says exactly this
        (`pbc::refuse_field`) -- but only if the field reaches it, and none of the periodic
        entry points takes one. So a `PM3(field=...)` on a periodic structure used to return
        field-free numbers with no indication that the field had been ignored.
        """
        if self.field is not None:
            raise RuntimeError(
                f"{what} cannot carry a uniform electric field: `-F.r` is not lattice-periodic, "
                "so the energy per cell would depend on which cell was chosen. Drop `field=`, "
                "or run the molecular path, which does carry one."
            )

    def _periodic_args(self, atoms):
        cell = np.asarray(atoms.get_cell(), dtype=float)
        if not np.any(cell):
            raise RuntimeError(
                "atoms.pbc is set but the cell is empty; a periodic calculation "
                "needs lattice vectors."
            )
        self._refuse_periodic_field("a periodic calculation")
        return (
            atoms.get_atomic_numbers(),
            atoms.get_positions(),  # A
            cell,
            [bool(v) for v in atoms.get_pbc()],
            self.kpts,
            self.charge,
            self.multiplicity,
            self.reference,
            self.method,
            self.smearing_ev,
            self.magnetization,
        )

    @staticmethod
    def _bound(atoms):
        """The Atoms to work on, or an actionable error saying there are none.

        Reached when an accessor is called with no argument before any calculation
        has bound an Atoms object (``self.atoms`` is still None). Every accessor
        that reaches past ``get_property`` has to go through this, or ``None``
        propagates into an AttributeError from somewhere unrelated.
        """
        if atoms is None:
            raise RuntimeError(
                "PM3 has no Atoms object yet. Either call the accessor through "
                "the Atoms (`atoms.get_potential_energy()`), pass one explicitly "
                "(`atoms.calc.get_frequencies(atoms)`), or evaluate a property "
                "first so the calculator binds to a structure."
            )
        return atoms

    def _args(self, atoms):
        atoms = self._bound(atoms)
        return (
            atoms.get_atomic_numbers(),
            atoms.get_positions(),  # A
            self.charge,
            self.multiplicity,
            self.reference,
            self.method,
            self.field,
        )

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        # Recorded before the work, so that `check_state` compares against the selectors these
        # results were actually produced with.
        self._selectors_at_calculate = self._selectors()

        if self._is_periodic(self.atoms):
            self._calculate_periodic(properties)
        else:
            self._calculate_molecular(properties)

        # Lazy: the Hessian is only built when explicitly requested (never during a plain
        # energy/forces cycle), then cached in `results` like any other ASE property.
        if "hessian" in properties:
            self._calculate_hessian()

    def _calculate_periodic(self, properties):
        # Everything is computed on every cycle, regardless of what was asked for.
        #
        # ASE requests one property at a time -- `get_potential_energy()` asks for "energy",
        # `get_forces()` asks for "forces" -- and re-enters `calculate` for each one it does not
        # already have. Computing only what was named therefore converges the *same SCF* two or
        # three times per geometry, which in a molecular-dynamics run is most of the cost. The
        # gradient is cheap next to the SCF, so producing it unasked is nearly free and turns
        # three SCFs per step into one.
        #
        # *One*, which this did not do. It called `periodic_forces` and then
        # `periodic_single_point`, and every field it read off the second is already on the
        # first -- so the stated saving was described here and not taken, at a cost of one full
        # SCF per evaluation. Worse, only the second call was passed `smearing_ev`, so on a metal
        # the energy and forces came from a strictly-filled solution and the charges from a
        # smeared one: two different self-consistent states reported as one result.
        del properties
        args = self._periodic_args(self.atoms)
        f = native.periodic_forces(*args)
        self.results["forces"] = np.asarray(f["forces_ev_per_angstrom"], dtype=float)
        stress = f["stress_ev_per_angstrom3"]
        if stress is not None:
            self.results["stress"] = np.asarray(stress, dtype=float)
        self.results["energy"] = f["energy_ev"]
        self.results["charges"] = np.asarray(f["charges"], dtype=float)
        self.results["heat_of_formation_kcal"] = f["heat_of_formation_kcal"]
        # ASE's `free_energy` is the **Mermin electronic** free energy `E - TS`: the potential
        # whose nuclear gradient the forces above are, once the occupations are fractional.
        # `get_potential_energy(force_consistent=True)` reads exactly this key. Without
        # smearing, `TS` is zero and it equals the energy.
        #
        # It is NOT a Gibbs free energy -- see `entropy_ts_ev` below and the note on the class.
        self.results["free_energy"] = f["free_energy_ev"]
        self.results["electronic_entropy_ts_ev"] = f["entropy_ts_ev"]
        # A periodic dipole is not defined without a surface convention, so none is reported
        # rather than a number that would look usable.
        self.results["gamma_margin_bohr"] = f["gamma_margin_bohr"]

    def _calculate_molecular(self, properties):
        if "stress" in properties:
            raise RuntimeError(
                "stress is only defined for a periodic system; set atoms.pbc and a cell."
            )
        # Forces unasked, for the reason given in `_calculate_periodic` -- and one SCF, for the
        # reason given there too: `native.forces` already returns the charges, the dipole and the
        # heat of formation that the second call was being made for.
        f = native.forces(*self._args(self.atoms))
        self.results["forces"] = np.asarray(f["forces_ev_per_angstrom"], dtype=float)
        self.results["energy"] = f["energy_ev"]
        # A molecule is filled by aufbau: the occupations are integers, there is no electronic
        # entropy, and the Mermin free energy is the energy. Reported so that
        # `get_potential_energy(force_consistent=True)` works on both paths rather than raising
        # on one of them.
        self.results["free_energy"] = f["energy_ev"]
        self.results["electronic_entropy_ts_ev"] = 0.0
        self.results["charges"] = np.asarray(f["charges"], dtype=float)
        self.results["dipole"] = np.asarray(f["dipole_debye"], dtype=float) * _DEBYE_TO_E_ANGSTROM
        self.results["heat_of_formation_kcal"] = f["heat_of_formation_kcal"]

    def _calculate_hessian(self):
        if self._is_periodic(self.atoms):
            raise RuntimeError(
                "the ASE hessian property is the molecular one; for a periodic cell use "
                "get_phonons(), which returns the Gamma-point frequencies."
            )
        h = native.hessian(*self._args(self.atoms))
        hess = np.asarray(h["hessian_hartree_per_bohr2"], dtype=float)
        self.results["hessian"] = hess * (Hartree / Bohr**2)  # Hartree/Bohr^2 -> eV/A^2

    # ---- standard ASE property accessors ----

    def get_forces(self, atoms=None):
        """Cartesian forces in **eV/A** (ASE-standard accessor).

        Delegates to ASE's cached property mechanism (:meth:`get_property`), so a
        calculation is triggered only when the geometry or parameters change.
        """
        return self.get_property("forces", atoms)

    def get_gradient(self, atoms=None):
        """Cartesian energy gradient in **eV/A** (the negative of forces)."""
        return -self.get_forces(atoms)

    def get_dipole_moment(self, atoms=None):
        """Dipole moment in **e.A**, molecular only.

        A periodic dipole is not defined without a surface convention -- the answer
        depends on where the cell is cut -- so this refuses rather than reporting the
        cell's own charge-times-position sum, which looks usable and is not. Use
        :meth:`get_berry_polarization` for the periodic quantity that *is* defined,
        or :meth:`get_born_charges` for its derivative.

        ASE's own refusal for an unset property names the property and nothing else;
        this one says why.
        """
        target = self._bound(atoms if atoms is not None else self.atoms)
        if self._is_periodic(target):
            raise PropertyNotImplementedError(
                "a dipole is not defined for a periodic cell without a surface convention; "
                "use get_berry_polarization() for the polarization, or get_born_charges() "
                "for its derivative with respect to displacement."
            )
        return self.get_property("dipole", atoms)

    def get_stress(self, atoms=None):
        """Stress in **eV/A^3** as ASE's 6-component Voigt vector.

        Only defined for a periodic structure -- a chain reports its axis, a slab
        its two in-plane components, both with exact zeros in the non-periodic
        directions. An isolated cell has no strain at all, and this raises rather
        than returning zeros there: a silently zero stress would let a
        variable-cell relaxation "converge" against something never computed.
        """
        return self.get_property("stress", atoms)

    def get_phonons(
        self, atoms=None, q=None, kpts=None, lo_to_direction=None, cphf_max_iter=None
    ):
        """Phonon frequencies in **cm^-1**, with their polarization vectors.

        With ``q`` left out this is the Gamma-point analytic Hessian, and the
        result carries ``acoustic_residual_cm`` -- the largest of the three
        acoustic frequencies, which should vanish and is reported so it can be
        checked rather than trusted.

        **Eigenvectors** come back alongside the frequencies. At Gamma they are
        ``modes``, real, one mode per row: ``modes[m][3 * a + i]`` is the
        mass-weighted displacement of atom ``a`` along axis ``i``, so the
        Cartesian displacement is that over ``sqrt(masses[a])``. At a wavevector
        they are complex and arrive as ``modes_real`` and ``modes_imag``, because
        the atoms in a cell move with a relative phase and discarding it would
        turn a travelling wave into a standing one.

        ``cphf_max_iter`` raises the coupled-perturbed iteration cap for the
        response behind the calculation; ``None`` keeps the default of 400.

        ``q`` is a fractional wavevector, one component per reciprocal lattice
        vector, and switches to density-functional perturbation theory: the
        phonon at that wavevector from this cell, electronic response included,
        without building a supercell. ``kpts`` gives the response a mesh to sum
        over instead of Gamma alone; it defaults to the calculator's own.

        Not an ASE property: ``ase.phonons`` builds its own force-constant matrix
        by finite displacement over a supercell, which this calculator serves
        through ``get_forces`` like any other. This is the direct analytic route.
        """
        atoms = self._bound(atoms if atoms is not None else self.atoms)
        if not self._is_periodic(atoms):
            raise RuntimeError("get_phonons needs a periodic structure (set atoms.pbc and a cell)")
        self._refuse_periodic_field("periodic phonons")
        # By keyword, not by position. Passed positionally, this silently handed `method` to
        # `reference` the moment `phonons` gained the `reference` argument its siblings already
        # had -- the kind of break that produces a confusing error at best and a wrong answer at
        # worst.
        mesh = self.kpts if kpts is None else kpts
        key = (
            None if q is None else tuple(float(v) for v in np.asarray(q).reshape(-1)),
            None if mesh is None else tuple(int(k) for k in np.asarray(mesh).reshape(-1)),
            None
            if lo_to_direction is None
            else tuple(float(v) for v in np.asarray(lo_to_direction).reshape(-1)),
            # In the key because it decides whether the response converged at all: without it, a
            # generous first call would serve its answer to a later, stricter one.
            cphf_max_iter,
        )
        return self._memo(
            "phonons",
            atoms,
            key,
            lambda: native.phonons(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                np.asarray(atoms.get_cell(), dtype=float),
                pbc=[bool(v) for v in atoms.get_pbc()],
                charge=self.charge,
                multiplicity=self.multiplicity,
                reference=self.reference,
                method=self.method,
                q=q,
                kpts=mesh,
                lo_to_direction=lo_to_direction,
                cphf_max_iter=cphf_max_iter,
            ),
        )

    def get_bands(self, path, atoms=None, kpts=None):
        """Band energies in **eV** along a path of fractional k-points.

        The path is evaluated non-self-consistently in the potential of a
        calculation converged on ``kpts`` (the calculator's own by default), so
        each point costs one diagonalization.
        """
        atoms = self._bound(atoms if atoms is not None else self.atoms)
        if not self._is_periodic(atoms):
            raise RuntimeError("get_bands needs a periodic structure (set atoms.pbc and a cell)")
        self._refuse_periodic_field("a band structure")
        return native.bands(
            atoms.get_atomic_numbers(),
            atoms.get_positions(),
            np.asarray(atoms.get_cell(), dtype=float),
            path,
            pbc=[bool(v) for v in atoms.get_pbc()],
            kpts=self.kpts if kpts is None else kpts,
            charge=self.charge,
            multiplicity=self.multiplicity,
            reference=self.reference,
            method=self.method,
            smearing_ev=self.smearing_ev,
            magnetization=self.magnetization,
        )

    def relax(self, atoms=None, fixed_cell=False, max_steps=200, force_tol=0.02,
              stress_tol=0.001):
        """Relax a periodic structure, atoms and lattice vectors together.

        ``force_tol`` is eV/A and ``stress_tol`` eV/A^3, the units the rest of
        this class uses. Returns the relaxed ``positions`` and ``cell`` in
        Angstrom rather than mutating ``atoms``, so the caller decides whether to
        keep the result.

        ASE's own optimizers move the atoms in a fixed cell through
        ``get_forces``; this is the direct route, and the only one here that
        relaxes the lattice.

        The returned dict carries ``positions``/``positions_angstrom``, ``cell``,
        ``energy_ev``/``energy_hartree``, ``heat_of_formation_kcal``, ``converged``
        and ``steps``/``iterations`` -- the same names :func:`pm3_rs.native.optimize`
        uses for a molecule.
        """
        atoms = self._bound(atoms if atoms is not None else self.atoms)
        if not self._is_periodic(atoms):
            raise RuntimeError("relax needs a periodic structure (set atoms.pbc and a cell)")
        self._refuse_periodic_field("a variable-cell relaxation")
        return native.relax(
            atoms.get_atomic_numbers(),
            atoms.get_positions(),
            np.asarray(atoms.get_cell(), dtype=float),
            pbc=[bool(v) for v in atoms.get_pbc()],
            charge=self.charge,
            multiplicity=self.multiplicity,
            reference=self.reference,
            method=self.method,
            max_steps=max_steps,
            force_tol=force_tol,
            stress_tol=stress_tol,
            fixed_cell=fixed_cell,
        )

    def divide_and_conquer(self, atoms=None, core_radius=3.2, buffer_radius=4.8,
                           smearing_ev=0.1, long_range_cutoff=None):
        """Partitioned SCF, for a system too large to diagonalize whole.

        Radii are in **Angstrom**. ``long_range_cutoff`` (also Angstrom, ``None``
        to leave it off) switches on the linear-scaling near field.

        Energies come back in eV rather than Hartree, as everywhere else in this
        class. ``dropped_pairs`` and ``largest_subsystem`` report what the
        partitioning traded away.

        A calculator-level ``field`` is carried on the molecular unscreened path
        and refused by the other two, in Rust. It used not to be passed at all,
        so the field was dropped and the result looked like an ordinary one.
        """
        atoms = self._bound(atoms if atoms is not None else self.atoms)
        cell = np.asarray(atoms.get_cell(), dtype=float) if self._is_periodic(atoms) else None
        result = native.divide_and_conquer(
            atoms.get_atomic_numbers(),
            atoms.get_positions(),
            cell=cell,
            pbc=[bool(v) for v in atoms.get_pbc()] if cell is not None else None,
            charge=self.charge,
            multiplicity=self.multiplicity,
            reference=self.reference,
            method=self.method,
            core_radius=core_radius,
            buffer_radius=buffer_radius,
            smearing_ev=smearing_ev,
            long_range_cutoff=long_range_cutoff,
            field=self.field,
        )
        return dict(result)

    def divide_and_conquer_forces(self, atoms=None, core_radius=3.2, buffer_radius=4.8,
                                  smearing_ev=0.1, long_range_cutoff=None):
        """Forces in **eV/A** -- and, for a cell, stress in eV/A^3 -- from a partitioned SCF.

        The arguments are :meth:`divide_and_conquer`'s. This is the accessor a
        molecular-dynamics run over a large system needs, and it had no Python
        route at all: ``dc_gradient`` and ``dc_periodic_gradient`` existed in
        Rust and reached nothing else.

        Not wired into ``get_forces``: that path is the full SCF, and which of
        the two a caller wants is a decision about accuracy rather than a
        detail, so it is made explicitly here.
        """
        atoms = self._bound(atoms if atoms is not None else self.atoms)
        cell = np.asarray(atoms.get_cell(), dtype=float) if self._is_periodic(atoms) else None
        result = native.divide_and_conquer_forces(
            atoms.get_atomic_numbers(),
            atoms.get_positions(),
            cell=cell,
            pbc=[bool(v) for v in atoms.get_pbc()] if cell is not None else None,
            charge=self.charge,
            multiplicity=self.multiplicity,
            reference=self.reference,
            method=self.method,
            core_radius=core_radius,
            buffer_radius=buffer_radius,
            smearing_ev=smearing_ev,
            long_range_cutoff=long_range_cutoff,
            field=self.field,
        )
        return dict(result)

    def divide_and_conquer_optimize(self, atoms=None, core_radius=3.2, buffer_radius=4.8,
                                    smearing_ev=0.1, long_range_cutoff=None, max_steps=200,
                                    force_tol=0.02):
        """Relax a molecule on the partitioned gradient.

        The arguments are :meth:`divide_and_conquer`'s, plus ``max_steps`` and
        ``force_tol`` (eV/A) from :meth:`relax`. Returns the relaxed
        ``positions``/``positions_angstrom`` in Angstrom rather than mutating
        ``atoms``, alongside ``energy_ev``/``energy_hartree``,
        ``heat_of_formation_kcal``, ``converged``, ``steps``/``iterations`` and
        ``n_subsystems`` -- the same names :meth:`relax` and
        :func:`pm3_rs.native.optimize` use.

        Molecular only, because the partitioned path has no cell gradient: a
        periodic structure would have to be relaxed at fixed cell without saying
        so, and :meth:`relax` is the periodic optimizer.

        The geometry this reaches is the buffer's minimum, not the method's --
        the partitioned gradient is not the exact derivative of the partitioned
        energy. Widen ``buffer_radius`` and re-run before believing a structure.
        """
        atoms = self._bound(atoms if atoms is not None else self.atoms)
        if self._is_periodic(atoms):
            raise RuntimeError(
                "divide_and_conquer_optimize is molecular: the partitioned path has no cell "
                "gradient, so a periodic structure would be relaxed at fixed cell without "
                "saying so. Use relax() for a periodic structure."
            )
        result = native.divide_and_conquer_optimize(
            atoms.get_atomic_numbers(),
            atoms.get_positions(),
            charge=self.charge,
            multiplicity=self.multiplicity,
            reference=self.reference,
            method=self.method,
            core_radius=core_radius,
            buffer_radius=buffer_radius,
            smearing_ev=smearing_ev,
            long_range_cutoff=long_range_cutoff,
            field=self.field,
            max_steps=max_steps,
            force_tol=force_tol,
        )
        return dict(result)

    def get_berry_polarization(self, atoms=None, strings=12, kpts=None):
        """Berry-phase polarization (e/Bohr^2), modulo its quantum.

        Present as an independent check on :meth:`get_born_charges`: it reaches the same ``Z*``
        through overlaps between neighbouring k-points, with no response equation in it.

        The result is defined only modulo ``quantum``. Only differences between two geometries
        mean anything, and they must be reduced onto the nearest branch -- see
        :func:`pm3_rs.native.berry_polarization` for how. Cached against the geometry, the string
        length and the transverse mesh, since all three change the answer.
        """
        atoms = self._bound(atoms if atoms is not None else self.atoms)
        if not self._is_periodic(atoms):
            raise RuntimeError(
                "a Berry phase needs a Brillouin zone to wind through; an isolated molecule's "
                "dipole is well defined and comes from get_dipole_moment()."
            )
        self._refuse_periodic_field("a Berry phase")
        mesh = tuple(kpts) if kpts is not None else (self.kpts if self.kpts else (1, 1, 1))
        return self._memo(
            "berry",
            atoms,
            (int(strings), tuple(int(v) for v in mesh)),
            lambda: dict(
                native.berry_polarization(
                    atoms.get_atomic_numbers(),
                    atoms.get_positions(),
                    np.asarray(atoms.get_cell(), dtype=float),
                    strings=int(strings),
                    kpts=[int(v) for v in mesh],
                    pbc=[bool(v) for v in atoms.get_pbc()],
                    charge=self.charge,
                    multiplicity=self.multiplicity,
                    reference=self.reference,
                    method=self.method,
                )
            ),
        )

    def get_finite_field(self, field, kpts, atoms=None):
        """A finite field along a periodic direction, by the Berry-phase electric enthalpy.

        ``field`` is in eV per (e*Bohr). Along a periodic direction ``E.R`` is not
        lattice-periodic and the ground state of ``H - E.R`` does not exist, so what is minimized
        is ``F = E - V E.P``. A field orthogonal to *every* lattice vector needs none of this and
        goes through ``PM3(field=...)``.

        ``kpts`` along each field direction is that direction's Berry-phase string length and is
        the convergence parameter; at least 3. Check ``resolved`` before reading the polarization
        vector -- an axis the mesh could not see contributes zero, which is not its value.

        **There is no force here.** The derivative of the enthalpy with respect to the nuclei is
        not implemented, so this cannot drive a relaxation or dynamics.
        """
        atoms = self._bound(atoms if atoms is not None else self.atoms)
        if not self._is_periodic(atoms):
            raise RuntimeError(
                "the Berry-phase finite field is for a field along a periodic direction; for a "
                "molecule pass PM3(field=...), which uses the ordinary -E.r coupling."
            )
        self._refuse_periodic_field("a Berry-phase finite field")
        return self._memo(
            "finite_field",
            atoms,
            (
                tuple(float(v) for v in np.asarray(field, dtype=float).reshape(-1)),
                tuple(int(v) for v in np.asarray(kpts).reshape(-1)),
            ),
            lambda: dict(
                native.finite_field(
                    atoms.get_atomic_numbers(),
                    atoms.get_positions(),
                    np.asarray(atoms.get_cell(), dtype=float),
                    field,
                    kpts,
                    pbc=[bool(v) for v in atoms.get_pbc()],
                    charge=self.charge,
                    multiplicity=self.multiplicity,
                    reference=self.reference,
                    method=self.method,
                )
            ),
        )

    def get_phonon_bands(self, supercell, path, atoms=None, points=12, enforce_asr=False):
        """Phonon dispersion from supercell force constants: one Hessian for the whole path.

        A supercell's Gamma point *is* a mesh of the primitive cell, so this reproduces the
        **mesh**-sampled response with the matching mesh, not a Gamma-only one.

        ``acoustic_sum_rule_residual`` comes back before it can be imposed, because it is what
        the truncation threw away.
        """
        atoms = self._bound(atoms if atoms is not None else self.atoms)
        if not self._is_periodic(atoms):
            raise RuntimeError(
                "force constants need a lattice to be cut out of; for a molecule use "
                "get_frequencies()."
            )
        self._refuse_periodic_field("phonon bands")
        return self._memo(
            "phonon_bands",
            atoms,
            (
                tuple(int(v) for v in np.asarray(supercell).reshape(-1)),
                tuple(tuple(float(c) for c in q) for q in path),
                int(points),
                bool(enforce_asr),
            ),
            lambda: dict(
                native.phonon_bands(
                    atoms.get_atomic_numbers(),
                    atoms.get_positions(),
                    np.asarray(atoms.get_cell(), dtype=float),
                    supercell,
                    path,
                    points=int(points),
                    pbc=[bool(v) for v in atoms.get_pbc()],
                    charge=self.charge,
                    multiplicity=self.multiplicity,
                    reference=self.reference,
                    method=self.method,
                    enforce_asr=bool(enforce_asr),
                )
            ),
        )

    def get_born_charges(self, atoms=None, enforce=False):
        """Born effective charges: one ``3 x 3`` tensor per atom, in elementary charges.

        ``Z*[a][alpha][beta]`` is the dipole the cell gains along ``alpha`` per unit displacement
        of atom ``a`` along ``beta``. Periodic only; a molecule's equivalent is
        :meth:`get_dipole_derivatives`.

        Returns the dict :func:`pm3_rs.native.born_charges` returns, including
        ``sum_rule_residual`` -- which is zero for an exact response and is the number to look at
        before trusting the rest. Cached against the geometry like the other heavy accessors.
        """
        atoms = self._bound(atoms if atoms is not None else self.atoms)
        if not self._is_periodic(atoms):
            raise RuntimeError(
                "Born effective charges need a periodic structure; for a molecule use "
                "get_dipole_derivatives(), which is the same quantity without a lattice."
            )
        self._refuse_periodic_field("Born effective charges")
        return self._memo(
            "born_charges",
            atoms,
            (bool(enforce),),
            lambda: dict(
                native.born_charges(
                    atoms.get_atomic_numbers(),
                    atoms.get_positions(),
                    np.asarray(atoms.get_cell(), dtype=float),
                    pbc=[bool(v) for v in atoms.get_pbc()],
                    charge=self.charge,
                    multiplicity=self.multiplicity,
                    reference=self.reference,
                    method=self.method,
                    enforce=enforce,
                )
            ),
        )

    def get_dielectric(self, atoms=None, include_ionic=False):
        """Electronic polarizability (Bohr^3) and, in 3D, ``eps_inf``.

        Returns the dict :func:`pm3_rs.native.dielectric` returns: ``polarizability`` always, and
        ``epsilon`` only for a fully periodic cell -- a slab has an area and a chain a length,
        and dividing by a supercell's vacuum padding would make the answer a statement about the
        padding. Cached against the geometry and against ``include_ionic``.

        ``epsilon`` is the **electronic** response at fixed nuclei.

        ``include_ionic=True`` adds ``epsilon_static`` (the static constant ``eps_0``, with the
        nuclei free to relax along each infrared-active mode), its two halves
        ``epsilon_electronic`` and ``epsilon_ionic``, and ``skipped_modes``. It costs a
        Gamma-point phonon calculation and a set of Born charges on top of the field response,
        and it is only meaningful at a **relaxed geometry** -- check ``skipped_modes``, where
        anything above the three acoustic modes means the structure is not a minimum.
        """
        atoms = self._bound(atoms if atoms is not None else self.atoms)
        if not self._is_periodic(atoms):
            raise RuntimeError(
                "a dielectric tensor needs a periodic structure; for a molecule the "
                "corresponding quantity is a finite-field polarizability, which this "
                "calculator does not compute directly."
            )
        self._refuse_periodic_field("a dielectric tensor")
        return self._memo(
            "dielectric",
            atoms,
            # Part of the memo key, not just an argument: the two calls return different key
            # sets, so a cache that ignored it would answer the second call from the first.
            (bool(include_ionic),),
            lambda: dict(
                native.dielectric(
                    atoms.get_atomic_numbers(),
                    atoms.get_positions(),
                    np.asarray(atoms.get_cell(), dtype=float),
                    pbc=[bool(v) for v in atoms.get_pbc()],
                    include_ionic=bool(include_ionic),
                    charge=self.charge,
                    multiplicity=self.multiplicity,
                    reference=self.reference,
                    method=self.method,
                )
            ),
        )

    def get_ir_spectrum(self, atoms=None):
        """Harmonic frequencies with infrared intensities (km/mol).

        Computed on demand and cached against the geometry, so asking twice costs one
        calculation. It is a Hessian plus three coupled-perturbed solves; :meth:`get_frequencies`
        is a Hessian on its own, and the two caches are separate, so asking for both costs two
        Hessians. Evaluate at a **stationary point**.

        Returns the dict `pm3_rs.native.ir_spectrum` returns, including the raw
        `3 x 3N` dipole-derivative tensor in elementary charges.
        """
        atoms = self._bound(atoms if atoms is not None else self.atoms)
        if self._is_periodic(atoms):
            raise RuntimeError(
                "the infrared spectrum is molecular; for a periodic cell use get_phonons()."
            )
        return self._memo(
            "ir_spectrum",
            atoms,
            (),
            lambda: native.ir_spectrum(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                charge=self.charge,
                multiplicity=self.multiplicity,
                reference=self.reference,
                method=self.method,
                field=self.field,
            ),
        )

    def get_dipole_derivatives(self, atoms=None):
        """The `3 x 3N` tensor `d(mu)/dR` in elementary charges.

        Three coupled-perturbed solves, not `3N`: the interchange theorem trades the nuclear
        perturbations for the field ones, so this costs the same whatever the molecule's size.
        No Hessian is built, which is what separates it from :meth:`get_ir_spectrum`.
        """
        atoms = self._bound(atoms if atoms is not None else self.atoms)
        if self._is_periodic(atoms):
            raise RuntimeError("the dipole derivatives are molecular")
        result = self._memo(
            "dipole",
            atoms,
            (),
            lambda: native.dipole(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                charge=self.charge,
                multiplicity=self.multiplicity,
                reference=self.reference,
                method=self.method,
                field=self.field,
            ),
        )
        return np.asarray(result["derivatives_e"], dtype=float)

    def get_dipole_operator(self, atoms=None):
        """The three `nao x nao` dipole moment matrices (Bohr, about the centre of mass).

        The same operator the reported dipole and the external-field coupling are built from --
        which is what makes `mu = -dE/dF` hold by construction rather than by coincidence.
        """
        atoms = self._bound(atoms if atoms is not None else self.atoms)
        result = self._memo(
            "dipole_operator",
            atoms,
            (),
            lambda: native.dipole(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                charge=self.charge,
                multiplicity=self.multiplicity,
                reference=self.reference,
                method=self.method,
                field=self.field,
                operator=True,
            ),
        )
        return np.asarray(result["operator_bohr"], dtype=float)

    def get_dynamical_matrix(self, q, atoms=None, kpts=None, rigid_ion=False):
        """`D(q)` itself, in eV/Bohr^2, as `(real, imag)` arrays of shape `(3N, 3N)`.

        :meth:`get_phonons` gives the frequencies; this gives what they are diagonalized from.
        Cached per `(q, kpts, rigid_ion)`, one result at a time -- a dispersion sweep moves
        through wavevectors and would otherwise keep every matrix it built.
        """
        atoms = self._bound(atoms if atoms is not None else self.atoms)
        if not self._is_periodic(atoms):
            raise RuntimeError("a dynamical matrix needs a periodic structure")
        self._refuse_periodic_field("a dynamical matrix")
        # The calculator's own mesh when the call site names none, exactly as `get_phonons` does.
        # Without this fallback the two accessors described different calculations: a
        # `PM3(kpts=(2,2,2))` gave `get_phonons` a mesh-summed response and `get_dynamical_matrix`
        # a Gamma-only one, with no argument at either call site to say so.
        mesh = self.kpts if kpts is None else kpts
        mesh = None if mesh is None else tuple(int(k) for k in np.asarray(mesh).reshape(-1))
        key = (tuple(float(v) for v in np.asarray(q).reshape(-1)), mesh, bool(rigid_ion))
        result = self._memo(
            "dynamical_matrix",
            atoms,
            key,
            lambda: native.dynamical_matrix(
                atoms.get_atomic_numbers(),
                atoms.get_positions(),
                np.asarray(atoms.get_cell(), dtype=float),
                q,
                pbc=[bool(v) for v in atoms.get_pbc()],
                kpts=mesh,
                charge=self.charge,
                multiplicity=self.multiplicity,
                reference=self.reference,
                method=self.method,
                rigid_ion=rigid_ion,
            ),
        )
        return (
            np.asarray(result["real"], dtype=float),
            np.asarray(result["imag"], dtype=float),
        )

    def write_molden(self, path, atoms=None, basis="gto"):
        """Write the converged wavefunction to ``path`` as a Molden file.

        Not an ASE property -- ASE has no notion of one -- so this recomputes the
        SCF each call rather than reading a cache, like :meth:`get_phonons`.

        Molecular only. A periodic wavefunction has no single set of molecular
        orbitals to write.

        ``basis`` is ``"gto"`` (the default, an even-tempered Gaussian expansion
        of the Slater orbitals, which is what viewers read) or ``"sto"`` (the
        exponents PM3 actually uses, for viewers that read ``[STO]``). Only the
        basis section differs; the orbitals are the same wavefunction.

        The calculator's ``field`` is applied, as it is by every sibling
        accessor. Omitting it wrote the field-free wavefunction to a file
        alongside energies that had been computed in the field -- two different
        states, with nothing in the Molden document to say which one it was.
        """
        atoms = self._bound(atoms if atoms is not None else self.atoms)
        if self._is_periodic(atoms):
            raise RuntimeError(
                "Molden export is molecular; a periodic wavefunction has no single set of "
                "molecular orbitals. Use a large cell with pbc off."
            )
        text = native.molden(
            atoms.get_atomic_numbers(),
            atoms.get_positions(),
            charge=self.charge,
            multiplicity=self.multiplicity,
            reference=self.reference,
            method=self.method,
            field=self.field,
            basis=basis,
        )
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def get_hessian(self, atoms=None):
        """Analytic Cartesian Hessian in **eV/A^2** (shape ``(3N, 3N)``).

        A declared but **lazy** ASE property: computed on demand via
        :meth:`get_property` (and cached) -- the coupled-perturbed SCF Hessian plus the
        classical D3/H4/X second derivatives for the correction variants. It is never
        built as part of an energy/forces evaluation.
        """
        return self.get_property("hessian", atoms)

    # ---- lazy accessors for properties outside the standard ASE property set ----

    def get_frequencies(self, atoms=None):
        """Harmonic vibrational frequencies in cm^-1 (evaluate at a minimum).

        Computed lazily on demand from the analytic Hessian. ``atoms`` defaults
        to the structure the calculator is currently bound to; pass one
        explicitly if no property has been evaluated yet.
        """
        atoms = self._bound(atoms if atoms is not None else self.atoms)
        vib = self._memo(
            "frequencies", atoms, (), lambda: native.frequencies(*self._args(atoms))
        )
        return np.asarray(vib["frequencies_cm"], dtype=float)


__all__ = ["PM3"]
