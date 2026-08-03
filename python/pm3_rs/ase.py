# SPDX-License-Identifier: GPL-3.0-or-later
"""ASE calculator for PM3.

Uses ASE's convention throughout: energies in **eV**, forces in **eV/Å**,
positions in **Å**, and the Hessian in **eV/Å²**. Internally it calls the
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
:meth:`get_forces`, …). The Hessian and vibrational frequencies are **not**
standard ASE properties and are computed lazily, on demand, via
:meth:`get_hessian` / :meth:`get_frequencies` (they are never part of a
``calculate`` cycle).
"""

from __future__ import annotations

import numpy as np

try:
    from ase.calculators.calculator import Calculator, all_changes
    from ase.units import Bohr, Hartree
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The PM3 ASE calculator requires ASE. Install with `pip install pm3-rs-python[ase]`."
    ) from exc

from . import native

# Debye → e·Å (ASE dipole unit).
_DEBYE_TO_E_ANGSTROM = 0.2081943


class PM3(Calculator):
    """PM3 semiempirical calculator (ASE units: eV, eV/Å, Å, eV/Å²).

    Parameters
    ----------
    charge : int
        Total (formal) molecular charge.
    multiplicity : int
        Spin multiplicity (2S+1); 1 = singlet, 2 = doublet, …
    reference : {"auto", "rhf", "uhf"}
        Restricted/unrestricted SCF reference (default "auto").
    method : {"pm3", "pm3-d3", "pm3-d3h4", "pm3-d3h4x"}
        PM3 correction variant (default "pm3").

    ``results`` is populated with ``energy`` (eV), ``forces`` (eV/Å),
    ``charges`` (Mulliken, e), ``dipole`` (e·Å) and ``heat_of_formation_kcal``
    (kcal/mol) — each declared in :attr:`implemented_properties` so it is reachable
    through ASE's ``get_property`` / ``get_charges`` / ``get_dipole_moment`` API.
    """

    # Every entry here is populated by `calculate`, so `get_property(name)` works for each.
    # `hessian` (eV/Å²) is declared so it is reachable through the standard ASE property API,
    # but it is computed **lazily** — only when a caller actually requests it (via
    # `get_hessian`/`get_property("hessian")`); a plain energy/forces cycle never builds it.
    implemented_properties = [
        "energy",
        "forces",
        "charges",
        "dipole",
        "heat_of_formation_kcal",
        "hessian",
    ]

    def __init__(
        self,
        charge: int = 0,
        multiplicity: int = 1,
        reference: str = "auto",
        method: str = "pm3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self.reference = str(reference)
        self.method = str(method)

    def _args(self, atoms):
        if atoms is None:
            # Reached when an accessor is called with no argument before any
            # calculation has bound an Atoms object (`self.atoms` is still
            # None). Raise something actionable instead of letting `None`
            # propagate into an AttributeError.
            raise RuntimeError(
                "PM3 has no Atoms object yet. Either call the accessor through "
                "the Atoms (`atoms.get_potential_energy()`), pass one explicitly "
                "(`atoms.calc.get_frequencies(atoms)`), or evaluate a property "
                "first so the calculator binds to a structure."
            )
        return (
            atoms.get_atomic_numbers(),
            atoms.get_positions(),  # Å
            self.charge,
            self.multiplicity,
            self.reference,
            self.method,
        )

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        if "forces" in properties:
            f = native.forces(*self._args(self.atoms))
            energy_ev = f["energy_ev"]
            self.results["forces"] = np.asarray(f["forces_ev_per_angstrom"], dtype=float)
        else:
            energy_ev = None

        sp = native.single_point(*self._args(self.atoms))
        if energy_ev is None:
            energy_ev = sp["energy_ev"]

        self.results["energy"] = energy_ev
        self.results["charges"] = np.asarray(sp["charges"], dtype=float)
        self.results["dipole"] = np.asarray(sp["dipole_debye"], dtype=float) * _DEBYE_TO_E_ANGSTROM
        self.results["heat_of_formation_kcal"] = sp["heat_of_formation_kcal"]

        # Lazy: the Hessian is only built when explicitly requested (never during a plain
        # energy/forces cycle), then cached in `results` like any other ASE property.
        if "hessian" in properties:
            h = native.hessian(*self._args(self.atoms))
            hess = np.asarray(h["hessian_hartree_per_bohr2"], dtype=float)
            self.results["hessian"] = hess * (Hartree / Bohr**2)  # Hartree/Bohr² → eV/Å²

    # ---- standard ASE property accessors ----

    def get_forces(self, atoms=None):
        """Cartesian forces in **eV/Å** (ASE-standard accessor).

        Delegates to ASE's cached property mechanism (:meth:`get_property`), so a
        calculation is triggered only when the geometry or parameters change.
        """
        return self.get_property("forces", atoms)

    def get_gradient(self, atoms=None):
        """Cartesian energy gradient in **eV/Å** (the negative of forces)."""
        return -self.get_forces(atoms)

    def get_hessian(self, atoms=None):
        """Analytic Cartesian Hessian in **eV/Å²** (shape ``(3N, 3N)``).

        A declared but **lazy** ASE property: computed on demand via
        :meth:`get_property` (and cached) — the coupled-perturbed SCF Hessian plus the
        classical D3/H4/X second derivatives for the correction variants. It is never
        built as part of an energy/forces evaluation.
        """
        return self.get_property("hessian", atoms)

    # ---- lazy accessors for properties outside the standard ASE property set ----

    def get_frequencies(self, atoms=None):
        """Harmonic vibrational frequencies in cm⁻¹ (evaluate at a minimum).

        Computed lazily on demand from the analytic Hessian. ``atoms`` defaults
        to the structure the calculator is currently bound to; pass one
        explicitly if no property has been evaluated yet.
        """
        atoms = atoms if atoms is not None else self.atoms
        vib = native.frequencies(*self._args(atoms))
        return np.asarray(vib["frequencies_cm"], dtype=float)


__all__ = ["PM3"]
