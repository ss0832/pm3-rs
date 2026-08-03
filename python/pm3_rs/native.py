# SPDX-License-Identifier: GPL-3.0-or-later
"""Native PM3 API (atomic units: Hartree, Bohr).

Thin wrapper over the compiled ``pm3_rs._native`` extension. Input coordinates
are Ångström; energies are returned in Hartree (and, for convenience, the heat
of formation in kcal/mol). This is the raw model surface; the ASE calculator
layer converts to eV/Å.

Every function takes ``charge`` and ``multiplicity``, a ``reference`` selecting
the SCF reference, and a ``method`` selecting the correction variant:

``reference``
    - ``"auto"`` (default): RHF for a closed shell, UHF for an open shell;
    - ``"rhf"``: force restricted (error if the system is open-shell);
    - ``"uhf"``: force unrestricted even for a singlet (spin-symmetry breaking).

``method``
    - ``"pm3"`` (default): plain PM3, no post-SCF corrections;
    - ``"pm3-d3"``: Grimme D3 dispersion;
    - ``"pm3-d3h4"``: D3 dispersion + the Řezáč H4 hydrogen-bond correction;
    - ``"pm3-d3h4x"``: D3 + H4 + the halogen-bond (X) correction.

The correction energy, gradient **and Hessian** all track ``method``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import _native


def _as_lists(numbers, positions):
    numbers = [int(z) for z in np.asarray(numbers).reshape(-1)]
    positions = np.asarray(positions, dtype=float).reshape(len(numbers), 3).tolist()
    return numbers, positions


def single_point(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
) -> dict:
    """PM3 single-point energy and properties.

    Parameters
    ----------
    numbers : sequence of int
        Atomic numbers, one per atom.
    positions : array-like, shape (N, 3)
        Cartesian coordinates in **Ångström**.
    charge : float
        Total molecular charge (electrons = Σ Z_valence − charge).
    multiplicity : int
        Spin multiplicity; 1 = singlet, 2 = doublet, …
    reference : {"auto", "rhf", "uhf"}
        SCF reference (see module docstring).
    method : {"pm3", "pm3-d3", "pm3-d3h4", "pm3-d3h4x"}
        Correction variant (see module docstring).

    Returns
    -------
    dict with keys ``energy_hartree``, ``energy_ev``, ``heat_of_formation_kcal``,
    ``electronic_ev``, ``core_ev``, ``charges`` (Mulliken, e),
    ``dipole_debye`` ([x, y, z]), ``homo_ev``, ``lumo_ev``, ``converged``,
    ``unrestricted``.
    """
    n, p = _as_lists(numbers, positions)
    return _native.single_point(
        n, p, float(charge), int(multiplicity), str(reference), str(method)
    )


def gradient(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
) -> dict:
    """PM3 energy and analytic nuclear **gradient** dE/dR (Hellmann–Feynman).

    Coordinates in **Ångström**. Returns ``energy_hartree``, ``energy_ev``,
    ``heat_of_formation_kcal``, ``gradient_hartree_per_bohr`` and
    ``gradient_ev_per_angstrom``. (Forces = −gradient; see :func:`forces`.)
    """
    n, p = _as_lists(numbers, positions)
    return _native.gradient(
        n, p, float(charge), int(multiplicity), str(reference), str(method)
    )


def forces(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
) -> dict:
    """PM3 energy and **forces** (= −dE/dR).

    Coordinates in **Ångström**. Returns ``energy_hartree``, ``energy_ev``,
    ``heat_of_formation_kcal``, ``forces_hartree_per_bohr`` and
    ``forces_ev_per_angstrom``.
    """
    n, p = _as_lists(numbers, positions)
    return _native.forces(
        n, p, float(charge), int(multiplicity), str(reference), str(method)
    )


def optimize(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
) -> dict:
    """L-BFGS geometry optimization on the analytic PM3 gradient.

    Coordinates in **Ångström**. Returns ``positions_angstrom``,
    ``energy_hartree``, ``heat_of_formation_kcal``, ``converged``, ``iterations``.
    """
    n, p = _as_lists(numbers, positions)
    return _native.optimize(
        n, p, float(charge), int(multiplicity), str(reference), str(method)
    )


def frequencies(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
) -> dict:
    """Harmonic vibrational frequencies from the analytic (CPHF) Hessian.

    Evaluate at a **stationary point** (optimize first). Coordinates in
    **Ångström**. Returns ``frequencies_cm`` (ascending; negatives are
    imaginary) and mass-weighted ``eigenvalues`` (eV/(Å²·amu)).
    """
    n, p = _as_lists(numbers, positions)
    return _native.frequencies(
        n, p, float(charge), int(multiplicity), str(reference), str(method)
    )


def hessian(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
) -> dict:
    """Analytic Cartesian Hessian from coupled-perturbed SCF (+ the classical
    D3/H4/X second derivatives for the correction variants).

    Coordinates in **Ångström**. Returns ``hessian_hartree_per_bohr2`` (a
    ``3N × 3N`` nested list, atomic units) and ``ndof``.
    """
    n, p = _as_lists(numbers, positions)
    return _native.hessian(
        n, p, float(charge), int(multiplicity), str(reference), str(method)
    )
