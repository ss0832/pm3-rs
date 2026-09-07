# SPDX-License-Identifier: GPL-3.0-or-later
"""Native PM3 API (atomic units: Hartree, Bohr).

Thin wrapper over the compiled ``pm3_rs._native`` extension. Input coordinates
are Angstrom; energies are returned in Hartree (and, for convenience, the heat
of formation in kcal/mol). This is the raw model surface; the ASE calculator
layer converts to eV/A.

Every function takes ``charge`` and ``multiplicity``, a ``reference`` selecting
the SCF reference, and a ``method`` selecting the correction variant:

``reference``
    - ``"auto"`` (default): RHF for a closed shell, UHF for an open shell;
    - ``"rhf"``: force restricted (error if the system is open-shell);
    - ``"uhf"``: force unrestricted even for a singlet (spin-symmetry breaking).

``method``
    - ``"pm3"`` (default): plain PM3, no post-SCF corrections;
    - ``"pm3-d3"``: Grimme D3 dispersion;
    - ``"pm3-d3h4"``: D3 dispersion + the Rezac H4 hydrogen-bond correction;
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
    field=None,
) -> dict:
    """PM3 single-point energy and properties.

    Parameters
    ----------
    numbers : sequence of int
        Atomic numbers, one per atom.
    positions : array-like, shape (N, 3)
        Cartesian coordinates in **Angstrom**.
    charge : float
        Total molecular charge (electrons = sum Z_valence - charge).
    multiplicity : int
        Spin multiplicity; 1 = singlet, 2 = doublet, ...
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
        n,
        p,
        float(charge),
        int(multiplicity),
        str(reference),
        str(method),
        None if field is None else [float(v) for v in field],
    )


def gradient(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    field=None,
) -> dict:
    """PM3 energy and analytic nuclear **gradient** dE/dR (Hellmann-Feynman).

    Coordinates in **Angstrom**. Returns ``energy_hartree``, ``energy_ev``,
    ``heat_of_formation_kcal``, ``gradient_hartree_per_bohr`` and
    ``gradient_ev_per_angstrom``. (Forces = -gradient; see :func:`forces`.)
    """
    n, p = _as_lists(numbers, positions)
    return _native.gradient(
        n,
        p,
        float(charge),
        int(multiplicity),
        str(reference),
        str(method),
        None if field is None else [float(v) for v in field],
    )


def forces(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    field=None,
) -> dict:
    """PM3 energy and **forces** (= -dE/dR).

    Coordinates in **Angstrom**. Returns ``energy_hartree``, ``energy_ev``,
    ``heat_of_formation_kcal``, ``forces_hartree_per_bohr`` and
    ``forces_ev_per_angstrom``.
    """
    n, p = _as_lists(numbers, positions)
    return _native.forces(
        n,
        p,
        float(charge),
        int(multiplicity),
        str(reference),
        str(method),
        None if field is None else [float(v) for v in field],
    )


def optimize(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    field=None,
    max_steps: int = 200,
    force_tol=None,
) -> dict:
    """L-BFGS geometry optimization on the analytic PM3 gradient.

    Coordinates in **Angstrom**. Returns ``positions_angstrom``, ``energy_hartree``,
    ``energy_ev``, ``heat_of_formation_kcal``, ``converged`` and ``iterations``.

    ``positions`` and ``steps`` are aliases for ``positions_angstrom`` and
    ``iterations``, so that this function and :func:`relax` -- the same operation on a
    periodic cell -- answer to the same names. Through 0.2.3 the two returned disjoint
    key sets and ``optimize(...)["energy_ev"]`` raised :exc:`KeyError` even though every
    other function in this module returns that key.

    ``force_tol`` is eV/Angstrom, as in :func:`relax` and
    :func:`divide_and_conquer_optimize`. It defaults to ``None``, meaning the optimizer's
    own 1e-3 eV/Bohr, rather than to those functions' 0.02 eV/Angstrom: the two differ by
    a factor of ten, and exposing a knob is not a reason to move the answer for callers
    who never touch it.
    """
    n, p = _as_lists(numbers, positions)
    return _native.optimize(
        n,
        p,
        float(charge),
        int(multiplicity),
        str(reference),
        str(method),
        None if field is None else [float(v) for v in field],
        int(max_steps),
        None if force_tol is None else float(force_tol),
    )


def orbitals(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    field=None,
) -> dict:
    """Molecular orbital energies, coefficients and occupations.

    Coordinates in **Angstrom**. All of this has been on the Rust ``Pm3Result`` since
    the beginning and none of it was reachable from Python: :func:`single_point` gave
    ``homo_ev`` and ``lumo_ev`` and nothing else, so the only way to see a spectrum or
    a coefficient was to write a Molden file and parse it back.

    ``mo_coefficients`` is ``nao x nmo`` with **AOs down the rows and MOs across the
    columns**, so ``mo_coefficients[i][m]`` is the weight of atomic orbital ``i`` in
    molecular orbital ``m``. ``ao_labels`` names the rows -- ``(atom index, element
    symbol, orbital)`` with orbital one of ``s``, ``px``, ``py``, ``pz`` -- because
    otherwise a coefficient is a number whose meaning the caller has to reconstruct
    from the element table, and getting that wrong is silent.

    Energies are ascending, so the occupied orbitals are the leading ``n_occupied``.
    ``homo_index`` and ``lumo_index`` are ``None`` rather than an index when a shell is
    empty or full, since ``0`` and ``nao`` would both read as a real frontier.

    ``homo_ev``, ``lumo_ev`` and ``gap_ev`` are taken across **both** spin channels: a
    radical's beta LUMO sits below its alpha one, so the alpha spectrum alone gives the
    wrong frontier.

    Keys: ``mo_energies_ev``, ``mo_energies_hartree``, ``mo_coefficients``,
    ``occupations``, ``n_occupied``, ``homo_index``, ``lumo_index``, ``homo_ev``,
    ``lumo_ev``, ``gap_ev``, ``ao_labels``, ``unrestricted``, ``n_beta``, and the beta
    set ``mo_energies_beta_ev``, ``mo_energies_beta_hartree``, ``mo_coefficients_beta``,
    ``occupations_beta`` -- present and ``None`` for a restricted run, so the key set
    does not depend on the shell.
    """
    n, p = _as_lists(numbers, positions)
    return _native.orbitals(
        n,
        p,
        float(charge),
        int(multiplicity),
        str(reference),
        str(method),
        None if field is None else [float(v) for v in field],
    )


def frequencies(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    field=None,
    cphf_max_iter=None,
) -> dict:
    """Harmonic vibrational frequencies from the analytic (CPHF) Hessian.

    Evaluate at a **stationary point** (optimize first). Coordinates in
    **Angstrom**. Returns ``frequencies_cm`` (ascending; negatives are
    imaginary) and mass-weighted ``eigenvalues`` (eV/(A^2.amu)).

    ``modes`` are the normal modes as the **columns** of a ``3N x 3N`` list of
    lists, ordered with ``frequencies_cm``, and they are in **mass-weighted**
    coordinates: the Cartesian displacement of mode ``m`` is
    ``modes[i][m] / sqrt(masses[i // 3])``. ``masses`` (amu, one per atom) is
    returned for exactly that division. Contracting a Cartesian quantity against
    a mode without it -- an infrared intensity above all -- gives numbers that
    look plausible and are systematically wrong, which is why the two travel
    together and why returning the frequencies alone put every use of a normal
    mode out of reach from Python.
    """
    n, p = _as_lists(numbers, positions)
    return _native.frequencies(
        n,
        p,
        float(charge),
        int(multiplicity),
        str(reference),
        str(method),
        None if field is None else [float(v) for v in field],
            cphf_max_iter,
    )


def hessian(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    field=None,
    cphf_max_iter=None,
) -> dict:
    """Analytic Cartesian Hessian from coupled-perturbed SCF (+ the classical
    D3/H4/X second derivatives for the correction variants).

    Coordinates in **Angstrom**. Returns ``hessian_hartree_per_bohr2`` (a
    ``3N x 3N`` nested list, atomic units) and ``ndof``.
    """
    n, p = _as_lists(numbers, positions)
    return _native.hessian(
        n,
        p,
        float(charge),
        int(multiplicity),
        str(reference),
        str(method),
        None if field is None else [float(v) for v in field],
            cphf_max_iter,
    )


def _cell_rows(cell):
    """Normalise a cell to three lattice-vector rows in Angstrom.

    Accepts anything ASE would: a 3x3 matrix, three lengths for an orthorhombic
    cell, or a single length for a cube.
    """
    array = np.asarray(cell, dtype=float)
    if array.shape == (3, 3):
        return array.tolist()
    if array.shape == (3,):
        return np.diag(array).tolist()
    if array.shape == ():
        return (np.eye(3) * float(array)).tolist()
    raise ValueError(
        "cell must be a 3x3 matrix, three lengths, or one length "
        f"(got shape {array.shape})"
    )


def periodic_single_point(
    numbers: Sequence[int],
    positions,
    cell,
    pbc=None,
    kpts=None,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    smearing_ev: float = 0.0,
    magnetization: str = "fixed",
) -> dict:
    """PM3 single point under periodic boundary conditions.

    Parameters
    ----------
    cell : array-like
        Lattice vectors as rows, in **Angstrom**.
    pbc : sequence of bool, optional
        Which directions are periodic. Defaults to all three. A slab or a chain
        keeps its non-periodic directions out of the volume, the reciprocal
        basis and the stress.
    kpts : sequence of int, optional
        Monkhorst-Pack divisions, Gamma-centred. ``None`` or ``(1, 1, 1)`` takes
        only the Gamma point.
    smearing_ev : float
        Fermi-Dirac broadening. Zero fills strictly by energy.
    magnetization : {"fixed", "free"}
        How the two spin channels share a Fermi level. ``"fixed"`` holds
        ``n_alpha - n_beta`` at the multiplicity's value, which is the periodic
        reading of a molecular multiplicity. ``"free"`` gives both spins one
        Fermi level and lets the moment come out where the electronic structure
        puts it -- the right convention for a magnetic solid, where the moment is
        an output rather than an input.

    Returns
    -------
    dict
        Energies per unit cell, Mulliken charges, ``gamma_margin_bohr``,
        ``band_gap_ev`` and ``fermi_ev``. The same keys come back from the
        Gamma-point and k-mesh branches; which of them are ``None`` differs.
        At Gamma there is no Fermi level -- orbitals are filled by aufbau -- so
        ``fermi_ev`` is ``None`` there rather than a midpoint dressed up as one.

    Notes
    -----
    ``free_energy_ev`` is the **Mermin electronic** free energy
    ``energy_ev - entropy_ts_ev``, and ``entropy_ts_ev`` is the ``T*S`` of the
    Fermi-Dirac occupations. With ``smearing_ev = 0`` the occupations are a step,
    ``T*S`` is exactly zero and the two energies coincide.

    This matters because the force is the gradient of the free energy, not of the
    energy, once the occupations are fractional -- so on a metal the two are the
    pair that belong together.

    **It is not a Gibbs free energy.** There is no zero-point energy, no
    vibrational partition function, no ``pV`` term and no nuclear entropy in it;
    ``G = H - T*S_total`` would need a normal-mode analysis this package does not
    perform. It is also not ``heat_of_formation_kcal``, which is MOPAC's
    parameterized dH_f at 298 K -- a fitted quantity rather than a computed
    thermodynamic potential. Three distinct things, deliberately named apart.

    Notes
    -----
    **Check ``gamma_margin_bohr`` on a Gamma-point run.** One k-point substitutes
    ``P(Gamma)`` for ``P(0, T)`` at every image, which is exact only when no image
    lies inside the exchange range. A cell one Bohr too narrow converges cleanly
    to an answer wrong by tens of eV. Positive is fine; negative is not. See
    ``docs/pbc.md``.

    Keys: ``energy_ev``, ``energy_hartree``, ``electronic_ev``, ``core_ev``,
    ``correction_ev``, ``ewald_ev``, ``heat_of_formation_kcal``, ``charges``,
    ``converged``, ``n_kpoints``, ``homo_ev``, ``lumo_ev``, ``band_gap_ev``,
    ``fermi_ev``, ``gamma_margin_bohr``, ``entropy_ts_ev``, ``free_energy_ev``,
    ``rescued_by``, ``charge_swing``.

    ``charge_swing`` is how far any atom's electron population moved *during* the
    iteration, in electrons. Small for a well-behaved SCF. A large value says the density
    passed through qualitatively different arrangements before settling -- charge
    sloshing -- and a converged answer reached that way may have landed on a spurious
    self-consistent branch.

    Rocksalt NaCl is the case that makes this concrete. On a ``2x2x2`` mesh it converges,
    with ``converged`` true and no error, to a state carrying **-2.10 electrons of charge
    on the sodium**; every mesh from ``3x3x3`` to ``7x7x7`` puts +0.16 to +0.24 there and
    agrees on the energy to 0.06 eV. The even mesh swings 1.44 electrons on the way,
    against 0.29 for the odd one. Nothing in the energy or the residual says the answer is
    nonsense; this does.

    ``rescued_by`` is ``None`` almost always. It is a sentence when the SCF ran out at
    the settings you asked for and a retry converged it -- currently only ever with Fermi
    smearing, and only kept when the occupations came out integral, which proves the
    smeared answer is the one the strict filling would have reached. **Read it when it is
    set** and corroborate the energy against another k-mesh: a convergence aid can find a
    different self-consistent solution, and on rocksalt NaCl the difference between two of
    them is 37 eV.
    """
    numbers, positions = _as_lists(numbers, positions)
    return _native.periodic_single_point(
        numbers,
        positions,
        _cell_rows(cell),
        None if pbc is None else [bool(v) for v in np.asarray(pbc).reshape(-1)],
        None if kpts is None else [int(v) for v in np.asarray(kpts).reshape(-1)],
        charge,
        multiplicity,
        reference,
        method,
        smearing_ev,
        magnetization,
    )


def periodic_forces(
    numbers: Sequence[int],
    positions,
    cell,
    pbc=None,
    kpts=None,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    smearing_ev: float = 0.0,
    magnetization: str = "fixed",
) -> dict:
    """Periodic forces (eV/Angstrom) and stress (Voigt, eV/Angstrom^3).

    Every periodic dimensionality reports a stress, with exact zeros in the
    non-periodic directions. It is ``None`` only for an isolated cell, which has no
    strain: reporting zero there would let a variable-cell relaxation "converge"
    against a stress that was never computed.

    ``smearing_ev`` and ``magnetization`` mean what they do in
    :func:`periodic_single_point`, and are here for a blunt reason: without them
    this function converged a strictly-filled SCF while its sibling converged a
    smeared one, so an ASE calculator built with a smearing took its energy and
    forces from one self-consistent solution and its charges from another.

    Keys: ``forces_ev_per_angstrom``, ``stress_ev_per_angstrom3``, ``energy_ev``,
    ``energy_hartree``, ``charges``, ``heat_of_formation_kcal``,
    ``gamma_margin_bohr``, ``entropy_ts_ev``, ``free_energy_ev``, ``converged``.
    """
    numbers, positions = _as_lists(numbers, positions)
    return _native.periodic_forces(
        numbers,
        positions,
        _cell_rows(cell),
        None if pbc is None else [bool(v) for v in np.asarray(pbc).reshape(-1)],
        None if kpts is None else [int(v) for v in np.asarray(kpts).reshape(-1)],
        charge,
        multiplicity,
        reference,
        method,
        smearing_ev,
        magnetization,
    )


def phonons(
    numbers: Sequence[int],
    positions,
    cell,
    pbc=None,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    q=None,
    kpts=None,
    smearing_ev: float = 0.0,
    lo_to_direction=None,
    cphf_max_iter=None,
) -> dict:
    """Phonon frequencies (cm^-1).

    With ``q`` left out this is the Gamma-point analytic Hessian, and
    ``acoustic_residual_cm`` is the largest of the three acoustic frequencies,
    which should vanish; it is reported so it can be checked rather than trusted.

    ``q`` is a fractional wavevector, one component per reciprocal lattice
    vector, and switches to density-functional perturbation theory: the phonon
    at that wavevector, from this cell, with the electronic response included.
    ``kpts`` gives the response a Monkhorst-Pack mesh to sum over instead of
    Gamma alone. There is no acoustic sum rule away from Gamma, so no residual
    is reported there; what comes back instead is ``hermitian_defect`` -- the
    largest departure from ``D(q)^dagger = D(q)`` before the matrix was
    symmetrized, which is the number that moves first if a phase is wrong -- and
    the ``masses`` the frequencies were weighted by.

    ``lo_to_direction`` adds the non-analytic term along that Cartesian
    direction. It is what makes the ``q -> 0`` limit of a polar crystal
    direction-dependent: without it the longitudinal and transverse optical
    branches stay degenerate at Gamma, which is wrong by an amount that is not
    small. Use it together with ``q=[0, 0, 0]``. Three-dimensional cells only --
    a slab's macroscopic field vanishes linearly in ``q`` and a chain's as
    ``q^2 ln(1/q)``, so neither has a splitting to add. The Born charges and
    ``eps_inf`` it needs are computed on the way.

    **Polarization vectors** come back with the frequencies. At Gamma they are
    ``modes``, real, one mode per row: ``modes[m][3 * a + i]`` is the
    mass-weighted displacement of atom ``a`` along axis ``i`` in mode ``m``, so
    the Cartesian displacement is that divided by ``sqrt(masses[a])``. At a
    wavevector they are complex -- the atoms in a cell move with a relative phase
    -- and arrive as ``modes_real`` and ``modes_imag`` with the same indexing.
    Dropping the imaginary part would silently turn a travelling wave into a
    standing one, which is why both halves are returned rather than a magnitude.

    A frequency says how fast a mode vibrates; the eigenvector says what moves in
    it, and it is what separates an optical branch from an acoustic one or shows
    which sublattice a soft mode belongs to. It was being computed and discarded.

    Keys, Gamma: ``frequencies_cm``, ``modes``, ``masses``,
    ``acoustic_residual_cm``, ``energy_ev``.
    At a wavevector: ``frequencies_cm``, ``modes_real``, ``modes_imag``,
    ``masses``, ``hermitian_defect``, ``q``, ``lo_to_direction``.
    """
    numbers, positions = _as_lists(numbers, positions)
    return _native.phonons(
        numbers,
        positions,
        _cell_rows(cell),
        None if pbc is None else [bool(v) for v in np.asarray(pbc).reshape(-1)],
        charge,
        multiplicity,
        reference,
        method,
        None if q is None else [float(v) for v in np.asarray(q).reshape(-1)],
        None if kpts is None else [int(v) for v in np.asarray(kpts).reshape(-1)],
        float(smearing_ev),
        None if lo_to_direction is None
        else [float(v) for v in np.asarray(lo_to_direction).reshape(-1)],
            cphf_max_iter,
    )


def divide_and_conquer(
    numbers: Sequence[int],
    positions,
    cell=None,
    pbc=None,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    core_radius: float = 3.2,
    buffer_radius: float = 4.8,
    smearing_ev: float = 0.1,
    long_range_cutoff=None,
    field=None,
) -> dict:
    """Divide-and-conquer single point, molecular or periodic.

    Parameters
    ----------
    core_radius, buffer_radius : float
        Angstrom. The buffer is the knob that matters: widening it must converge
        the result onto the full diagonalization. If it does not, the system is
        not near-sighted -- it is metallic, or the gap has closed -- and
        divide-and-conquer is the wrong method rather than a poorly converged one.
    cell : array-like, optional
        Passing a cell runs the periodic (Gamma-point) path.
    long_range_cutoff : float, optional
        Angstrom. Splits the Coulomb term the way the periodic path does: full
        pair tables inside the cutoff, a point-charge model outside, so the
        two-electron work grows with the system rather than with its square.
        ``None`` (the default) keeps the exact tables, because the split is not
        free -- it costs a measured 28 ueV per atom at the usual handover, flat in
        system size. Molecular runs only: the periodic path takes its long range
        from the lattice sum, so passing both a cell and a cutoff is refused
        rather than quietly running the unscreened route.
    field : sequence of float, optional
        Uniform external field in volts per Angstrom. Molecular and unscreened
        only -- the periodic path and the screened near-field path refuse one,
        because `-F.r` is not lattice-periodic and the screened machinery does not
        carry it.

    Returns
    -------
    dict
        ``energy_ev``, ``energy_hartree``, ``heat_of_formation_kcal``,
        ``charges``, ``converged``, ``fermi_ev``, ``n_subsystems``, plus
        ``dropped_pairs`` and ``largest_subsystem``, which say what the
        partitioning traded: how many density elements were set to zero, and
        how big the problems actually solved were.
    """
    numbers, positions = _as_lists(numbers, positions)
    return _native.divide_and_conquer(
        numbers,
        positions,
        None if cell is None else _cell_rows(cell),
        None if pbc is None else [bool(v) for v in np.asarray(pbc).reshape(-1)],
        charge,
        multiplicity,
        reference,
        method,
        core_radius,
        buffer_radius,
        smearing_ev,
        long_range_cutoff,
        None if field is None else [float(v) for v in np.asarray(field).reshape(-1)],
    )


def divide_and_conquer_optimize(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    core_radius: float = 3.2,
    buffer_radius: float = 4.8,
    smearing_ev: float = 0.1,
    long_range_cutoff=None,
    field=None,
    max_steps: int = 200,
    force_tol: float = 0.02,
) -> dict:
    """Geometry optimization on the divide-and-conquer gradient.

    The same L-BFGS as :func:`optimize`, driven by
    :func:`divide_and_conquer_forces` instead of a full diagonalization -- for the
    case the method exists for, where a full diagonalization per line-search trial
    is not affordable.

    Molecular only. The periodic partitioned path has no cell gradient, so a cell
    would have to be held fixed silently; :func:`relax` is the periodic optimizer.

    Arguments are :func:`divide_and_conquer`'s, plus ``max_steps`` and
    ``force_tol`` (eV/Angstrom) from :func:`optimize`.

    Returns
    -------
    dict
        ``positions_angstrom`` (and ``positions``), ``energy_ev``,
        ``energy_hartree``, ``heat_of_formation_kcal``, ``converged``,
        ``iterations`` (and ``steps``), ``n_subsystems``.

    The caveat on :func:`divide_and_conquer_forces` carries through and gets
    worse here: a partitioned gradient is not the exact derivative of the
    partitioned energy, so the geometry this converges to is the buffer's
    minimum, not the method's. Widen ``buffer_radius`` and re-run before
    believing a structure.
    """
    numbers, positions = _as_lists(numbers, positions)
    return _native.divide_and_conquer_optimize(
        numbers,
        positions,
        charge,
        multiplicity,
        reference,
        method,
        core_radius,
        buffer_radius,
        smearing_ev,
        long_range_cutoff,
        None if field is None else [float(v) for v in np.asarray(field).reshape(-1)],
        int(max_steps),
        float(force_tol),
    )


def divide_and_conquer_forces(
    numbers: Sequence[int],
    positions,
    cell=None,
    pbc=None,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    core_radius: float = 3.2,
    buffer_radius: float = 4.8,
    smearing_ev: float = 0.1,
    long_range_cutoff=None,
    field=None,
) -> dict:
    """Forces (eV/Angstrom) -- and, with a cell, stress -- from a partitioned SCF.

    The divide-and-conquer analogue of :func:`periodic_forces`, and separate from
    :func:`divide_and_conquer` for the same reason that one is separate from
    :func:`periodic_single_point`: this method exists for systems large enough
    that a gradient is worth asking for rather than producing unasked.

    Arguments are :func:`divide_and_conquer`'s. Returns
    ``forces_ev_per_angstrom``, ``stress_ev_per_angstrom3`` (``None`` without a
    cell) and the same energy and charge keys: ``energy_ev``, ``energy_hartree``,
    ``heat_of_formation_kcal``, ``charges``, ``converged``, ``n_subsystems``.

    One caveat, stated rather than hidden: a divide-and-conquer density is not
    variational -- it is assembled, not minimized -- so the usual argument that
    the first-order energy error vanishes at the SCF solution does not apply, and
    the gradient carries the density's own truncation error rather than its
    square. It converges with the buffer the same way the energy does.
    """
    numbers, positions = _as_lists(numbers, positions)
    return _native.divide_and_conquer_forces(
        numbers,
        positions,
        None if cell is None else _cell_rows(cell),
        None if pbc is None else [bool(v) for v in np.asarray(pbc).reshape(-1)],
        charge,
        multiplicity,
        reference,
        method,
        core_radius,
        buffer_radius,
        smearing_ev,
        long_range_cutoff,
        None if field is None else [float(v) for v in np.asarray(field).reshape(-1)],
    )


def born_charges(
    numbers: Sequence[int],
    positions,
    cell,
    pbc=None,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    enforce: bool = False,
    cphf_max_iter=None,
) -> dict:
    """Born effective charges: one ``3 x 3`` tensor per atom, in elementary charges.

    ``Z*[a][alpha][beta]`` is the dipole the cell acquires along ``alpha`` per unit
    displacement of atom ``a`` along ``beta``. It is what carries the long-range
    electrostatics of a lattice vibration: without it a polar crystal's
    longitudinal and transverse optical branches stay degenerate as ``q -> 0``,
    which is wrong by an amount that is not small.

    Periodic only. An isolated molecule's equivalent is the atomic polar tensor,
    :func:`dipole`'s ``derivatives_e``.

    ``sum_rule_residual`` is the largest ``|sum_a Z*_a|``. Translating the whole
    crystal produces no dipole, so this is zero for an exact response -- it is the
    number that says whether the coupled-perturbed solve converged. Reported
    rather than enforced, because flattening it silently would hide exactly that.
    Pass ``enforce=True`` to remove the mean violation after reading it.

    A caveat worth knowing: the sum rule alone cannot tell a correct response from
    an absent one, because the static charges of a neutral cell already sum to
    zero. The crate's own tests therefore also check ``Z*`` against a central
    difference of the cell dipole.

    Keys: ``born_charges``, ``sum_rule_residual``, ``gamma_margin_bohr``.
    """
    numbers, positions = _as_lists(numbers, positions)
    return _native.born_charges(
        numbers,
        positions,
        _cell_rows(cell),
        None if pbc is None else [bool(v) for v in np.asarray(pbc).reshape(-1)],
        float(charge),
        int(multiplicity),
        str(reference),
        str(method),
        bool(enforce),
            cphf_max_iter,
    )


def dielectric(
    numbers: Sequence[int],
    positions,
    cell,
    pbc=None,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    include_ionic: bool = False,
    cphf_max_iter=None,
) -> dict:
    """Electronic polarizability and, in 3D, the dielectric tensor.

    ``polarizability`` is ``alpha`` in Bohr^3 and exists in every
    dimensionality. ``epsilon`` is ``1 + 4*pi*alpha/V`` and is ``None`` for a
    chain or a slab: ``V`` would be a length or an area there, and dividing by a
    supercell's vacuum padding would make the answer a statement about the
    padding rather than about the material.

    ``epsilon`` is the **electronic** (clamped-ion, high-frequency) response
    ``eps_inf``, which holds the nuclei still.

    ``include_ionic=True`` adds four more keys: ``epsilon_static`` (the static
    constant ``eps_0``, which lets the nuclei relax along each infrared-active
    mode), its two halves ``epsilon_electronic`` and ``epsilon_ionic``, and
    ``skipped_modes``. It is off by default because it costs a Gamma-point
    phonon calculation and a set of Born charges on top of the field response.

    ``eps_0`` is only meaningful at a **relaxed geometry**. The ionic term sums
    ``(4 pi / V) sum_m Z* Z* / omega_m^2`` over the modes, so it diverges as a
    mode softens and is undefined for an imaginary one. ``skipped_modes`` counts
    the modes with ``omega^2 <= 0`` that were left out: three of them are the
    acoustic branch and are expected, and **more than three means the structure
    is not a minimum** and the ionic term is missing whatever those modes would
    have contributed -- which for a soft mode is most of it. Relax first, and
    check the count rather than trusting the tensor.

    A caveat worth stating: PM3 was parameterized against molecular heats of
    formation, geometries, dipoles and ionization potentials, not against
    solid-state dielectric response. These are the correct tensors *of this
    model*, which is a different claim from agreement with a measurement.

    Keys: ``polarizability``, ``epsilon``, ``gamma_margin_bohr``; with
    ``include_ionic=True`` also ``electronic``, ``ionic`` and ``skipped_modes``.
    """
    numbers, positions = _as_lists(numbers, positions)
    return _native.dielectric(
        numbers,
        positions,
        _cell_rows(cell),
        None if pbc is None else [bool(v) for v in np.asarray(pbc).reshape(-1)],
        float(charge),
        int(multiplicity),
        str(reference),
        str(method),
        bool(include_ionic),
            cphf_max_iter,
    )


def berry_polarization(
    numbers: Sequence[int],
    positions,
    cell,
    strings: int = 12,
    kpts=None,
    pbc=None,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
) -> dict:
    """Berry-phase polarization (e/Bohr^2), the modern theory.

    ``strings`` is the number of k-points along each Brillouin-zone string and
    is the convergence parameter: the answer must become independent of it.
    ``kpts`` is the transverse sampling.

    Polarization is defined only **modulo** ``quantum``, one per lattice vector.
    That is the physics, not a defect -- a different branch of the logarithm
    assigns the electrons to a different unit cell. Only *differences* mean
    anything, and a plain subtraction of two ``total`` values is off by exactly
    one quantum whenever the two landed on different branches. Reduce instead::

        d = numpy.asarray(after["total"]) - numpy.asarray(before["total"])
        for q in numpy.asarray(before["quantum"]):
            d -= q * numpy.round(numpy.dot(d, q) / numpy.dot(q, q))

    Three-dimensional cells only, since the quantum is ``e a/V``.

    This exists as an independent check on :func:`born_charges`: it reaches the
    same ``Z*`` through overlaps between neighbouring k-points, with no response
    equation anywhere in it. The two differ by the intra-atomic ``s``-``p``
    moment the phase cannot carry, which places every orbital at its atom.

    Keys: ``total``, ``electronic``, ``ionic``, ``phase``, ``quantum``,
    ``string_length``, ``gamma_margin_bohr``.
    """
    numbers, positions = _as_lists(numbers, positions)
    return _native.berry_polarization(
        numbers,
        positions,
        _cell_rows(cell),
        int(strings),
        None if kpts is None else [int(v) for v in np.asarray(kpts).reshape(-1)],
        None if pbc is None else [bool(v) for v in np.asarray(pbc).reshape(-1)],
        float(charge),
        int(multiplicity),
        str(reference),
        str(method),
    )


def phonon_bands(
    numbers: Sequence[int],
    positions,
    cell,
    supercell,
    path,
    points: int = 12,
    pbc=None,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    enforce_asr: bool = False,
) -> dict:
    """Phonon dispersion from supercell force constants.

    One Hessian of the supercell buys the whole path, where direct DFPT pays per
    wavevector. ``supercell`` is the replication; ``path`` is a list of
    fractional wavevectors and ``points`` interpolates that many per segment.

    A supercell's Gamma point *is* a mesh of the primitive cell, so what this
    reproduces is the **mesh**-sampled response with the matching mesh, not a
    Gamma-only one.

    ``acoustic_sum_rule_residual`` is reported before it can be imposed: it is
    what the truncation threw away, and flattening it unseen would hide force
    constants that are simply wrong. Pass ``enforce_asr=True`` once you have
    looked at it.

    Keys: ``q``, ``frequencies_cm``, ``supercell``, ``acoustic_sum_rule_residual``,
    ``gamma_margin_bohr``.
    """
    numbers, positions = _as_lists(numbers, positions)
    return _native.phonon_bands(
        numbers,
        positions,
        _cell_rows(cell),
        [int(v) for v in np.asarray(supercell).reshape(-1)],
        [[float(c) for c in np.asarray(q, dtype=float).reshape(-1)] for q in path],
        int(points),
        None if pbc is None else [bool(v) for v in np.asarray(pbc).reshape(-1)],
        float(charge),
        int(multiplicity),
        str(reference),
        str(method),
        bool(enforce_asr),
    )


def finite_field(
    numbers: Sequence[int],
    positions,
    cell,
    field,
    kpts,
    pbc=None,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    tol: float = 1.0e-8,
    max_iter: int = 60,
    mixing: float = 0.5,
) -> dict:
    """A finite electric field along a periodic direction (Berry-phase enthalpy).

    ``field`` is in eV per (e*Bohr). A component along a lattice vector is what
    needs this: ``E.R`` is not lattice-periodic there, the spectrum has no lower
    bound, and the ground state of ``H - E.R`` does not exist. What is minimized
    instead is the electric enthalpy ``F = E - Omega E.P`` with ``P`` the
    Berry-phase polarization. A field orthogonal to *every* lattice vector needs
    none of this and goes through ``Pm3Options::field`` / ``PM3(field=...)``.

    ``kpts`` is the mesh; its division along each field direction is that
    direction's Berry-phase string length, and is the convergence parameter the
    answer must become independent of. At least 3 along any direction the field
    has a component on -- two points cannot resolve a winding.

    Returns ``energy``, ``enthalpy_ev``, ``polarization`` and its two halves,
    ``phase`` (in turns), ``iterations``, ``converged``, ``resolved`` and
    ``gamma_margin_bohr``.

    ``resolved`` says, per axis, whether the mesh had the three k-points a Berry
    phase needs. An unresolved axis contributes **zero** to ``phase`` and to
    ``electronic_polarization``, which is not the same as its contribution being
    zero: a ``[6, 1, 1]`` mesh resolves ``x`` alone. That is enough to measure
    ``alpha_xx`` -- the other components cancel in a ``+E``/``-E`` difference --
    and not enough to read the polarization vector itself.

    Restricted, gapped, three-dimensional cells only; an open shell would need
    each spin manifold's phase separately. **There is no force here**: the
    derivative of the enthalpy with respect to the nuclei is not implemented, so
    this returns a state, not a relaxation.

    Polarization is defined modulo the quantum ``e a/Omega``. Compare two runs by
    differencing them at a field small enough that the branch cannot have
    changed, as ``alpha = Omega dP/dE`` does, rather than reading one in
    isolation.

    Keys: ``energy_ev`` (also as ``energy``, this function's older name for it),
    ``enthalpy_ev``, ``polarization``, ``electronic_polarization``,
    ``ionic_polarization``, ``resolved``, ``phase``, ``field``, ``converged``,
    ``iterations``, ``gamma_margin_bohr``.
    """
    numbers, positions = _as_lists(numbers, positions)
    return _native.finite_field(
        numbers,
        positions,
        _cell_rows(cell),
        [float(v) for v in np.asarray(field, dtype=float).reshape(-1)],
        [int(v) for v in np.asarray(kpts).reshape(-1)],
        None if pbc is None else [bool(v) for v in np.asarray(pbc).reshape(-1)],
        float(charge),
        int(multiplicity),
        str(reference),
        str(method),
        float(tol),
        int(max_iter),
        float(mixing),
    )


def molden(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    field=None,
    basis: str = "gto",
) -> str:
    """The converged wavefunction as a Molden document (a string).

    Coordinates in **Angstrom**. Returns the file's text rather than writing it,
    so the caller chooses the destination.

    ``basis`` selects the section the basis is written into:

    ``"gto"`` (default)
        The Slater orbitals as even-tempered Gaussian expansions, because almost
        no viewer reads Molden's ``[STO]`` section. The fit's overlap defect is
        at most 3e-5.
    ``"sto"``
        The Slater exponents PM3 actually uses, in ``[STO]``. Exact rather than
        fitted, and readable by fewer viewers -- for the ones that do, and for
        anyone who wants the model's own exponents rather than a fit to them.

    The two differ only in the basis section: the ``[MO]`` block is the same
    wavefunction either way, and a test asserts it.

    Note what is being drawn, in both cases: PM3 assumes its AO basis is
    orthonormal, and the real Slater basis is not, so these are the conventional
    semiempirical orbitals -- the same ones MOPAC's ``VECTORS`` and ``GRAPHF``
    produce -- not the orbitals of the Slater basis itself.
    """
    n, p = _as_lists(numbers, positions)
    return _native.molden(
        n,
        p,
        float(charge),
        int(multiplicity),
        str(reference),
        str(method),
        None if field is None else [float(v) for v in field],
        str(basis),
    )


def ir_spectrum(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    field=None,
    cphf_max_iter=None,
) -> dict:
    """Infrared spectrum from the analytic dipole derivatives.

    Evaluate at a **stationary point** (optimize first). Coordinates in
    **Angstrom**. Returns ``frequencies_cm`` (as :func:`frequencies`),
    ``intensities_km_per_mol`` aligned with them, and
    ``dipole_derivatives_e`` -- the raw ``3 x 3N`` tensor in elementary charges,
    row = dipole axis, column = 3*atom + axis, and ``ndof``.

    The tensor costs three coupled-perturbed solves regardless of system size,
    not one per nuclear coordinate; see the Rust module for why.
    """
    n, p = _as_lists(numbers, positions)
    return _native.ir_spectrum(
        n,
        p,
        float(charge),
        int(multiplicity),
        str(reference),
        str(method),
        None if field is None else [float(v) for v in field],
            cphf_max_iter,
    )


def dipole(
    numbers: Sequence[int],
    positions,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    field=None,
    operator: bool = False,
) -> dict:
    """The dipole operator and everything derived from it.

    The reported dipole and the external field's coupling are the same matrix,
    which is what makes ``mu = dE/dF`` hold by construction rather than by
    coincidence. Returns ``dipole_debye``, ``dipole_e_angstrom`` and
    ``centre_of_mass_angstrom`` (the origin both are about, MOPAC's convention),
    and the ``3 x 3N`` derivative tensor ``derivatives_e`` in elementary charges.

    That tensor costs **three** coupled-perturbed solves rather than ``3N``, by
    the interchange theorem, so it is the same price whatever the molecule's
    size. No Hessian is built; :func:`ir_spectrum` is what pairs it with one.

    ``operator=True`` adds ``operator_bohr``: the three ``nao x nao`` moment
    matrices themselves, in Bohr about the same centre.
    """
    n, p = _as_lists(numbers, positions)
    return _native.dipole(
        n,
        p,
        float(charge),
        int(multiplicity),
        str(reference),
        str(method),
        None if field is None else [float(v) for v in field],
        bool(operator),
    )


def dynamical_matrix(
    numbers: Sequence[int],
    positions,
    cell,
    q,
    pbc=None,
    kpts=None,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    rigid_ion: bool = False,
    smearing_ev: float = 0.0,
) -> dict:
    """``D(q)`` itself, in eV/Bohr^2, alongside the frequencies it gives.

    :func:`phonons` with a ``q`` returns the frequencies alone; here the matrix
    comes too, as ``real`` and ``imag``, each ``3N x 3N``, for a caller building
    a dispersion, applying their own masses, or checking an identity.

    ``masses`` (amu, in the order the matrix indexes them) and
    ``frequencies_cm`` travel with it. Without the masses the matrix cannot be
    mass-weighted from Python at all -- the isotope-averaged values are this
    crate's, not ``ase``'s -- which left the frequencies at a wavevector
    reachable only from Rust.

    ``hermitian_defect`` is the largest departure from ``D(q)^dagger = D(q)``
    **before** the matrix was symmetrized. It is reported rather than hidden
    because a wrong assembly shows there first: a defect that grows with ``q``
    means the response is not the adjoint it should be.

    ``rigid_ion=True`` leaves the electronic response out, giving the
    fixed-density part alone -- which is the half a supercell finite difference
    can be compared against directly.
    """
    numbers, positions = _as_lists(numbers, positions)
    return _native.dynamical_matrix(
        numbers,
        positions,
        _cell_rows(cell),
        [float(v) for v in np.asarray(q).reshape(-1)],
        None if pbc is None else [bool(v) for v in np.asarray(pbc).reshape(-1)],
        None if kpts is None else [int(v) for v in np.asarray(kpts).reshape(-1)],
        float(charge),
        int(multiplicity),
        str(reference),
        str(method),
        bool(rigid_ion),
        float(smearing_ev),
    )


def bands(
    numbers: Sequence[int],
    positions,
    cell,
    path,
    pbc=None,
    kpts=None,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    smearing_ev: float = 0.0,
    magnetization: str = "fixed",
) -> dict:
    """Band energies (eV) along a path of fractional k-points.

    The path is evaluated in the potential of a calculation converged on
    ``kpts``, non-self-consistently, so points off the mesh cost one
    diagonalization each and nothing more. ``smearing_ev`` and ``magnetization``
    belong to that underlying mesh calculation, and are here because it is one:
    without them a band structure of a metal was converged under strict filling
    no matter what the caller had used for the energy, and the ``fermi_ev`` it
    returned was a different calculation's.

    Returns ``bands_ev`` (one ascending list per path point), ``bands_beta_ev``
    (``None`` for a restricted run), ``fermi_ev``, ``kpoints`` (the path, as
    fractional coordinates) and ``distances_per_bohr`` -- the cumulative
    Cartesian path length, which is the horizontal axis a band plot needs if
    segments of different reciprocal length are to look different.
    """
    numbers, positions = _as_lists(numbers, positions)
    return _native.bands(
        numbers,
        positions,
        _cell_rows(cell),
        [[float(v) for v in np.asarray(point).reshape(-1)] for point in path],
        None if pbc is None else [bool(v) for v in np.asarray(pbc).reshape(-1)],
        None if kpts is None else [int(v) for v in np.asarray(kpts).reshape(-1)],
        float(charge),
        int(multiplicity),
        str(reference),
        str(method),
        float(smearing_ev),
        str(magnetization),
    )


def relax(
    numbers: Sequence[int],
    positions,
    cell,
    pbc=None,
    kpts=None,
    charge: float = 0.0,
    multiplicity: int = 1,
    reference: str = "auto",
    method: str = "pm3",
    max_steps: int = 200,
    force_tol: float = 0.02,
    stress_tol: float = 0.001,
    fixed_cell: bool = False,
) -> dict:
    """Relax a periodic structure, atoms and lattice vectors together.

    ``force_tol`` is eV/Angstrom and ``stress_tol`` eV/Angstrom^3. Pass
    ``fixed_cell=True`` to move the atoms only. An isolated cell is refused,
    having no strain to relax against.

    Returns ``positions`` and ``cell`` in Angstrom, plus ``energy_ev``,
    ``energy_hartree``, ``heat_of_formation_kcal``, ``converged`` and ``steps``.

    ``positions_angstrom`` and ``iterations`` are aliases for ``positions`` and
    ``steps``, so that this function and :func:`optimize` -- the same operation on a
    molecule -- answer to the same names. Through 0.2.3 the two returned disjoint key
    sets, and the heat of formation was missing here entirely.
    """
    numbers, positions = _as_lists(numbers, positions)
    return _native.relax(
        numbers,
        positions,
        _cell_rows(cell),
        None if pbc is None else [bool(v) for v in np.asarray(pbc).reshape(-1)],
        None if kpts is None else [int(v) for v in np.asarray(kpts).reshape(-1)],
        float(charge),
        int(multiplicity),
        str(reference),
        str(method),
        int(max_steps),
        float(force_tol),
        float(stress_tol),
        bool(fixed_cell),
    )
