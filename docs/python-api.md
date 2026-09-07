<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# pm3-rs — Python API

Two layers are provided:

1. **`pm3_rs.native`** — a thin wrapper over the compiled Rust extension
   (`pm3_rs._native`). Coordinates are Ångström; energies are returned in
   **Hartree** (atomic units), with convenience eV and kcal/mol fields. This is
   the raw model surface.
2. **`pm3_rs.ase.PM3`** — an [ASE](https://wiki.fysik.dtu.dk/ase/) `Calculator`
   using ASE conventions throughout (**eV**, **eV/Å**, **Å**, **eV/Å²**). ASE is
   an optional dependency.

## Installation

```bash
pip install pm3-rs-python          # native API only
pip install "pm3-rs-python[ase]"   # + the ASE calculator
```

Or, from a checkout, build the extension into the current environment:

```bash
maturin develop --release --features python
```

## Common arguments

Every function/calculator accepts the same model selectors:

| argument       | values                                            | meaning |
|----------------|---------------------------------------------------|---------|
| `charge`       | int                                               | total (formal) molecular charge |
| `multiplicity` | int (`1`=singlet, `2`=doublet, …)                 | spin multiplicity `2S+1` |
| `reference`    | `"auto"` (default), `"rhf"`, `"uhf"`              | SCF reference; `auto` = RHF closed shell / UHF open shell |
| `method`       | `"pm3"` (default), `"pm3-d3"`, `"pm3-d3h4"`, `"pm3-d3h4x"`, each optionally `+mmok` | correction variant |
| `field`        | `[x, y, z]` in **V/Å**, or `None` (default)       | uniform external electric field; molecular only |

The `method` variant selects the post-SCF classical corrections — Grimme **D3**
dispersion, the Řezáč **H4** hydrogen-bond term, and the **X** halogen-bond term.
They contribute to the energy, the analytic gradient **and** the analytic Hessian.

### `+mmok`, and why MOPAC's default is not PM3

Appending `+mmok` — `method="pm3+mmok"`, `method="pm3-d3h4+mmok"` — switches on
MOPAC's `MMOK` correction: a classical `K·sin²(O=C–N–H)` term, `K = 7.1853
kcal/mol` per amide hydrogen, added to the heat of formation after the SCF to
restore the peptide rotation barrier that an NDDO Hamiltonian has no term for.

**It is off by default here and on by default in MOPAC.** That difference is
deliberate: the term is not part of PM3, so a heat of formation with it in is not
comparable to a published PM3 number. It matters only for amides, and there it
matters a lot — on acetamide it moves the heat of formation by 2.3 kcal/mol while
leaving the density, the dipole and every orbital energy untouched, which is
exactly what makes it easy to mistake for a discrepancy.

So: to compare against a MOPAC run, either give MOPAC `NOMM` or give pm3-rs
`+mmok`. `tests/data/mopac_oracle.tsv` takes the first route.

`field` is volts per Ångström, the unit MOPAC's own `FIELD=` keyword takes, and
reaches the energy, the analytic gradient and the analytic Hessian for both
references. It is refused under periodic boundary conditions, where `−f·r` is not
lattice-periodic and the energy per cell would depend on which cell was chosen.
A **charged** molecule in a field has no minimum — the net force never vanishes —
so `optimize` warns and runs rather than converging.

---

## `pm3_rs.native`

All functions take `(numbers, positions, charge=0, multiplicity=1, reference="auto", method="pm3")`,
where `numbers` is a length-`N` sequence of atomic numbers and `positions` an
`(N, 3)` array in **Ångström**. Each returns a `dict`.

### `single_point(...) -> dict`
Keys: `energy_hartree`, `energy_ev`, `heat_of_formation_kcal`, `electronic_ev`,
`core_ev`, `charges` (Mulliken, e), `dipole_debye` (`[x, y, z]`), `homo_ev`,
`lumo_ev`, `converged`, `unrestricted`.

`homo_ev` and `lumo_ev` are the frontier of **both** spin channels. For an
unrestricted run the β LUMO usually lies below the α one, so an α-only reading —
which is what this returned through 0.2.1 — reports a gap that is not the gap.

`converged` on any single-point-like call is always `True`: a run that does not
converge raises rather than returning a result with it set to `False`. The key is
kept so a reader need not know that to trust the answer. `relax`'s `converged` is
a real flag — the optimizer can run out of steps and still hand back a geometry.

### `gradient(...) -> dict`
Analytic Hellmann–Feynman nuclear gradient `dE/dR`. Keys: `energy_hartree`,
`energy_ev`, `heat_of_formation_kcal`, `gradient_hartree_per_bohr` (`(N, 3)`),
`gradient_ev_per_angstrom` (`(N, 3)`).

### `forces(...) -> dict`
Forces `= −dE/dR`. Keys: `energy_hartree`, `energy_ev`, `heat_of_formation_kcal`,
`forces_hartree_per_bohr`, `forces_ev_per_angstrom`.

### `optimize(...) -> dict`
L-BFGS geometry optimization on the analytic gradient. Keys:
`positions_angstrom` (`(N, 3)`), `energy_hartree`, `energy_ev`,
`heat_of_formation_kcal`, `converged`, `iterations`, plus `positions` and
`steps` as aliases for the first and the last.

The aliases exist because this function and [`relax`](#relaxnumbers-positions-cell--fixed_cellfalse---dict)
— the same operation on a periodic cell — used to return **disjoint** key sets:
this one had `positions_angstrom` / `energy_hartree` / `iterations` and no
`energy_ev`, `relax` had `positions` / `energy_ev` / `steps` and no heat of
formation. Reading the docs for one and applying them to the other ended a
structure optimization with a `KeyError`, in either direction. Both names are
now carried by both functions; `tests/test_api_key_contract.py` holds every
native function to the keys it documents.

### `frequencies(...) -> dict`
Harmonic vibrational frequencies from the analytic (CPHF) Hessian. Evaluate at a
**stationary point** (optimize first). Keys: `frequencies_cm` (ascending;
negatives are imaginary) and mass-weighted `eigenvalues` (eV/(Å²·amu)).

### `hessian(...) -> dict`
Analytic Cartesian Hessian (coupled-perturbed SCF + the classical D3/H4/X second
derivatives for the correction variants). Keys: `hessian_hartree_per_bohr2`
(a `3N × 3N` nested list, atomic units) and `ndof`.

### Example

```python
import numpy as np
from pm3_rs import native

numbers = [8, 1, 1]                       # water: O, H, H
positions = np.array([[0.00, 0.00, 0.00],
                      [0.96, 0.00, 0.00],
                      [-0.24, 0.93, 0.00]])

sp = native.single_point(numbers, positions)
print(sp["heat_of_formation_kcal"], "kcal/mol")

# PM3-D3H4 optimization, then frequencies at the minimum
opt = native.optimize(numbers, positions, method="pm3-d3h4")
freqs = native.frequencies(numbers, opt["positions_angstrom"], method="pm3-d3h4")
print(freqs["frequencies_cm"])

# A cation (NH4+ — closed-shell singlet, so `reference="auto"` picks RHF)
nh4_positions = np.array([[ 0.00,  0.00,  0.00],
                          [ 0.63,  0.63,  0.63],
                          [-0.63, -0.63,  0.63],
                          [-0.63,  0.63, -0.63],
                          [ 0.63, -0.63, -0.63]])
g = native.gradient([7, 1, 1, 1, 1], nh4_positions, charge=1, multiplicity=1)

# An open-shell radical: multiplicity 2 selects UHF automatically.
ch3_positions = np.array([[ 0.0000,  0.000, 0.0],
                          [ 0.0000,  1.078, 0.0],
                          [ 0.9336, -0.539, 0.0],
                          [-0.9336, -0.539, 0.0]])
ch3 = native.single_point([6, 1, 1, 1], ch3_positions, multiplicity=2)
assert ch3["unrestricted"]
```

The top-level module re-exports the native functions for convenience:
`from pm3_rs import single_point, gradient, forces, optimize, frequencies, hessian`.

---

## `pm3_rs.ase.PM3`

An ASE `Calculator`. Units follow ASE: **eV**, **eV/Å**, **Å**, **eV/Å²**.

```python
PM3(charge=0, multiplicity=1, reference="auto", method="pm3",
    kpts=None, smearing_ev=0.0, magnetization="fixed", field=None, **kwargs)
```

`kpts` is a Monkhorst–Pack mesh, `smearing_ev` the Fermi–Dirac broadening, and
`magnetization` either `"fixed"` (hold the multiplicity's moment) or `"free"`
(one Fermi level for both spins, the moment an output). All three reach every
periodic accessor, including the energy and forces — `periodic_forces` took no
smearing before 0.2.2, so a smeared calculator reported an energy and forces
from one self-consistent solution and charges from another.

`field` is in **volts per Ångström** and is molecular only. Under periodic
boundary conditions it is **refused rather than dropped**: `−F·r` is not
lattice-periodic, so the energy per cell would depend on which cell was chosen.

`implemented_properties` = `["energy", "free_energy",
"electronic_entropy_ts_ev", "forces", "charges", "dipole",
"heat_of_formation_kcal", "hessian", "stress"]`. Energy, forces, charges and
dipole are computed eagerly and served through ASE's standard accessors
(`get_potential_energy`, `get_forces`, `get_charges`, `get_dipole_moment`, and
`get_property("heat_of_formation_kcal")`).

### Three energies that are not each other

| key | what it is |
|---|---|
| `energy` | the electronic energy `E` |
| `free_energy` | the **Mermin electronic** free energy `E − TS` |
| `heat_of_formation_kcal` | MOPAC's parameterized ΔH°f at 298 K |

`free_energy` is what ASE means by the name and what
`get_potential_energy(force_consistent=True)` returns. With Fermi–Dirac smearing
the variational functional is `E − TS`, so **that** is the potential whose
nuclear gradient `get_forces()` returns — reporting `E` beside those forces makes
the pair inconsistent by `∂(TS)/∂R`. With `smearing_ev = 0` the occupations are a
step, `TS` is exactly zero, and `free_energy == energy`; that covers every
molecular calculation. `electronic_entropy_ts_ev` carries the `TS` itself.

**None of these is a Gibbs free energy.** `G = H − T·S_total` needs a zero-point
energy, a vibrational partition function and a `pV` term, and this package
computes none of them — no normal-mode thermochemistry exists here. The entropy
in `free_energy` is the *electronic* entropy of fractional band occupations at a
fictitious electronic temperature (`smearing_ev` is `k_BT` for that temperature,
chosen to make a metal's occupations converge), not a thermodynamic entropy of
the nuclei. And `heat_of_formation_kcal` is a fitted quantity, not a computed
thermodynamic potential at all.

The **Hessian is lazy**: it is declared as a property but built only when
requested (`get_hessian()` / `get_property("hessian")`), never as part of an
energy/forces cycle, and then cached like any other ASE property.

| method | returns |
|--------|---------|
| `get_potential_energy(atoms=None)` | energy (eV) — ASE base class |
| `get_forces(atoms=None)`           | forces `(N, 3)` (eV/Å) |
| `get_charges(atoms=None)`          | Mulliken charges (e) — ASE base class |
| `get_dipole_moment(atoms=None)`    | dipole (e·Å) — ASE base class |
| `get_hessian(atoms=None)`          | Cartesian Hessian `(3N, 3N)` (eV/Å²), lazy |
| `get_frequencies(atoms=None)`      | harmonic frequencies (cm⁻¹) |
| `get_ir_spectrum(atoms=None)`      | frequencies with infrared intensities (km/mol) |
| `get_dipole_derivatives(atoms=None)` | the `3 × 3N` tensor `∂μ/∂R` (e), no Hessian |
| `get_dipole_operator(atoms=None)`  | the three `nao × nao` moment matrices (Bohr) |
| `write_molden(path, atoms=None)`   | the converged orbitals, for a viewer |
| `get_phonons(atoms=None, q=None, kpts=None)` | phonons at Γ, or at a wavevector (periodic) |
| `get_dynamical_matrix(q, atoms=None, kpts=None, rigid_ion=False)` | `D(q)` as `(real, imag)` (periodic) |
| `get_bands(path, atoms=None, kpts=None)` | band energies (eV) along a k-path, with `distances_per_bohr` for the axis (periodic) |
| `relax(atoms=None, fixed_cell=False, max_steps=200, force_tol=0.02, stress_tol=0.001)` | variable-cell relaxation (periodic) |
| `divide_and_conquer(atoms=None, ...)` | partitioned SCF, for a system too large to diagonalize whole |
| `divide_and_conquer_forces(atoms=None, ...)` | forces (eV/Å) and, for a cell, stress, from that partitioned density |

`get_frequencies` returns `modes` and `masses` alongside the frequencies. The
modes are in **mass-weighted** coordinates, so the Cartesian displacement of mode
`m` is `modes[i][m] / √masses[i // 3]` — which is why the masses come with them,
and why returning the frequencies alone put every use of a normal mode out of
reach from Python.

`get_dynamical_matrix` falls back to the calculator's own `kpts` when the call
site names none, as `get_phonons` does; before 0.2.2 it did not, and the two
accessors could describe different calculations for the same calculator.

None of these are ASE properties: `ase.vibrations` and `ase.phonons` build their
own force constants out of `get_forces`, and ASE's optimizers move atoms through
`get_forces` and `get_stress`, which this calculator serves like any other. These
are the direct routes.

**They are lazy, and cached.** Nothing here is computed as part of an
energy/forces cycle — `calculate` produces energy, forces, charges and dipole and
stops. Each accessor computes on first ask and then hands back the same result
until the geometry, the cell, or a model selector changes, which is the same test
ASE's own `check_state` makes. One result is kept per accessor, keyed on its
arguments too, so a dispersion sweep does not accumulate a dynamical matrix per
wavevector.

The caches are separate. `get_frequencies` and `get_ir_spectrum` each build a
Hessian, so asking for both costs two; `get_dipole_derivatives` builds none.

`atoms=None` means "the structure the calculator is already bound to". Assigning
`atoms.calc = PM3()` does not bind it — ASE binds on the first evaluation — so
either go through the `Atoms` (`atoms.get_potential_energy()`, the normal ASE
pattern) or pass the structure explicitly (`atoms.calc.get_frequencies(atoms)`).
Calling an accessor with neither raises a `RuntimeError` saying so.

### Example

```python
from ase.build import molecule
from ase.optimize import BFGS
from pm3_rs.ase import PM3

atoms = molecule("H2O")
atoms.calc = PM3(method="pm3-d3h4")

BFGS(atoms).run(fmax=0.02)             # optimize (uses get_forces)
print(atoms.get_potential_energy())    # eV
print(atoms.calc.get_frequencies())    # cm^-1  (Hessian built lazily on request)

hess = atoms.calc.get_hessian()        # (9, 9) eV/Å²
```

> ASE's own `Vibrations` class computes the Hessian by finite differences of the
> forces. `get_hessian()` / `get_frequencies()` instead use pm3-rs's **analytic**
> Hessian directly, which is faster and free of finite-difference noise.


## Periodic boundary conditions

Four more native functions, and the ASE calculator switches automatically when
`atoms.pbc` is set. Cells are given in **Ångström**, like positions — one number
for a cube, three for an orthorhombic cell, or a 3×3 matrix of lattice vectors as
rows.

### `periodic_single_point(numbers, positions, cell, pbc=None, kpts=None, ...) -> dict`

Energies **per unit cell**: `energy_ev`, `electronic_ev`, `core_ev`,
`correction_ev`, `ewald_ev`, `heat_of_formation_kcal`, plus `charges`,
`converged` and `n_kpoints`. A Γ-point run also returns `gamma_margin_bohr`; a
k-point run returns `band_gap_ev` and `fermi_ev`.

> **Check `gamma_margin_bohr`.** One k-point substitutes `P(Γ)` for `P(0, T)` at
> every image, which is exact only when no image lies inside the exchange range.
> A cell one Bohr too narrow converges cleanly to an answer wrong by tens of eV.
> Positive is fine; negative means use `kpts` or a larger supercell. See
> [`pbc.md`](pbc.md).

### `periodic_forces(...) -> dict`

`forces_ev_per_angstrom` (shape `(N, 3)`) and `stress_ev_per_angstrom3` — the
6-component Voigt vector ASE expects. Every periodic dimensionality reports one,
with exact zeros in the non-periodic directions; only an isolated cell returns
`None`, having no strain at all.

### `phonons(numbers, positions, cell, ..., q=None, kpts=None) -> dict`

With `q` left out: Γ-point `frequencies_cm` from the analytic Hessian, and
`acoustic_residual_cm` — the largest of the three acoustic frequencies, which
should vanish and is reported so it can be checked rather than trusted.

`q` is a fractional wavevector, one component per reciprocal lattice vector, and
switches to density-functional perturbation theory: the phonon at that wavevector
from this cell, electronic response included, without building a supercell. Open
and closed shell, plain PM3 and every corrected variant, any periodic
dimensionality — a component along a non-periodic axis is refused. `kpts` gives
the response a Monkhorst–Pack mesh to sum over instead of Γ alone. There is no
acoustic sum rule away from Γ, so no residual is reported there.

**The polarization vectors come back with the frequencies.** A frequency says how
fast a mode vibrates; the eigenvector says what moves in it, which is what
separates an optical branch from an acoustic one and what a visualization draws.

| sampling | key | shape | meaning |
|---|---|---|---|
| Γ | `modes` | `3N × 3N`, real | `modes[m][3*a + i]` is the **mass-weighted** displacement of atom `a` along axis `i` in mode `m` |
| at `q` | `modes_real`, `modes_imag` | same, two halves | complex, because atoms in a cell move with a relative phase |

`masses` (amu, atom order) is what de-weights them: the Cartesian displacement is
`modes[m][3*a + i] / sqrt(masses[a])`. Both halves are returned at finite `q`
rather than a magnitude — collapsing them turns a travelling wave into a standing
one.

`cphf_max_iter` raises the coupled-perturbed iteration cap behind the response;
`None` keeps the default of 400.

```python
import numpy as np
from pm3_rs import native
chain = dict(numbers=[8, 1, 1],
             positions=[[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]],
             cell=[[6.0, 0, 0], [0, 20.0, 0], [0, 0, 20.0]],
             pbc=[True, False, False])
zone_boundary = native.phonons(**chain, q=[0.5, 0.0, 0.0])["frequencies_cm"]

# What actually moves in the highest Γ mode.
gamma = native.phonons(**chain)
top = int(np.argmax(gamma["frequencies_cm"]))
displacement = (np.asarray(gamma["modes"][top]).reshape(-1, 3)
                / np.sqrt(gamma["masses"])[:, None])
```

### `dipole(numbers, positions, ..., operator=False) -> dict`

The dipole operator and everything derived from it. `dipole_debye` and
`dipole_e_angstrom` (about the centre of mass, MOPAC's convention),
`centre_of_mass_angstrom`, and the `3 × 3N` derivative tensor `derivatives_e` in
elementary charges. `operator=True` adds `operator_bohr`: the three `nao × nao`
moment matrices themselves.

The tensor costs **three** coupled-perturbed solves rather than `3N`, by the
interchange theorem, so it is the same price whatever the molecule's size. No
Hessian is built — `ir_spectrum` is what pairs it with one.

### `dynamical_matrix(numbers, positions, cell, q, ..., rigid_ion=False) -> dict`

`D(q)` itself in eV/Bohr², as `real` and `imag` arrays of shape `(3N, 3N)`, for a
caller building a dispersion, applying their own masses, or checking an identity.
`hermitian_defect` is the largest departure from `D(q)† = D(q)` **before** the
matrix was symmetrized — reported rather than hidden, because a wrong assembly
shows there first. `rigid_ion=True` leaves the electronic response out.

`masses` (amu, in the order the matrix indexes them) and `frequencies_cm` come
with it. The masses are not a convenience: they are the crate's own
isotope-averaged values and nothing else on this surface hands them out, so
without them the matrix cannot be mass-weighted from Python at all. Returning
`D(q)` alone is what made the frequencies at a wavevector look like a Rust-only
feature. `frequencies_cm` is what `phonons(..., q=...)` reports, from the same
diagonalization.

### `bands(numbers, positions, cell, path, ..., kpts=None) -> dict`

Band energies (eV) along a path of fractional k-points, evaluated
non-self-consistently in the potential of a calculation converged on `kpts`.
Returns `bands_ev` (one ascending list per path point), `bands_beta_ev` (`None`
for a restricted run) and `fermi_ev`.

### `relax(numbers, positions, cell, ..., fixed_cell=False) -> dict`

Variable-cell relaxation: the atoms and the lattice vectors that exist.
`force_tol` is eV/Å and `stress_tol` eV/Å³. Returns the relaxed `positions` and
`cell` in Ångström, plus `energy_ev`, `energy_hartree`,
`heat_of_formation_kcal`, `converged` and `steps`, with `positions_angstrom` and
`iterations` as aliases. An isolated cell is refused, having no strain to relax
against. See [`optimize`](#optimize---dict) for why both sets of names are
carried.

`stress_tol` is a *density*, so converting it is the **cube** of the length
conversion. Through 0.2.1 it went to the optimizer unconverted while `force_tol`
beside it was converted, which made the applied tolerance 6.75× looser than the
documented one and returned `converged=True` for cells whose stress had never
reached the threshold.

### `divide_and_conquer(numbers, positions, cell=None, ...) -> dict`

Linear-scaling-style partitioned SCF, molecular or (with a `cell`) periodic.
`buffer_radius` is the knob that matters: widening it must converge the result
onto the full diagonalization. `dropped_pairs` and `largest_subsystem` report
what the partitioning traded.

`long_range_cutoff` is the **molecular** near-field split; the periodic path
takes its long range from the lattice sum and does not read it, so passing both
a cell and a cutoff is refused rather than quietly running the unscreened route.

### `divide_and_conquer_optimize(numbers, positions, ...) -> dict`

The same L-BFGS as `optimize`, driven by the partitioned gradient instead of a
full diagonalization — for the case the method exists for, where a full
diagonalization per line-search trial is not affordable. Arguments are
`divide_and_conquer`'s plus `max_steps` and `force_tol` (eV/Å).

Returns the same names as `optimize` and `relax` — `positions_angstrom` /
`positions`, `energy_ev` / `energy_hartree`, `heat_of_formation_kcal`,
`converged`, `iterations` / `steps` — plus `n_subsystems`.

Molecular only: the partitioned path has no cell gradient, so a periodic
structure would be relaxed at fixed cell without saying so. Use `relax` there.

The caveat on `divide_and_conquer_forces` gets worse here. A partitioned gradient
is not the exact derivative of the partitioned energy, so what this converges to
is the **buffer's** minimum, not the method's. Widen `buffer_radius` and re-run
before believing a structure.

### `born_charges(numbers, positions, cell, ..., enforce=False) -> dict`

`Z*[a][α][β] = ∂(Ω P_α)/∂u_{a,β}` — the dipole a cell acquires per unit
displacement of one atom — as one `3 × 3` tensor per atom in elementary charges.
Periodic only; a molecule's equivalent is `dipole(...)["derivatives_e"]`.

`sum_rule_residual` is the largest `|Σ_a Z*_a|`. Translating the whole crystal
produces no dipole, so it is zero for an exact response and is the number that
says whether the coupled-perturbed solve converged. Reported rather than
enforced; `enforce=True` removes the mean violation *after* you have read it.

Worth knowing: that sum rule cannot by itself distinguish a correct response from
an absent one, because the static charges of a neutral cell already sum to zero.
The crate's tests therefore also check `Z*` against a central difference of the
model's own cell dipole.

### `dielectric(numbers, positions, cell, ..., include_ionic=False) -> dict`

`polarizability` is `α` in Bohr³, available in every dimensionality. `epsilon` is
`ε∞ = 1 + 4πα/Ω` and is `None` for a chain or a slab, where `Ω` would be a length
or an area and dividing by a supercell's vacuum padding would make the answer a
statement about the padding. `gamma_margin_bohr` comes back with them, because a
negative margin makes this the response of a ground state one k-point cannot
represent and nothing else in the result would say so.

`epsilon` is the **electronic** (clamped-ion) response, pinned by agreeing with
a finite field on the isolated molecule to 0.2%.

`include_ionic=True` adds four keys: `epsilon_static` (`ε₀`, with the nuclei
free to relax along each infrared-active mode), its halves
`epsilon_electronic` and `epsilon_ionic`, and `skipped_modes`. It costs a
Γ-point phonon run and a set of Born charges on top of the field response.

`ε₀` is only meaningful at a **relaxed geometry**. The ionic term weights each
mode by `1/ω²`, so it diverges as a mode softens and is undefined for an
imaginary one. `skipped_modes` counts what was left out: three are the acoustic
branch, and more than three means the structure is not a minimum and `ε₀` is
missing whatever those modes carried. Check the count rather than trusting the
tensor.

`ε₀`, `ε∞`, `Z*` and the LO–TO term satisfy Lyddane–Sachs–Teller to a quotient
of `1.000000` over six optical modes. That is a consistency check between two
constructions rather than independent physics — both carry a `4π/Ω`, a Hartree
conversion and a mass weighting, in different arrangements — which is exactly
why a wrong factor in one shows up.

### `berry_polarization(numbers, positions, cell, strings=12, kpts=None, ...) -> dict`

Berry-phase polarization in `e/Bohr²`, the modern theory. Returns `total`,
`electronic`, `ionic`, `phase` (in turns), `quantum`, `string_length` and
`gamma_margin_bohr`.

`strings` is the k-points per Brillouin-zone string and is the convergence
parameter — the answer must become independent of it. 3D only, since the
quantum is `e a/V`.

**Only differences are meaningful.** The result is defined modulo `quantum`, one
per lattice vector; that is the physics, not a defect, since a different branch
assigns the electrons to a different unit cell. A plain subtraction of two
`total` values is off by exactly one quantum whenever the two landed on
different branches, which for a finite displacement is common. Reduce instead:

```python
d = np.asarray(after["total"]) - np.asarray(before["total"])
for q in np.asarray(before["quantum"]):
    d -= q * np.round(np.dot(d, q) / np.dot(q, q))
```

This exists as an independent check on `born_charges`: same `Z*`, reached
through overlaps between neighbouring k-points with no response equation in it.
The two differ by the intra-atomic `s`–`p` moment the phase cannot carry —
measured at `0.147 e` on HF, falling to `1.9e-4` when that term is removed from
the CPHF side.

### `finite_field(numbers, positions, cell, field, kpts, ...) -> dict`

A finite field **along** a periodic direction, by the Nunes–Gonze electric
enthalpy `F = E − Ω 𝓔·P`. `field` is in eV per `e·Bohr`.

Along a periodic direction `𝓔·R` is not lattice-periodic, the spectrum has no
lower bound, and the ground state of `H − 𝓔·R` does not exist. A field
orthogonal to *every* lattice vector needs none of this and goes through
`PM3(field=...)`.

`kpts` along each field direction is that direction's string length and is the
convergence parameter; at least 3. `resolved` says which axes the mesh could
see — an unresolved axis contributes zero to `electronic_polarization`, which is
not the same as its contribution being zero.

Restricted, gapped, 3D only. **There is no force**: the derivative of the
enthalpy with respect to the nuclei is not implemented.

It reproduces the CPHF polarizability to 1 part in 10⁴ on hydrogen, where both
routes carry the same position operator.

### `phonon_bands(numbers, positions, cell, supercell, path, points=12, ...) -> dict`

Phonon dispersion from supercell force constants: one Hessian buys the whole
path, where direct DFPT pays per wavevector. Returns `q`, `frequencies_cm`,
`supercell`, `acoustic_sum_rule_residual` and `gamma_margin_bohr`.

A supercell's Γ point *is* a mesh of the primitive cell, so this reproduces the
**mesh**-sampled response with the matching mesh, not a Γ-only one. Measured:
0.011 cm⁻¹ against the matching mesh at a cell where the Γ margin is positive.

`acoustic_sum_rule_residual` is reported before it can be imposed, because it is
what the truncation threw away. `enforce_asr=True` flattens it once you have
looked.

### LO–TO splitting: `phonons(..., q=[0,0,0], lo_to_direction=[1,0,0])`

Adds the non-analytic term
`(4π/Ω)(q̂·Z*_a)(q̂·Z*_b)/(q̂·ε∞·q̂)` along that Cartesian direction, which is what
makes the `q → 0` limit of a polar crystal direction-dependent. Without it the
longitudinal and transverse optical branches stay degenerate at Γ, which is wrong
by an amount that is not small. Three-dimensional cells only: a slab's
macroscopic field vanishes linearly in `q` and a chain's as `q² ln(1/q)`, so
neither has a splitting to add.

Do **not** add it to a finite-`q` `dynamical_matrix`. That one already carries
the long-range channel.

### `divide_and_conquer_forces(numbers, positions, cell=None, ...) -> dict`

The same arguments, returning `forces_ev_per_angstrom` and — with a cell —
`stress_ev_per_angstrom3` in Voigt order. Separate from `divide_and_conquer` for
the reason `periodic_forces` is separate from `periodic_single_point`: this
method exists for systems large enough that a gradient is worth asking for
rather than producing unasked.

A divide-and-conquer density is not variational — it is assembled, not minimized
— so the gradient carries the density's own truncation error rather than its
square. It converges with the buffer the same way the energy does, and a buffer
reaching the whole system reproduces the full gradient exactly.

### Example

```python
import numpy as np
import pm3_rs

numbers = [8, 1, 1]
positions = [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.24, 0.9278, 0.0]]

# A 10 Å cube, Gamma point only.
r = pm3_rs.periodic_single_point(numbers, positions, 10.0)
print(r["energy_ev"], r["gamma_margin_bohr"])   # per cell; margin must be > 0

# A 2x2x2 Monkhorst-Pack mesh on a narrower cell, where Gamma alone would not do.
r = pm3_rs.periodic_single_point(numbers, positions, 6.5, kpts=(2, 2, 2))
print(r["energy_ev"], r["band_gap_ev"])

# Forces and stress.
f = pm3_rs.periodic_forces(numbers, positions, 10.0)
print(np.asarray(f["forces_ev_per_angstrom"]).shape)   # (3, 3)
print(np.asarray(f["stress_ev_per_angstrom3"]).shape)  # (6,) Voigt

# Divide and conquer on something bigger.
big_numbers = numbers * 20
big_positions = [[x + 3.2 * i, y, z] for i in range(20) for x, y, z in positions]
d = pm3_rs.divide_and_conquer(big_numbers, big_positions, buffer_radius=5.0)
print(d["energy_ev"], d["n_subsystems"], d["dropped_pairs"])
```

## The ASE calculator under periodic boundary conditions

Set `atoms.pbc` and a cell; the calculator takes the periodic path on its own.
It reads `atoms.pbc` rather than a constructor flag, because an ASE user who sets
`pbc` expects it to be honoured.

```python
from ase import Atoms
from pm3_rs.ase import PM3

atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
atoms.set_cell([10.0, 10.0, 10.0])
atoms.set_pbc(True)
atoms.calc = PM3()                      # or PM3(kpts=(2, 2, 2))

print(atoms.get_potential_energy())     # eV per cell
print(atoms.get_stress())               # (6,) Voigt, eV/Å³
print(atoms.calc.get_phonons(atoms))    # Gamma-point frequencies, cm^-1
```

`stress` is in `implemented_properties`, so variable-cell relaxation through
`FrechetCellFilter` and barostatted molecular dynamics both work:

```python
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS

BFGS(FrechetCellFilter(atoms)).run(fmax=0.05)
```

A molecule (no `pbc`) raises on `get_stress()` rather than returning zeros:
zeros would be a claim, not an absence. Every *periodic* dimensionality now has
one — a chain reports its axis, a slab its two in-plane components — with exact
zeros in the non-periodic directions, so a masked barostat or a
`FrechetCellFilter` moves only the cell vectors that exist.