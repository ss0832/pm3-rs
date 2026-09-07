<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# pm3-rs — Rust API

The crate exposes the full PM3 model as a library (`pm3_rs`). Add it to a project:

```toml
[dependencies]
pm3-rs = "0.1"
```

## Units

Positions are stored internally in **Bohr** (`Molecule::from_xyz_*` reads Ångström
and converts). Energies are in **eV**; the nuclear gradient is **eV/Bohr**, the
Hessian **eV/Bohr²**, and the heat of formation is reported in **kcal/mol**.
Vibrational frequencies are **cm⁻¹**.

## Building a molecule

```rust
use pm3_rs::system::Molecule;

// From an XYZ string (Ångström); the second argument is the total charge.
let mol = Molecule::from_xyz_str("3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n", 0.0)?;

// …or from a file, or built directly:
let mol = Molecule::from_xyz_file("water.xyz", 0.0)?;
let mol = Molecule::new(atoms).with_charge(1.0).with_multiplicity(2);
```

`Molecule { atoms: Vec<Atom>, charge: f64, multiplicity: usize }`, where
`Atom { z: u8, position: Vec3 /* Bohr */ }`.

`run_pm3` uses charge and multiplicity from `Molecule` when the corresponding
option remains at its default. Setting either value explicitly in
`Pm3Options` is also supported; conflicting non-default molecule/option values
return `Pm3Error::InvalidInput` rather than silently choosing one.

## Parameters and options

```rust
use pm3_rs::params::Pm3Parameters;
use pm3_rs::scf::{Pm3Options, Reference};
use pm3_rs::corrections::Variant;

let params = Pm3Parameters::standard()?;   // embedded MOPAC v23.2.5 parameter set

let options = Pm3Options {
    charge: 0.0,
    multiplicity: 1,
    reference: Reference::Auto,             // Auto | Rhf | Uhf
    variant: Variant::Pm3D3H4,              // Pm3 | Pm3D3 | Pm3D3H4 | Pm3D3H4X
    ..Pm3Options::default()
};
```

`Variant::parse("pm3-d3h4")` maps method strings to the enum. Other notable
`Pm3Options` fields: `max_scf`, `e_tol`, `p_tol`, `level_shift_ev`, `damping`,
and `hessian_cutoff`.

### Which SCF solution you get

The SCF equations have more than one solution, and iteration finds whichever the
starting density and the accelerator lead to — a converged excited solution obeys
the aufbau principle among its own eigenvalues and reports itself converged. Two
options govern this:

```rust
use pm3_rs::scf::{ScfGuess, ScfStability};

let options = Pm3Options {
    guess: ScfGuess::Auto,            // Auto (= SAD) | Sad | Core | SymmetryBroken
    stability: ScfStability::Auto,    // Auto | Off | Always
    ..Pm3Options::default()
};
```

`ScfStability::Auto`, the default, re-solves from other starting points when the
converged state's frontier gap is under 6 eV and keeps the lowest solution. It
fires on about 3% of ordinary molecules and is what makes the whole 189-molecule
MOPAC oracle agree; without it, seven of those molecules come back up to 279
kcal/mol high. `Pm3Result::scf_paths_tried` and `Pm3Result::scf_improvement_ev`
report whether it ran and whether it changed the answer.

### `mmok`

`Pm3Options::mmok` (default `false`) adds MOPAC's `MMOK` amide correction. See
[`python-api.md`](python-api.md) for what it is and why the default differs from
MOPAC's; from Rust it is a plain boolean rather than a `+mmok` method suffix.

Three soft memory budgets bound the allocations that grow with system size.
Each reports a `Pm3Error::ResourceLimit` (or degrades) instead of letting the
allocator abort the process; set one to `0` only when an external memory limit
is already enforced.

| Field | Default | Bounds |
|---|---|---|
| `integral_memory_mb` | 4096 MiB (env `PM3_MAX_PAIR_CACHE_MB`) | the resident `O(N^2)` two-electron pair cache, sized in closed form before allocation |
| `scf_memory_mb` | 512 MiB | the SCF accelerator history; the DIIS depth is trimmed and A-DIIS falls back to CDIIS to fit |
| `hessian_memory_mb` | 1024 MiB | Hessian concurrency and the CPHF occupied-virtual workspace |

Trimming the accelerator history changes only the SCF path, not its fixed
point: the converged energy, density, and every derived quantity are unchanged.

## Single point

```rust
use pm3_rs::scf::run_pm3;

let r = run_pm3(&mol, &params, &options)?;   // -> Pm3Result
println!("ΔHf = {} kcal/mol", r.heat_of_formation_kcal);
println!("E   = {} eV", r.total_ev);
println!("charges = {:?}", r.charges);       // Mulliken (e)
```

`Pm3Result` carries `total_ev`, `electronic_ev`, `core_ev`,
`heat_of_formation_kcal`, `charges`, `dipole_debye`, `mo_energies`, `mo_coeff`,
`density`, `homo_ev`, `lumo_ev`, `n_occ`, `iterations`, `converged`,
`unrestricted`.

A convenience wrapper bundles parameters + options:

```rust
use pm3_rs::scf::Pm3Calculator;
let calc = Pm3Calculator::with_options(params.clone(), options.clone());
let r = calc.calculate(&mol)?;
```

## Analytic gradient

```rust
use pm3_rs::gradient::closed_form_gradient;

let g = closed_form_gradient(&mol, &params, &options)?;   // -> GradientResult
for f in &g.forces  { /* eV/Bohr, = −gradient */ }
for d in &g.gradient { /* eV/Bohr */ }
println!("|g|_max = {}", g.max_gradient);
```

The gradient is fully analytic (Hellmann–Feynman + closed-form integral
derivatives), including the analytic D3/H4/X correction gradient for the
correction variants. Open-shell systems use the spin-resolved path automatically.

## Analytic Hessian and frequencies

```rust
use pm3_rs::hessian::{analytic_hessian, vibrational_analysis};

// 3N × 3N Cartesian Hessian (eV/Bohr²); `step` is the displacement used by the
// numerical fallback paths (capped bonds, open-shell d systems).
let h = analytic_hessian(&mol, &params, &options, 1.0e-3)?;

// Optimise first, then analyse the harmonic modes at the stationary point.
let modes = vibrational_analysis(&opt_mol, &params, &options, 1.0e-3)?;
for f in &modes.frequencies_cm { /* cm^-1, ascending; negative = imaginary */ }
```

The Hessian combines the closed-form CPHF (coupled-perturbed SCF) response with
the exact analytic D3/H4/X second derivatives for the correction variants. A
`numerical_hessian` (central differences of the analytic gradient) is available
as an independent reference. `VibrationalModes { hessian, frequencies_cm,
eigenvalues }`.

Translations and rotations are removed by **projection**, not by being small: the
rigid-body subspace is discovered from the geometry — two rotations for a linear
molecule, none for a single atom — and those modes come back as exactly `0.0`.
`n_rigid` says how many were found and `rigid_residual_cm` is the largest `|ω|`
they carried **before** the projection, which is the diagnostic for the Hessian's
quality rather than a number the projection has already flattened.

`Pm3Options::cphf_max_iter` (default 400) caps the coupled-perturbed solve. It
was a hard-coded constant, so a stiff response could only be rescued by editing
the crate; raise it first when a Hessian fails with a large response residual and
the SCF under it converged cleanly.

## Geometry optimization (L-BFGS)

```rust
use pm3_rs::optimizer::{optimize, OptOptions};

let res = optimize(&mol, &params, &options, &OptOptions::default())?;  // -> OptResult
let opt_mol = res.molecule;                 // optimised geometry (Bohr)
println!("converged in {} steps", res.iterations);
println!("ΔHf = {} kcal/mol", res.scf.heat_of_formation_kcal);
```

`OptOptions { max_iter, gtol /* eV/Bohr */, grad_step, history }`.
`OptResult { molecule, scf, converged, iterations, trajectory }`.

## Periodic boundary conditions

A `Molecule` becomes periodic when it carries a `Cell`; the molecular entry
points are unaffected and stay bit-identical when `cell` is `None`.

```rust
use pm3_rs::{Cell, Molecule};
use pm3_rs::pbc::gamma::{run_gamma, PeriodicOptions};
use pm3_rs::pbc::gradient::periodic_gradient;
use pm3_rs::pbc::optimize::{relax, CellRelaxation, PeriodicOptOptions};

let mut mol = Molecule::from_xyz_file("water.xyz", 0.0)?;
mol.cell = Some(Cell::cubic(20.0)?);              // Bohr; periodic in x, y, z

let scf = run_gamma(&mol, &params, &options, &PeriodicOptions::default())?;
println!("{} eV per cell", scf.total_ev);

let g = periodic_gradient(&mol, &params, &options, &PeriodicOptions::default())?;
let stress = g.stress;                            // Option<Mat3>, eV/Bohr³

let relaxed = relax(&mol, &params, &options, &PeriodicOptions::default(),
                    &PeriodicOptOptions { cell: CellRelaxation::Variable,
                                          ..PeriodicOptOptions::default() })?;
```

`Cell::new(a, b, c, [bool; 3])` builds 1D, 2D, and 3D cells; the non-periodic
directions never enter the measure, the reciprocal basis, or the stress, so a
slab's vacuum thickness cannot affect a result. `PeriodicResult` mirrors
`Pm3Result` and adds `correction_ev`, `ewald_ev`, `spin_density`, and
`gamma_margin`. `PeriodicGradient` carries `forces`, `gradient`, `virial`, and
`stress`; the last two are `None` only for an isolated cell, which has no strain.
A chain reports its axis, a slab its two in-plane components, and both put exact
zeros in the non-periodic directions.

**`gamma_margin` decides whether the result means anything.** It is the
narrowest periodic width minus `PeriodicOptions::short_range_cutoff`, and it
must be positive: one k-point substitutes `P(Γ)` for `P(0, T)` at every image,
which is exact only when no image lies inside the exchange range. A cell one
Bohr too narrow converges cleanly to an answer wrong by tens of eV. See
[`pbc.md`](pbc.md) for the derivation, the measured numbers, and the treatment
of charged cells, corrections, and reduced dimensionality.

## Phonons at a wavevector

`pbc::dfpt` gives the dynamical matrix and the phonon frequencies at any `q`
from the primitive cell, without a supercell. Every item is re-exported at the
crate root.

```rust
use pm3_rs::{dynamical_matrix, dynamical_matrix_on_mesh, phonon_frequencies,
             rigid_ion_dynamical_matrix, KpointOptions};
use pm3_rs::pbc::gamma::PeriodicOptions;

let periodic = PeriodicOptions::default();
let q = [0.25, 0.0, 0.0];                              // fractional

let d = dynamical_matrix(&mol, &params, &options, &periodic, q)?;
assert!(d.hermitian_defect < 1e-6);                    // reported, not assumed
let freqs = phonon_frequencies(&mol, &params, &options, &periodic, q)?;  // cm⁻¹

// The response over a mesh rather than Γ alone.
let mesh = KpointOptions::mesh([4, 4, 4]);
let dense = dynamical_matrix_on_mesh(&mol, &params, &options, &periodic, &mesh, q)?;

// The fixed-density part by itself, for separating the rigid-ion and response halves.
let rigid = rigid_ion_dynamical_matrix(&mol, &params, &options, &periodic, q)?;
```

Open and closed shell, plain PM3 and every corrected variant, any periodic
dimensionality — a `q` component along a non-periodic axis is refused rather
than summed. `DynamicalMatrix::hermitian_defect` is the largest departure from
`D(q)† = D(q)` before the matrix is symmetrized; it is a property of the
assembly, and reporting it is what makes a wrong one visible.

## Response properties

```rust
use pm3_rs::{
    born_charges, born_charge_sum_rule_residual, enforce_born_sum_rule,
    polarizability, dielectric_tensor, static_dielectric_tensor,
    add_non_analytic, non_analytic_term,
    ForceConstants, q_path, berry_polarization,
};

let z = born_charges(&molecule, &params, &options, &periodic)?;
let residual = born_charge_sum_rule_residual(&z);   // report, then decide

let eps = dielectric_tensor(&molecule, &params, &options, &periodic)?.epsilon;
let full = static_dielectric_tensor(&molecule, &params, &options, &periodic)?;
// full.epsilon == full.electronic + full.ionic; full.skipped_modes says how
// much of the ionic sum is missing, and three is the acoustic branch.

let mut d = pm3_rs::dynamical_matrix(&molecule, &params, &options, &periodic, [0.0; 3])?;
add_non_analytic(&mut d, &z, eps, [1.0, 0.0, 0.0], &molecule)?;   // LO-TO

let fc = ForceConstants::from_supercell(&molecule, &params, &options, &periodic, [2, 2, 2])?;
let bands = fc.band_structure(&q_path(&corners, 12))?;            // one Hessian
```

All Γ-point paths, and all of them only as good as the ground state under them:
a `gamma_margin` at or below zero means one k-point cannot represent the cell,
and the response inherits that silently.

`berry_polarization` reaches `Z*` by a different formalism entirely and exists
to check the above. Take differences through `BerryPolarization::difference` —
polarization is defined modulo `e a/Ω`, and a plain subtraction is off by
exactly one quantum whenever the two branches differ.

```rust
use pm3_rs::{run_finite_field, FiniteFieldOptions};

let result = run_finite_field(
    &molecule, &params, &options, &periodic,
    [6, 1, 1],                      // the mesh; its division along a field
    Vec3::new(1.0e-3, 0.0, 0.0),    // direction is that string's length
    &FiniteFieldOptions::default(),
)?;
result.enthalpy_ev;   // E − Ω 𝓔·P, the quantity actually minimized
result.resolved;      // which axes the mesh could see a phase along
```

A field **along** a periodic direction cannot be `−𝓔·r`: that is not
lattice-periodic, and the ground state of `H − 𝓔·R` does not exist. This
minimizes the Nunes–Gonze electric enthalpy instead. Restricted, gapped, 3D
only, and there is **no force** — the nuclear derivative of the enthalpy is not
implemented. A field orthogonal to every lattice vector needs none of this and
stays on `Pm3Options::field`.

## Dipole, infrared and Molden

```rust
use pm3_rs::{centre_of_mass, dipole_matrix, dipole_derivatives, ir_spectrum,
             molden_string};

let com = centre_of_mass(&mol, &params)?;
let m = dipole_matrix(&mol, &params, &basis, com)?;     // [Matrix; 3], the operator
let tensor = dipole_derivatives(&mol, &params, &options)?;   // 3 × 3N, in e
let spectrum = ir_spectrum(&mol, &params, &options, 1e-3)?;  // cm⁻¹ and km/mol
std::fs::write("water.molden", molden_string(&mol, &params, &scf)?)?;
```

`dipole_derivatives` costs **three** coupled-perturbed solves, not `3N`: the
interchange theorem trades the nuclear perturbations for the field ones. The
same operator backs the reported dipole and the external-field coupling, which
is what makes `μ = −∂E/∂F` hold by construction rather than by coincidence.

An external field is `Pm3Options::field`, in eV/Bohr with the sign that makes
`E = E₀ + μ·F`, and reaches the energy, the analytic gradient and the analytic
Hessian for both references. It is molecular only.

## Errors

All fallible entry points return `pm3_rs::error::Result<T>` (alias for
`Result<T, pm3_rs::error::Pm3Error>`), covering XYZ parse errors, missing
parameters, and non-convergence.

## Command-line interface

The crate also ships a CLI (`pm3_rs_cli`):

```bash
pm3_rs_cli energy      water.xyz --method pm3-d3h4 --charge 0 --multiplicity 1
pm3_rs_cli gradient    water.xyz
pm3_rs_cli charges     water.xyz          # Mulliken charges + dipole
pm3_rs_cli optimize    water.xyz          # writes water.pm3opt.xyz
pm3_rs_cli hessian     water.xyz          # Cartesian Hessian
pm3_rs_cli frequencies water.xyz --method pm3-d3h4
pm3_rs_cli phonons     crystal.xyz --cell 10.0                  # Gamma point
pm3_rs_cli phonons     crystal.xyz --cell 10.0 --q 0.25,0,0     # DFPT at a wavevector
pm3_rs_cli phonons     crystal.xyz --cell 10.0 --q 0.25,0,0 --rigid-ion
pm3_rs_cli orbitals    water.xyz --coefficients   # MO energies, occupations, HOMO/LUMO
pm3_rs_cli molden      water.xyz --output out.molden
pm3_rs_cli optimize    crystal.xyz --cell 6.0 --pbc xz --relax-cell
pm3_rs_cli optimize    big.xyz --dc 4.8           # partitioned geometry optimization
```

`--q` is in fractions of the reciprocal lattice vectors and needs a `--cell`;
with `--kpts` the response is summed over that mesh, each point paired with
`k + q`. `--rigid-ion` reports the fixed-density half of `D(q)` alone, which is
what a supercell finite difference can be compared against directly.

`--pbc` accepts every spelling of the same thing: `1,0,1`, `101`, `xz`, `x,z`,
`true,false,true`, and `none` for an isolated cell.

`optimize` on a cell takes `--relax-cell` / `--fixed-cell` (the default, stated),
`--max-steps`, `--force-tol` (eV/Å), `--stress-tol` (eV/Å³) and `--pressure`.
Which one ran is printed, because "optimized" without saying what moved is read
once and misremembered. The saved geometry carries its lattice as extended-XYZ
`Lattice="..."`, so a variable-cell result is not silently thrown away.

`molden` writes `[GTO]` by default — an even-tempered Gaussian expansion of the
Slater orbitals, which is what viewers read — and `--sto` writes the exponents PM3
actually uses instead. `--output` says where the file goes.

`--dc <radius>` runs the partitioned SCF, and works with `energy`, `charges` and
`optimize`. Every other command refuses it by name rather than quietly returning a
single point.

Flags: `--charge <q>`, `--multiplicity <m>`, `--method <pm3|pm3-d3|pm3-d3h4|pm3-d3h4x>`,
`--no-diis`. Coordinates in the XYZ input/output are Ångström.
