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
```

Flags: `--charge <q>`, `--multiplicity <m>`, `--method <pm3|pm3-d3|pm3-d3h4|pm3-d3h4x>`,
`--no-diis`. Coordinates in the XYZ input/output are Ångström.
