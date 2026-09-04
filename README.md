# pm3-rs

`pm3-rs` is a Rust implementation of the PM3 semiempirical NDDO method
(Stewart 1989), with the PM3 Hamiltonian, the MOPAC v23.2.5 PM3 parameter
tables, and the PM3 core-core repulsion — for molecules and for periodic solids.

The project provides:

- RHF and UHF single points, heats of formation, Mulliken charges, and dipoles
- analytic gradients and CPHF/UCPHF Hessians
- L-BFGS geometry optimization and harmonic frequencies
- PM3-D3, PM3-D3H4, and PM3-D3H4X post-SCF variants
- MOPAC special atoms: `Cb` (capped bond), `+`, `-`, and La-Lu Sparkles
- a **uniform external electric field** (molecular): energy, analytic gradient
  and analytic Hessian, RHF and UHF
- **infrared intensities**, and **Molden** output of the orbitals
- **periodic boundary conditions** in 1D, 2D and 3D — Ewald electrostatics,
  Γ-point and k-point SCF, analytic forces, analytic stress in every periodic
  dimensionality, Γ-point phonons, charged cells, and lattice-summed
  corrections ([`docs/pbc.md`](docs/pbc.md))
- the **dynamical matrix at arbitrary `q`**, from the primitive cell, with the
  electronic response — over a k-mesh or at Γ alone;
  `rigid_ion_dynamical_matrix` gives the fixed-density part by itself
  ([`docs/pbc.md`](docs/pbc.md))
- **Born effective charges**, the **electronic and static dielectric tensors**
  (`ε∞` and `ε₀`), **LO–TO splitting**, and **supercell phonon dispersion** —
  all contractions of that same response ([`docs/pbc.md`](docs/pbc.md))
- **Berry-phase polarization**, present as an independent check on the charges
  above rather than as a feature in itself, and a **finite electric field along
  a periodic direction** built on it (the Nunes–Gonze electric enthalpy), where
  the ordinary `−𝓔·r` coupling has no ground state to find
- the **Mermin electronic free energy** `E − TS` under Fermi smearing, which is
  what ASE's `free_energy` and `force_consistent=True` mean — and which is *not*
  a Gibbs free energy: no zero-point energy, no vibrational partition function,
  no `pV`, no nuclear entropy
- **divide-and-conquer** partitioned SCF, molecular and periodic, with an
  optional linear-scaling near field
  ([`docs/divide-and-conquer.md`](docs/divide-and-conquer.md))
- Rust library plus the `pm3-rs-python` native and ASE Python package, and one
  `pm3-rs` command that `pip install` and `cargo install` both put on your path

Linear algebra uses `faer`; no external BLAS/LAPACK installation is required.


## Supported PM3 elements

The embedded MOPAC v23.2.5 PM3 set covers H-Ca, Zn-Sr, Cd-Ba, and Hg-Bi.
PM3 uses an s/p valence basis for these elements; elements without PM3
parameters return an explicit missing-parameter error. La-Lu are represented by
MOPAC's zero-orbital trivalent Sparkle model. Atomic-number codes 102, 104, and
106 provide `Cb`, `+1`, and `-1`, respectively.

## Correction variants

| Method | Post-SCF correction |
|---|---|
| `PM3` | none |
| `PM3-D3` | PM3 zero-damping D3 |
| `PM3-D3H4` | refitted D3 + PM3 H4 + H-H repulsion |
| `PM3-D3H4X` | PM3-D3H4 + X halogen-bond term |

Correction energies, gradients, and Hessians use the same scalar-generic
implementation. The H4 path includes continuous water, ammonium, and
carboxylate scaling. PDB-residue-name-only HIP/GUA overrides from the Cuby
interface are not applied because the Rust/Python/XYZ APIs do not carry residue
metadata.

## Validation

Base PM3 calculations are regression-tested against MOPAC v23.2.5. In addition
to frozen molecular regressions, the oracle sweep covers every supported PM3
element, every La-Lu Sparkle, and the `Cb`, `+`, and `-` atom codes (60 cases).
For each case it compares the heat of formation, every Cartesian gradient
component, and the complete Cartesian Hessian matrix. The PM3-D3 constants are
checked against the public MOPAC 5.022mn implementation; PM3-D3H4 constants
follow the published D3H4 parameterization. See
`tools/oracle/PM3_VALIDATION.md`.

Separately, every code block and stated guarantee in `README.md`,
`docs/rust-api.md`, and `docs/python-api.md` is executed as a test —
`tests/api_surface.rs` for the Rust API and `tests/test_python_api.py` for the
Python and ASE APIs — so a documented example that stops working fails the
build.

## Build and test

```sh
cargo build --release
cargo test --all-targets --all-features
```

Python and ASE (needs a virtualenv with `numpy`, `ase`, `pytest`):

```sh
maturin develop --release --features python
python -m pytest tests/test_python.py tests/test_python_api.py
```

A `pip install` puts a `pm3-rs` command on your path; a `cargo build` produces
the same interface as a binary named `pm3_rs_cli`. They are the same compiled
implementation — the Python entry point hands `sys.argv` straight to it — so
neither can drift from the other.

```sh
pm3-rs energy water.xyz
pm3-rs gradient water.xyz
pm3-rs optimize water.xyz
pm3-rs frequencies water.pm3opt.xyz
pm3-rs charges water.xyz --charge 0 --multiplicity 1
pm3-rs energy dimer.xyz --method PM3-D3H4X
```

## Rust API

```rust
use pm3_rs::{run_pm3, Molecule, Pm3Options, Pm3Parameters};

let molecule = Molecule::from_xyz_file("water.xyz", 0.0)?;
let parameters = Pm3Parameters::standard()?;
let result = run_pm3(&molecule, &parameters, &Pm3Options::default())?;
println!("heat of formation = {} kcal/mol", result.heat_of_formation_kcal);
```

## Python and ASE

The Python distribution is `pm3-rs-python`; its import package is `pm3_rs`.

```sh
pip install pm3-rs-python
pip install "pm3-rs-python[ase]"
```

```python
import numpy as np
import pm3_rs

numbers = [8, 1, 1]
positions = np.array([
    [0.0, 0.0, 0.0],
    [0.9584, 0.0, 0.0],
    [-0.24, 0.9278, 0.0],
])

result = pm3_rs.single_point(numbers, positions, method="pm3")
gradient = pm3_rs.gradient(numbers, positions, method="pm3-d3h4")
```

```python
from ase.build import molecule
from pm3_rs.ase import PM3

atoms = molecule("H2O")
atoms.calc = PM3(method="pm3-d3h4")
print(atoms.get_potential_energy())
print(atoms.get_forces())
```

## Periodic boundary conditions

```bash
pm3-rs energy  crystal.xyz --cell 10.0                 # Gamma point
pm3-rs energy  crystal.xyz --cell 6.5 --kpts 4,4,4     # 4x4x4 mesh
pm3-rs stress  crystal.xyz --cell 10.0
pm3-rs phonons crystal.xyz --cell 10.0                 # Gamma point
pm3-rs phonons crystal.xyz --cell 10.0 --q 0.25,0,0    # DFPT at a wavevector
pm3-rs phonon-bands crystal.xyz --cell 6.0 --supercell 2,2,2   # dispersion
pm3-rs born   crystal.xyz --cell 6.0                   # Born effective charges
pm3-rs dielectric crystal.xyz --cell 6.0 --static      # eps_inf and eps_0
pm3-rs berry  crystal.xyz --cell 6.0                   # Berry-phase polarization
pm3-rs finite-field crystal.xyz --cell 6.0 --field 0.001,0,0 --kpts 6,1,1
pm3-rs energy  big.xyz --dc 5.0                        # divide and conquer
```

```python
from ase import Atoms
from pm3_rs.ase import PM3

atoms.set_cell([10.0, 10.0, 10.0])
atoms.set_pbc(True)
atoms.calc = PM3()                    # or PM3(kpts=(4, 4, 4))
atoms.get_potential_energy()          # eV per cell
atoms.get_stress()                    # (6,) Voigt, eV/A^3
atoms.calc.get_born_charges(atoms)    # Z*, with its sum-rule residual
atoms.calc.get_dielectric(atoms, include_ionic=True)   # eps_inf and eps_0
```

> **One caveat worth reading before trusting a Γ-point number.** A single
> k-point substitutes `P(Γ)` for `P(0, T)` at *every* image, which is exact only
> when no image lies inside the exchange range. A cell one Bohr too narrow
> converges cleanly, in the usual number of iterations, to an answer wrong by
> tens of eV — nothing in the SCF reacts. Every periodic result reports a
> `gamma_margin`; it must be positive, or you need `--kpts` or a larger
> supercell. See [`docs/pbc.md`](docs/pbc.md).

## Units

- internal model: eV and Bohr
- Rust/Python native results: atomic-unit fields plus eV and kcal/mol conveniences
- ASE: eV, Angstrom, eV/Angstrom, and eV/Angstrom^2

## References

- J. J. P. Stewart, *J. Comput. Chem.* **10**, 209-220 (1989),
  DOI 10.1002/jcc.540100208.
- J. J. P. Stewart, *J. Comput. Chem.* **10**, 221-264 (1989),
  DOI 10.1002/jcc.540100209.
- S. Grimme et al., *J. Chem. Phys.* **132**, 154104 (2010).
- J. Rezac and P. Hobza, *J. Chem. Theory Comput.* **8**, 141-151 (2012),
  DOI 10.1021/ct200751e.

License: GPL-3.0-or-later. Parameter and algorithm provenance is recorded in
`THIRD_PARTY_NOTICES.md`.
