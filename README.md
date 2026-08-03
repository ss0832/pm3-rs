# pm3-rs

`pm3-rs` is a Rust implementation of the molecular PM3 semiempirical NDDO
method (Stewart 1989), with the PM3 Hamiltonian, the MOPAC v23.2.5 PM3
parameter tables, and the PM3 core-core repulsion.

The project provides:

- RHF and UHF single points, heats of formation, Mulliken charges, and dipoles
- analytic gradients and CPHF/UCPHF Hessians
- L-BFGS geometry optimization and harmonic frequencies
- PM3-D3, PM3-D3H4, and PM3-D3H4X post-SCF variants
- MOPAC special atoms: `Cb` (capped bond), `+`, `-`, and La-Lu Sparkles
- Rust library/CLI plus the `pm3-rs-python` native and ASE Python package

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

The binary is named `pm3_rs_cli`:

```sh
pm3_rs_cli energy water.xyz
pm3_rs_cli gradient water.xyz
pm3_rs_cli optimize water.xyz
pm3_rs_cli frequencies water.pm3opt.xyz
pm3_rs_cli charges water.xyz --charge 0 --multiplicity 1
pm3_rs_cli energy dimer.xyz --method PM3-D3H4X
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
