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
| `method`       | `"pm3"` (default), `"pm3-d3"`, `"pm3-d3h4"`, `"pm3-d3h4x"` | correction variant |

The `method` variant selects the post-SCF classical corrections — Grimme **D3**
dispersion, the Řezáč **H4** hydrogen-bond term, and the **X** halogen-bond term.
They contribute to the energy, the analytic gradient **and** the analytic Hessian.

---

## `pm3_rs.native`

All functions take `(numbers, positions, charge=0, multiplicity=1, reference="auto", method="pm3")`,
where `numbers` is a length-`N` sequence of atomic numbers and `positions` an
`(N, 3)` array in **Ångström**. Each returns a `dict`.

### `single_point(...) -> dict`
Keys: `energy_hartree`, `energy_ev`, `heat_of_formation_kcal`, `electronic_ev`,
`core_ev`, `charges` (Mulliken, e), `dipole_debye` (`[x, y, z]`), `homo_ev`,
`lumo_ev`, `converged`, `unrestricted`.

### `gradient(...) -> dict`
Analytic Hellmann–Feynman nuclear gradient `dE/dR`. Keys: `energy_hartree`,
`energy_ev`, `heat_of_formation_kcal`, `gradient_hartree_per_bohr` (`(N, 3)`),
`gradient_ev_per_angstrom` (`(N, 3)`).

### `forces(...) -> dict`
Forces `= −dE/dR`. Keys: `energy_hartree`, `energy_ev`, `heat_of_formation_kcal`,
`forces_hartree_per_bohr`, `forces_ev_per_angstrom`.

### `optimize(...) -> dict`
L-BFGS geometry optimization on the analytic gradient. Keys:
`positions_angstrom` (`(N, 3)`), `energy_hartree`, `heat_of_formation_kcal`,
`converged`, `iterations`.

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
PM3(charge=0, multiplicity=1, reference="auto", method="pm3", **kwargs)
```

`implemented_properties` = `["energy", "forces", "charges", "dipole",
"heat_of_formation_kcal", "hessian"]`. Energy, forces, charges and dipole are
computed eagerly and served through ASE's standard accessors
(`get_potential_energy`, `get_forces`, `get_charges`, `get_dipole_moment`, and
`get_property("heat_of_formation_kcal")`).

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
