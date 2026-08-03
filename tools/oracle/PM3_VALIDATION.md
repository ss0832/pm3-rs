# PM3 validation record

## Oracle

- executable: MOPAC v23.2.5 Windows build
- method: `PM3 PRECISE AUX(PRECISION=9)`
- harness: `tools/oracle/run_mopac.py`
- frozen Rust comparisons: `tests/molecules.rs`

The MOPAC executable is not redistributed. Place it under the gitignored
`tools/oracle/mopac/` directory or set `MOPAC_EXE` when importing the harness.

## Single-point heats of formation

All values are kcal/mol at the frozen XYZ geometries.

| System | MOPAC v23.2.5 | pm3-rs regression tolerance |
|---|---:|---:|
| H2O | -53.2301514568744 | 0.005 |
| CH4 | -13.0256696855340 | 0.005 |
| NH3 | -2.73858108540571 | 0.005 |
| H2CO | -33.9081151338050 | 0.005 |
| H2S | -0.235687164627052 | 0.005 |
| HCl | -20.4516034561493 | 0.005 |
| CH3, UHF doublet | 28.0055283157567 | 0.005 |
| GdF3, Gd(III) Sparkle | 61.2557665835629 | 0.005 |
| H2O + MOPAC `+` atom | -47.2642956996679 | 0.005 |
| H2O + MOPAC `-` atom | -41.3884183096980 | 0.005 |

The point atoms retain charges exactly +1 and -1. In GdF3, the zero-orbital Gd
Sparkle carries +3 and each fluorine carries -1 to numerical precision.

## Derivatives, optimization, and frequencies

- Distorted-water analytic gradient is checked against MOPAC and against a
  full-SCF finite-difference gradient.
- Optimized H2O heat of formation: -53.4330121104622 kcal/mol.
- MOPAC FORCE frequencies at the exact pm3-rs optimized geometry:
  1743.46, 3868.68, and 3989.81 cm^-1. The regression tolerance is 2 cm^-1.
- Analytic correction Hessians are independently checked against finite
  differences of the analytic correction gradient.

## Exhaustive supported-atom audit

`all_element_validation.py` exercises all 60 supported atom-code cases:

- 42 ordinary PM3 elements: H-Ca, Zn-Sr, Cd-Ba, and Hg-Bi;
- 15 La-Lu trivalent Sparkles;
- MOPAC special atoms `Cb`, `+`, and `-`.

For every case, the same input geometry is used by both programs and the audit
compares the heat of formation, all `3N` Cartesian gradient components, and all
`(3N)^2` Cartesian Hessian elements. The reference Hessian in
`ALL_ELEMENTS_RESULTS.json` is a central difference of MOPAC analytic gradients
with a 0.001 Angstrom displacement and `NOREOR`.

The 60/60 completed sweep has these maximum absolute differences:

| Quantity | Maximum absolute difference |
|---|---:|
| Heat of formation | 0.0839253 kcal/mol |
| Cartesian gradient | 1.60353e-4 eV/Bohr |
| Cartesian Hessian | 2.23658e-2 eV/Bohr^2 |

The energy maximum is the `Cb` fixture, whose heat of formation is about
-1.171e8 kcal/mol because MOPAC deliberately uses a -9,999,999 eV resonance
sentinel; its relative energy error is below 1e-9. The Hessian maximum comes
from differentiating finite-precision MOPAC gradient output for heavy atoms.
Using a 0.005 Angstrom difference for As/Br/I reduces their maximum to
4.925e-3 eV/Bohr^2. Direct MOPAC FORCE comparisons are retained in
`ALL_ELEMENTS_FORCE_RESULTS.json`; on non-stationary geometries they include
MOPAC's rotational/translational treatment and are therefore not the primary
Cartesian derivative oracle.

## PM3-D3 family

MOPAC v23.2.5 does not accept PM3-D3/D3H4 keywords. Validation is split into
source verification and derivative consistency:

- PM3-D3 constants are frozen in a unit test against public MOPAC 5.022mn
  `anad3.f`: `s6=1`, `s8=0.612`, `rs6=1.345`, `rs8=1`, `alpha6=14`,
  `alpha8=16`.
- PM3-D3H4 uses `s6=1`, `s8=0`, `rs6=0.90`, `alpha6=22`, `alpha8=24`,
  the PM3-specific H4 donor/acceptor coefficients, continuous water/NH4+/COO-
  multipliers, and the PM3 H-H repulsion parameters.
- PM3-D3H4X adds the published X halogen-bond correction.
- Unit tests cover the coefficient sets and energy signs; the analytic Hessian
  test covers D3 coordination-number coupling and H4/H-H terms.

## Documented-API conformance

Independently of the numerical oracle, every code block and stated guarantee in
`README.md`, `docs/rust-api.md`, and `docs/python-api.md` is executed as a test:

- `tests/api_surface.rs` — the Rust library surface (constructors, option
  fields, result fields, units, error variants, the L-BFGS/Hessian/gradient
  contracts) and the README quick-start.
- `tests/test_python_api.py` — `pm3_rs.native` dict keys and unit conversions,
  the `pm3_rs.ase.PM3` calculator (ASE units, lazy Hessian, accessors), and both
  documented example blocks.
- The CLI is exercised over every documented subcommand, flag, and error path.

## Reproduction

```powershell
$env:CARGO_BUILD_JOBS='1'
cargo test --all-targets --all-features
python tools/oracle/run_mopac.py examples/water.xyz --method PM3
```

```powershell
maturin develop --release --features python
python -m pytest tests/test_python.py tests/test_python_api.py
```
