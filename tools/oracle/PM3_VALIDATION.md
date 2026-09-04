# PM3 validation record

## Oracle

- executable: MOPAC v23.2.5 Windows build
- method: `PM3 PRECISE AUX(PRECISION=9)`
- harness: `tools/oracle/run_mopac.py`
- frozen Rust comparisons: `tests/molecules.rs`
- reproduction audit and known deviations: `tools/oracle/PM3_AUDIT.md`

The MOPAC executable is not redistributed. Run
`python tools/oracle/fetch_mopac.py` to download the
official portable archive into the gitignored `tools/oracle/mopac/` directory (its
SHA-256 is pinned in the script), or set `MOPAC_EXE` when importing the harness.

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
compares the heat of formation, all `3N` Cartesian gradient components, all
`(3N)^2` Cartesian Hessian elements, the Mulliken charges, the dipole vector,
and every molecular-orbital energy. The reference Hessian in
`ALL_ELEMENTS_RESULTS.json` is a central difference of MOPAC analytic gradients
with a 0.001 Angstrom displacement and `NOREOR`.

The 60/60 completed sweep has these maximum absolute differences, re-run at
v0.2.1 and unchanged from v0.2.0 in every digit that matters:

| Quantity | Maximum absolute difference |
|---|---:|
| Heat of formation | 0.0839224 kcal/mol |
| Cartesian gradient | 1.60352e-4 eV/Bohr |
| Cartesian Hessian | 2.23657e-2 eV/Bohr^2 |
| Mulliken charge | 5.85332e-5 e |
| Dipole component | 2.49155e-4 D |
| MO energy | 1.53202e-4 eV |

The energy maximum is the `Cb` fixture, whose heat of formation is about
-1.171e8 kcal/mol because MOPAC deliberately uses a -9,999,999 eV resonance
sentinel; its relative energy error is below 1e-9. The Hessian maximum comes
from differentiating finite-precision MOPAC gradient output for heavy atoms.
Using a 0.005 Angstrom difference for As/Br/I reduces their maximum to
4.925e-3 eV/Bohr^2. Every remaining outlier is an element with valence principal
quantum number >= 4; that deviation is characterised and bounded in
`tools/oracle/PM3_AUDIT.md`.

`ALL_ELEMENTS_FORCE_RESULTS.json` repeats the sweep with MOPAC's analytic
`FORCE` Hessian instead of the finite-difference reference. Its Hessian
differences are larger because MOPAC's `FORCE` output on a non-stationary
geometry carries MOPAC's rotational/translational projection, which the plain
Cartesian second derivative does not; it is therefore not the primary derivative
oracle. Its *energies* are the same quantity as the finite-difference sweep's and
must agree with them — an earlier copy of the file disagreed by 48 kcal/mol on
`SbH3` because it had been generated from a superseded planar fixture, which is
recorded in `PM3_AUDIT.md`.

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

## External electric field

MOPAC's `FIELD=(x,y,z)` keyword applies a uniform field in **volts per Ångström**
with `E = E₀ + μ·F` — the plus sign following from MOPAC reporting the dipole in
the chemistry convention, pointing from negative to positive. The unit and the
sign were **measured, not assumed**: `tests/molecules.rs`
(`an_external_field_matches_the_mopac_field_keyword`) freezes MOPAC's own heat of
formation for water at `FIELD=0.001`, `0.002` and `0.004` and reproduces all
eight digits MOPAC prints.

`internal_consistency.py` adds the identity that needs no oracle:
`μ = ∂E/∂F`, by central difference of the CLI's own energy against the dipole
the CLI reports. Neutral systems only — `∂E/∂F` is about the field's origin, the
coordinate origin, while the reported dipole is about the centre of mass, and
for an ion those differ by `Q` times their separation.

Over the 60-case sweep the worst disagreement is **4.0e-6 D** (`KH`), against
dipoles of order 1–10 D. The single exception is the `Cb` capped-bond fixture at
6.3e-2 D, which is the same sentinel artefact that gives it a gradient residual
of 2.6e6 eV/Bohr: MOPAC assigns `Cb` a −9,999,999 eV resonance integral, so every
finite difference of its energy is dominated by that number rather than by
physics. It is characterised in `PM3_AUDIT.md`. Excluding it, the sweep's worst
analytic-gradient-versus-finite-difference residual is 8.2e-5 eV/Bohr.

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

```bash
cargo test --all-targets --all-features
python tools/oracle/run_mopac.py examples/water.xyz --method PM3
```

```bash
maturin develop --release --features python
python -m pytest tests/
```
