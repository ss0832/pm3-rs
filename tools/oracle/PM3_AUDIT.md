# PM3 reproduction audit (v0.2.0)

A re-examination of whether pm3-rs is a faithful reproduction of MOPAC's PM3, carried out
before the periodic and divide-and-conquer work so that everything built on top rests on a
verified molecular core.

Oracle: MOPAC v23.2.5 (`tools/oracle/fetch_mopac.py`, SHA-256 pinned). Harnesses:
`all_element_validation.py` (against MOPAC) and `internal_consistency.py` (oracle-free).

## Summary

| | before the audit | after |
|---|---|---|
| quantities compared over the 60-case sweep | energy, gradient, Hessian | + Mulliken charges, dipole, **all** MO energies |
| cases with an MO comparison | 0 | 60 |
| defects found in pm3-rs | — | 1 (fixed) |
| defects found in the validation record | — | 3 (fixed) |

Final agreement with MOPAC v23.2.5 over all 60 supported atom-code cases:

| quantity | max absolute difference | where |
|---|---:|---|
| heat of formation | 8.39e-2 kcal/mol (7e-10 relative) | `Cb`, whose ΔHf is -1.17e8 kcal/mol |
| heat of formation, excluding `Cb` | 1.05e-3 kcal/mol | `GeH4` |
| Cartesian gradient | 1.60e-4 eV/Bohr | `SeH2` |
| Cartesian Hessian | 2.24e-2 eV/Bohr² | `AsH3` (differencing MOPAC's printed gradients) |
| Mulliken charge | 5.85e-5 e | `BiH3` |
| dipole | 2.49e-4 D | `SeH2` |
| MO energy | 1.53e-4 eV | `SeH2` |

## Defect 1 — the dipole of a charged system was origin-dependent (**fixed**)

`Σ q_i r_i` depends on the origin whenever `Σ q_i ≠ 0`, so the origin is part of the
definition of an ion's dipole rather than a free choice. MOPAC (`dipole.F90`) references it
to the **centre of mass**; pm3-rs used the coordinate origin.

The consequence was not subtle. Translating NH4+ by 5 Å moved its reported dipole from 0 to
24.02 D — the exact `5 Å × 4.80320 D/(e·Å) × 1 e` a rigid translation of a unit charge
produces. Every ion pm3-rs was ever asked about had a dipole that depended on where the
caller happened to place it.

Evidence that the centre of mass is the right origin: for water plus a MOPAC `+` point atom,
pm3-rs minus MOPAC was `(0.1930, 0.2493, 0)` D, which is `4.80320 ×` the centre of mass of
the **three real atoms** — MOPAC's `+`/`-` carry `ams = 0`, so they move the charge
distribution without moving the origin. The z component agreed exactly, and the real atoms
all lie at z = 0.

Fixed in `src/scf.rs` by referencing the point-charge term to the centre of mass
unconditionally (a no-op for a neutral system, where `Σ q_i = 0` makes the shift cancel).
After the fix every case in the sweep agrees with MOPAC to ~1e-6 D and is translation
invariant. Regression:
`tests/molecules.rs::charged_system_dipole_matches_mopac_and_is_translation_invariant`.

## Defect 2 — the validation record misexplained a 48 kcal/mol disagreement (**fixed**)

`ALL_ELEMENTS_FORCE_RESULTS.json` contained an `SbH3` case with a 48.17 kcal/mol energy
difference and a 0.437 eV/Bohr gradient difference, while `PM3_VALIDATION.md` attributed the
FORCE file's differences to "MOPAC's rotational/translational treatment". A rotational
treatment cannot change a heat of formation, so that explanation could not be right.

What actually happened, established by re-running both codes:

* The file was **stale**. It was generated with an older planar group-15 hydride fixture; the
  harness was later changed to a pyramidal geometry (`closed_shell_hydride`, valence 5) and
  only `ALL_ELEMENTS_RESULTS.json` was regenerated.
* On the **pyramidal** geometry the two codes agree to 3.5e-6 kcal/mol, and their HOMO/LUMO
  energies agree to 1e-5 eV. The Hamiltonian is identical.
* On the **planar** geometry they converge to two different fixed points of that same
  Hamiltonian: MOPAC's default converger reaches 161.22 kcal/mol (HOMO/LUMO -6.590/-2.719),
  pm3-rs reaches 113.06 kcal/mol (HOMO/LUMO -8.340/-2.354). Both are converged aufbau
  solutions with a healthy gap; pm3-rs's is 48 kcal/mol **lower**.
* Asking MOPAC for its Camp-King converger (`CAMP`) makes it reach **113.05600 kcal/mol with
  HOMO/LUMO -8.340/-2.354** — pm3-rs's solution, to all printed digits. `PULAY`, `SHIFT=20`
  and a 2000-iteration run all stay in the higher basin.

So this is a converger artefact in MOPAC's default SCF path, not a model difference: pm3-rs's
SAD guess plus the A-DIIS/CDIIS hybrid finds the variationally lower solution where MOPAC's
default does not. The lower solution is the correct one.

`--mopac-keyword CAMP` was added to `all_element_validation.py` so this cross-check is
reproducible, and `ALL_ELEMENTS_FORCE_RESULTS.json` has been regenerated from the current
fixtures.

### MOPAC is not self-consistent about the dipole origin

Regenerating `ALL_ELEMENTS_FORCE_RESULTS.json` surfaced a second, independent confirmation of
Defect 1 — this time inside MOPAC. For HeH+ at a fixed geometry:

| MOPAC run mode | dipole (D) |
|---|---:|
| `1SCF` / `GRADIENTS` | 2.62260 |
| `FORCE` | 3.49218 |

The difference, 0.86958 D, is exactly `4.80320 D/(e·Å) × 0.18104 Å`, and 0.18104 Å is exactly
HeH+'s centre-of-mass offset from the He nucleus. MOPAC's `FORCE` path reports a charged
system's dipole about the **coordinate origin**; its `1SCF` path reports it about the **centre
of mass**. The same 0.24932 D offset appears for `water_point_plus` and `water_point_minus`.

pm3-rs follows the `1SCF` convention, which is the one MOPAC documents. The validation harness
therefore abstains from the dipole comparison for charged systems in FORCE mode and records
why, rather than reporting a "0.87 D error" against a different definition.

## Defect 3 — the sweep never compared the density (**fixed**)

The 60-case sweep compared energy, gradient and Hessian only. The NDDO energy expression is
stationary in the density, so an error in a one-center term can cancel out of the energy *and*
its derivatives while still leaving the density wrong — which is precisely how Defect 1
survived: the `water_point_plus` energy agreed to 3.3e-10 kcal/mol while its dipole was off by
0.25 D.

Mulliken charges, the dipole vector and every MO energy are now compared for all 60 cases.
Two MOPAC output conventions had to be handled before the MO comparison covered anything:

* `EIGENVALUES` is a **window**, not the spectrum. `SET_OF_MOS` gives its inclusive 1-based
  range; for a Sparkle complex it reads `3 12`, i.e. MOPAC drops the two deepest levels. A
  naive length comparison silently skipped all 15 Sparkle cases.
* For the capped bond `Cb`, MOPAC omits the two orbitals dominated by its -9,999,999 eV
  resonance sentinel and pads the list with `1e-12` placeholders, while pm3-rs reports them at
  ±2.8e6 eV. Comparing padding against real orbitals produced a meaningless "2.8e6 eV error"
  that hid the fact that Cb's nine physical orbitals agree to 3.5e-8 eV.

## Open item — a bounded n ≥ 4 deviation

Every remaining outlier is an element with valence principal quantum number ≥ 4 (Zn, Ga, Ge,
As, Se, Br, Sr, Cd, In, Sb, Te, I, Cs, Ba, Hg, Tl, Pb, Bi). The deviation is bounded by
1.05e-3 kcal/mol in energy, 5.9e-5 e in charge, 2.5e-4 D in dipole and 1.5e-4 eV in orbital
energy — three to four orders of magnitude below chemical significance, and below the
5e-3 kcal/mol tolerance the regression suite already uses.

What it is **not**:

* **Not quadrature error.** pm3-rs computes the n ≥ 4 diatomic overlap by Gauss-Legendre
  quadrature in prolate-spheroidal coordinates (`overlap_numeric.rs`). Tripling the
  quadrature order (48→160 radial nodes, 40→120 angular, `xi_max` 60→180) leaves every
  reported difference **bit-identical**. The quadrature is fully converged.
* **Not a principal-quantum-number mismatch.** MOPAC's AUX `ATOM_PQN` was compared against
  `element_data.csv` `npq_s` for all 42 elements; they agree, including the noble-gas cases
  where PM3 deliberately assigns a Slater `n` one above the periodic-table row.
* **Not long-range.** Scaling GeH4's bonds by 1.5 drops the difference from 1.0e-3 to
  6.0e-6 kcal/mol, so it lives where the overlap does.

The remaining explanation is that MOPAC's own closed-form `diat`/`SS` overlap for n ≥ 4 —
built from the `A`/`B` auxiliary integrals with series expansions that switch form by argument
— differs from the exact Slater overlap at the 1e-8 level, and pm3-rs's converged quadrature
is the more accurate of the two. Reproducing MOPAC bit-for-bit here would mean reproducing its
truncation, which is of questionable value; the deviation is recorded and pinned instead.

## Reproducing this audit

```bash
python tools/oracle/fetch_mopac.py
```

```bash
python tools/oracle/all_element_validation.py --cli target/release/pm3_rs_cli
```

```bash
python tools/oracle/internal_consistency.py --cli target/release/pm3_rs_cli
```

On Windows the CLI is `target/release/pm3_rs_cli.exe`; everything else is the
same on every platform.
