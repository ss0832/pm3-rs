# pm3-rs scope

## In scope

- Molecular PM3 with the MOPAC v23.2.5 element and pair parameter tables.
- RHF/UHF SCF selected from charge and multiplicity.
- Electronic energy, core-core energy, heat of formation, Mulliken charges,
  dipole, orbital energies, analytic gradient, CPHF/UCPHF Hessian, geometry
  optimization, and harmonic frequencies.
- PM3-D3, PM3-D3H4, and PM3-D3H4X classical post-SCF corrections, including
  their analytic gradient and Hessian contributions.
- PM3 elements H-Ca, Zn-Sr, Cd-Ba, and Hg-Bi using the PM3 s/p basis.
- MOPAC La-Lu trivalent Sparkles and special atom codes `Cb`, `+`, and `-`.
- Rust, CLI, Python-native, and ASE interfaces.

## Out of scope

- Periodic boundary conditions.
- Elements without a MOPAC PM3 parameterization; they return an error.
- MOPAC methods other than PM3 and the listed correction variants.
- PDB-residue-name-only HIP/GUA H4 overrides. Standard continuous water,
  ammonium, and carboxylate H4 scaling remains enabled for all inputs.

## Oracle validation

MOPAC v23.2.5 is the executable oracle for plain PM3 and the special-atom
paths. Tests freeze heats of formation for water, methane, ammonia,
formaldehyde, H2S, HCl, CH3, GdF3, and water interacting with `+` and `-`. A
second exhaustive oracle covers all 42 ordinary PM3 elements, 15 La-Lu
Sparkles, and `Cb`, `+`, and `-`. It compares energy, all gradient components,
and every element of the Cartesian Hessian. Water gradient, optimized minimum,
and harmonic frequencies are also checked.

Current MOPAC v23.2.5 does not expose the historical PM3-D3/D3H4 method
keywords. Correction coefficients are therefore source-verified against public
MOPAC 5.022mn and the D3H4 reference implementation, while derivative paths are
checked against finite differences in Rust.

Exact values and tolerances are recorded in `tools/oracle/PM3_VALIDATION.md` and
`tests/molecules.rs`.

## Units

- Internal: eV and Bohr, using MOPAC's 2018 CODATA model constants.
- Rust/Python native: atomic-unit fields with eV/kcal conveniences.
- ASE: eV, Angstrom, eV/Angstrom, and eV/Angstrom^2.
