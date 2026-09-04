# pm3-rs scope

## In scope

- Molecular PM3 with the MOPAC v23.2.5 element and pair parameter tables.
- RHF/UHF SCF selected from charge and multiplicity.
- Electronic energy, core-core energy, heat of formation, Mulliken charges,
  dipole, orbital energies and coefficients, analytic gradient, CPHF/UCPHF
  Hessian, geometry optimization, and harmonic frequencies.
- **A uniform external electric field**, molecular only: energy, analytic
  gradient and analytic Hessian, RHF and UHF. Refused under periodic boundary
  conditions, where `−f·r` is not lattice-periodic, and refused by the periodic
  machinery even on an isolated cell, which does not carry one.
- **The Mermin electronic free energy** `A = E − TS` for a smeared k-point
  calculation (`KpointResult::free_energy_ev`, ASE's `free_energy`). It is what
  the Hellmann–Feynman force is the gradient of once occupations are fractional,
  which is why reporting `E` beside those forces was inconsistent by `∂(TS)/∂R`.
  **Not a Gibbs free energy**: no zero-point energy, no vibrational partition
  function, no `pV`, no nuclear entropy. `smearing_ev` is `k_B T` for a
  fictitious electronic temperature that makes a metal's occupations converge,
  not for the temperature of an experiment.
- **A finite field along a periodic direction** (`pbc::finite_field`), where
  that refusal is unavoidable rather than conservative: the ground state of
  `H − 𝓔·R` on a lattice does not exist. What is minimized instead is the
  Nunes–Gonze electric enthalpy `F = E − Ω 𝓔·P`, with `P` the Berry-phase
  polarization. Restricted, gapped, three-dimensional cells; energy and
  polarization only, **no force**.
- **Infrared intensities** and the dipole-derivative tensor `∂μ/∂R` behind them,
  RHF and UHF: three coupled-perturbed solves rather than `3N`, with the
  translations and rotations projected out of the mass-weighted modes.
- **Molden wavefunction output**: the raw ZDO coefficients, in MOPAC's own
  `VECTORS`/`GRAPHF` convention, over a Gaussian expansion of each Slater
  function derived at run time rather than transcribed.
- PM3-D3, PM3-D3H4, and PM3-D3H4X classical post-SCF corrections, including
  their analytic gradient and Hessian contributions.
- PM3 elements H-Ca, Zn-Sr, Cd-Ba, and Hg-Bi using the PM3 s/p basis.
- MOPAC La-Lu trivalent Sparkles and special atom codes `Cb`, `+`, and `-`.
- Rust, CLI, Python-native, and ASE interfaces.
- Periodic boundary conditions in 1D, 2D, and 3D: Γ-point and k-point RHF/UHF
  energy, analytic forces, and the Γ-point analytic Hessian with phonons. Ewald
  electrostatics (3D tinfoil, 2D Parry, 1D direct, 0D direct), neutral and
  charged cells, lattice-summed D3/H4/X corrections, and band structures. See
  `docs/pbc.md`, in particular the Γ-point validity condition.
- **Born effective charges** (`pbc::born`), from the same coupled-perturbed
  response the Γ-point force constants use. Checked against their acoustic sum
  rule, against a central difference of the model's own cell dipole, and for
  independence of where the cell origin was put.
- **Electronic polarizability and dielectric tensor** (`pbc::dielectric`).
  `α` in every dimensionality; `ε∞ = 1 + 4πα/Ω` only in 3D, since a slab has an
  area and a chain a length. Pinned by agreeing with a finite field on the
  isolated molecule to 0.2%.
- **LO–TO splitting** (`pbc::lo_to`), the non-analytic term
  `(4π/Ω)(q̂·Z*)(q̂·Z*)/(q̂·ε∞·q̂)` that makes the `q → 0` limit of a polar
  crystal direction-dependent. Its prefactor is **measured**, not transcribed:
  the rigid-ion lattice sum still contains the macroscopic term, so the closed
  form is checked against that sum's own `q → 0` limit — a comparison a code
  that reaches `D(q)` by interpolating truncated force constants cannot make.
- **Analytic stress and variable-cell relaxation in every periodic
  dimensionality.** A chain reports its axis, a slab its two in-plane
  components, a crystal all nine, with exact zeros where there is no periodic
  direction to strain. `relax` moves the cell vectors that exist and leaves a
  slab's vacuum thickness alone. Only an isolated cell refuses, having no strain
  rather than an underived derivative.
- Isolated systems as the zero-dimensional case of the same machinery
  (`Cell::isolated`). This gives a large molecule the crystal's `O(N)` near
  field instead of the dense `O(N²)` pair cache, at the cost of the documented
  Klopman–Ohno switch: 3 µeV per atom, measured, and flat in system size.
  The far field beyond 80 Bohr runs over a **multipole tree** rather than every
  pair: an octree whose nodes carry the combined moments of everything beneath
  them, accepted on an absolute error bound rather than an opening angle, and
  only where the pairwise rule would already have taken every cloud inside them —
  so the near field is unchanged to the bit. The far-field term count grows with
  a measured log-log slope of 1.58 against the pairwise sum's exact 2.00. Below
  about 160 Bohr across there is no far field at all and none of this applies.
- Divide-and-conquer partitioned SCF, molecular and Γ-point periodic, RHF and
  UHF, with forces and stress. See `docs/divide-and-conquer.md`.
- Charged cells in every dimensionality: a uniform neutralizing background in
  3D and 2D, a neutralizing line charge in 1D. Absolute energies of charged
  cells are convention-dependent and are not comparable across cells; the 1D
  convention is size-consistent, so a chain and its own doubled cell agree.
- Sparkles, the `+`/`-` point atoms and `d` shells in the periodic path.
- **DFPT at arbitrary `q`** (`pbc::dfpt`): the dynamical matrix and phonon
  frequencies at any wavevector, from the primitive cell, including the
  electronic response. Every periodic dimensionality — a `q` component along a
  non-periodic axis is refused rather than summed. Open and closed shell, plain
  PM3 and every corrected variant. `rigid_ion_dynamical_matrix` gives the
  fixed-density part alone.
- **The classical corrections at finite `q`.** Their image-resolved second
  derivative comes from the same forward-mode arithmetic the `Γ` one uses, with
  each cluster entry's displacement scaled by its own `e^{iq·T}`; the energy is
  never decomposed into pairs, so D3's coordination-number coupling and H4's
  donor–hydrogen–acceptor triples are carried rather than dropped. An image's
  coordination number is computed honestly wherever the cluster is wide enough
  around it and copied from its parent beyond that — the copy is exact at `Γ`
  and an approximation away from it, and the folding residual it leaves is a
  relative `4e-7`, falling with the cluster radius.
- **A k-mesh under DFPT** (`dynamical_matrix_on_mesh`,
  `phonon_frequencies_on_mesh`). The response runs the *unreduced* mesh even
  where the ground-state SCF reduced it by time reversal: time reversal maps the
  coupled pair `(k, k+q)` onto a pair at `−q`, so the irreducible sum with
  doubled weights is the wrong number wherever `q ≠ −q`. Passing no mesh keeps
  the Γ-only sampling, which is what the Γ ground state supports.
- A `pm3-rs` command installed by `pip`, backed by the same compiled CLI the
  standalone executable uses.

## Out of scope

- Genuine linear scaling. `DcOptions::long_range_cutoff` makes the near field
  `O(N)` — pair tables inside a cutoff, a point-charge model outside — and the
  density algebra is linear (`dc::pattern`), and past 80 Bohr each atom's charge
  cloud is collapsed onto its multipole moments instead of being summed site by
  site. The measured log-log slope is **1.16** over 360 to 2880 atoms — not 1,
  and just past the 1.15 the plan set. It buys **86×** over full
  diagonalization at 2880 atoms, at about 28 µeV per atom, a figure that does
  not grow with the system. The *memory* is linear: the eight `nao × nao`
  matrices the SCF loop held are stored on the sparsity pattern instead, at a
  measured log-log slope of 1.06 against the dense form's 2.00. See
  `docs/divide-and-conquer.md` for the per-size table and the component
  breakdown.

  The 1.16 is a re-measurement. The figure here used to be 1.27, taken before
  the multipole tree for the far field existed; that tree is what closed the
  gap, and the last three points alone give 1.17, so this is where it settles
  rather than a number still falling with size.

- Second derivatives under divide-and-conquer. Energy, gradient and stress
  only; a partitioned CPHF response is not defined by the current partitioning.

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

There is no external oracle for periodic PM3. Correctness there rests on
internal consistency instead — supercell folding, the Madelung constant, Ewald
against direct lattice summation in each dimension, the isolated-molecule limit,
and analytic derivatives against finite differences. `docs/pbc.md` lists what is
checked and to what tolerance.

## Units

- Internal: eV and Bohr, using MOPAC's 2018 CODATA model constants.
- Rust/Python native: atomic-unit fields with eV/kcal conveniences.
- ASE: eV, Angstrom, eV/Angstrom, and eV/Angstrom^2.
