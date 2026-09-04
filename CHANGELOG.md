# Changelog

## 0.2.3 — 2026-09-03

One performance fix, found by running real crystals rather than by reading code,
and two reporting defects it turned up on the way.

### Faster

- **Born charges and Γ-point phonons are ~2.4× faster**, and the numbers are
  unchanged. The long-range term of the bare perturbation ran two lattice sums
  over the site pairs *inside* the per-degree-of-freedom loop — `6N` sums where
  two suffice. The displacements do not depend on the Cartesian axis at all, so
  each was recomputed three times per atom for nothing, and across atoms the
  sets are disjoint slices of one site-by-site table. It is now built once
  (`pbc::dfpt::LongRangeKernels`).

  At 12 atoms a Born-charge run went from 62.2 s to 26.3 s; the Rust suite's
  periodic tests roughly halved.

  Worth recording how it was found, because three earlier attempts missed it.
  Replacing the `3N` coupled-perturbed solves with three by the interchange
  theorem changed the wall clock by **nothing**; hoisting the neighbour list out
  of the bare perturbations changed **nothing**; parallelising the three field
  solves changed **nothing**. Timing the pieces put 85% of the run in one
  untimed call. The lesson is in `examples/born_profile.rs`, which decomposes
  the cost with public entry points that differ in one term each.

### Fixed

- **`Pm3Error::ScfNotConverged` always reported `error: NaN`.** It was
  hardcoded at all four sites — the Γ path, the k-point path and both molecular
  paths — so the field meant to say how far off a run was never did, and `NaN`
  reads as a numerical blow-up when the truth was usually a slow tail. Diamond's
  conventional cell reports `3.854e-4` against a `1e-7` tolerance; it was not
  diverging at all.
- **A failed Γ-point SCF never mentioned the Γ margin**, which is usually the
  cause. Below it one k-point cannot represent the cell and no amount of damping
  helps. The message now gives the margin and says to use a k-mesh or a larger
  supercell. Measured on NaCl's conventional cell, where the Γ answer and the
  4×4×4 one differ by **421 eV**.

### Housekeeping

- `cargo fmt` across the tree: 0.2.2 shipped 26 files that rustfmt disagreed
  with, all of them new or edited in that release.

## 0.2.2 — 2026-09-03

The response properties a dynamical matrix was already most of the way to, and a
Berry phase to check them from outside. The largest fix is none of those: a
coupled-perturbed solver that had been returning unconverged responses under a
converged label, which every analytic Hessian and infrared intensity this crate
has produced was built on.

### Added

- **Born effective charges** (`pbc::born`, `pm3_rs.born_charges`, CLI `born`,
  `PM3.get_born_charges`), reported with the acoustic sum-rule residual rather
  than with the rule imposed.
- **Polarizability and the dielectric tensors** (`pbc::dielectric`,
  `pm3_rs.dielectric`, CLI `dielectric`, `PM3.get_dielectric`): `ε∞` for a
  fully periodic cell, `α` in every dimensionality, and — with `include_ionic`
  / `--static` — the static tensor `ε₀`. `skipped_modes` says how much of the
  ionic sum is missing, since at an unrelaxed geometry the answer is incomplete
  rather than merely odd.
- **LO–TO splitting** (`pbc::lo_to`, `phonons(lo_to_direction=)`,
  `--lo-to`). The coefficient was not transcribed: it was fixed against this
  crate's own finite-`q` DFPT in the `q → 0` limit, where the ratio converges to
  1 as `q²` (0.56 → 0.89 → 0.973).
- **Supercell force constants and phonon dispersion** (`pbc::phonon`,
  `ForceConstants`, `pm3_rs.phonon_bands`, CLI `phonon-bands`,
  `PM3.get_phonon_bands`). One Hessian buys the whole band structure. `Φ(T)`
  lives in a sorted vector, not a hash map: the Fourier sum's accumulation order
  is otherwise run-dependent, which in a near-degenerate mode is not a
  last-digit effect.
- **Berry-phase polarization** (`pbc::berry`, `pm3_rs.berry_polarization`, CLI
  `berry`, `PM3.get_berry_polarization`), as an independent check on the above
  rather than as a feature in itself.
- **A finite electric field along a periodic direction** (`pbc::finite_field`,
  `pm3_rs.finite_field`, CLI `finite-field`, `PM3.get_finite_field`), by the
  Nunes–Gonze electric enthalpy `F = E − Ω 𝓔·P`. `𝓔·R` is not lattice-periodic
  along a periodic axis, so `H − 𝓔·R` has no ground state there; a field
  orthogonal to every lattice vector still goes through `Pm3Options::field`.
  Reproduces the CPHF polarizability to 1 part in 10⁴. Restricted, gapped, 3D
  only, and there is no force.
- **`DfptOptions` / `LongRange` / `DfptResult`**, `force_constants_at_q`,
  `frequencies_at_q`, and the response density (`PhononResponse`) everything
  above contracts.
- **The Mermin electronic free energy**: `KpointResult::entropy_ts_ev` and
  `free_energy_ev`, and ASE's `free_energy`, so
  `get_potential_energy(force_consistent=True)` works. Documented in four places
  as *not* a Gibbs energy — no zero-point energy, no vibrational partition
  function, no `pV`, no nuclear entropy.
- **`divide_and_conquer_forces`**, `periodic_hessian`, `magnetization`,
  `smearing_ev`, CLI `--reference`, and the mode vectors and masses
  `frequencies` and `dynamical_matrix` had been withholding.

### Fixed

- **CPHF returned unconverged responses as converged.** All four paths ran to an
  iteration limit and returned `Ok`. Adding the check failed eight existing
  tests, which exposed the cause: `hessian.rs`'s own DIIS divided `B_ij` by each
  residual's magnitude, replacing the constraint `Σc = 1` with a different one.
  The correct implementation was already in `scf.rs`. Water's response went from
  160 passes to **3** (residual `8.1e-17`); the Rust suite from 38.7 s to 31.4 s.
- **`D(q)` diverged as `1/q²`** in a polar cell — the acoustic sum rule read
  220.1 at `q = 0.0125`, and water in a 10 Bohr cell put its lowest mode at
  −4705 cm⁻¹. The `G = 0` term of the bare long-range perturbation was scaled by
  the displaced atom's own charge rather than the cell's, so `Σ_a Q_a = 0` never
  cancelled it. Fixed by excluding the macroscopic field from the
  self-consistent response and restoring it analytically; the term restored is
  exactly `pbc::lo_to`. No existing DFPT test ran below `q = 0.2`.
- **`dynamical_matrix_on_mesh` built its skeleton from the Γ density**, so
  passing a mesh corrected the response and left the dominant term wrong.
- **ASE ran two SCF calculations per step and stitched them together**, taking
  energy and forces from an unsmeared one and charges from a smeared one — on a
  metal, different states. Now one call; the Python suite went 99 s → 51 s.
- **UHF CPHF had no DIIS**, so the hardest case was the one without acceleration.
- **Open-shell HOMO/LUMO ignored the β spectrum.**
- **`stress_tol` crossed the Python boundary without its unit conversion**,
  leaving the periodic optimizer 6.75× looser than documented and returning
  relaxed cells with `converged=True`.
- **`long_range_cutoff` was silently ignored by periodic divide-and-conquer**,
  and `--kpts` by three CLI paths that cannot sample a mesh. Both now refuse.
- **ASE's result cache was not invalidated by attribute changes**, so setting
  `atoms.calc.charge = 1` returned the neutral energy.
- **`vibrational_analysis` divided by zero mass** for MOPAC's sparkles.

### Fixed in the new code, by cross-checks rather than by review

- **The field operator was half its correct size.** `M = i λ (W₊ − W₋) C†` is
  one-sided — `M|v⟩ = 0` for a virtual `v` — so it holds the whole
  virtual-occupied block and none of the occupied-virtual one. The conventional
  `½(M + M†)` therefore halves the block a linear response is made of. Measured
  at a ratio of `0.5001` against the CPHF polarizability; `M + M†` gives
  `1.0003`.
- **The finite field computed polarization only along the axes the field
  touched.** Those are not the axes that carry polarization: a zero field came
  back with an electronic polarization of exactly zero. `resolved` now reports
  which axes the mesh could see.
- **The CLI handed `run_finite_field` a negated field.** `--field` is stored
  with the sign that makes `E = E₀ + μ·F` hold for the molecular `−𝓔·r`
  coupling. Nothing failed — the loop converged and `∂P/∂𝓔` was backwards.

### Wiring

A layer-by-layer audit found four features present in some layers and not
others, the same shape of gap as the "DFPT is Rust-only" report:
`berry_polarization` was Rust-only, supercell dispersion was CLI-only, the
finite field reached neither ASE nor the CLI, and LO–TO was unreachable from the
CLI. All four are now in every layer.

### Measured, and not fixed

- **The correction Hessian is still `O(N⁴)`.** Bounding the D3/H4 sums at the
  radii the periodic path uses was tried and reverted: it cost `2.5e-4` eV at
  375 atoms and bought nothing, the log-log slope staying at **1.959**. The sums
  are `for i { for j { if r > cutoff { continue } } }`, so a radius bounds the
  range and not the work — and 30 Bohr encloses ~2000 atoms of liquid water,
  more than any system measured. See `examples/correction_cutoff.rs` and
  `examples/correction_scaling.rs`.
- **Divide-and-conquer is at slope 1.16**, against an acceptance criterion of
  1.15, over 360 to 2880 atoms — 86× over full diagonalization at 2880 atoms, at
  28 µeV per atom, a figure flat in system size.
- **A cliff at the Γ-point margin.** Crossing `DEFAULT_SHORT_RANGE_CUTOFF`
  (14 Bohr) steps a water box's energy by **35 eV** between 7.4080 and 7.4085 Å.
  Documented behaviour rather than a defect — below the margin the SCF converges
  cleanly to a well-defined wrong answer — but the size of the step is not
  obvious from that sentence, and a test fixture of this crate's own was sitting
  on the wrong side of it. See `examples/cell_continuity.rs`.

### Two Berry-phase conventions worth writing down

Both were derived against this crate's own gauge rather than taken from a
published form, because getting either wrong returns a plausible number rather
than an error.

- **No closure factor on the last link.** `bloch_fock` carries the phase on the
  lattice translation alone, so `H(k + G) = H(k)` exactly and the coefficients at
  `k₀ + G` *are* those at `k₀`. An extra `e^{−iG·τ}` put fluorine's Born charge
  at `+21.8 e` against a true `−0.33`.
- **The sign of the electronic term.** With `e^{−ib·τ}` in the overlap, the
  textbook `−(e/Ω) φ a` counts that sign twice. Fixed by the single-orbital
  limit; getting it wrong gives `+14.4`.

With both right, the Berry and CPHF Born charges differ by `0.147 e` — and
removing the intra-atomic `dd` moment the phase omits (`PM3_BORN_NO_DD=1`)
collapses that to `1.9e-4`. The two formalisms agree on everything except one
identified physical term.

## 0.2.1 — 2026-09-01

Phonons everywhere the Γ point already worked, an external electric field, and
the wavefunction outputs that go with it. Three defects that had shipped in
0.2.0 are fixed; all of them were invisible at `q = 0`, which is where every
test that could have caught them ran.

### Added

- **Analytic stress for a slab.** Every periodic dimensionality now reports one:
  a chain its axis, a slab its two in-plane components, a crystal all nine, with
  exact zeros in the non-periodic directions. `relax` relaxes the cell vectors
  that exist and leaves a slab's vacuum thickness alone; only an isolated cell
  refuses, having no strain rather than an underived derivative.
- **A uniform external electric field** (`Pm3Options::field`), molecular only —
  energy, analytic gradient and analytic Hessian. Validated against MOPAC's own
  `FIELD=` keyword to all eight digits MOPAC prints. Refused under periodic
  boundary conditions, where `−f·r` is not lattice-periodic.
- **A shared dipole operator** (`pm3_rs::dipole`). The reported dipole and the
  field coupling are the same matrix, which is what makes `μ = −∂E/∂F` hold by
  construction rather than by coincidence.
- **Molden wavefunction output** (`pm3_rs::molden`, the CLI's `molden` command,
  `write_molden` in both Python layers). The coefficients are the raw ZDO ones,
  which is MOPAC's own `VECTORS`/`GRAPHF` convention and makes the comparison
  against it direct; the Slater functions are expanded in Gaussians derived at
  run time by a regularized linear least squares rather than transcribed from a
  table, to a measured overlap deficit of `2e-5` at worst.
- **β orbitals on `Pm3Result`** (`mo_energies_beta`, `mo_coeff_beta`, `n_beta`),
  which an unrestricted Molden file needs and which the UHF Hessian was
  re-diagonalizing to recover.
- **Infrared intensities** (`pm3_rs::ir`): the dipole-derivative tensor
  `∂μ/∂R` from **three** coupled-perturbed solves rather than `3N`, by the
  interchange theorem, with the nuclear term the electronic part alone would
  miss; and from it the per-mode spectrum in km/mol, translations and rotations
  projected out of the mass-weighted modes. RHF and UHF, both finite-difference
  checked. The km/mol conversion is computed from its SI inputs rather than
  transcribed, and checked against the published 974.88.
- **Phonons for a spin-polarized cell.** The response is written over a list of
  spin channels rather than as a pair of code paths: a closed shell is one
  channel holding the total density at half exchange strength with two electrons
  per state, an open shell two channels each holding its own at full strength
  with one. Those are the same number when the spins are equal, which keeps the
  restricted path at its old cost — one diagonalization per k-point, not two —
  and pins it, every closed-shell value being unchanged. Validated against
  central differences of the periodic UHF force, which shares none of the
  response machinery.
- **The classical corrections at finite `q`.** `D(q)` carries the D3/H4/X terms
  instead of refusing them. Their contribution is the bilinear form
  `Σ_{T,T'} e^{−iq·T} e^{+iq·T'} ∂²E_cell/∂x_{(κ,T)}∂x_{(κ',T')}`, obtained by
  scaling each cluster entry's displacement by its own weight and taking a
  forward-mode second derivative of the whole cluster energy — never of a pair,
  so D3's coordination-number coupling and H4's donor–hydrogen–acceptor triples
  come along. Four real evaluations per entry, `Dual2` being real.
- **Every new feature reaches every API.** The external field, the phonons at a
  wavevector, the band structure and the variable-cell relaxation are now
  callable from `pm3_rs`, `pm3_rs.native`, the ASE calculator and (for the
  field) the CLI's `--field`; `divide_and_conquer` gained the ASE accessor it
  never had. The Python layers take the field in **volts per Angstrom**, the unit
  MOPAC's own `FIELD=` keyword uses. `dipole`, `ir`, `molden` and `pbc::dfpt`
  are re-exported from the crate root and pinned by `tests/api_surface.rs`,
  which covered none of them.
- **The divide-and-conquer SCF's memory is linear in the system.** The eight
  `nao × nao` matrices the loop held — two densities, two Fock matrices, four
  workspaces — live on the sparsity pattern now. Every one of them was already
  read back only through that pattern, and the subsystem gather looks exactly
  where some subsystem holds both orbitals, so nothing outside it was ever
  consulted: no number changes, and `a_reaching_buffer_reproduces_the_full_result_exactly`
  still holds to the bit. The Fock build writes into the pattern directly instead
  of filling an array to have most of it ignored. Measured on a chain of waters,
  8 → 128 molecules: dense grows 0.14 → 36.0 MiB, a log-log slope of exactly
  2.00; on the pattern, 0.141 → 5.06 MiB, slope **1.06**.
- **`pm3_rs::dipole` and `pbc::dfpt` reach Python.** `dipole(...)` returns the
  operator, the centre of mass it is taken about, and the `3 × 3N` derivative
  tensor — three coupled-perturbed solves, not `3N`, and no Hessian, which is
  what separates it from `ir_spectrum`. `dynamical_matrix(...)` returns `D(q)`
  itself rather than only the frequencies it is diagonalized to, with the
  Hermitian defect it was assembled at. Both reach `pm3_rs`, `pm3_rs.native` and
  the ASE calculator.
- **The heavy ASE accessors are cached.** They were already lazy — nothing but
  energy, forces, charges and dipole is computed in a `calculate` cycle — but
  each recomputed on every ask, so a caller who wanted frequencies and then an
  infrared spectrum paid for two Hessians and got no warning. Each now keeps its
  most recent result against the geometry and parameters it was computed at, and
  the phonon and dynamical-matrix caches are keyed on the wavevector and mesh as
  well, so a dispersion sweep neither reuses the wrong answer nor accumulates
  every matrix it built.
- **A multipole tree for the isolated far field.** The point-multipole sum past
  the 80 Bohr handover was quadratic in atoms; it now runs over an octree whose
  nodes carry the combined moments of everything beneath them, so a distant group
  is one term rather than many. The translation up the tree is exact — a node's
  moments *are* its clouds', shifted — so the only error is the expansion's own
  truncation, which a test checks against the definition rather than against the
  recursion that produced it.

  A node is accepted only when `d − s > MULTIPOLE_RADIUS`, so by the triangle
  inequality every cloud inside it was already far by the pairwise rule and **the
  near field is exactly the set it was**. The second condition is an *absolute*
  error bound, not a Barnes–Hut opening angle: the pairwise rule it has to match
  is absolute, and a fixed angle is far looser where it matters — `θ = 0.3` moved
  the energy by 1.5 meV against a 50 µeV tolerance. Measured by counting terms
  rather than by a clock, since this machine runs other work: over a block
  growing 125 → 1000 clouds the far-field count grows with a log-log slope of
  **1.58** against the pairwise sum's exact 2.00, doing 18% of the pairwise work
  at the larger size.
- **An independent check on the phased lattice sum.** `ewald_phased` shipped in
  0.2.0 with four tests, all of which compare it against itself.
  `ewald_reference::direct_phased_sum` compares it against a direct sum over
  whole cells that shares none of its algebra — value, gradient and Hessian, to
  better than `1e-6` at a generic interior wavevector.

- **Phonons off Γ in one and two dimensions.** The phased lattice sum, the phased
  Parry slab and the phased direct chain now all exist, so a slab or a wire has a
  dynamical matrix at any wavevector in its periodic subspace — a component along
  a non-periodic axis is refused rather than quietly summed. `slab_kernel` gained
  its second `z` derivative (and its first unit test), and `ewald_atom_hessian`
  covers 1D and 2D through the phased sum at `q = 0` instead of a second
  implementation.

### Fixed

- **`D(q)` was not Hermitian**, by an amount that was exactly zero at `q = 0` and
  grew linearly with `q` — a tenth of an eV/Bohr² out of twenty-five by
  `q = (0.2, 0, 0)`. The first-order density's contribution to the multipole
  charges read only the lower triangle and doubled it. That is the same number as
  the symmetrized form for a ground-state density, which is real and symmetric,
  and a different number for a complex first-order one; it also stopped the
  charge map being the adjoint of the potential map, which is what Hermiticity
  rests on. Every identity test in the file ran on the rigid-ion matrix, so none
  of them could see it.
- **The response solve diverged at a general wavevector and said nothing.** It
  reached `1e30` in its two hundred passes and returned what it was holding. Two
  changes: it now refuses rather than returns, and it extrapolates over a history
  instead of substituting the equation into itself. The response is linear, so
  the plain iteration converges only where the spectral radius of `χ₀K` is below
  one, and at a general `q` it is not; damping only rescales that eigenvalue.
  Extrapolation solves the linear system on the Krylov subspace the history
  spans, which does not care. Every wavevector probed now converges to `1e-10`.
- **The response summed the time-reversal-irreducible mesh.** The ground state may
  — `P(−k) = P(k)*` — and the response may not: time reversal maps the coupled
  pair `(k, k+q)` onto a pair at `−q`, so doubling the irreducible weights is the
  wrong sum wherever `q ≠ −q`. The response now regenerates the full mesh, while
  still sharing the potential the reduced SCF converged.
- **The external field was half-wired into three paths.** The open-shell
  skeleton (`skeleton_fock_ov_spin`) never carried the field's first derivative,
  so a UHF Hessian in a field came back symmetric, with six near-zero modes, and
  short by the cross term `Tr[(∂P/∂R)(∂H'/∂R)]` — measured at `3e-3 eV/Bohr²`
  for the methyl radical, sixteen times the finite-difference tolerance. The
  divide-and-conquer energy omitted the field's nuclear half `−Σ_A Z_A R_A·f`,
  and its gradient omitted `−q_A f`. `analytic_gradient` dropped the field
  entirely. The screened long-range path and the periodic path now refuse a
  field rather than ignore one — including on an isolated cell, where the field
  is well-defined but that machinery still does not carry it.
- **Every periodic image was given the reference cell's coordination-number
  response.** `cn[image] = cn[parent]` is exact at `Γ`, where all copies move
  alike, and wrong under a phased displacement, where the image's neighbours
  move by different amounts than its parent's do. It was therefore invisible to
  every test that existed: at `q = 0` it is not an approximation. An image is
  now given its own coordination number wherever the cluster is wide enough
  around it for that number to be right — a triangle-inequality test against the
  cluster radius — and the parent's beyond. Restoring the blanket copy fails the
  supercell folding test by four thousand times its tolerance.
- **The phased reciprocal sum carried a conjugated phase**, `e^{+i(G+q)·d}` where
  Poisson summation gives `e^{−i(G+q)·d}`, which made `D(q)` wrong in its
  imaginary parts at a generic interior `q` in 3D. Every existing test was
  structurally blind to it: the oracle contraction is identically real over `±d`
  pairs, the folding test's `q = b/2` makes the shifted set negation-symmetric,
  and the internal identities conjugate both halves together. Pinned now by the
  α-independence of `Im Φ_q` and by `Φ_q(d+T₀) = e^{−iq·T₀} Φ_q(d)`.
- **The shipped Python is now ASCII.** Sixteen docstrings could not be printed on
  a legacy-codepage console: the package imported and computed correctly, and
  `help()` on it raised `UnicodeEncodeError` on Japanese, Chinese or Korean
  Windows. Installing was never affected — verified by building the sdist and
  installing it under a forced `cp932` locale — but a documented API surface was
  unusable. Two tests keep it that way.
- **The divide-and-conquer radii disagreed between layers by a factor of 1.9.**
  `DcOptions` defaults to 6.0 and 9.0 *Bohr*; the Python signature took Ångström
  and defaulted to 6.0 and 9.0, so a caller who omitted the argument silently got
  subsystems nearly twice the intended size. The CLI and `pm3_rs.native` were
  right; the extension's own defaults are now the same radii.
- **`long_range_cutoff` was unreachable from `import pm3_rs`** — the whole
  linear-scaling path existed and could not be switched on.
- **`phonons` alone had no `reference` argument**, so an open-shell cell could
  not be asked for. ASE's forwarding is now by keyword, since adding the argument
  immediately exposed a positional call handing `method` to `reference`.
- The type stub declared `optimize(max_iter, gtol)`, arguments that never
  existed. A test now compares every stub signature against the extension, and
  another asserts `pm3_rs.native` can pass every argument the extension accepts.

## 0.2.0 — 2026-08-27

Periodic boundary conditions. Molecular results are unchanged except for one
bug fix, noted below.

### Added — periodic boundary conditions

- `Cell` on `Molecule`, covering 1D chains, 2D slabs, and 3D crystals. The
  non-periodic directions never enter the measure, the reciprocal basis, or the
  stress, so a slab's vacuum thickness cannot affect a result.
- Γ-point periodic SCF (`pbc::gamma::run_gamma`), RHF and UHF, with analytic
  forces and analytic stress (`pbc::gradient::periodic_gradient`) and
  fixed- or variable-cell relaxation (`pbc::optimize::relax`).
- Ewald electrostatics with the dimension dependence confined to the reciprocal
  sum: 3D tinfoil, exact 2D Parry slab, a cell-grouped 1D direct sum, and a
  plain owner-excluded pair sum at zero dimensions. Charged cells are supported
  in every dimensionality — a uniform neutralizing background in 3D and 2D, a
  neutralizing line charge in 1D — and the neutralizer's potential enters the
  Fock matrix rather than only correcting the energy afterwards.
- D3, H4, X, and the simple hydrogen-bond correction are lattice summed,
  including the D3 coordination number over images.
- `PeriodicResult::gamma_margin` reports the Γ-point validity condition — see
  `docs/pbc.md`. A cell narrower than the exchange cutoff converges cleanly to
  an answer that is wrong by tens of eV, and nothing else reveals it.
- **k-point sampling** (`pbc::kscf::run_kpoints`), RHF and UHF, neutral and
  charged: Γ-centred Monkhorst–Pack meshes with exact time-reversal reduction,
  a global Fermi level with optional Fermi–Dirac smearing, fixed or free
  magnetization, and band structures along a path. Sampling more than one `k`
  is what lets `P(0, T)` decay with `T`, which lifts the Γ-point cell-width
  condition above.
- New `cmatrix` module: complex Hermitian matrices and eigensolver. ZDO makes
  `S(k) = I`, so no generalized eigenproblem is needed.
- **Γ-point analytic Hessian and phonons** (`pbc::hessian`), with the Ewald
  second derivative (`pbc::ewald_hessian`) and the periodic CPHF response. The
  acoustic sum rule holds before enforcement, so enforcement cleans up rounding
  rather than hiding a misplaced term.
- **Divide and conquer** (`dc`), molecular and Γ-point periodic, RHF and UHF:
  Yang–Lee partitioning with a single global chemical potential, plus forces and
  stress from the partitioned density.
- **Zero dimensions as a member of the same family** (`Cell::isolated`). An
  isolated system is the case with no images, where the lattice sum is a plain
  owner-excluded pair sum. Running a molecule through the periodic path gives it
  the crystal's `O(N)` near field — neighbour-list pair tables plus a
  point-charge model outside the cutoff — instead of a dense `O(N²)` pair cache.
  It reproduces molecular PM3 to 3 µeV per atom with the default switch, and to
  `1e-8` eV total with the switch pushed past the molecule; the error is
  intensive, staying put from 12 to 288 atoms rather than accumulating.
- **A linear-scaling near field for molecular divide and conquer**
  (`DcOptions::long_range_cutoff`, `None` by default). Same split at zero
  dimensions. Measured at 960 atoms: 7.0× faster than full diagonalization
  against 3.2× for the dense path, with the log-log slope down from 2.03 to
  1.27. Left off by default because the switch costs a measured 28 µeV per atom,
  about a hundred times the divide-and-conquer truncation at the default buffer.
- **A multipole far field for isolated systems.** Beyond 80 Bohr two atoms now
  interact through their net charge, dipole and second moment rather than through
  all six hundred and twenty-five of their auxiliary site pairs, with the
  potential carried back out to the sites by a Taylor expansion of matching
  order. The self-consistent field also stops computing site gradients it never
  reads. Together these took the Fock build at 960 atoms from 2.52 s to 1.22 s
  and the whole run from 4.9 s to 3.0 s, with no measurable change in the answer
  — the divide-and-conquer error stays at 28.2 µeV/atom, and `pbc::ewald`'s own
  test puts the collapse at 8 µeV over forty-eight atoms.
  `PM3_DC_PROFILE=1` prints the loop's split into Fock build, subsystem solves
  and density assembly.
- **Analytic stress in 1D.** 1D is summed directly rather than through
  reciprocal space, so its virial is the ordinary pair virial and comes free
  with the gradients. Only the axial component exists; the projection onto the
  strains a cell actually has now happens once, for every dimensionality, rather
  than being left to each sum.
- **Charged 1D cells**, at Γ and at any k mesh, through a uniform neutralizing
  line charge. The subtraction is in terms of the physical extent summed rather
  than the image count — `H_N + ln(L/L₀)` with `L₀` a fixed reference — which is
  what makes it size-consistent. `Q² H_N / L` alone gives each cell a stable,
  plausible energy while a chain and its own doubled cell disagree by eV.
- **Sparkles, point atoms and `d` shells in the periodic path.** An atom with no
  orbitals has no electronic multipoles, so its realization is the nucleus it
  already has; a `d` element's realization comes from MNDO-d's own multipole
  table, which has only `l ≤ 2` and so is carried in full rather than truncated.
  (PM3 itself parameterizes all forty-two of its elements on an s/p basis, so
  the `d` path is unreachable through the standard tables and its test builds
  its own subject.)
- **DFPT at arbitrary `q`** (`pbc::dfpt`), with the phased lattice sum it needs
  (`pbc::phased`): the dynamical matrix and phonon frequencies at any
  wavevector, from the primitive cell, at a cost that does not depend on `q`.
  The electrons' response is included, and it is most of the answer — on water
  it turns a rigid-ion force constant of 5.96 eV/Bohr² into 0.56.
  `rigid_ion_dynamical_matrix` gives the fixed-density part alone.

  Validated at two levels. `D(0)` against `pbc::hessian::periodic_hessian`, an
  independent implementation with no phases in it, itself checked against finite
  differences; and `D(q)` by folding, a doubled cell's Γ-point force constants
  holding the primitive cell's `D(0)` and `D(zone boundary)` between them. Each
  constituent is separately checked against something built differently, which
  is what made three real bugs findable rather than merely visible: a lattice
  sum that skipped same-atom images, a phased kernel with no Ewald self term,
  and an exchange handed a total density where it wanted a spin one.

  The response samples one k-point, `Γ`, paired with `q` — the sampling the
  ground state used, carrying the same `gamma_margin` condition. 3D, closed
  shell, plain PM3.
- New `docs/pbc.md` and `docs/divide-and-conquer.md`.

### Added — interfaces

- Python: `periodic_single_point`, `periodic_forces`, `phonons` and
  `divide_and_conquer`, all taking a cell in Ångström.
- ASE: the calculator switches to the periodic path from `atoms.pbc`, adds
  `stress` to `implemented_properties` (6-component Voigt, eV/Å³) and gains
  `get_phonons`. A molecule or a slab raises on `get_stress()` rather than
  returning zeros, because zeros would be a claim rather than an absence.
- CLI: `--cell`, `--pbc`, `--kpts`, `--dc`, `--dc-core`, and the `stress`,
  `phonons` and `bands` subcommands. A Γ-point run that violates the validity
  condition prints a warning naming it.
- Packaging: PEP 639 licence metadata, classifiers, project URLs, keywords,
  a `py.typed` marker and `_native.pyi` stubs, and GitHub Actions workflows for
  CI and for wheels published through PyPI Trusted Publishing.
- **`pip install` puts the `pm3-rs` command on your path.** The CLI moved out of
  `src/bin` and into the library, taking its argument vector rather than reading
  the process environment, so a PyO3 wrapper can hand it `sys.argv`. `pip` and
  `cargo install` give the identical interface rather than two that drift.
  Verified by installing the source distribution into a clean virtualenv,
  compiling from scratch, and running the installed command.

### Fixed

- **XYZ files with a byte-order mark.** Notepad and PowerShell's
  `-Encoding utf8` both put one in front of the atom count, which produced
  `invalid XYZ atom count: 3` on a file whose first line was visibly `3`. A
  leading mark is now stripped; a second one is still an error, because that is a
  malformed file rather than a Windows editor.
- **Charged-system dipole origin.** The dipole was referenced to the coordinate
  origin rather than to the centre of mass, so for any system with a net charge
  it depended on where the molecule sat: translating NH₄⁺ by 5 Å moved its
  dipole from 0 to 24.02 D. It now matches MOPAC's `dipole.F90` (centre of mass,
  with `+`/`−` point atoms carrying zero mass) for all 60 oracle cases to ~1e-6 D.
  Neutral systems are unaffected, the dipole being origin-independent there.
- `erfc` was computed as `1 − erf`, which cancels catastrophically: 1.16e-8
  relative error at `x = 3.9`. It now uses a continued fraction above `x = 2`,
  giving 3.9e-14 against scipy over 157 points. (New code only; no molecular
  result used `erfc`.)
- The DIIS coefficient solve normalizes its Gram matrix before the pivot test.
  The threshold was absolute while `⟨E_i,E_j⟩ ~ ‖E‖²` shrinks quadratically, so
  a converging run could be handed a matrix of ~1e-10 entries and get back
  wildly amplified coefficients — DIIS converging nicely and then walking back
  out into a limit cycle. The coefficients are invariant under the scaling, so
  no converged result changes.

- **The Γ-point periodic energy belonged to no state.** It was reported as
  `½(P·H + P·F)` with `F` the *extrapolated* Fock — a combination of history
  matrices — paired with the freshly diagonalized density, while the density
  actually returned was the damped and extrapolated one. Three different
  objects. The convergence test could not see it: a stable set of DIIS weights
  makes a wrong energy stop moving as convincingly as a right one. Worst case
  measured, 1.03 eV on a chain of 92 water molecules; usually far below
  tolerance, which is why it survived. The k-point path was checked for the same
  defect and does not have it — its accelerator extrapolates the density, not
  the Fock.
- **The `B` auxiliary integrals lost seven digits just above `|x| = 0.5`.**
  MOPAC's closed-form recursion multiplies its cancellation by `k/|x|` at every
  step, and its power series stops at `0.5`; at `x = −0.51`, `B₉` was off by a
  relative `5e-7` against the integral itself. Below `1e-6` the `x → 0` branch
  returned constants, so a differentiating scalar got a zero derivative where
  `dB₁/dx = −2/3`. Both go away by summing the defining series directly out to
  `|x| = 3` — absolutely convergent, every term independent of every other,
  exact at `x = 0` including its derivative — and the `x → 0` special case
  disappears rather than being widened. Every frozen MOPAC value is unchanged.
- **The Slater overlap went NaN beyond about 500 Bohr.** `A_k` carries
  `e^{−r(ζ_a+ζ_b)/2}` and `B_k` carries `e^{+r|ζ_a−ζ_b|/2}`; the overlap is their
  product, which is tiny, but the two factors separately are `0` and `∞`, and
  `B` overflows first. For an O–H pair that happens at 501 Bohr, and `0 · ∞` then
  propagated silently through `H_core`. This affects **any** calculation
  containing a 500 Bohr separation, not only a periodic or partitioned one; it
  surfaced here because divide-and-conquer was the first thing run on a system
  that large. The overlap is now cut where its own decaying factor is below
  `1e-130`, so no overlap that could matter is affected and every frozen MOPAC
  value is unchanged.

### Performance

- **The periodic SCF now uses the molecular path's accelerator.** Plain CDIIS
  interpolates the Fock with unconstrained weights and has no notion of the
  energy going down, so only damping held it and how much damping is enough
  grows with the system: on a chain of identical waters it converged at 56
  molecules, failed at 58, converged again at 66, then failed at every size
  beyond. Restricted periodic runs now use A-DIIS until the commutator is small
  and CDIIS after, sharing `crate::scf`'s history rather than a second copy.
  Every size converges, in 25 iterations where damped CDIIS needed 110.
- **The Ewald sum caches everything that depends on the geometry** — the
  reciprocal enumeration, the neighbour list, and the `cos(G·r)`/`sin(G·r)` of
  every site against every `G`. An SCF changes only the charges. Capped at
  256 MiB, above which the trigonometry is recomputed.
- **Divide and conquer no longer does `O(N²)` work on its own structural
  zeros.** A partitioned density is zero wherever no subsystem holds both
  orbitals — the approximation the method makes, not a rounding effect. DIIS,
  damping, the RMS change and the energy trace now run over a recorded sparsity
  pattern, and every matrix the loop touches is allocated once. At 960 atoms the
  DIIS history alone fell from 7.17 s to 0.19 s and the whole run from 25.5 s to
  6.5 s, with the converged energies unchanged to every printed digit.
- **The ASE calculator computes forces unasked.** ASE requests one property at a
  time and re-enters `calculate` for each, so naming only what was asked
  converged the same SCF two or three times per MD step. With the Ewald cache, a
  periodic single point went from 250 ms to 53 ms and an MD step from ~520 ms to
  74 ms.
- Periodic SCF: CDIIS on the `[F, P]` commutator, replacing plain damping
  (about 38 iterations → a dozen, each costing a full lattice sum).
- Periodic UHF starts from core-Hamiltonian orbitals occupied to the two aufbau
  counts rather than a spin-scaled atomic-density guess. The scaled guess starts
  inside the spin-symmetric subspace, which for the methyl radical contains a
  stationary point 4.8 eV above the UHF minimum: damping needed ~25 wasted
  cycles to escape it and CDIIS converged straight onto it. Methyl in a cell
  now takes 9 iterations rather than 96, and reaches the correct solution.
- The Ewald reciprocal sums enumerate one member of each `±G` pair and double.
  Every quantity they produce is even under `G → −G`.

## 0.1.2

Performance and memory release. No change to any computed PM3 quantity: the
MOPAC v23.2.5 oracle regressions (heats of formation, charges, gradients,
optimized geometries, frequencies, special atoms, Sparkles) are unchanged.

### Performance

- The two-center Fock build — the `O(N²)` part of every SCF and CPHF iteration —
  now runs batched-parallel over atom pairs instead of in a single serial loop.
- The two-center Coulomb term is contracted as two packed mat-vecs over the
  `(μν)`/`(λσ)` orbital-pair indices rather than a four-index loop: 100 instead
  of 256 multiply-adds per sp/sp pair, with identical arithmetic.
- The SCF accelerator keeps its `⟨E_i,E_j⟩` and `⟨D_i,F_j⟩` Gram matrices
  incrementally, evaluating only the new row and column each iteration. The
  A-DIIS difference matrices `D_i − D_n` / `F_j − F_n` are no longer
  materialized, removing `2 × depth` full `nao × nao` temporaries per iteration.
- `[F,P]` is formed with one matrix product instead of two (`F` and `P` are
  symmetric, so `PF = (FP)ᵀ`), and is skipped entirely when no accelerator runs.
- The two-electron pair table is a single flat buffer instead of a `Vec<Vec<_>>`,
  removing one heap allocation per packed row and making each row contiguous.
- The classical-correction (D3/H4/X) Hessian off-diagonal loop runs on rayon.
- Elementwise reductions (`frobenius_dot`, RMS density change, history
  combination) go parallel above 2^18 elements.
- `symmetric_eigen` skips building a permutation when `faer` already returns
  ascending eigenvalues.
- 900-atom water cluster (nao = 1800), 16 cores: 35.7 s → 17.7 s wall.

### Memory

- The pair cache no longer retains the electron–core attraction blocks
  `e1b`/`e2a` (2 × 81 `f64` per pair). They are consumed while `H_core` is
  assembled and never referenced again; holding them cost 1.3 KiB per pair
  (3.7 GiB for a 2400-atom system) for no benefit.
- New `Pm3Options::integral_memory_mb` (env `PM3_MAX_PAIR_CACHE_MB`, default
  4096 MiB): the `O(N²)` pair cache size is computed in closed form before the
  allocation and reported as a `ResourceLimit` error if it exceeds the budget.
- New `Pm3Options::scf_memory_mb` (default 512 MiB): bounds the SCF accelerator
  history. The DIIS depth is reduced to fit, and A-DIIS degrades to CDIIS (which
  needs no density history) rather than exceeding the budget.
- 900-atom water cluster peak resident set: ≈ 1.4 GiB → 0.75 GiB.

### Documented-API verification

Every code block and stated guarantee in `README.md`, `docs/rust-api.md`, and
`docs/python-api.md` is now executed as a test, and the CLI is exercised over
every documented subcommand, flag, and error path.

- `tests/api_surface.rs` (new, 15 tests) — the Rust surface: constructors,
  `Pm3Options` fields and defaults, the `Pm3Result`/`GradientResult`/
  `VibrationalModes`/`OptResult` field lists, unit conventions, `Variant::parse`,
  reference selection, the memory budgets, and the documented error variants.
- `tests/test_python_api.py` (new, 19 tests) — `pm3_rs.native` dict keys and
  Hartree↔eV / Bohr↔Å conversions, the `pm3_rs.ase.PM3` calculator (ASE units,
  lazy Hessian, every accessor), and both documented example blocks.

Fixed along the way:

- **`pm3_rs.ase.PM3`**: every accessor (`get_potential_energy`, `get_forces`,
  `get_gradient`, `get_hessian`, `get_frequencies`) raised
  `AttributeError: 'NoneType' object has no attribute 'get_atomic_numbers'` when
  called with no argument before the calculator had been bound to a structure.
  They now raise a `RuntimeError` that names the three ways to fix it.
- `docs/python-api.md`: the cation example was labelled "forced UHF doublet" but
  passed `multiplicity=1` (NH4+ is a closed-shell singlet), and its
  `nh4_positions` was never defined, so the block could not be run as printed.
  Corrected, given a runnable geometry, and extended with an open-shell example.
- `docs/python-api.md`: documented what `atoms=None` means for the ASE
  accessors — assigning `atoms.calc = PM3()` does not bind the structure.

### Documentation

- Removed the claims that this crate is derived from or structured after sibling
  Rust projects; the PM3 Hamiltonian, parameters, and derivations are documented
  against MOPAC v23.2.5 and the primary literature.
- `third_party/pyseqm/NOTICE` added — `THIRD_PARTY_NOTICES.md` referenced a
  `third_party/pyseqm/LICENSE` that was not present.
- Corrected the MOPAC reference values quoted in the `scf.rs` unit-test comments
  for water and the methyl radical; they did not match the asserted PM3 values.
- `docs/rust-api.md`: documented the three memory budgets and corrected the
  description of the Hessian `step` argument.

## 0.1.1

Initial `pm3-rs` release, backed by the PM3 Hamiltonian and the MOPAC v23.2.5
PM3 parameter tables.

- Rust implementation; linear algebra via `faer`, no BLAS/LAPACK dependency.
- RHF/UHF energies, heats of formation, charges, dipoles, analytic gradients,
  CPHF/UCPHF Hessians, optimization, and frequencies.
- PM3-D3, PM3-D3H4, and PM3-D3H4X correction variants.
- MOPAC special atoms `Cb`, `+`, `-`, and La-Lu trivalent Sparkles.
- Rust library `pm3_rs`, CLI `pm3_rs_cli`, and Python distribution
  `pm3-rs-python` with native and ASE APIs.
- MOPAC v23.2.5 oracle regressions for closed-shell molecules, an open-shell
  radical, a Sparkle complex, point charges, gradients, optimization, and
  frequencies.
- Configurable Hessian workspace budget and bounded integral-cache construction.
- Release profile uses fat LTO with one code-generation unit.
