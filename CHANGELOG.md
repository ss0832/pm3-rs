# Changelog

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
