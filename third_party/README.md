# Third-party attribution index

`pm3-rs` is GPL-3.0-or-later. Five upstream sets contribute to it, in ways that
differ enough to be worth separating: one supplies parameters *and* equations
*and* a validation oracle, two supply parameters, one supplies equations only,
and one supplies **code that is compiled into everything this project ships**.
This table says which is which, so a reader does not have to infer the
obligation from the presence of a file.

**No third-party source code is vendored in this repository.** Every Rust file
here is original. What crosses the boundary *in the source tree* is published
numeric parameters and published working equations.

The binaries are the other case, and the distinction is the one that matters for
what is owed. `_native.pyd` and the `pm3-rs` executable are statically linked, so
the whole Rust dependency graph — around sixty crates — is compiled into them.
Nothing scientific comes from those crates and all of their code ships, which is
the exact inverse of every other row.

| Directory | Upstream | License | What `pm3-rs` takes | Where it lands |
|---|---|---|---|---|
| [`mopac/`](mopac/NOTICE) | [openmopac/mopac](https://github.com/openmopac/mopac) v23.2.5 | Apache-2.0 ([text](mopac/LICENSE)) | PM3 element/pair/Sparkle parameters; NDDO and core-core reference equations; the executable validation oracle | `src/data/*.csv`, `src/params.rs`, `src/integrals.rs`, `src/repulsion.rs`, `tools/oracle/` |
| [`pyseqm/`](pyseqm/NOTICE) | [lanl/PYSEQM](https://github.com/lanl/PYSEQM) | BSD-3-Clause | Closed-form s/p two-center integral and diatomic rotation equations | `src/integrals.rs`, `src/overlap.rs`, `src/rotations.rs` |
| [`dftd3/`](dftd3/NOTICE) | Grimme D3, via MOPAC | Apache-2.0 (as distributed) | C6 reference table, `r0ab` cutoff radii, `r2r4`/`rcov` per element | `src/data/d3_*.csv`, `src/corrections/d3.rs` |
| [`h_bonds4/`](h_bonds4/NOTICE) | Řezáč & Hobza H4/H-H | see NOTICE | PM3-specific H4 and H-H repulsion coefficients | `src/corrections/h4.rs` |
| [`rust-crates/`](rust-crates/NOTICE) | `faer`, `rayon`, `pyo3` and their transitive closure (~60 crates) | mostly MIT; also Apache-2.0, BSD-2-Clause, Zlib, Unicode-3.0, Apache-2.0-WITH-LLVM-exception ([texts](rust-crates/LICENSES.txt)) | linear algebra, parallelism, the Python bindings — **object code**, statically linked | `_native.pyd`, the `pm3-rs` executable |

Each directory's `NOTICE` records what was taken, from which upstream file, and
under what terms. A `LICENSE` file sits beside it wherever the upstream terms
require the license text to travel with the material.

`rust-crates/` is generated rather than written: run
`python tools/collect_rust_notices.py` after changing a dependency.
`tests/attribution.rs` fails if the recorded graph and the real one disagree, so
adding a crate without regenerating is a test failure rather than a silent
omission.

## Why `pyseqm/` has no `LICENSE` file

BSD-3-Clause requires the copyright notice and license text to be retained *in
redistributions of source or binary form*. No PySEQM source is redistributed
here — the Rust kernels are original code reproducing the same published NDDO
working equations — so there is nothing for the clause to attach to, and
vendoring a license text for material that is not present would misdescribe the
situation rather than clarify it.

`pyseqm/NOTICE` carries the citation and the standing instruction: if PySEQM
source is ever vendored, the verbatim upstream `LICENSE`, copyright line
included, must be copied to `third_party/pyseqm/LICENSE` in the same commit.

`mopac/` does carry a `LICENSE`, because parameter *data* extracted from MOPAC
is redistributed, and Apache-2.0 section 4 asks for the license to travel with
it.

## The scientific citations

Attribution here is a licensing record. The papers behind the model — Stewart's
two PM3 papers, Grimme's D3, Řezáč and Hobza's H4 and X, Zhou et al. on PySEQM —
are cited in [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) with DOIs,
and in the module documentation of the code that implements them.
