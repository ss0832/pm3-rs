# Performance baseline — v0.1.2 code, before any v0.2.0 change

Machine: Windows 11, 16 logical cores. Release profile (`opt-level=3`, fat LTO, 1 codegen unit).
Command: `target/release/pm3_rs_cli <cmd> <file>`. Wall-clock, single run, warm filesystem.

Geometries: `examples/bench102.xyz` ships with the crate; the water clusters are generated
reproducibly by `tools/bench/water_cluster.py <n_molecules> [seed]` (default seed 20260826).

| case            | atoms | nao  | command  | seconds |
|-----------------|-------|------|----------|---------|
| bench102        |  102  |  ~   | energy   |  0.99   |
| water50         |  150  |  300 | energy   |  0.60   |
| water150        |  450  |  900 | energy   |  3.99   |
| water300        |  900  | 1800 | energy   | 26.64   |
| water150        |  450  |  900 | gradient |  4.44   |

Observed scaling between 450 and 900 atoms: 3.99 s -> 26.64 s, i.e. N^2.7 — the
O(N^3) diagonalization and O(N^2) Fock build dominating, as expected for the dense
molecular path. This is the number the divide-and-conquer work has to beat: the M7
acceptance criterion is a log-log slope <= 1.15 for the DC path over a decade of N.

Test-suite baseline at the same commit: 99 Rust tests (70 lib + 15 api_surface + 14
molecules) and 27 Python tests, all passing. Full `cargo test --release` takes 508 s,
which is why `[profile.quick]` was added for the development loop.