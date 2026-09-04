// SPDX-License-Identifier: GPL-3.0-or-later
//! Dump `erf`, `erfc` and `erfcx` on a grid, one `x erf erfc erfcx` row per line.
//!
//! This exists so the error-function implementation can be re-validated against an
//! independent high-quality reference rather than only against its own identities:
//! `tools/oracle/check_erf.py` pipes this through `scipy.special` and reports the worst
//! relative disagreement. scipy is not a dependency of pm3-rs, which is why the comparison
//! lives outside the test suite.
//!
//!     cargo run --release --example erf_probe | python tools/oracle/check_erf.py

fn main() {
    let mut xs: Vec<f64> = (0..=140).map(|k| 0.05 * k as f64).collect();
    xs.extend([
        1e-8, 1e-4, 0.5, 1.5, 1.85, 2.0, 2.5, 3.5, 3.9, 4.0, 4.1, 5.0, 8.0, 12.0, 20.0, 26.0,
    ]);
    for x in xs {
        println!(
            "{:.17e} {:.17e} {:.17e} {:.17e}",
            x,
            pm3_rs::special::erf(x),
            pm3_rs::special::erfc(x),
            pm3_rs::special::erfcx(x)
        );
    }
}
