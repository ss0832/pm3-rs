// SPDX-License-Identifier: GPL-3.0-or-later

//! One `D(q)` at the wavevector given on the command line, for tracing.
//!
//! ```text
//! PM3_DFPT_CHARGE_TRACE=1 cargo run --release --example charge_at_q -- 0.05
//! ```
//!
//! Exists so that the induced-charge trace can be attributed to a single `q`: the probe that
//! sweeps several wavevectors in one process interleaves their traces and cannot be read.

use pm3_rs::pbc::gamma::PeriodicOptions;
use pm3_rs::{Cell, Molecule, Pm3Options, Pm3Parameters};

fn main() {
    let fraction: f64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(0.05);

    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options::default();
    let periodic = PeriodicOptions::default();
    let mut molecule = Molecule::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n",
        0.0,
    )
    .unwrap();
    molecule.cell = Some(Cell::cubic(10.0).unwrap());

    let d = pm3_rs::dynamical_matrix(
        &molecule,
        &params,
        &options,
        &periodic,
        [fraction, 0.0, 0.0],
    )
    .unwrap();
    let ndof = d.matrix.rows;
    let nat = ndof / 3;
    let mut worst = 0.0_f64;
    for row in 0..ndof {
        for beta in 0..3 {
            let mut total = faer::c64::new(0.0, 0.0);
            for atom in 0..nat {
                total += d.matrix[(row, 3 * atom + beta)];
            }
            worst = worst.max(total.norm());
        }
    }
    println!("q={fraction} acoustic-sum-rule residual {worst:.6e}");
}
