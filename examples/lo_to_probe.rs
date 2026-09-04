// SPDX-License-Identifier: GPL-3.0-or-later

//! Check the non-analytic term's prefactor against the lattice sum that contains it.
//!
//! ```text
//! cargo run --release --example lo_to_probe
//! ```
//!
//! # Why the rigid-ion matrix is the one to check against
//!
//! `pbc::dfpt`'s *response* now runs on the microscopic kernel — the `G = 0` member is excluded,
//! because keeping it inside a self-consistent solve makes the acoustic sum rule diverge. The
//! **rigid-ion** matrix keeps it: there is no response there to amplify it, and charge neutrality
//! gives the fixed-charge lattice sum a finite `q → 0` limit.
//!
//! So the rigid-ion matrix contains exactly the macroscopic term, unscreened and with the bare
//! net charges. Its `q → 0` limit minus its `q = 0` value must equal
//!
//! ```text
//! (4π/Ω) (q̂·Q_a)(q̂·Q_b) / (q̂·1·q̂)
//! ```
//!
//! which is [`pm3_rs::non_analytic_term`] evaluated with `Z*_a = Q_a δ` and `ε = 1`. Nothing is
//! shared between the two but the SCF charges, so agreement pins the `4π/Ω` and the
//! Hartree-per-eV conversion — the two factors a closed form written twice cannot check.

// A note on `clippy::needless_range_loop`, allowed below.
//
// The loop variables here are Cartesian directions (`alpha`, `beta`, `axis`), atom indices, or
// the rows and columns of a matrix being eliminated. The index *is* the meaning: `for alpha in
// 0..3` says which direction, where `for (alpha, row) in out.iter_mut().enumerate()` says it
// less clearly and no more safely. Several of these loops also index two different tensors by
// the same direction, which no single iterator expresses.
#![allow(clippy::needless_range_loop)]

use pm3_rs::pbc::gamma::{run_gamma, PeriodicOptions};
use pm3_rs::{Cell, Molecule, Pm3Options, Pm3Parameters, Vec3};

fn main() {
    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options::default();
    let periodic = PeriodicOptions::default();

    let mut molecule = Molecule::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n",
        0.0,
    )
    .unwrap();
    molecule.cell = Some(Cell::cubic(10.0).unwrap());

    // Point charges: `Z*_a = Q_a δ_αβ`, and no electronic screening, so `ε = 1`.
    let scf = run_gamma(&molecule, &params, &options, &periodic).unwrap();
    let rigid_charges: Vec<[[f64; 3]; 3]> = scf
        .charges
        .iter()
        .map(|q| {
            let mut z = [[0.0; 3]; 3];
            for a in 0..3 {
                z[a][a] = *q;
            }
            z
        })
        .collect();
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let closed = pm3_rs::non_analytic_term(
        &molecule,
        Vec3::new(1.0, 0.0, 0.0),
        &rigid_charges,
        identity,
    )
    .unwrap();

    let at_zero =
        pm3_rs::rigid_ion_dynamical_matrix(&molecule, &params, &options, &periodic, [0.0; 3])
            .unwrap();

    println!("Mulliken charges: {:?}", round_all(&scf.charges));
    println!(
        "{:>9} {:>16} {:>16} {:>10}",
        "q (frac)", "D(q)-D(0) worst", "closed form", "ratio"
    );
    for fraction in [0.05, 0.025, 0.0125, 0.00625] {
        let d = pm3_rs::rigid_ion_dynamical_matrix(
            &molecule,
            &params,
            &options,
            &periodic,
            [fraction, 0.0, 0.0],
        )
        .unwrap();
        // Compare on the element where the closed form is largest, so the ratio is not dominated
        // by a near-zero entry.
        let ndof = d.matrix.rows;
        let mut best = (0usize, 0usize, 0.0_f64);
        for i in 0..ndof {
            for j in 0..ndof {
                if closed[(i, j)].abs() > best.2 {
                    best = (i, j, closed[(i, j)].abs());
                }
            }
        }
        let (i, j, _) = best;
        let measured = d.matrix[(i, j)].re - at_zero.matrix[(i, j)].re;
        println!(
            "{:>9.5} {:>16.6} {:>16.6} {:>10.4}",
            fraction,
            measured,
            closed[(i, j)],
            measured / closed[(i, j)]
        );
    }
}

fn round_all(values: &[f64]) -> Vec<f64> {
    values.iter().map(|v| (v * 1e5).round() / 1e5).collect()
}
