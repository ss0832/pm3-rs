// SPDX-License-Identifier: GPL-3.0-or-later

//! Where does a Born-charge run actually spend its time?
//!
//! ```text
//! cargo run --release --example born_profile
//! ```
//!
//! Two guesses have already been wrong. Replacing the `3N` coupled-perturbed solves with three by
//! the interchange theorem changed nothing, and hoisting the neighbour list out of the `3N` bare
//! perturbations changed nothing. So this decomposes the cost with public entry points that
//! differ in exactly one term each:
//!
//! | entry point | ground-state SCF | bare perturbations | solves |
//! |---|---|---|---|
//! | `run_gamma` | yes | no | 0 |
//! | `polarizability` | yes | no | 3 |
//! | `born_charges` | yes | `3N` | `3N` |
//!
//! Subtracting them in that order separates the SCF, the response solve, and whatever is left.

use std::time::Instant;

use pm3_rs::pbc::gamma::PeriodicOptions;
use pm3_rs::{Cell, Molecule, Pm3Options, Pm3Parameters, Vec3};

fn chain(n: usize) -> Molecule {
    let mut xyz = format!("{}\nwater chain\n", 3 * n);
    for i in 0..n {
        let x = 3.0 * i as f64;
        xyz.push_str(&format!("O {x} 0.0 0.0\n"));
        xyz.push_str(&format!("H {} 0.0 0.0\n", x + 0.96));
        xyz.push_str(&format!("H {} 0.93 0.0\n", x - 0.24));
    }
    let mut molecule = Molecule::from_xyz_str(&xyz, 0.0).unwrap();
    let edge = 3.0 * n as f64 * 1.8897261254578281 + 16.0;
    molecule.cell = Some(
        Cell::new(
            Vec3::new(edge, 0.0, 0.0),
            Vec3::new(0.0, 18.0, 0.0),
            Vec3::new(0.0, 0.0, 18.0),
            [true, true, true],
        )
        .unwrap(),
    );
    molecule
}

fn main() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let options = Pm3Options::default();

    println!(
        "{:>6} {:>9} {:>12} {:>10} {:>12} {:>12}",
        "atoms", "scf(s)", "alpha(s)", "born(s)", "per solve", "born - 3N*solve"
    );
    for n in 1..=4 {
        let molecule = chain(n);
        let nat = molecule.atoms.len();
        let ndof = 3 * nat;

        let t = Instant::now();
        pm3_rs::run_gamma(&molecule, &params, &options, &periodic).unwrap();
        let scf = t.elapsed().as_secs_f64();

        let t = Instant::now();
        pm3_rs::polarizability(&molecule, &params, &options, &periodic).unwrap();
        let alpha = t.elapsed().as_secs_f64();

        let t = Instant::now();
        pm3_rs::born_charges(&molecule, &params, &options, &periodic).unwrap();
        let born = t.elapsed().as_secs_f64();

        // `alpha` is one SCF plus three solves, so a solve costs `(alpha - scf)/3`.
        let per_solve = (alpha - scf) / 3.0;
        // What a Born run costs beyond its SCF and its `3N` solves. If this is most of the
        // total, the solves are not the target.
        let remainder = born - scf - ndof as f64 * per_solve;
        println!(
            "{nat:>6} {scf:>9.3} {alpha:>12.3} {born:>10.3} {per_solve:>12.4} {remainder:>12.3}"
        );
    }
}
