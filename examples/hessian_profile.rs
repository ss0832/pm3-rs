// SPDX-License-Identifier: GPL-3.0-or-later

//! **Where a Γ-point Hessian's time goes, against system size.**
//!
//! The Ewald finding in 0.2.4 came from measuring rather than reading, after three planned
//! optimizations had each been aimed at something that turned out to be under 1% of the run. The
//! Hessian is the next thing that is slow — a 64-atom supercell takes minutes where its SCF takes
//! seconds — so it gets the same treatment before anything is changed.
//!
//! Three phases, and reading the code does not settle which dominates:
//!
//! * the **skeleton**, second-order AD over every pair in the cutoff, `O(N_pairs)` with a large
//!   constant;
//! * the **Ewald and classical** second derivatives, one lattice sum each;
//! * the **response**, `3N` coupled-perturbed solves, each a fixed-point iteration over the
//!   occupied–virtual space — the term everyone assumes is the expensive one.
//!
//! ```text
//! cargo run --release --example hessian_profile
//! ```
//!
//! Runs with `PM3_HESSIAN_PROFILE=1` set internally, so the per-phase lines come from
//! `pbc::hessian` itself rather than from a stopwatch wrapped around the outside.

use std::time::Instant;

use pm3_rs::pbc::gamma::PeriodicOptions;
use pm3_rs::pbc::hessian::periodic_hessian;
use pm3_rs::{Atom, Cell, Molecule, Pm3Options, Pm3Parameters, Vec3};

const A: f64 = 1.8897261254578281;
/// Wide enough that the Γ margin is positive at every size below.
const EDGE_ANGSTROM: f64 = 18.0;

/// `copies` waters on a cubic grid, 3.5 Å apart — the same fixture `periodic_profile` uses, so
/// the two measurements are about the same systems.
fn grid(copies: usize) -> Molecule {
    const SPACING: f64 = 3.5;
    let per_side = (copies as f64).cbrt().ceil() as usize;
    let span = SPACING * per_side.saturating_sub(1) as f64;
    let origin = 0.5 * (EDGE_ANGSTROM - span);

    let mut atoms = Vec::with_capacity(copies * 3);
    for i in 0..copies {
        let x = origin + SPACING * (i % per_side) as f64;
        let y = origin + SPACING * ((i / per_side) % per_side) as f64;
        let z = origin + SPACING * (i / (per_side * per_side)) as f64;
        atoms.push(Atom {
            z: 8,
            position: Vec3::new(x, y, z) * A,
        });
        atoms.push(Atom {
            z: 1,
            position: Vec3::new(x + 0.9584, y, z) * A,
        });
        atoms.push(Atom {
            z: 1,
            position: Vec3::new(x - 0.24, y + 0.9278, z) * A,
        });
    }
    let edge = EDGE_ANGSTROM * A;
    let mut molecule = Molecule::new(atoms);
    molecule.cell = Some(
        Cell::new(
            Vec3::new(edge, 0.0, 0.0),
            Vec3::new(0.0, edge, 0.0),
            Vec3::new(0.0, 0.0, edge),
            [true; 3],
        )
        .expect("a cubic cell"),
    );
    molecule
}

fn main() {
    // The per-phase lines come from inside `pbc::hessian`, so the split is of the real call and
    // not of a re-implementation of it.
    std::env::set_var("PM3_HESSIAN_PROFILE", "1");

    let params = Pm3Parameters::standard().expect("parameters");
    let options = Pm3Options {
        max_scf: 400,
        ..Pm3Options::default()
    };
    let periodic = PeriodicOptions::default();

    println!("Γ-point Hessian cost against system size, in a {EDGE_ANGSTROM} A cube\n");
    println!(
        "  {:>7}  {:>6}  {:>10}  {:>10}",
        "waters", "atoms", "total (s)", "exponent"
    );

    let mut previous: Option<(f64, f64)> = None;
    for copies in [2usize, 4, 8, 12, 16] {
        let molecule = grid(copies);
        let atoms = molecule.atoms.len() as f64;
        eprintln!("--- {} atoms", atoms as usize);
        let started = Instant::now();
        match periodic_hessian(&molecule, &params, &options, &periodic) {
            Ok(_) => {
                let seconds = started.elapsed().as_secs_f64();
                let exponent = previous.map(|(n0, t0)| (seconds / t0).ln() / (atoms / n0).ln());
                println!(
                    "  {:>7}  {:>6}  {:>10.2}  {:>10}",
                    copies,
                    atoms as usize,
                    seconds,
                    exponent.map_or("--".to_string(), |e| format!("{e:.2}")),
                );
                previous = Some((atoms, seconds));
            }
            Err(error) => {
                println!(
                    "  {:>7}  {:>6}  failed after {:.0} s: {error}",
                    copies,
                    atoms as usize,
                    started.elapsed().as_secs_f64()
                );
            }
        }
    }
    println!(
        "\nThe per-phase lines above each row are from `pbc::hessian` itself. An exponent near 2\n\
         is pair work; near 3 is the occupied-virtual algebra of the response."
    );
}
