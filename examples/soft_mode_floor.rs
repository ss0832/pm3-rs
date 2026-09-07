// SPDX-License-Identifier: GPL-3.0-or-later

//! What projecting the acoustic branch out does to the mass-weighted Γ eigenvalues.
//!
//! ```text
//! cargo run --release --example soft_mode_floor
//! ```
//!
//! `static_dielectric_tensor` weights each mode by `1/ω²`, so it must exclude the acoustic branch.
//! Through 0.2.3 it did that with a floor — `ω² ≤ 1e-6` — because an acoustic mode that lands a
//! hair *above* zero contributes an enormous term rather than none, and testing `ω² ≤ 0` would
//! let it through whenever the arithmetic noise came out positive. The floor was chosen from a
//! measured gap, and the gap was wide: this program is what measured it.
//!
//! It is gone now. The three translations are exact null vectors of `D(0)`, so they are projected
//! out before diagonalization and land at **exactly** zero rather than at `±1e-15`; the sum then
//! tests `ω² ≤ 0` and no magnitude is involved. This prints both columns — before and after — so
//! the change is visible rather than asserted, and so the gap is still on the record.
//!
//! The reason it matters is not the acoustic branch, which was never in danger. It is the other
//! side of the floor: a genuinely soft mode in a cell near a phase transition is a real mode with
//! a small `ω²`, and a constant chosen on water in a box has no business deciding it is noise.

use pm3_rs::pbc::gamma::PeriodicOptions;
use pm3_rs::{Cell, Molecule, Pm3Options, Pm3Parameters, Vec3};

const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";
const ANGSTROM_TO_BOHR: f64 = 1.8897261254578281;

fn boxed(angstrom: f64) -> Molecule {
    let mut molecule = Molecule::from_xyz_str(WATER, 0.0).unwrap();
    let edge = angstrom * ANGSTROM_TO_BOHR;
    molecule.cell = Some(
        Cell::new(
            Vec3::new(edge, 0.0, 0.0),
            Vec3::new(0.0, edge, 0.0),
            Vec3::new(0.0, 0.0, edge),
            [true, true, true],
        )
        .unwrap(),
    );
    molecule
}

fn main() {
    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options::default();
    let periodic = PeriodicOptions::default();

    println!("mass-weighted Gamma eigenvalues, three lowest, eV/(A^2 amu)");
    println!("`skipped` is what static_dielectric_tensor leaves out of the 1/w^2 sum.");
    println!();
    println!(
        "{:>10}  {:>7}  {:>32}  {:>32}",
        "cell (A)", "skipped", "unprojected", "acoustic projected out"
    );

    let mut widths: Vec<f64> = vec![6.0, 9.0, 12.0];
    // A fine scan across the width where a mode appeared at -1.1e-1 and vanished 0.0015 A later.
    // A real instability turns on gradually; a discontinuity that size over that interval is a
    // cutoff crossing an image shell, which is a bug rather than physics.
    for step in 0..25 {
        widths.push(7.400 + 0.001 * step as f64);
    }
    widths.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for angstrom in widths {
        let molecule = boxed(angstrom);
        let dynamical =
            pm3_rs::dynamical_matrix(&molecule, &params, &options, &periodic, [0.0; 3]).unwrap();
        let ndof = dynamical.matrix.rows;

        // The same mass weighting `static_dielectric_tensor` applies, so these are the numbers it
        // compares against the floor rather than something merely proportional to them.
        let mut weighted = dynamical.matrix.clone();
        for row in 0..ndof {
            for col in 0..ndof {
                let scale = (dynamical.masses[row / 3] * dynamical.masses[col / 3]).sqrt();
                if scale > 0.0 {
                    weighted[(row, col)] /= scale;
                }
            }
        }
        let (bare, _) = pm3_rs::cmatrix::hermitian_eigen(&weighted).unwrap();
        let mut before: Vec<f64> = (0..ndof).map(|i| bare[i]).collect();
        before.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // The same projection `static_dielectric_tensor` applies, so this column is what it
        // actually sums over rather than something merely similar.
        let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
        let acoustic = pm3_rs::rigid::rigid_body_basis(
            &positions,
            &dynamical.masses,
            pm3_rs::rigid::RigidMotions::TranslationsOnly,
        );
        pm3_rs::rigid::project_out_hermitian(&acoustic, &mut weighted);
        let (projected, vectors) = pm3_rs::cmatrix::hermitian_eigen(&weighted).unwrap();
        let mut after: Vec<f64> = (0..ndof).map(|i| projected[i]).collect();
        for index in pm3_rs::rigid::rigid_mode_indices_complex(&acoustic, &vectors) {
            after[index] = 0.0;
        }
        after.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let skipped = after.iter().filter(|v| **v <= 0.0).count();
        let show = |values: &[f64]| {
            values
                .iter()
                .take(3)
                .map(|v| format!("{v:+.3e}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        println!(
            "{angstrom:>10.5}  {skipped:>7}  {:>32}  {:>32}",
            show(&before),
            show(&after)
        );
    }
}
