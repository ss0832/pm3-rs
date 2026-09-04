// SPDX-License-Identifier: GPL-3.0-or-later

//! Where the soft-mode floor sits between the acoustic branch and the softest real mode.
//!
//! ```text
//! cargo run --release --example soft_mode_floor
//! ```
//!
//! `static_dielectric_tensor` weights each mode by `1/ω²`, so it must exclude the acoustic branch
//! -- and cannot do it by testing `ω² ≤ 0`, because an acoustic mode that lands a hair above zero
//! then contributes an enormous term instead of none. The threshold is therefore a floor, and a
//! floor is only defensible if there is a gap to put it in. This prints the mass-weighted Γ
//! eigenvalues on both sides of it so the constant is chosen from the gap rather than guessed.

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

    println!("floor = {:e} (eV/(A^2 amu))", pm3_rs::SOFT_MODE_FLOOR);
    println!();
    println!(
        "{:>10}  {:>9}  mass-weighted Gamma eigenvalues, six lowest",
        "cell (A)", "skipped"
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
        let (eigenvalues, _) = pm3_rs::cmatrix::hermitian_eigen(&weighted).unwrap();
        let mut values: Vec<f64> = (0..ndof).map(|i| eigenvalues[i]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let skipped = values
            .iter()
            .filter(|v| **v <= pm3_rs::SOFT_MODE_FLOOR)
            .count();
        let text: Vec<String> = values.iter().take(6).map(|v| format!("{v:+.3e}")).collect();
        println!("{angstrom:>10.5}  {skipped:>9}  {}", text.join(" "));
    }
}
