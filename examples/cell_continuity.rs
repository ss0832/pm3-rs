// SPDX-License-Identifier: GPL-3.0-or-later

//! Is the periodic energy a continuous function of the cell edge?
//!
//! ```text
//! cargo run --release --example cell_continuity
//! ```
//!
//! Not entirely. There is one cliff, and it is worth knowing exactly where it is.
//!
//! The Γ-point condition is that every periodic width exceed
//! [`pm3_rs::pbc::gamma::DEFAULT_SHORT_RANGE_CUTOFF`], which is 14 Bohr. Crossing it does not
//! degrade the answer gradually -- it switches which images the resonance sum admits, and the
//! energy of this water box steps by **35 eV** between 7.4080 and 7.4085 Angstrom (14.0 Bohr),
//! with the slope going from `+5.27` to `+0.45` eV/Angstrom. Both branches are smooth; they are
//! simply different functions, and only the upper one is the Γ-point approximation being used
//! inside its validity condition.
//!
//! This is the documented behaviour rather than a defect -- `pbc/gamma.rs` says the SCF
//! "converges cleanly to a well-defined wrong answer" below the margin, and `gamma_margin` is
//! reported everywhere for this reason. The point of keeping this example is that the size of
//! the step is not obvious from that sentence: a cell chosen at the boundary, or a variable-cell
//! relaxation that wanders across it, does not produce a slightly worse number.
//!
//! Prints the energy against cell edge on a fine grid, with the first and second differences, so
//! the step and the two smooth branches either side of it are visible at once.

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

    let step = 0.0005;
    let widths: Vec<f64> = (0..40).map(|i| 7.400 + step * i as f64).collect();

    let energies: Vec<f64> = widths
        .iter()
        .map(|a| {
            pm3_rs::run_gamma(&boxed(*a), &params, &options, &periodic)
                .map(|r| r.total_ev)
                .unwrap_or(f64::NAN)
        })
        .collect();

    println!(
        "{:>10} {:>18} {:>14} {:>14}",
        "cell (A)", "energy (eV)", "d/dA", "d2/dA2"
    );
    for i in 0..widths.len() {
        let first = if i == 0 {
            f64::NAN
        } else {
            (energies[i] - energies[i - 1]) / step
        };
        let second = if i == 0 || i + 1 == widths.len() {
            f64::NAN
        } else {
            (energies[i + 1] - 2.0 * energies[i] + energies[i - 1]) / (step * step)
        };
        println!(
            "{:>10.4} {:>18.10} {:>14.5} {:>14.3e}",
            widths[i], energies[i], first, second
        );
    }
}
