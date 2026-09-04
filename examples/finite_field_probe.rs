// SPDX-License-Identifier: GPL-3.0-or-later

//! Does the Berry-phase finite field reproduce the CPHF polarizability?
//!
//! ```text
//! cargo run --release --example finite_field_probe
//! ```
//!
//! `α = Ω ∂P/∂𝓔` by central differences of `run_finite_field`, against
//! `pbc::dielectric::polarizability`. The two share the SCF and nothing else, so a factor of two,
//! a missing `J`, or a sign in the field operator shows up here and in no other test.
//!
//! Units: `P` is `e/Bohr²`, `Ω` is `Bohr³`, `𝓔` is `eV/(e·Bohr)`, so `Ω ∂P/∂𝓔` comes out in
//! `e²Bohr²/eV`. Multiplying by `HARTREE_TO_EV` gives `e²Bohr²/Hartree = Bohr³`, since
//! `e²/Hartree = Bohr` — the same conversion `dielectric.rs` applies for the same reason.

use pm3_rs::pbc::gamma::PeriodicOptions;
use pm3_rs::{Cell, FiniteFieldOptions, Molecule, Pm3Options, Pm3Parameters, Vec3};

const HARTREE_TO_EV: f64 = 27.211386245988;
const HF: &str = "2\nhydrogen fluoride\nF 0.0 0.0 0.0\nH 0.93 0.0 0.0\n";
/// Hydrogen has no `p` orbitals, so `dd` is zero for every atom here and the two routes carry
/// exactly the same position operator with no switch needed.
const H2: &str = "2\nhydrogen\nH 0.0 0.0 0.0\nH 0.74 0.0 0.0\n";

fn boxed(xyz: &str, edge: f64) -> Molecule {
    let mut molecule = Molecule::from_xyz_str(xyz, 0.0).unwrap();
    molecule.cell = Some(Cell::cubic(edge).unwrap());
    molecule
}

fn main() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let options = Pm3Options {
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 600,
        ..Pm3Options::default()
    };
    let ff = FiniteFieldOptions::default();
    let strength = 2.0e-4;

    // The CPHF route is a Gamma-point response and the finite field needs at least three points
    // along the field axis, so the two cannot be made to sample the same Brillouin zone. If the
    // gap between them is that sampling, it has to close as the cell grows and the bands flatten;
    // if it is a factor in the field operator, it will not move.
    println!(
        "{:>10} {:>8} {:>14} {:>14} {:>8}",
        "edge(Bohr)", "mesh", "finite field", "CPHF (Gamma)", "ratio"
    );
    for (label, xyz) in [("H2 (dd = 0)", H2), ("HF (dd on F)", HF)] {
        println!("--- {label} ---");
        for edge in [12.0, 16.0, 20.0, 26.0, 32.0] {
            let molecule = boxed(xyz, edge);
            let volume = molecule.cell.unwrap().measure();
            let cphf = match pm3_rs::polarizability(&molecule, &params, &options, &periodic) {
                Ok(a) => a,
                Err(e) => {
                    println!("{edge:>10.1} CPHF failed: {e}");
                    continue;
                }
            };
            for divisions in [[4, 1, 1], [8, 1, 1]] {
                let at = |sign: f64| {
                    let field = Vec3::new(sign * strength, 0.0, 0.0);
                    pm3_rs::run_finite_field(
                        &molecule, &params, &options, &periodic, divisions, field, &ff,
                    )
                };
                match (at(-1.0), at(1.0)) {
                    (Ok(m), Ok(p)) => {
                        // Differenced through the polarization directly: the two states differ by a
                        // field this small, so the branch cannot have changed between them.
                        let dp = (p.polarization - m.polarization) / (2.0 * strength);
                        let alpha_xx = dp.x * volume * HARTREE_TO_EV;
                        println!(
                            "{edge:>10.1} {:>8} {alpha_xx:>14.5} {:>14.5} {:>8.4}",
                            format!("{}x1x1", divisions[0]),
                            cphf[0][0],
                            alpha_xx / cphf[0][0]
                        );
                    }
                    (Err(e), _) | (_, Err(e)) => println!("{edge:>10.1} {divisions:?}: {e}"),
                }
            }
        }
        println!();
    }
}
