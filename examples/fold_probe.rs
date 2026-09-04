// SPDX-License-Identifier: GPL-3.0-or-later

//! Which comparison the supercell force constants should be held to.
//!
//! ```text
//! cargo run --release --example fold_probe
//! ```
//!
//! A supercell's Γ point *is* a mesh of the primitive cell: an `n×1×1` supercell resolves the
//! primitive's `n` wavevectors along that axis. So its Hessian carries a response sampled on that
//! mesh, and comparing it against a Γ-only primitive response tests the sampling rather than the
//! transform. This prints both comparisons, at two cell widths, so the right identity is chosen
//! from evidence rather than from argument.

use pm3_rs::pbc::gamma::PeriodicOptions;
use pm3_rs::{Cell, ForceConstants, Molecule, Pm3Options, Pm3Parameters, Vec3};

fn chain(axis_bohr: f64) -> Molecule {
    let mut molecule = Molecule::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n",
        0.0,
    )
    .unwrap();
    molecule.cell = Some(
        Cell::new(
            Vec3::new(axis_bohr, 0.0, 0.0),
            Vec3::new(0.0, 24.0, 0.0),
            Vec3::new(0.0, 0.0, 24.0),
            [true, false, false],
        )
        .unwrap(),
    );
    molecule
}

fn worst(a: &[f64], b: &[f64]) -> f64 {
    let mut x = a.to_vec();
    let mut y = b.to_vec();
    x.sort_by(|p, q| p.partial_cmp(q).unwrap());
    y.sort_by(|p, q| p.partial_cmp(q).unwrap());
    x.iter()
        .zip(&y)
        .fold(0.0_f64, |m, (p, q)| m.max((p - q).abs()))
}

fn main() {
    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options::default();
    let periodic = PeriodicOptions::default();

    for axis in [6.0, 9.0, 12.0, 16.0] {
        let molecule = chain(axis);
        let margin = pm3_rs::run_gamma(&molecule, &params, &options, &periodic)
            .map(|r| r.gamma_margin)
            .unwrap_or(f64::NAN);
        let constants =
            ForceConstants::from_supercell(&molecule, &params, &options, &periodic, [2, 1, 1])
                .unwrap();

        println!("=== chain repeat {axis} Bohr (Gamma margin {margin:.2}) ===");
        println!(
            "{:>10} {:>18} {:>18}",
            "q", "vs Gamma DFPT", "vs 2x1x1-mesh DFPT"
        );
        for q in constants.commensurate_q() {
            let supercell = constants.frequencies(q).unwrap();
            let gamma =
                pm3_rs::phonon_frequencies(&molecule, &params, &options, &periodic, q).unwrap();
            let mesh = pm3_rs::phonon_frequencies_on_mesh(
                &molecule,
                &params,
                &options,
                &periodic,
                &pm3_rs::KpointOptions::mesh([2, 1, 1]),
                q,
            )
            .unwrap();
            println!(
                "{:>10.3} {:>18.3} {:>18.3}",
                q[0],
                worst(&supercell, &gamma),
                worst(&supercell, &mesh)
            );
        }
        println!();
    }
}
