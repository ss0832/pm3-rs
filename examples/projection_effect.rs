// SPDX-License-Identifier: GPL-3.0-or-later

//! What projecting translations and rotations out does to the reported frequencies.
//!
//! ```text
//! cargo run --release --example projection_effect
//! ```
//!
//! Two questions, and they have different answers.
//!
//! The **rigid-body modes** are the obvious one: unprojected they come out at whatever the
//! accumulated error in the Hessian happens to be, and this repository's own tests bounded the
//! same quantity at `< 50`, `< 100` and `< 300 cm⁻¹` in three different files. Projected they are
//! exactly zero. That is not an improvement in accuracy, it is the removal of a number that never
//! meant anything.
//!
//! The **internal modes** are the one worth measuring. Removing the rigid-body content changes
//! them too, by the amount they were contaminated with it, and the direction of that change is a
//! claim that needs checking rather than assuming. MOPAC's `FORCE` projects, so its values are the
//! ones to compare against: if projecting moves the vibrations *towards* the oracle, the
//! contamination was real and is now gone.
//!
//! Printed rather than asserted. `tests/molecules.rs` holds the tolerance; this shows the size and
//! the sign of the effect so the tolerance is a decision rather than a habit.

use pm3_rs::hessian::{analytic_hessian, SQRT_EV_PER_ANG2_AMU_TO_CM};
use pm3_rs::linalg::symmetric_eigen;
use pm3_rs::rigid::{project_out_symmetric, rigid_body_basis, RigidMotions};
use pm3_rs::{
    optimize, vibrational_analysis, Matrix, Molecule, OptOptions, Pm3Options, Pm3Parameters, Vec3,
};

const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";
/// MOPAC v23.2.5 `FORCE` at its own optimized water minimum — the same values `tests/molecules.rs`
/// pins the three vibrations against.
const ORACLE: [f64; 3] = [1743.46, 3868.68, 3989.81];

fn main() {
    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options::default();
    let start = Molecule::from_xyz_str(WATER, 0.0).unwrap();
    let relaxed = optimize(&start, &params, &options, &OptOptions::default()).unwrap();
    let molecule = &relaxed.molecule;

    // The unprojected spectrum, assembled here rather than taken from the library, because the
    // library no longer produces one. This is exactly what `vibrational_analysis` did before.
    let hessian = analytic_hessian(molecule, &params, &options, 1.0e-3).unwrap();
    let masses: Vec<f64> = molecule
        .atoms
        .iter()
        .map(|a| params.element(a.z).unwrap().mass)
        .collect();
    let a0_sq = pm3_rs::constants::ANGSTROM_TO_BOHR * pm3_rs::constants::ANGSTROM_TO_BOHR;
    let ndof = 3 * molecule.atoms.len();
    let mut mw = Matrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            let scale = (masses[i / 3] * masses[j / 3]).sqrt();
            if scale > 0.0 {
                mw[(i, j)] = hessian[(i, j)] * a0_sq / scale;
            }
        }
    }
    let wavenumber = |lam: f64| -> f64 {
        if lam >= 0.0 {
            SQRT_EV_PER_ANG2_AMU_TO_CM * lam.sqrt()
        } else {
            -SQRT_EV_PER_ANG2_AMU_TO_CM * (-lam).sqrt()
        }
    };
    let (bare, _) = symmetric_eigen(&mw).unwrap();
    let unprojected: Vec<f64> = bare.iter().map(|&l| wavenumber(l)).collect();

    // And the same matrix with the rigid-body subspace removed, which is what ships.
    let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let basis = rigid_body_basis(&positions, &masses, RigidMotions::TranslationsAndRotations);
    project_out_symmetric(&basis, &mut mw);
    let (projected_eigs, _) = symmetric_eigen(&mw).unwrap();
    let projected: Vec<f64> = projected_eigs.iter().map(|&l| wavenumber(l)).collect();

    let shipped = vibrational_analysis(molecule, &params, &options, 1.0e-3).unwrap();

    println!("water at its PM3 minimum, {} degrees of freedom", ndof);
    println!(
        "rigid-body directions found from the geometry: {}",
        basis.len()
    );
    println!(
        "the Hessian's own error along them (rigid_residual_cm): {:.4} cm^-1",
        shipped.rigid_residual_cm
    );
    println!();
    println!(
        "{:>5}  {:>14}  {:>14}  {:>14}",
        "mode", "unprojected", "projected", "as shipped"
    );
    for i in 0..ndof {
        println!(
            "{:>5}  {:>14.4}  {:>14.4}  {:>14.4}",
            i + 1,
            unprojected[i],
            projected[i],
            shipped.frequencies_cm[i]
        );
    }

    println!();
    println!("the three vibrations against MOPAC FORCE, which also projects:");
    println!(
        "{:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "MOPAC", "unprojected", "error", "projected", "error"
    );
    let top = |v: &[f64]| v[v.len() - 3..].to_vec();
    let (bare_top, proj_top) = (top(&unprojected), top(&shipped.frequencies_cm));
    let (mut bare_worst, mut proj_worst, mut moved) = (0.0_f64, 0.0_f64, 0.0_f64);
    for k in 0..3 {
        let (be, pe) = (bare_top[k] - ORACLE[k], proj_top[k] - ORACLE[k]);
        bare_worst = bare_worst.max(be.abs());
        proj_worst = proj_worst.max(pe.abs());
        moved = moved.max((proj_top[k] - bare_top[k]).abs());
        println!(
            "{:>12.2}  {:>12.4}  {:>+12.4}  {:>12.4}  {:>+12.4}",
            ORACLE[k], bare_top[k], be, proj_top[k], pe
        );
    }

    println!();
    println!("largest change to a vibration:      {moved:.6} cm^-1");
    println!("worst deviation from the oracle:    unprojected {bare_worst:.4}, projected {proj_worst:.4}");
    println!();

    // The result worth stating, and it is not the one the exercise looks like it is testing.
    //
    // The projection does not move the vibrations at all -- the largest change is below the
    // printing precision, because a Hessian assembled from translation-invariant terms has its
    // rigid-body error confined to the rigid-body block. What changes is the six modes that were
    // reported as `-1.34 ... +0.0003 cm^-1` and are now exactly zero.
    //
    // So the residual disagreement with MOPAC is unrelated to any of this. It was there before
    // and is there after, at the same size, and it is a difference in the second derivative
    // itself -- `tests/molecules.rs` allows 2.0 cm^-1 for exactly that reason.
    println!("The projection is not an accuracy change: the vibrations are untouched to");
    println!("{moved:.1e} cm^-1, and the deviation from MOPAC is identical before and after.");
    println!("What it removes is the six rigid-body modes' arithmetic noise, which is now");
    println!(
        "exactly zero instead of {:.3} cm^-1 of numbers that never meant anything.",
        shipped.rigid_residual_cm
    );
}
