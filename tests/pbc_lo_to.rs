// SPDX-License-Identifier: GPL-3.0-or-later

//! The non-analytic term, and the one check that actually pins its coefficient.
//!
//! # Why this crate can check it and an interpolating one cannot
//!
//! A code that reaches `D(q)` by Fourier-transforming truncated real-space force constants has no
//! independent handle on the non-analytic term: the truncation is exactly what removes the
//! long-range tail, so the only thing available to compare the closed form against is the same
//! closed form written a second time. That checks the transcription and not the physics — a wrong
//! `4π/Ω`, a missing Hartree-per-eV conversion or a contraction over the wrong index of `Z*` all
//! survive it.
//!
//! `pbc::dfpt` computes `D(q)` at any wavevector from the phased lattice sum, so the macroscopic
//! term is *in* the rigid-ion matrix, unscreened and with the bare net charges. Its `q → 0` limit
//! is therefore something to measure the closed form against, and
//! [`the_prefactor_matches_the_lattice_sum_that_contains_it`] does.

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

const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";

fn options() -> Pm3Options {
    Pm3Options::default()
}

fn cube(edge_bohr: f64) -> Molecule {
    let mut molecule = Molecule::from_xyz_str(WATER, 0.0).unwrap();
    molecule.cell = Some(Cell::cubic(edge_bohr).unwrap());
    molecule
}

/// Point charges as Born tensors: `Z*_a = Q_a δ_αβ`, the rigid-ion limit.
fn point_charge_tensors(charges: &[f64]) -> Vec<[[f64; 3]; 3]> {
    charges
        .iter()
        .map(|q| {
            let mut z = [[0.0; 3]; 3];
            for a in 0..3 {
                z[a][a] = *q;
            }
            z
        })
        .collect()
}

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// The closed form is the `q → 0` limit of the lattice sum that contains it.
///
/// The rigid-ion matrix keeps the macroscopic (`G = 0`) member of the phased reciprocal sum — it
/// has no self-consistent response to amplify it, and charge neutrality gives the fixed-charge
/// sum a finite limit. So `D_rigid(q) − D_rigid(0)` approaches the non-analytic term built from
/// the bare charges with no screening, and the approach is `O(q²)` because everything else in the
/// matrix is analytic in `q`.
///
/// This is the test that pins `4π/Ω` and the Hartree-per-eV conversion. Nothing is shared between
/// the two sides but the converged charges.
#[test]
fn the_prefactor_matches_the_lattice_sum_that_contains_it() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = cube(10.0);

    let scf = run_gamma(&molecule, &params, &options(), &periodic).unwrap();
    let closed = pm3_rs::non_analytic_term(
        &molecule,
        Vec3::new(1.0, 0.0, 0.0),
        &point_charge_tensors(&scf.charges),
        IDENTITY,
    )
    .unwrap();

    // The element the closed form is largest on, so the ratio is not a near-zero quotient.
    let ndof = closed.rows;
    let mut best = (0usize, 0usize, 0.0_f64);
    for i in 0..ndof {
        for j in 0..ndof {
            if closed[(i, j)].abs() > best.2 {
                best = (i, j, closed[(i, j)].abs());
            }
        }
    }
    let (row, column, size) = best;
    assert!(
        size > 1.0e-4,
        "the non-analytic term is {size:.3e} here, too small to measure a ratio against"
    );

    let at_zero =
        pm3_rs::rigid_ion_dynamical_matrix(&molecule, &params, &options(), &periodic, [0.0; 3])
            .unwrap();

    // Two wavevectors, because the claim is about a *limit*: the ratio must approach one, and
    // approach it faster than the wavevector shrinks. A single point could be anything.
    let ratio_at = |fraction: f64| -> f64 {
        let d = pm3_rs::rigid_ion_dynamical_matrix(
            &molecule,
            &params,
            &options(),
            &periodic,
            [fraction, 0.0, 0.0],
        )
        .unwrap();
        (d.matrix[(row, column)].re - at_zero.matrix[(row, column)].re) / closed[(row, column)]
    };

    let coarse = ratio_at(0.0125);
    let fine = ratio_at(0.00625);
    assert!(
        (fine - 1.0).abs() < 0.05,
        "the lattice sum's macroscopic term is {fine:.4} of the closed form at q = 0.00625; a \
         missing Hartree conversion would put it at {:.4}",
        1.0 / 27.211_386
    );
    assert!(
        (fine - 1.0).abs() < (coarse - 1.0).abs(),
        "the ratio must converge as q shrinks: {coarse:.4} at 0.0125, {fine:.4} at 0.00625"
    );
}

/// A rank-one positive semi-definite update cannot lower an eigenvalue.
///
/// `D_NA = c · v v^T` with `c = 4π/(Ω q̂·ε·q̂) > 0`, so adding it can only raise the spectrum.
/// Asserted on the eigenvalues rather than on frequencies, because the square root's sign
/// convention for an imaginary mode reverses the ordering and would make a true statement look
/// false.
#[test]
fn the_non_analytic_term_only_raises_eigenvalues() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = cube(10.0);

    let born = pm3_rs::born_charges(&molecule, &params, &options(), &periodic).unwrap();
    let tensors = pm3_rs::dielectric_tensor(&molecule, &params, &options(), &periodic).unwrap();

    let mut with =
        pm3_rs::dynamical_matrix(&molecule, &params, &options(), &periodic, [0.0; 3]).unwrap();
    let without = with.clone();
    pm3_rs::add_non_analytic(
        &mut with,
        &molecule,
        Vec3::new(1.0, 0.0, 0.0),
        &born,
        tensors.epsilon,
    )
    .unwrap();

    let before = pm3_rs::frequencies_of(&without).unwrap();
    let after = pm3_rs::frequencies_of(&with).unwrap();
    // Frequencies are the signed square roots of the eigenvalues, and the map is monotone, so an
    // ordered comparison of them is an ordered comparison of the eigenvalues.
    //
    // The tolerance is the acoustic floor, not machine epsilon. Three modes at Γ are zero by
    // translation invariance and come out at a few times `1e-5 cm⁻¹`; the square root magnifies
    // the last bits of a near-zero eigenvalue and flips their sign freely. A thousandth of a
    // wavenumber is far below the splitting the next assertion requires (more than one) and far
    // above that noise.
    const ACOUSTIC_FLOOR_CM: f64 = 1.0e-3;
    for (b, a) in before.iter().zip(&after) {
        assert!(
            a >= &(b - ACOUSTIC_FLOOR_CM),
            "a positive semi-definite update lowered a mode: {b} -> {a}"
        );
    }
    assert!(
        after.iter().zip(&before).any(|(a, b)| a - b > 1.0e-6),
        "the term changed nothing; water in a 10 Bohr cube should show a splitting"
    );
}

/// The limit is direction dependent — which is the whole point of the term.
#[test]
fn the_limit_depends_on_the_direction_it_is_approached_from() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = cube(10.0);

    let born = pm3_rs::born_charges(&molecule, &params, &options(), &periodic).unwrap();
    let tensors = pm3_rs::dielectric_tensor(&molecule, &params, &options(), &periodic).unwrap();

    let along = |direction: Vec3| -> Vec<f64> {
        let mut d =
            pm3_rs::dynamical_matrix(&molecule, &params, &options(), &periodic, [0.0; 3]).unwrap();
        pm3_rs::add_non_analytic(&mut d, &molecule, direction, &born, tensors.epsilon).unwrap();
        pm3_rs::frequencies_of(&d).unwrap()
    };

    let x = along(Vec3::new(1.0, 0.0, 0.0));
    let y = along(Vec3::new(0.0, 1.0, 0.0));
    let moved = x
        .iter()
        .zip(&y)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        moved > 1.0,
        "the two directions give the same spectrum to {moved:.3e} cm^-1; there is no splitting \
         to speak of here"
    );
}

/// `q = 0` exactly is not a direction, and a slab or a chain has no splitting to add.
#[test]
fn what_the_term_refuses() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = cube(10.0);
    let born = pm3_rs::born_charges(&molecule, &params, &options(), &periodic).unwrap();

    let error = pm3_rs::non_analytic_term(&molecule, Vec3::zero(), &born, IDENTITY)
        .expect_err("the zero vector is not a direction");
    assert!(error.to_string().contains("direction"), "{error}");

    let mut chain = Molecule::from_xyz_str(WATER, 0.0).unwrap();
    chain.cell = Some(
        Cell::new(
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(0.0, 30.0, 0.0),
            Vec3::new(0.0, 0.0, 30.0),
            [true, false, false],
        )
        .unwrap(),
    );
    let error = pm3_rs::non_analytic_term(&chain, Vec3::new(1.0, 0.0, 0.0), &born, IDENTITY)
        .expect_err("LO-TO splitting is three-dimensional");
    assert!(error.to_string().contains("three-dimensional"), "{error}");

    // And a wrong number of tensors is caught rather than indexed past.
    let error =
        pm3_rs::non_analytic_term(&molecule, Vec3::new(1.0, 0.0, 0.0), &born[..1], IDENTITY)
            .expect_err("one tensor per atom");
    assert!(error.to_string().contains("Born-charge tensors"), "{error}");
}
