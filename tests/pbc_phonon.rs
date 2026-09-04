// SPDX-License-Identifier: GPL-3.0-or-later

//! Supercell force constants, and the identity that ties them to the direct response.
//!
//! The load-bearing test is [`the_supercell_reproduces_the_direct_response_at_commensurate_q`].
//! The two routes to `D(q)` share the SCF and nothing else — one displaces atoms in a supercell
//! and Fourier transforms, the other solves a coupled-perturbed response in a primitive cell — so
//! agreement at the wavevectors both can represent is a real check on both.
//!
//! Everything else here is a property of the transform: the acoustic modes at Γ, determinism,
//! and what the module refuses.

use pm3_rs::pbc::gamma::PeriodicOptions;
use pm3_rs::{
    build_supercell, q_path, Cell, ForceConstants, Molecule, Pm3Options, Pm3Parameters, Vec3,
};

const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";

fn options() -> Pm3Options {
    Pm3Options::default()
}

/// A chain of waters, periodic along `x` only — small enough that a supercell Hessian is cheap.
fn chain(axis_bohr: f64) -> Molecule {
    let mut molecule = Molecule::from_xyz_str(WATER, 0.0).unwrap();
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

/// The supercell reproduces the direct response at the `q` it can represent.
///
/// # Which direct response
///
/// The **mesh** one, with the mesh the supercell corresponds to. A supercell's Γ point *is* a
/// mesh of the primitive cell — an `n×1×1` supercell resolves the primitive's `n` wavevectors
/// along that axis — so its Hessian carries a response sampled on that mesh. Holding it against a
/// Γ-only primitive response would be testing the sampling, not the transform.
///
/// That was measured rather than assumed. On this chain at four widths:
///
/// | repeat | Γ margin | vs Γ-sampled | vs `2×1×1`-sampled |
/// |---|---|---|---|
/// | 6 Bohr | −8.0 | 772.9 | 324.4 |
/// | 12 Bohr | −2.0 | 16.2 | 217.5 |
/// | 16 Bohr | **+2.0** | 13.6 | **0.011** |
///
/// Both comparisons are meaningless while the Γ margin is negative, because then neither
/// sampling is adequate and the disagreement is the model's, not the transform's. Once it turns
/// positive the mesh comparison collapses to a hundredth of a wavenumber and the Γ one does not —
/// which is the two routes agreeing exactly, and the reason the cell here is 16 Bohr.
///
/// Compared as sorted sets, so the result does not depend on the two routes ordering degenerate
/// modes the same way.
#[test]
fn the_supercell_reproduces_the_direct_response_at_commensurate_q() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = chain(16.0);

    // Non-vacuity: the identity is only meaningful where one k-point is enough for the primitive
    // cell, and this asserts that it is rather than hoping.
    let margin = pm3_rs::run_gamma(&molecule, &params, &options(), &periodic)
        .unwrap()
        .gamma_margin;
    assert!(
        margin > 0.0,
        "the Gamma margin is {margin:.2} Bohr; below zero neither sampling is adequate and this \
         comparison tests the model rather than the transform"
    );

    let constants =
        ForceConstants::from_supercell(&molecule, &params, &options(), &periodic, [2, 1, 1])
            .unwrap();
    assert_eq!(constants.supercell(), [2, 1, 1]);
    assert_eq!(constants.commensurate_q().len(), 2);

    let mesh = pm3_rs::KpointOptions::mesh([2, 1, 1]);
    for q in constants.commensurate_q() {
        let mut from_supercell = constants.frequencies(q).unwrap();
        let mut direct =
            pm3_rs::phonon_frequencies_on_mesh(&molecule, &params, &options(), &periodic, &mesh, q)
                .unwrap();
        from_supercell.sort_by(|a, b| a.partial_cmp(b).unwrap());
        direct.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let worst = from_supercell
            .iter()
            .zip(&direct)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            worst < 2.0,
            "at q = {q:?} the supercell gives {from_supercell:?} and the mesh-sampled response \
             {direct:?} (worst {worst:.4} cm^-1)"
        );
    }
}

/// Three acoustic modes go to zero at Γ, because translating the crystal costs nothing.
#[test]
fn the_acoustic_modes_vanish_at_gamma() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = chain(16.0);

    let mut constants =
        ForceConstants::from_supercell(&molecule, &params, &options(), &periodic, [2, 1, 1])
            .unwrap();

    // Reported before it is imposed: the residual is what the truncation threw away, and
    // flattening it without looking would hide a set of force constants that is simply wrong.
    let residual = constants.acoustic_sum_rule_residual();
    assert!(
        residual.is_finite(),
        "the sum-rule residual should be a number, got {residual}"
    );
    constants.enforce_acoustic_sum_rule();
    assert!(
        constants.acoustic_sum_rule_residual() < 1.0e-9,
        "imposing the rule should zero it"
    );

    let mut frequencies = constants.frequencies([0.0, 0.0, 0.0]).unwrap();
    frequencies.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
    for value in &frequencies[..3] {
        assert!(
            value.abs() < 30.0,
            "an acoustic mode came out at {value} cm^-1; the three lowest are {:?}",
            &frequencies[..3]
        );
    }
}

/// The same force constants give the same numbers, bit for bit, every time.
///
/// A Fourier sum whose accumulation order varies between runs moves the last digits, and in a
/// near-degenerate mode that is not a last-digit effect. The blocks are kept in a sorted vector
/// rather than a hash map for exactly this reason, and this is the test that would notice if that
/// changed.
#[test]
fn the_transform_is_bit_reproducible() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = chain(16.0);
    let constants =
        ForceConstants::from_supercell(&molecule, &params, &options(), &periodic, [2, 1, 1])
            .unwrap();

    let q = [0.3, 0.0, 0.0];
    let first = constants.frequencies(q).unwrap();
    for _ in 0..4 {
        let again = constants.frequencies(q).unwrap();
        assert_eq!(
            first, again,
            "the same transform gave two different answers"
        );
    }
}

/// A band structure is a path of transforms, and costs one Hessian for all of it.
#[test]
fn a_band_structure_disperses() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = chain(16.0);
    let constants =
        ForceConstants::from_supercell(&molecule, &params, &options(), &periodic, [2, 1, 1])
            .unwrap();

    let path = q_path(&[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], 8);
    assert_eq!(path.len(), 9, "eight per segment plus the final corner");
    let bands = constants.band_structure(&path).unwrap();
    assert_eq!(bands.len(), path.len());
    assert!(bands
        .iter()
        .all(|row| row.len() == 3 * molecule.atoms.len()));

    // Something has to move along the path, or the "dispersion" is a flat line.
    let mut moved = 0.0_f64;
    for branch in 0..bands[0].len() {
        let values: Vec<f64> = bands.iter().map(|row| row[branch]).collect();
        let low = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let high = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        moved = moved.max(high - low);
    }
    assert!(
        moved > 1.0,
        "no branch disperses by more than {moved:.3} cm^-1"
    );
}

/// Replication along a non-periodic axis is refused rather than stacking copies.
#[test]
fn what_the_supercell_refuses() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = chain(16.0);

    let error = build_supercell(&molecule, [2, 2, 1])
        .expect_err("axis 1 is not periodic, so there is nothing to replicate along");
    assert!(error.to_string().contains("not periodic"), "{error}");

    let error =
        build_supercell(&molecule, [0, 1, 1]).expect_err("a division of zero is not a supercell");
    assert!(error.to_string().contains("zero"), "{error}");

    let isolated = Molecule::from_xyz_str(WATER, 0.0).unwrap();
    let error =
        ForceConstants::from_supercell(&isolated, &params, &options(), &periodic, [2, 1, 1])
            .expect_err("force constants need a lattice");
    assert!(error.to_string().contains("cell"), "{error}");

    // And a legitimate replication really does multiply the atoms and the lattice vector.
    let tripled = build_supercell(&molecule, [3, 1, 1]).unwrap();
    assert_eq!(tripled.atoms.len(), 3 * molecule.atoms.len());
    let original = molecule.cell.unwrap().vector(0).x;
    assert!((tripled.cell.unwrap().vector(0).x - 3.0 * original).abs() < 1.0e-12);
    // The non-periodic axes are untouched, which is what makes the refusal above the right call
    // rather than a missing feature.
    assert!((tripled.cell.unwrap().vector(1).y - molecule.cell.unwrap().vector(1).y).abs() < 1e-12);
}
