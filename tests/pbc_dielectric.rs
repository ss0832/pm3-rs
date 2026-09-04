// SPDX-License-Identifier: GPL-3.0-or-later

//! Polarizability and the electronic dielectric tensor.
//!
//! The load-bearing test here is [`a_molecule_in_a_box_has_its_own_finite_field_polarizability`].
//! Everything else — symmetry, positive-definiteness, origin independence — is a *structural*
//! property that survives a wrong overall factor, so none of it can catch the one mistake this
//! quantity is most prone to: the response is solved in eV and Bohr and the answer is wanted in
//! atomic units, and dropping the Hartree-per-eV conversion leaves `ε∞ − 1` smaller by 27.21
//! while remaining symmetric, positive-definite and origin-independent.
//!
//! Comparing against a finite field on the isolated molecule shares nothing with the
//! coupled-perturbed path but the SCF, so it sees the factor immediately.

// A note on `clippy::needless_range_loop`, allowed below.
//
// The loop variables here are Cartesian directions (`alpha`, `beta`, `axis`), atom indices, or
// the rows and columns of a matrix being eliminated. The index *is* the meaning: `for alpha in
// 0..3` says which direction, where `for (alpha, row) in out.iter_mut().enumerate()` says it
// less clearly and no more safely. Several of these loops also index two different tensors by
// the same direction, which no single iterator expresses.
#![allow(clippy::needless_range_loop)]

use pm3_rs::pbc::dielectric::{dielectric_origin_sensitivity, dielectric_tensor, polarizability};
use pm3_rs::pbc::gamma::PeriodicOptions;
use pm3_rs::{Cell, Molecule, Pm3Options, Pm3Parameters, Vec3};

const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";

/// eV per Hartree, the conversion the response needs on its way out.
const HARTREE_TO_EV: f64 = 27.211_386_245_988;

fn options() -> Pm3Options {
    Pm3Options::default()
}

fn boxed(xyz: &str, edge_bohr: f64) -> Molecule {
    let mut molecule = Molecule::from_xyz_str(xyz, 0.0).unwrap();
    molecule.cell = Some(Cell::cubic(edge_bohr).unwrap());
    molecule
}

/// `α` from a finite field on the **isolated** molecule, in Bohr³.
///
/// Shares nothing with the periodic coupled-perturbed path except the SCF itself: this displaces
/// the Hamiltonian and re-converges, where the other solves a linear response.
fn finite_field_polarizability(xyz: &str) -> [[f64; 3]; 3] {
    let params = Pm3Parameters::standard().unwrap();
    let molecule = Molecule::from_xyz_str(xyz, 0.0).unwrap();
    // Small enough to stay linear, large enough to clear the SCF's own convergence noise.
    let step = 1.0e-4;
    let mut alpha = [[0.0_f64; 3]; 3];
    for beta in 0..3 {
        let dipole_at = |strength: f64| -> Vec3 {
            let mut field = [0.0; 3];
            field[beta] = strength;
            let opts = Pm3Options {
                field: Some(Vec3::new(field[0], field[1], field[2])),
                ..Pm3Options::default()
            };
            let scf = pm3_rs::run_pm3(&molecule, &params, &opts).unwrap();
            // Debye back to e·Bohr, the units the periodic path works in.
            scf.dipole_debye / 2.541_746_473
        };
        let plus = dipole_at(step);
        let minus = dipole_at(-step);
        for alpha_index in 0..3 {
            let d = (plus.to_array()[alpha_index] - minus.to_array()[alpha_index]) / (2.0 * step);
            // e·Bohr per (eV/Bohr) is e·Bohr²/eV; Bohr³ is per (Hartree/(e·Bohr)).
            alpha[alpha_index][beta] = d * HARTREE_TO_EV;
        }
    }
    alpha
}

/// A molecule in a large box has the polarizability it has on its own.
///
/// This is the test that pins the units. The two routes are a linear response under periodic
/// boundary conditions and a finite field on an isolated molecule; they agree only if the
/// conversion out of the response's internal eV/Bohr is right.
#[test]
fn a_molecule_in_a_box_has_its_own_finite_field_polarizability() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    // Large enough that the images barely see each other; the residual is the water lattice's
    // own dipole–dipole interaction, which falls as 1/L³.
    let molecule = boxed(WATER, 20.0);

    let periodic_alpha = polarizability(&molecule, &params, &options(), &periodic).unwrap();
    let molecular_alpha = finite_field_polarizability(WATER);

    // Non-vacuity: water is genuinely polarizable, so agreement means something.
    let size = molecular_alpha
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        size > 1.0,
        "the finite-field polarizability is {size:.3e} Bohr^3, too small to test against"
    );

    let mut worst = 0.0_f64;
    for a in 0..3 {
        for b in 0..3 {
            worst = worst.max((periodic_alpha[a][b] - molecular_alpha[a][b]).abs());
        }
    }
    // Measured at 0.2% on this cell: the residual is the water lattice's own dipole–dipole
    // interaction at 20 Bohr, which falls as 1/L³. Three percent leaves room for that and for
    // platform arithmetic while staying four hundred times tighter than the 27.21 a dropped
    // Hartree conversion would cost.
    assert!(
        worst < 0.03 * size,
        "the periodic and finite-field polarizabilities differ by {worst:.4} Bohr^3 out of \
         {size:.4}; a ratio of {:.2} would be the Hartree conversion",
        molecular_alpha[0][0] / periodic_alpha[0][0].max(1e-30)
    );
}

/// `α` is symmetric and positive-definite, and `ε∞ ≥ 1`.
#[test]
fn the_dielectric_tensor_is_physical() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = boxed(WATER, 14.0);

    let tensors = dielectric_tensor(&molecule, &params, &options(), &periodic).unwrap();
    let alpha = tensors.polarizability;

    for a in 0..3 {
        for b in 0..3 {
            assert!(
                (alpha[a][b] - alpha[b][a]).abs() < 1.0e-6 * (1.0 + alpha[a][a].abs()),
                "alpha is not symmetric at ({a},{b}): {} vs {}",
                alpha[a][b],
                alpha[b][a]
            );
        }
        assert!(
            alpha[a][a] > 0.0,
            "a diagonal polarizability came out negative"
        );
        assert!(
            tensors.epsilon[a][a] > 1.0,
            "epsilon_infinity must exceed 1: {}",
            tensors.epsilon[a][a]
        );
    }
}

/// The answer does not move when the cell origin does.
///
/// The position operator used here is not a well-defined periodic operator, and the argument that
/// the *response* is nevertheless well defined is an argument. This measures it.
#[test]
fn the_polarizability_does_not_move_with_the_origin() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = boxed(WATER, 14.0);

    let moved = dielectric_origin_sensitivity(
        &molecule,
        &params,
        &options(),
        &periodic,
        Vec3::new(1.7, -0.9, 0.4),
    )
    .unwrap();
    let size = polarizability(&molecule, &params, &options(), &periodic)
        .unwrap()
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        moved < 1.0e-6 * size.max(1.0),
        "a 1.7 Bohr origin shift moved alpha by {moved:.3e} out of {size:.3e}"
    );
}

/// The static tensor is `ε∞` plus a positive lattice term, and it says what it left out.
#[test]
fn the_static_tensor_adds_a_positive_ionic_term() {
    use pm3_rs::static_dielectric_tensor;

    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = boxed(WATER, 12.0);

    let full = static_dielectric_tensor(&molecule, &params, &options(), &periodic).unwrap();

    // The electronic half is exactly what `dielectric_tensor` returns on its own.
    let electronic = dielectric_tensor(&molecule, &params, &options(), &periodic).unwrap();
    for a in 0..3 {
        for b in 0..3 {
            assert!((full.electronic[a][b] - electronic.epsilon[a][b]).abs() < 1e-12);
            assert!(
                (full.epsilon[a][b] - full.electronic[a][b] - full.ionic[a][b]).abs() < 1e-12,
                "the two halves must add up to the whole"
            );
            assert!(
                (full.ionic[a][b] - full.ionic[b][a]).abs() < 1e-8,
                "the ionic term is a sum of outer products and must be symmetric"
            );
        }
        // `Z̄ Z̄ / ω²` with `ω² > 0` is positive semi-definite, so the diagonal cannot be negative
        // and the static constant cannot fall below the electronic one.
        assert!(
            full.ionic[a][a] >= -1e-10,
            "a diagonal ionic contribution came out negative: {}",
            full.ionic[a][a]
        );
        assert!(full.epsilon[a][a] >= full.electronic[a][a] - 1e-10);
    }

    // Three acoustic modes are expected to be skipped. More than three means the geometry is not
    // a minimum, and the count is reported so that is visible rather than inferred from an odd
    // number. This water is not relaxed, so the assertion is that the count is *reported*, not
    // that it is three.
    assert!(
        full.skipped_modes >= 3,
        "at least the three acoustic modes should be skipped, got {}",
        full.skipped_modes
    );
    assert!(full.epsilon.iter().flatten().all(|v| v.is_finite()));
}

/// Lyddane–Sachs–Teller: `ε₀/ε∞` along a direction is the ratio of LO to TO frequencies squared.
///
/// `(q̂·ε₀·q̂)/(q̂·ε∞·q̂) = Π_m ω_LO,m² / ω_TO,m²`, stated along a direction rather than as a
/// determinant because these tensors are anisotropic.
///
/// # What this establishes, and what it does not
///
/// Not independent physics. LST is an algebraic consequence of the two constructions being
/// tested — the ionic sum in `ε₀` and the non-analytic term that raises the LO branch are built
/// from the same `Z*` and the same `ε∞` — so it *has* to hold if both are implemented correctly.
///
/// That is exactly why it is worth asserting. Both carry a `4π/Ω`, a Hartree-per-eV conversion
/// and a mass weighting, applied in different places and in different combinations; a wrong
/// factor in one and not the other breaks the identity immediately, and each formula on its own
/// has nothing to be checked against. It comes out at a quotient of `1.000000` over six optical
/// modes, so the tolerance is set where a real discrepancy would show rather than where a
/// plausible one might hide.
///
/// The Born charges and `ε∞` that both sides share are pinned separately — against a finite
/// difference of the cell dipole and against an isolated-molecule finite field respectively.
#[test]
fn lyddane_sachs_teller_holds_along_a_direction() {
    use pm3_rs::static_dielectric_tensor;

    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    // Relaxed first: LST is a statement about a minimum, and an imaginary mode has no `ω²` to
    // put in the product. Without this the test would be measuring an unrelaxed geometry.
    let molecule = boxed(WATER, 12.0);
    let relaxed = pm3_rs::relax(
        &molecule,
        &params,
        &options(),
        &periodic,
        &pm3_rs::PeriodicOptOptions {
            cell: pm3_rs::CellRelaxation::Fixed,
            max_iter: 300,
            ..Default::default()
        },
    )
    .unwrap()
    .molecule;

    let static_tensor = static_dielectric_tensor(&relaxed, &params, &options(), &periodic).unwrap();
    if static_tensor.skipped_modes > 3 {
        // An unconverged relaxation leaves a soft mode behind. Saying so beats asserting an
        // identity that cannot hold without it.
        eprintln!(
            "skipping: {} modes have omega^2 <= 0, so the structure is not a minimum",
            static_tensor.skipped_modes
        );
        return;
    }

    let born = pm3_rs::born_charges(&relaxed, &params, &options(), &periodic).unwrap();
    let epsilon = dielectric_tensor(&relaxed, &params, &options(), &periodic)
        .unwrap()
        .epsilon;

    let direction = Vec3::new(1.0, 0.0, 0.0);
    let unit = direction / direction.norm();
    let project = |t: [[f64; 3]; 3]| -> f64 {
        let q = unit.to_array();
        (0..3)
            .flat_map(|a| (0..3).map(move |b| (a, b)))
            .map(|(a, b)| q[a] * t[a][b] * q[b])
            .sum()
    };
    let ratio_from_tensors = project(static_tensor.epsilon) / project(epsilon);

    // Transverse: `D(0)` as computed, which is the transverse limit. Longitudinal: the same with
    // the non-analytic term along `q̂`.
    let transverse =
        pm3_rs::dynamical_matrix(&relaxed, &params, &options(), &periodic, [0.0; 3]).unwrap();
    let mut longitudinal = transverse.clone();
    pm3_rs::add_non_analytic(&mut longitudinal, &relaxed, direction, &born, epsilon).unwrap();

    let to = pm3_rs::frequencies_of(&transverse).unwrap();
    let lo = pm3_rs::frequencies_of(&longitudinal).unwrap();

    // Optical modes only: the three acoustic ones are zero in both and would divide by zero.
    let mut product = 1.0_f64;
    let mut counted = 0;
    for (t, l) in to.iter().zip(&lo) {
        if *t > 1.0 && *l > 1.0 {
            product *= (l / t) * (l / t);
            counted += 1;
        }
    }
    assert!(
        counted >= 3,
        "only {counted} optical modes to multiply over"
    );
    // Printed as well as asserted: this is the identity that ties the whole set together, and the
    // number it comes out at is worth seeing rather than only its pass/fail.
    eprintln!(
        "LST over {counted} optical modes: frequency product {product:.6}, \
         tensor ratio {ratio_from_tensors:.6}, quotient {:.6}",
        product / ratio_from_tensors
    );
    assert!(
        (product / ratio_from_tensors - 1.0).abs() < 1.0e-5,
        "Lyddane-Sachs-Teller: the frequency product gives {product:.6} and the tensors give \
         {ratio_from_tensors:.6}. These are two arrangements of the same `4*pi/Omega`, Hartree \
         conversion and mass weighting, so a discrepancy is a factor wrong in one of them."
    );
}

/// A slab or a chain is refused, with the alternative named.
///
/// `ε∞ = 1 + 4πα/Ω` needs a volume. A supercell's vacuum padding would supply a number, and it
/// would be a statement about the padding rather than about the material.
#[test]
fn a_low_dimensional_cell_is_refused_but_still_has_a_polarizability() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();

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

    let error = dielectric_tensor(&chain, &params, &options(), &periodic)
        .expect_err("a chain has a length, not a volume");
    let text = error.to_string();
    assert!(
        text.contains("polarizability"),
        "the refusal should name what to use instead: {text}"
    );

    // And the polarizability itself is available there.
    let alpha = polarizability(&chain, &params, &options(), &periodic).unwrap();
    assert!(alpha.iter().flat_map(|r| r.iter()).all(|v| v.is_finite()));
    assert!(alpha[0][0] > 0.0);
}
