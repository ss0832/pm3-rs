// SPDX-License-Identifier: GPL-3.0-or-later

//! Born effective charges: the identities that say the response is right.
//!
//! There is no external oracle for periodic PM3, so every check here is an identity the model
//! must satisfy exactly, or a comparison against a quantity computed a different way.
//!
//! One of them is chosen specifically because the obvious check cannot fail. `Σ_a Z*_a = 0`
//! holds for an implementation that returns only the rigid point charges and no electronic
//! response at all, since the net charges of a neutral cell already sum to zero. So the
//! homonuclear chain below is the one that separates "the response is right" from "the response
//! is missing": its true `Z*` is near zero only *because* the electrons cancel the point-charge
//! term.

// A note on `clippy::needless_range_loop`, allowed below.
//
// The loop variables here are Cartesian directions (`alpha`, `beta`, `axis`), atom indices, or
// the rows and columns of a matrix being eliminated. The index *is* the meaning: `for alpha in
// 0..3` says which direction, where `for (alpha, row) in out.iter_mut().enumerate()` says it
// less clearly and no more safely. Several of these loops also index two different tensors by
// the same direction, which no single iterator expresses.
#![allow(clippy::needless_range_loop)]

use pm3_rs::pbc::born::{born_charge_sum_rule_residual, born_charges, enforce_born_sum_rule};
use pm3_rs::pbc::gamma::PeriodicOptions;
use pm3_rs::{Cell, Molecule, Pm3Options, Pm3Parameters, Vec3};

const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";

/// A hydrogen molecule, which is homonuclear and therefore has no static charge to move.
const HYDROGEN: &str = "2\nH2\nH 0.0 0.0 0.0\nH 0.7414 0.0 0.0\n";

fn cell_of(xyz: &str, edge_bohr: f64) -> Molecule {
    let mut molecule = Molecule::from_xyz_str(xyz, 0.0).unwrap();
    molecule.cell = Some(Cell::cubic(edge_bohr).unwrap());
    molecule
}

fn chain_of(xyz: &str, axis_bohr: f64) -> Molecule {
    let mut molecule = Molecule::from_xyz_str(xyz, 0.0).unwrap();
    molecule.cell = Some(
        Cell::new(
            Vec3::new(axis_bohr, 0.0, 0.0),
            Vec3::new(0.0, 40.0, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, false, false],
        )
        .unwrap(),
    );
    molecule
}

fn options() -> Pm3Options {
    Pm3Options::default()
}

/// Translating the whole crystal produces no dipole, so `Σ_a Z*_a = 0`.
///
/// This follows from charge conservation and nothing else, which makes a violation a bug in the
/// response rather than a physical effect.
#[test]
fn the_born_charges_obey_their_acoustic_sum_rule() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();

    for molecule in [cell_of(WATER, 14.0), chain_of(WATER, 6.0)] {
        let born = born_charges(&molecule, &params, &options(), &periodic).unwrap();
        assert_eq!(born.len(), molecule.atoms.len());
        let residual = born_charge_sum_rule_residual(&born);
        assert!(
            residual < 1.0e-5,
            "the Born charges do not sum to zero: {residual:.3e}"
        );
    }
}

/// A chain of hydrogen molecules has Born charges of exactly zero, by symmetry.
///
/// Measured, not assumed: every component comes out below `1e-9`. Both hydrogens carry no net
/// charge, so the `Q_a δ_αβ` term vanishes — and the inversion centre at each bond midpoint makes
/// the charge-transfer derivative vanish too, so nothing is left. Hydrogen has no `p` shell, so
/// the hybridization term is structurally absent as well.
///
/// The value of this case is as a *falsifier*: a response contributing spuriously — a sign error,
/// a missing conjugation, a doubled triangle — would put something here, and the acoustic sum
/// rule would not notice, because a spurious contribution that is equal and opposite on the two
/// atoms still sums to zero.
#[test]
fn a_homonuclear_chain_has_no_born_charge_at_all() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = chain_of(HYDROGEN, 5.0);

    let born = born_charges(&molecule, &params, &options(), &periodic).unwrap();
    let worst = born
        .iter()
        .flat_map(|z| z.iter().flat_map(|row| row.iter()))
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        worst < 1.0e-9,
        "symmetry forbids a Born charge here; got {worst:.3e}"
    );
}

/// The charges do not depend on where the cell origin was put.
///
/// `Σ_b R_b Q_b` is origin dependent and the polarization of a periodic solid is defined only
/// modulo a quantum — but its *derivative* is not, because `Σ_b ∂Q_b/∂u_a = 0` makes the origin
/// dependence cancel term by term. That is the argument the module makes; this measures it.
#[test]
fn the_born_charges_do_not_move_with_the_cell_origin() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();

    let here = cell_of(WATER, 14.0);
    let mut shifted = here.clone();
    let offset = Vec3::new(1.7, -0.9, 0.4);
    for atom in &mut shifted.atoms {
        atom.position += offset;
    }

    let a = born_charges(&here, &params, &options(), &periodic).unwrap();
    let b = born_charges(&shifted, &params, &options(), &periodic).unwrap();

    let mut worst = 0.0_f64;
    for (za, zb) in a.iter().zip(&b) {
        for alpha in 0..3 {
            for beta in 0..3 {
                worst = worst.max((za[alpha][beta] - zb[alpha][beta]).abs());
            }
        }
    }
    assert!(
        worst < 1.0e-6,
        "a {offset:?} Bohr shift of the origin moved the Born charges by {worst:.3e}"
    );
}

/// Imposing the sum rule removes the residual and moves nothing else much.
#[test]
fn enforcing_the_sum_rule_zeroes_it() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = cell_of(WATER, 14.0);

    let mut born = born_charges(&molecule, &params, &options(), &periodic).unwrap();
    let before = born.clone();
    enforce_born_sum_rule(&mut born);
    assert!(born_charge_sum_rule_residual(&born) < 1.0e-12);

    // The correction is the residual spread over the atoms, so it cannot be larger than that.
    let moved = born
        .iter()
        .zip(&before)
        .flat_map(|(x, y)| (0..3).flat_map(move |a| (0..3).map(move |b| (x[a][b] - y[a][b]).abs())))
        .fold(0.0_f64, f64::max);
    assert!(
        moved < 1.0e-5,
        "enforcing the rule changed a charge by {moved:.3e}, which means the response was not \
         converged rather than merely rounded"
    );
}

/// The cell dipole this model defines, `Σ_b Q_b R_b − Σ_b 2 dd_b P_{s,p_α}`.
///
/// Written out here rather than called, on purpose: a finite-difference check against the same
/// code that the analytic derivative is built from would confirm nothing. This is the definition
/// the module's doc comment states, transcribed independently.
fn cell_dipole(
    molecule: &Molecule,
    params: &Pm3Parameters,
    periodic: &PeriodicOptions,
) -> [f64; 3] {
    let scf = pm3_rs::run_gamma(molecule, params, &options(), periodic).unwrap();
    let basis = pm3_rs::basis::Basis::build(molecule, params).unwrap();
    let mut dipole = [0.0_f64; 3];
    for (index, atom) in molecule.atoms.iter().enumerate() {
        let offset = basis.atom_offset[index];
        let norb = basis.atom_norb[index];
        if norb == 0 {
            continue;
        }
        let element = params.element(atom.z).unwrap();
        let population: f64 = (0..norb)
            .map(|mu| scf.density[(offset + mu, offset + mu)])
            .sum();
        let charge = element.core_charge - population;
        for alpha in 0..3 {
            dipole[alpha] += charge * atom.position.to_array()[alpha];
            if element.has_p() {
                dipole[alpha] -= 2.0 * element.dd * scf.density[(offset, offset + alpha + 1)];
            }
        }
    }
    dipole
}

/// `Z*` is the derivative of the cell dipole, and central differences say so.
///
/// This is the test that decides whether the coupled-perturbed response reaches the answer.
/// The sum rule cannot: it holds for an implementation that returns the static charges alone.
/// A finite difference of the dipole shares nothing with the analytic path but the SCF itself.
#[test]
fn the_born_charges_match_a_finite_difference_of_the_cell_dipole() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = cell_of(WATER, 14.0);
    let analytic = born_charges(&molecule, &params, &options(), &periodic).unwrap();

    let step = 1.0e-4;
    let mut largest = 0.0_f64;
    let mut response_size = 0.0_f64;
    for atom in 0..molecule.atoms.len() {
        for beta in 0..3 {
            let shift = |delta: f64| {
                let mut moved = molecule.clone();
                let mut position = moved.atoms[atom].position.to_array();
                position[beta] += delta;
                moved.atoms[atom].position = Vec3::new(position[0], position[1], position[2]);
                cell_dipole(&moved, &params, &periodic)
            };
            let plus = shift(step);
            let minus = shift(-step);
            for alpha in 0..3 {
                let numerical = (plus[alpha] - minus[alpha]) / (2.0 * step);
                largest = largest.max((analytic[atom][alpha][beta] - numerical).abs());
                response_size = response_size.max(numerical.abs());
            }
        }
    }
    assert!(
        largest < 2.0e-4,
        "analytic and finite-difference Born charges differ by {largest:.3e}"
    );
    // Non-vacuity: the derivative is not zero, so agreeing with it means something.
    assert!(
        response_size > 0.1,
        "the cell dipole barely moves ({response_size:.3e}); this system cannot test the response"
    );
}

/// An isolated molecule is refused, and the message names the molecular equivalent.
#[test]
fn a_molecule_is_refused_with_the_alternative_named() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = Molecule::from_xyz_str(WATER, 0.0).unwrap();

    let error = born_charges(&molecule, &params, &options(), &periodic)
        .expect_err("a Born charge is a property of a lattice");
    let text = error.to_string();
    assert!(
        text.contains("cell") || text.contains("dipole_derivatives"),
        "unhelpful message: {text}"
    );
}
