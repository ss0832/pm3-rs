// SPDX-License-Identifier: GPL-3.0-or-later

//! The open-shell response, and the identity that pins its spin bookkeeping.
//!
//! Every response in this crate is solved per spin channel and summed. A restricted calculation
//! has one channel carrying `P` at half exchange strength; an unrestricted one has two, each
//! carrying its own `P^σ` at full strength. The two arrangements are supposed to give the same
//! number for a closed shell, and that is the whole of what makes the restricted path an
//! optimization rather than a different method.
//!
//! So the load-bearing test here is [`forcing_uhf_on_a_closed_shell_reproduces_the_restricted_response`]:
//! force the unrestricted path onto a system that does not need it and require the answer not to
//! move. A spin factor that is wrong by the obvious amounts -- keeping one channel instead of
//! two, or summing two half-strength channels -- lands at exactly half or exactly twice the
//! restricted answer, which is a difference no tolerance hides.
//!
//! `src/pbc/dfpt.rs` already tests the unrestricted *force constants* against a finite difference
//! of the spin-aware periodic gradient. What it does not cover is the response density itself,
//! which is what the v0.2.2 properties -- Born charges, the polarizability, the dielectric
//! tensors -- are contractions of. That is what this file is for.

use pm3_rs::pbc::gamma::PeriodicOptions;
use pm3_rs::{Cell, Molecule, Pm3Options, Pm3Parameters, Reference};

/// Water: closed shell, and polar enough to have a response worth comparing.
const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";
/// Methyl radical: a genuine doublet.
const METHYL: &str = "4\nmethyl\nC 0.0 0.0 0.05\nH 1.09 0.0 0.0\nH -0.545 0.944 0.0\n\
                      H -0.545 -0.944 0.0\n";

/// A cube whose edge clears [`pm3_rs::pbc::gamma::DEFAULT_SHORT_RANGE_CUTOFF`] with room to spare.
///
/// 18 Bohr against a 14 Bohr cutoff. The margin is asserted rather than assumed in every test
/// below, because below it the SCF converges cleanly to a well-defined wrong answer and a
/// response built on that inherits the error silently.
const EDGE_BOHR: f64 = 18.0;

fn boxed(xyz: &str, edge: f64) -> Molecule {
    let mut molecule = Molecule::from_xyz_str(xyz, 0.0).unwrap();
    molecule.cell = Some(Cell::cubic(edge).unwrap());
    molecule
}

fn options(multiplicity: usize, reference: Reference) -> Pm3Options {
    Pm3Options {
        multiplicity,
        reference,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 600,
        ..Pm3Options::default()
    }
}

/// The margin, asserted so a cell change cannot quietly move a test into the invalid regime.
fn assert_valid_gamma(molecule: &Molecule, params: &Pm3Parameters, options: &Pm3Options) {
    let periodic = PeriodicOptions::default();
    let margin = pm3_rs::run_gamma(molecule, params, options, &periodic)
        .unwrap()
        .gamma_margin;
    assert!(
        margin > 0.0,
        "the Gamma margin is {margin:.3} Bohr; one k-point is not enough and this would be a \
         response of the wrong ground state"
    );
}

/// **The spin-bookkeeping test.** Forced UHF on a closed shell reproduces the restricted response.
///
/// Not a tautology: the two paths do genuinely different arithmetic. The restricted path solves
/// one channel holding `P` and scales exchange by `½`; the unrestricted path solves two channels,
/// each holding its own `P^σ` at full exchange strength, and sums the resulting first-order
/// densities. They agree only if the factor of two in the channel count and the factor of a half
/// in the exchange scale are both right and cancel.
///
/// Asserted on all three v0.2.2 properties rather than one, because they contract the same
/// response against different things: `Z*` against on-site charges and dipole terms, `alpha`
/// against the position operator. A spin error common to both would still show in each.
///
/// # Verified to fail
///
/// Changing the spin sum in `dfpt.rs` to keep one channel instead of two makes this fail by
/// `1.372e-1 e` against a largest `Z*` component of `0.627` -- so the tolerance sits four orders
/// of magnitude below the error it exists to catch. The other two tests in this file passed under
/// that same sabotage, which is why this one is the load-bearing one.
#[test]
fn forcing_uhf_on_a_closed_shell_reproduces_the_restricted_response() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = boxed(WATER, EDGE_BOHR);

    let restricted = options(1, Reference::Rhf);
    let forced = options(1, Reference::Uhf);
    assert_valid_gamma(&molecule, &params, &restricted);

    // Non-vacuity, and the assertion this test would be worthless without: `Reference::Uhf` on a
    // singlet has to actually take the unrestricted path. If it fell back to the restricted one
    // the comparison below would be a matrix against itself.
    let rhf_run = pm3_rs::run_gamma(&molecule, &params, &restricted, &periodic).unwrap();
    let uhf_run = pm3_rs::run_gamma(&molecule, &params, &forced, &periodic).unwrap();
    assert!(!rhf_run.unrestricted, "`Reference::Rhf` took the spin path");
    assert!(
        uhf_run.unrestricted,
        "`Reference::Uhf` on a singlet fell back to the restricted path, so this test compares \
         the restricted answer against itself"
    );
    // Same ground state, so any difference below is the response and not the SCF.
    assert!(
        (rhf_run.total_ev - uhf_run.total_ev).abs() < 1.0e-7,
        "the two references converged to different states ({:.9} vs {:.9} eV); this water is \
         breaking symmetry and the response comparison would be measuring that instead",
        rhf_run.total_ev,
        uhf_run.total_ev
    );

    let z_rhf = pm3_rs::born_charges(&molecule, &params, &restricted, &periodic).unwrap();
    let z_uhf = pm3_rs::born_charges(&molecule, &params, &forced, &periodic).unwrap();
    let mut worst_z = 0.0_f64;
    for (a, b) in z_rhf.iter().zip(&z_uhf) {
        for alpha in 0..3 {
            for beta in 0..3 {
                worst_z = worst_z.max((a[alpha][beta] - b[alpha][beta]).abs());
            }
        }
    }
    // A scale so the failure says *which* mistake it is rather than only that there was one.
    let largest = z_rhf
        .iter()
        .flat_map(|z| z.iter().flat_map(|row| row.iter()))
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        worst_z < 1.0e-7,
        "Born charges differ by {worst_z:.3e} e between the restricted and unrestricted paths \
         (largest Z* component is {largest:.4}). A ratio near 2 or ½ against that is a channel \
         count or an exchange scale, not a convergence difference."
    );

    let a_rhf = pm3_rs::polarizability(&molecule, &params, &restricted, &periodic).unwrap();
    let a_uhf = pm3_rs::polarizability(&molecule, &params, &forced, &periodic).unwrap();
    let mut worst_alpha = 0.0_f64;
    for alpha in 0..3 {
        for beta in 0..3 {
            worst_alpha = worst_alpha.max((a_rhf[alpha][beta] - a_uhf[alpha][beta]).abs());
        }
    }
    assert!(
        worst_alpha < 1.0e-6,
        "the polarizability differs by {worst_alpha:.3e} Bohr^3 between the two paths; the \
         restricted diagonal is {:?}",
        [a_rhf[0][0], a_rhf[1][1], a_rhf[2][2]]
    );

    let e_rhf = pm3_rs::dielectric_tensor(&molecule, &params, &restricted, &periodic).unwrap();
    let e_uhf = pm3_rs::dielectric_tensor(&molecule, &params, &forced, &periodic).unwrap();
    for alpha in 0..3 {
        for beta in 0..3 {
            let difference = (e_rhf.epsilon[alpha][beta] - e_uhf.epsilon[alpha][beta]).abs();
            assert!(
                difference < 1.0e-9,
                "eps_inf[{alpha}][{beta}] differs by {difference:.3e} between the two paths"
            );
        }
    }
    // And the response was not simply zero on both sides, which would satisfy every assertion
    // above without testing anything.
    assert!(
        largest > 1.0e-3,
        "the largest Z* component is {largest:.3e}; there is no response here to compare"
    );
    assert!(
        a_rhf[0][0] > 1.0e-3,
        "the polarizability is zero; nothing was tested"
    );
}

/// The dynamical matrix agrees between the two paths too, at a wavevector rather than at Gamma.
///
/// Separate from the properties above because it goes through the force-constant contraction
/// rather than the response density, and at finite `q` the channels carry Bloch phases. A spin
/// factor that survived `q = 0` by symmetry would have to survive this as well.
#[test]
fn the_two_paths_agree_at_a_wavevector() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = boxed(WATER, EDGE_BOHR);
    let restricted = options(1, Reference::Rhf);
    let forced = options(1, Reference::Uhf);
    assert_valid_gamma(&molecule, &params, &restricted);

    let q = [0.25, 0.0, 0.0];
    let d_rhf = pm3_rs::dynamical_matrix(&molecule, &params, &restricted, &periodic, q).unwrap();
    let d_uhf = pm3_rs::dynamical_matrix(&molecule, &params, &forced, &periodic, q).unwrap();

    let mut f_rhf = pm3_rs::frequencies_of(&d_rhf).unwrap();
    let mut f_uhf = pm3_rs::frequencies_of(&d_uhf).unwrap();
    f_rhf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    f_uhf.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let worst = f_rhf
        .iter()
        .zip(&f_uhf)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst < 1.0e-3,
        "D(q) differs by {worst:.4} cm^-1 between the restricted and unrestricted paths at \
         q = {q:?}: {f_rhf:?} vs {f_uhf:?}"
    );
    // Non-vacuity: a matrix of zeros would agree perfectly.
    assert!(
        f_rhf.iter().any(|f| *f > 100.0),
        "no mode above 100 cm^-1; there is nothing here for the two paths to agree about"
    );
}

/// A genuine doublet has Born charges, and they obey the acoustic sum rule.
///
/// The closed-shell comparisons above pin the spin factors; this says the machinery runs at all
/// on a cell that has no restricted answer to fall back on.
///
/// What the sum rule does **not** do is check the spin bookkeeping. Dropping one of the two
/// channels leaves this test passing, because the rule is a statement about translational
/// invariance and survives the response being scaled. It is worth asserting for what it does
/// cover -- that the open-shell path produces a coherent response at all -- but the factor of two
/// is caught upstairs, not here, and reading this as a check on it would be a mistake.
#[test]
fn an_open_shell_cell_has_born_charges_that_obey_the_sum_rule() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = boxed(METHYL, EDGE_BOHR);
    let doublet = options(2, Reference::Auto);
    assert_valid_gamma(&molecule, &params, &doublet);

    assert!(
        pm3_rs::run_gamma(&molecule, &params, &doublet, &periodic)
            .unwrap()
            .unrestricted,
        "the fixture converged closed-shell, so this proves nothing about the spin path"
    );

    let born = pm3_rs::born_charges(&molecule, &params, &doublet, &periodic).unwrap();
    assert_eq!(born.len(), molecule.atoms.len());
    let residual = pm3_rs::born_charge_sum_rule_residual(&born);
    assert!(
        residual < 1.0e-6,
        "sum_a Z*_a comes out at {residual:.3e} e for the doublet; translating the crystal must \
         produce no dipole whatever the spin state"
    );

    // Carbon and hydrogen must not come out with the same tensor, or this is a point-charge
    // model wearing a response's clothes.
    let spread = (0..3)
        .map(|alpha| (born[0][alpha][alpha] - born[1][alpha][alpha]).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        spread > 1.0e-3,
        "carbon and hydrogen have the same diagonal Z* to {spread:.3e}; nothing responded"
    );
}
