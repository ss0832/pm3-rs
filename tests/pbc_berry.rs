// SPDX-License-Identifier: GPL-3.0-or-later

//! Berry-phase polarization, and what it is for: checking the CPHF response from outside.
//!
//! The Born charges this crate reports come from coupled-perturbed theory --
//! `Z*_a = Q_a δ + Σ_b R_b ∂Q_b/∂u_a + Σ_b (−2 dd_b) ΔP_b(s, p)` -- and every check on them so
//! far has been internal to that machinery or to the identities it satisfies. The Berry phase
//! reaches the same number through nothing in common: a product of overlaps between neighbouring
//! k points, with no response equation anywhere in it.
//!
//! So [`berry_charges_agree_with_the_coupled_perturbed_ones`] is the point of this file. The rest
//! establish that the phase itself is trustworthy enough for that comparison to mean something --
//! that it converges in the string length, and that it handles the branch ambiguity correctly
//! rather than by luck.

use pm3_rs::pbc::gamma::PeriodicOptions;
use pm3_rs::{Cell, KpointOptions, Molecule, Pm3Options, Pm3Parameters, Vec3};

/// A polar cell: two unlike atoms, so there is a polarization to find.
const HF: &str = "2\nhydrogen fluoride\nF 0.0 0.0 0.0\nH 0.93 0.0 0.0\n";
const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";

/// 16 Bohr, clear of the 14 Bohr short-range cutoff.
const EDGE_BOHR: f64 = 16.0;

fn boxed(xyz: &str, edge: f64) -> Molecule {
    let mut molecule = Molecule::from_xyz_str(xyz, 0.0).unwrap();
    molecule.cell = Some(Cell::cubic(edge).unwrap());
    molecule
}

fn options() -> Pm3Options {
    Pm3Options {
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 600,
        ..Pm3Options::default()
    }
}

fn assert_valid_gamma(molecule: &Molecule, params: &Pm3Parameters, options: &Pm3Options) {
    let margin = pm3_rs::run_gamma(molecule, params, options, &PeriodicOptions::default())
        .unwrap()
        .gamma_margin;
    assert!(margin > 0.0, "the Gamma margin is {margin:.3} Bohr");
}

/// **The cross-check.** `Z* = Ω ∂P/∂u` from the Berry phase, against the CPHF Born charges.
///
/// Two formalisms with nothing shared but the converged ground state. A sign convention, a factor
/// of two for spin, a `4π`, a wrong `Ω` -- each of these is invisible inside one route and
/// obvious across the pair.
///
/// # What they do not share, and how much of the gap that accounts for
///
/// The Berry phase puts every orbital at its atom's position -- the same approximation the dipole
/// operator makes on its diagonal. What it drops is the intra-atomic `s`–`p` hybridization moment
/// `dd_a`, which the CPHF route carries at `born.rs`'s third term. So these are not two
/// computations of one number.
///
/// The gap is therefore **all** of that term, or it is a bug, and guessing which was not good
/// enough. Running this with `PM3_BORN_NO_DD=1`, which removes exactly that term from the CPHF
/// side, collapses the disagreement from `0.147` to `1.9e-4 e`:
///
/// | component | Berry | CPHF | CPHF without `dd` |
/// |---|---|---|---|
/// | `xx` (along the bond) | −0.38924 | −0.32600 | −0.38905 |
/// | `yy` = `zz` (across it) | −0.16529 | −0.31258 | −0.16528 |
///
/// So every other thing the two routes could have disagreed about -- the sign of the phase, the
/// factor of two for spin, `Ω`, the quantum, the branch reduction, the string discretization --
/// agrees to two parts in ten thousand. The residue is one identified physical term, and its
/// direction dependence is the expected one: the longitudinal charge transfer is a term both
/// routes carry, while the transverse response is almost entirely on-site `s`–`p` polarization,
/// which is why the relative gap is small along the bond and large across it.
///
/// Both routes independently satisfy the acoustic sum rule (Berry `3.3e-13`, CPHF `8.9e-16`),
/// which is what says each is internally consistent and the difference is between them rather
/// than inside one.
///
/// The tolerance below is therefore set to admit the `dd` term and nothing larger.
#[test]
fn berry_charges_agree_with_the_coupled_perturbed_ones() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let options = options();
    let molecule = boxed(HF, EDGE_BOHR);
    assert_valid_gamma(&molecule, &params, &options);

    let kopts = KpointOptions::mesh([1, 1, 1]);
    let strings = 12;
    let volume = molecule.cell.unwrap().measure();
    let step = 0.01;

    let cphf = pm3_rs::born_charges(&molecule, &params, &options, &periodic).unwrap();

    // `Z*_a[α][β] = Ω ∂P_α / ∂u_aβ`, by central differences of the polarization. Differenced
    // through `BerryPolarization::difference`, which reduces onto the nearest branch -- a plain
    // subtraction of `total` is off by exactly one quantum whenever the two land differently,
    // and for a displacement this size that happens.
    let displaced = |atom: usize, direction: usize, sign: f64| {
        let mut shifted = molecule.clone();
        let mut delta = [0.0; 3];
        delta[direction] = sign * step;
        shifted.atoms[atom].position += Vec3::new(delta[0], delta[1], delta[2]);
        pm3_rs::berry_polarization(&shifted, &params, &options, &periodic, &kopts, strings).unwrap()
    };

    let mut worst = 0.0_f64;
    let mut worst_pair = (0, 0, 0.0, 0.0);
    let mut berry_z = vec![[[0.0_f64; 3]; 3]; molecule.atoms.len()];
    for atom in 0..molecule.atoms.len() {
        for beta in 0..3 {
            let minus = displaced(atom, beta, -1.0);
            let plus = displaced(atom, beta, 1.0);
            let dp = minus.difference(&plus) * (volume / (2.0 * step));
            for (alpha, berry) in [dp.x, dp.y, dp.z].into_iter().enumerate() {
                berry_z[atom][alpha][beta] = berry;
                let reference = cphf[atom][alpha][beta];
                let gap = (berry - reference).abs();
                if gap > worst {
                    worst = gap;
                    worst_pair = (atom, alpha, berry, reference);
                }
            }
        }
    }

    // Printed in full, because "the worst component" alone cannot distinguish a systematic factor
    // from a term one route omits, and that distinction is the whole point of the comparison.
    for atom in 0..molecule.atoms.len() {
        for alpha in 0..3 {
            eprintln!(
                "atom {atom} row {alpha}: Berry [{:+.5} {:+.5} {:+.5}]  CPHF [{:+.5} {:+.5} {:+.5}]",
                berry_z[atom][alpha][0],
                berry_z[atom][alpha][1],
                berry_z[atom][alpha][2],
                cphf[atom][alpha][0],
                cphf[atom][alpha][1],
                cphf[atom][alpha][2],
            );
        }
    }
    // The sum rule for the Berry route, computed the same way as for the CPHF one. Both must
    // vanish: it is a property of the exact response, not of either formalism.
    let mut berry_residual = 0.0_f64;
    for alpha in 0..3 {
        for beta in 0..3 {
            let total: f64 = berry_z.iter().map(|z| z[alpha][beta]).sum();
            berry_residual = berry_residual.max(total.abs());
        }
    }
    eprintln!(
        "sum rules: Berry {berry_residual:.3e}, CPHF {:.3e}",
        pm3_rs::born_charge_sum_rule_residual(&cphf)
    );

    // Non-vacuity: HF is polar, so there is a real charge here to agree about. Without this the
    // comparison passes on two routes that both return zero.
    let largest = cphf
        .iter()
        .flat_map(|z| z.iter().flat_map(|row| row.iter()))
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        largest > 0.1,
        "the largest CPHF Z* component is {largest:.4}; this cell has no polarization to check"
    );

    // The Berry route must satisfy the sum rule on its own. This is the check that says its
    // internal consistency does not depend on the CPHF route being right.
    assert!(
        berry_residual < 1.0e-9,
        "the Berry-phase charges sum to {berry_residual:.3e} e; translating the crystal must \
         produce no dipole whichever formalism measured it"
    );

    let (atom, alpha, berry, reference) = worst_pair;
    assert!(
        worst < 0.2,
        "the Berry-phase Z* and the CPHF Z* differ by {worst:.5} e (largest component {largest:.4}). \
         The worst is atom {atom}, direction {alpha}: Berry {berry:+.5} against CPHF \
         {reference:+.5}. These routes differ by the intra-atomic `dd` moment the phase omits and \
         by nothing else -- measured at 1.9e-4 with `PM3_BORN_NO_DD=1` -- so a gap this size is \
         that term, and a gap much larger is a sign, a spin factor, or a volume."
    );
    eprintln!("Berry vs CPHF Z*: worst {worst:.6} e against a largest component of {largest:.4}");
}

/// The phase converges in the string length, which is what makes it a number rather than a guess.
///
/// A discretized Berry phase is exact only in the limit of a dense string. If the answer still
/// moved between 8 and 16 points, the cross-check above would be comparing the CPHF charge
/// against a discretization error.
#[test]
fn the_phase_converges_in_the_string_length() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let options = options();
    let molecule = boxed(HF, EDGE_BOHR);
    let kopts = KpointOptions::mesh([1, 1, 1]);

    let at = |strings: usize| {
        pm3_rs::berry_polarization(&molecule, &params, &options, &periodic, &kopts, strings)
            .unwrap()
    };
    let coarse = at(6);
    let medium = at(12);
    let fine = at(24);

    // Compared through `difference`, so a branch change between two string lengths is not read as
    // a convergence failure.
    let first = coarse.difference(&medium).norm();
    let second = medium.difference(&fine).norm();
    assert!(
        second <= first + 1.0e-12,
        "refining the string from 12 to 24 points moved the polarization by {second:.3e}, more \
         than the 6-to-12 step did ({first:.3e}); this is not converging"
    );
    assert!(
        second < 1.0e-4,
        "12 to 24 points still moves the polarization by {second:.3e} e/Bohr^2; the string is too \
         coarse for the cross-check to mean anything"
    );
    eprintln!("string convergence: 6->12 {first:.3e}, 12->24 {second:.3e}");
}

/// The quantum is reported, and a difference is reduced onto the nearest branch.
///
/// The branch ambiguity is the physics of the modern theory, not a rounding problem: two
/// polarizations differing by an integer combination of the quanta are the same state. This
/// checks that `difference` actually performs that reduction, by handing it a displacement large
/// enough to cross a branch and requiring the answer to stay small.
#[test]
fn the_branch_ambiguity_is_handled_rather_than_ignored() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let options = options();
    let molecule = boxed(WATER, EDGE_BOHR);
    let kopts = KpointOptions::mesh([1, 1, 1]);

    let reference =
        pm3_rs::berry_polarization(&molecule, &params, &options, &periodic, &kopts, 12).unwrap();

    // The quantum along each axis is `a_α / Ω`, which for a cubic cell is `1/edge²` along each
    // Cartesian direction.
    let volume = molecule.cell.unwrap().measure();
    for axis in 0..3 {
        let expected = molecule.cell.unwrap().vector(axis) / volume;
        let got = reference.quantum[axis];
        assert!(
            (got - expected).norm() < 1.0e-12,
            "quantum along {axis} is {got:?}, expected {expected:?}"
        );
    }
    assert_eq!(reference.string_length, 12);

    // Adding a whole quantum is the same physical state, and `difference` must say so.
    let mut shifted = reference.clone();
    shifted.total = reference.total + reference.quantum[0] * 3.0 - reference.quantum[1];
    let delta = reference.difference(&shifted);
    assert!(
        delta.norm() < 1.0e-12,
        "adding integer quanta moved the reduced difference by {:.3e}; the reduction is not \
         happening",
        delta.norm()
    );

    // And the total really is the sum of the two halves it reports.
    let sum = reference.electronic + reference.ionic;
    assert!((sum - reference.total).norm() < 1.0e-12);
    // The ionic half is not zero -- a cell of neutral atoms at the origin would make every
    // assertion above pass without any of this being exercised.
    assert!(
        reference.ionic.norm() > 1.0e-6,
        "the ionic polarization is zero"
    );
}

/// What the Berry phase refuses.
#[test]
fn what_the_berry_phase_refuses() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let options = options();
    let kopts = KpointOptions::mesh([1, 1, 1]);

    // A chain: the quantum is `e a/Ω` and there is no volume.
    let mut chain = Molecule::from_xyz_str(HF, 0.0).unwrap();
    chain.cell = Some(
        Cell::new(
            Vec3::new(EDGE_BOHR, 0.0, 0.0),
            Vec3::new(0.0, 30.0, 0.0),
            Vec3::new(0.0, 0.0, 30.0),
            [true, false, false],
        )
        .unwrap(),
    );
    let error = pm3_rs::berry_polarization(&chain, &params, &options, &periodic, &kopts, 12)
        .expect_err("a chain has no volume for the quantum to divide by");
    assert!(error.to_string().contains("three-dimensional"), "{error}");

    // An isolated molecule has no lattice at all.
    let isolated = Molecule::from_xyz_str(HF, 0.0).unwrap();
    let error = pm3_rs::berry_polarization(&isolated, &params, &options, &periodic, &kopts, 12)
        .expect_err("a Berry phase needs a Brillouin zone to wind through");
    assert!(error.to_string().contains("cell"), "{error}");

    // Two points cannot resolve a winding.
    let molecule = boxed(HF, EDGE_BOHR);
    let error = pm3_rs::berry_polarization(&molecule, &params, &options, &periodic, &kopts, 2)
        .expect_err("a string of two points is not a string");
    assert!(error.to_string().contains("3 k points"), "{error}");
}
