// SPDX-License-Identifier: GPL-3.0-or-later
//! PM3 heats of formation / charges for small molecules, validated against the
//! MOPAC v23.2.5 oracle (values baked in as constants, generated with
//! `tools/oracle/run_mopac.py`). Tolerances are tight (≤ 5e-3 kcal/mol) because
//! pm3-rs reproduces MOPAC's PM3 energy to ~1e-6 kcal/mol on these systems.

use pm3_rs::optimizer::{optimize, OptOptions};
use pm3_rs::params::Pm3Parameters;
use pm3_rs::scf::{run_pm3, Pm3Options};
use pm3_rs::system::Molecule;
use pm3_rs::{analytic_hessian, numerical_hessian, vibrational_analysis};

fn hof(xyz: &str, charge: f64, mult: usize) -> pm3_rs::scf::Pm3Result {
    let mol = Molecule::from_xyz_str(xyz, charge)
        .unwrap()
        .with_multiplicity(mult);
    let params = Pm3Parameters::standard().unwrap();
    let opts = Pm3Options {
        charge,
        multiplicity: mult,
        ..Pm3Options::default()
    };
    run_pm3(&mol, &params, &opts).unwrap()
}

const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";
const METHANE: &str = "5\nmethane\nC 0.0 0.0 0.0\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\nH -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n";
const AMMONIA: &str = "4\nammonia\nN 0.0 0.0 0.1147\nH 0.0 0.9383 -0.2677\nH 0.8125 -0.4691 -0.2677\nH -0.8125 -0.4691 -0.2677\n";
const FORMALDEHYDE: &str = "4\nformaldehyde\nC 0.0 0.0 -0.529\nO 0.0 0.0 0.674\nH 0.0 0.9387 -1.117\nH 0.0 -0.9387 -1.117\n";
const CH3: &str = "4\nmethyl radical\nC 0.0 0.0 0.0\nH 0.0 1.078 0.0\nH 0.9336 -0.539 0.0\nH -0.9336 -0.539 0.0\n";

#[test]
fn water_single_point_matches_mopac() {
    let r = hof(WATER, 0.0, 1);
    assert!(r.converged);
    // MOPAC v23.2.5 PM3 at this geometry.
    assert!(
        (r.heat_of_formation_kcal - (-53.2301514568744)).abs() < 5e-3,
        "water HoF {} kcal/mol",
        r.heat_of_formation_kcal
    );
    // MOPAC net charges: O -0.6095, H +0.3048, H +0.3047.
    assert!(r.charges[0] < -0.3 && r.charges[1] > 0.1 && r.charges[2] > 0.1);
    assert!(r.charges.iter().sum::<f64>().abs() < 1e-6);
}

#[test]
fn methane_matches_mopac() {
    let r = hof(METHANE, 0.0, 1);
    assert!(r.converged);
    // MOPAC v23.2.5 PM3 at this geometry.
    assert!(
        (r.heat_of_formation_kcal - (-13.025669685534)).abs() < 5e-3,
        "methane HoF {}",
        r.heat_of_formation_kcal
    );
}

#[test]
fn ammonia_matches_mopac() {
    let r = hof(AMMONIA, 0.0, 1);
    assert!(r.converged);
    // MOPAC v23.2.5 PM3 at this geometry.
    assert!(
        (r.heat_of_formation_kcal - (-2.73858108540571)).abs() < 5e-3,
        "ammonia HoF {}",
        r.heat_of_formation_kcal
    );
}

#[test]
fn ammonia_analytic_hessian_matches_gradient_difference() {
    let molecule = Molecule::from_xyz_str(AMMONIA, 0.0).unwrap();
    let parameters = Pm3Parameters::standard().unwrap();
    let options = Pm3Options::default();
    let analytic = analytic_hessian(&molecule, &parameters, &options, 1.0e-3).unwrap();
    let numerical = numerical_hessian(&molecule, &parameters, &options, 1.0e-3).unwrap();
    let mut max_difference: f64 = 0.0;
    for i in 0..analytic.rows {
        for j in 0..analytic.cols {
            max_difference = max_difference.max((analytic[(i, j)] - numerical[(i, j)]).abs());
        }
    }
    assert!(
        max_difference < 5.0e-3,
        "NH3 analytic/gradient-difference Hessian mismatch: {max_difference:.6e} eV/Bohr^2"
    );
}

#[test]
fn formaldehyde_carbonyl_polarization() {
    let r = hof(FORMALDEHYDE, 0.0, 1);
    assert!(r.converged);
    // MOPAC v23.2.5 PM3 at this geometry.
    assert!(
        (r.heat_of_formation_kcal - (-33.908115133805)).abs() < 5e-3,
        "formaldehyde HoF {}",
        r.heat_of_formation_kcal
    );
    // Carbonyl polarization: O negative, C positive.
    assert!(r.charges[1] < -0.2, "O charge {}", r.charges[1]);
    assert!(r.charges[0] > 0.1, "C charge {}", r.charges[0]);
    assert!(r.charges.iter().sum::<f64>().abs() < 1e-6);
}

#[test]
fn water_optimizes_to_mopac_minimum() {
    let mol = Molecule::from_xyz_str(WATER, 0.0).unwrap();
    let params = Pm3Parameters::standard().unwrap();
    let opts = Pm3Options::default();
    let res = optimize(&mol, &params, &opts, &OptOptions::default()).unwrap();
    assert!(res.converged);
    // MOPAC v23.2.5 PM3 optimized heat of formation.
    assert!(
        (res.scf.heat_of_formation_kcal - (-53.4330121104622)).abs() < 5e-3,
        "optimized water HoF {}",
        res.scf.heat_of_formation_kcal
    );
}

#[test]
fn water_frequencies_match_mopac() {
    // Optimize, then compute harmonic frequencies; compare the three genuine
    // vibrations to MOPAC's FORCE run at the same minimum.
    let mol = Molecule::from_xyz_str(WATER, 0.0).unwrap();
    let params = Pm3Parameters::standard().unwrap();
    let opts = Pm3Options::default();
    let res = optimize(&mol, &params, &opts, &OptOptions::default()).unwrap();
    let vib = vibrational_analysis(&res.molecule, &params, &opts, 1.0e-3).unwrap();
    // The three highest modes are bend + two O-H stretches.
    let mut f: Vec<f64> = vib.frequencies_cm.clone();
    f.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let top3 = &f[f.len() - 3..];
    // MOPAC FORCE at the same optimized geometry.
    for (got, want) in top3.iter().zip([1743.46, 3868.68, 3989.81]) {
        assert!((got - want).abs() < 2.0, "frequency {got} vs MOPAC {want}");
    }
}

const H2S: &str = "3\nh2s\nS 0.0 0.0 0.103\nH 0.0 0.974 -0.823\nH 0.0 -0.974 -0.823\n";
const HCL: &str = "2\nhcl\nCl 0.0 0.0 0.0\nH 0.0 0.0 1.275\n";
const TICL4: &str = "5\nticl4\nTi 0.0 0.0 0.0\nCl 1.252 1.252 1.252\nCl -1.252 -1.252 1.252\nCl -1.252 1.252 -1.252\nCl 1.252 -1.252 -1.252\n";

#[test]
fn h2s_matches_mopac() {
    let r = hof(H2S, 0.0, 1);
    assert!(r.converged);
    // MOPAC v23.2.5 PM3 at this geometry (s/p basis).
    assert!(
        (r.heat_of_formation_kcal - (-0.235687164627052)).abs() < 5e-3,
        "H2S HoF {}",
        r.heat_of_formation_kcal
    );
}

#[test]
fn hcl_matches_mopac() {
    let r = hof(HCL, 0.0, 1);
    assert!(r.converged);
    // MOPAC v23.2.5 PM3 at this geometry (s/p basis).
    assert!(
        (r.heat_of_formation_kcal - (-20.4516034561493)).abs() < 5e-3,
        "HCl HoF {}",
        r.heat_of_formation_kcal
    );
}

#[test]
fn unsupported_titanium_is_reported() {
    // Titanium has no parameter in the MOPAC PM3 model.
    let molecule = Molecule::from_xyz_str(TICL4, 0.0).unwrap();
    let params = Pm3Parameters::standard().unwrap();
    let error = run_pm3(&molecule, &params, &Pm3Options::default()).unwrap_err();
    assert!(error.to_string().contains("Z=22"));
}

#[test]
fn gdf3_lanthanide_sparkle_runs() {
    // Gadolinium trifluoride: Gd is modelled as a 3+ sparkle (no valence
    // orbitals). The SCF must converge and give the correct polarity
    // (Gd strongly positive, F negative), with charge conservation.
    const GDF3: &str =
        "4\ngdf3\nGd 0.0 0.0 0.0\nF 2.1 0.0 0.0\nF -1.05 1.818653 0.0\nF -1.05 -1.818653 0.0\n";
    let r = hof(GDF3, 0.0, 1);
    assert!(r.converged);
    assert!(
        (r.heat_of_formation_kcal - 61.2557665835629).abs() < 5e-3,
        "GdF3 HoF {}",
        r.heat_of_formation_kcal
    );
    assert!(r.charges[0] > 1.5, "Gd sparkle charge {}", r.charges[0]);
    for f in &r.charges[1..] {
        assert!(*f < 0.0, "F charge {f}");
    }
    assert!(r.charges.iter().sum::<f64>().abs() < 1e-6);
}

#[test]
fn mopac_point_charges_match_oracle() {
    let plus = "4\nwater plus\nO 0 0 0\nH 0.9584 0 0\nH -0.24 0.9278 0\n+ 0 0 3\n";
    let minus = "4\nwater minus\nO 0 0 0\nH 0.9584 0 0\nH -0.24 0.9278 0\n- 0 0 3\n";
    let plus_result = hof(plus, 1.0, 1);
    let minus_result = hof(minus, -1.0, 1);
    assert!(
        (plus_result.heat_of_formation_kcal - (-47.2642956996679)).abs() < 5e-3,
        "plus HoF {}",
        plus_result.heat_of_formation_kcal
    );
    assert!(
        (minus_result.heat_of_formation_kcal - (-41.388418309698)).abs() < 5e-3,
        "minus HoF {}",
        minus_result.heat_of_formation_kcal
    );
    assert!((plus_result.charges[3] - 1.0).abs() < 1e-12);
    assert!((minus_result.charges[3] + 1.0).abs() < 1e-12);
}

#[test]
fn mopac_special_atom_parameters_are_available() {
    use pm3_rs::{symbol_to_z, z_to_symbol};

    assert_eq!(symbol_to_z("Cb"), Some(102));
    assert_eq!(symbol_to_z("+"), Some(104));
    assert_eq!(symbol_to_z("-"), Some(106));
    assert_eq!(z_to_symbol(102), Some("Cb"));
    let parameters = Pm3Parameters::standard().unwrap();
    let capped_bond = parameters.element(102).unwrap();
    // MOPAC assigns the Cb center one s and three formal p AOs.  The p
    // resonance parameters are zero, but the orbitals participate in the
    // historical capped-bond density and `capcor` convention.
    assert_eq!(capped_bond.n_orb, 4);
    assert_eq!(capped_bond.core_charge, 1.0);
    assert_eq!(parameters.element(104).unwrap().n_orb, 0);
    assert_eq!(parameters.element(104).unwrap().core_charge, 1.0);
    assert_eq!(parameters.element(106).unwrap().n_orb, 0);
    assert_eq!(parameters.element(106).unwrap().core_charge, -1.0);
}

#[test]
fn methyl_radical_uhf_matches_mopac() {
    let r = hof(CH3, 0.0, 2);
    assert!(r.converged);
    assert!(r.unrestricted, "CH3 radical should use UHF");
    // MOPAC v23.2.5 PM3 UHF doublet at this geometry.
    assert!(
        (r.heat_of_formation_kcal - 28.0055283157567).abs() < 5e-3,
        "methyl radical HoF {}",
        r.heat_of_formation_kcal
    );
}
