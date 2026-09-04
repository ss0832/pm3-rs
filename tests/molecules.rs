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

/// The dipole of a **charged** system depends on the origin, so the origin is part of the
/// definition. MOPAC (`dipole.F90`) references it to the centre of mass, with its point atoms
/// `+`/`-` carrying zero mass. pm3-rs referenced it to the coordinate origin instead, which made
/// the reported dipole of every ion depend on where the molecule happened to sit: translating
/// NH4+ by 5 Å moved its dipole from 0 to 24.02 D. Fixed in 0.2.0; these are the MOPAC v23.2.5
/// values at the same geometries.
#[test]
fn charged_system_dipole_matches_mopac_and_is_translation_invariant() {
    // NH4+ is tetrahedral, so its dipole is exactly zero wherever it sits.
    const NH4_PLUS: &str = "5\nammonium\nN 0 0 0\nH 0.63 0.63 0.63\nH -0.63 -0.63 0.63\nH -0.63 0.63 -0.63\nH 0.63 -0.63 -0.63\n";
    const NH4_PLUS_SHIFTED: &str = "5\nammonium shifted\nN 5 0 0\nH 5.63 0.63 0.63\nH 4.37 -0.63 0.63\nH 4.37 0.63 -0.63\nH 5.63 -0.63 -0.63\n";

    let at_origin = hof(NH4_PLUS, 1.0, 1);
    let shifted = hof(NH4_PLUS_SHIFTED, 1.0, 1);
    // MOPAC reports 8.0e-11 D at both placements.
    assert!(
        at_origin.dipole_magnitude < 1.0e-8,
        "NH4+ dipole at the origin should vanish, got {} D",
        at_origin.dipole_magnitude
    );
    assert!(
        shifted.dipole_magnitude < 1.0e-8,
        "NH4+ dipole must not depend on where the ion sits, got {} D after a 5 Å shift",
        shifted.dipole_magnitude
    );
    // Same heat of formation either way (a pure translation cannot change it).
    assert!((at_origin.heat_of_formation_kcal - shifted.heat_of_formation_kcal).abs() < 1e-6);

    // Water plus a MOPAC `+` point atom: a charged system with a genuinely non-zero dipole,
    // and one whose zero-mass point atom must shift the charge distribution without moving
    // the centre of mass. MOPAC v23.2.5: (0.92295852, 1.19207248, 14.38633218) D.
    let plus = "4\nwater plus\nO 0 0 0\nH 0.9584 0 0\nH -0.24 0.9278 0\n+ 0 0 3\n";
    let r = hof(plus, 1.0, 1);
    let reference = [
        0.922_958_525_869_74,
        1.192_072_477_432_1,
        14.386_332_179_826,
    ];
    let got = [r.dipole_debye.x, r.dipole_debye.y, r.dipole_debye.z];
    for (axis, (value, expected)) in got.iter().zip(&reference).enumerate() {
        assert!(
            (value - expected).abs() < 1.0e-5,
            "dipole component {axis}: pm3-rs {value} vs MOPAC {expected}"
        );
    }
}

/// Every element with valence principal quantum number ≥ 4 carries a small, systematic
/// deviation from MOPAC — pm3-rs evaluates their diatomic overlap by converged quadrature
/// while MOPAC's closed-form `diat`/`SS` uses series expansions that differ at the 1e-8
/// level. The deviation is characterised in `tools/oracle/PM3_AUDIT.md`; it is bounded by
/// 1.05e-3 kcal/mol, which is three orders of magnitude below chemical significance. This
/// test pins that bound so it cannot silently grow, and would also catch a regression that
/// accidentally made the heavy-element path much *worse*.
#[test]
fn heavy_element_deviation_from_mopac_stays_bounded() {
    // MOPAC v23.2.5 heats of formation at these exact geometries (the fixtures
    // `tools/oracle/all_element_validation.py` generates for Ge, Se and Bi).
    const CASES: [(&str, f64); 3] = [
        (
            "5\nGeH4\nGe 0 0 0\nH 0.9153888518 0.9153888518 0.9153888518\nH -0.9153888518 -0.9153888518 0.9153888518\nH -0.9153888518 0.9153888518 -0.9153888518\nH 0.9153888518 -0.9153888518 -0.9153888518\n",
            36.517_700_295_622_9,
        ),
        (
            "3\nSeH2\nSe 0 0 0\nH 1.5855 0 0\nH -0.396375 1.5351537739 0\n",
            27.719_425_002_056_6,
        ),
        (
            "4\nBiH3\nBi 0 0 0\nH 1.7929287290 0 0.56385\nH -0.8964643645 1.5527218265 0.56385\nH -0.8964643645 -1.5527218265 0.56385\n",
            45.042_139_218_554,
        ),
    ];
    // The audit's measured bound, with a little headroom; tightening this is a real
    // improvement, loosening it means the heavy-element path changed.
    const BOUND_KCAL: f64 = 2.0e-3;
    for (xyz, mopac) in CASES {
        let r = hof(xyz, 0.0, 1);
        assert!(r.converged);
        let difference = (r.heat_of_formation_kcal - mopac).abs();
        assert!(
            difference < BOUND_KCAL,
            "heavy-element deviation grew: pm3-rs {} vs MOPAC {mopac} ({difference:.3e} kcal/mol)",
            r.heat_of_formation_kcal
        );
    }
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

/// The external electric field, against MOPAC's own `FIELD=` keyword.
///
/// MOPAC's units and sign are not documented in a way worth trusting, so they were **measured**:
/// running water at `FIELD=(f,0,0)` and differencing the heat of formation gives
/// `dE/df = +0.22451 eV` per unit, against a dipole of `0.22458 e.Angstrom`. So MOPAC's field is
/// in **volts per Angstrom** and carries the sign convention `E = E0 + mu.F` -- the opposite of
/// the physical one, in which a dipole aligned with the field is *stabilized*.
///
/// This crate uses the physical convention, `E = E0 - mu.f`, with `f` in eV per Bohr per
/// elementary charge, so the two are related by
///
/// ```text
/// f = -F * BOHR_TO_ANGSTROM
/// ```
///
/// and the frozen numbers below are MOPAC's, to every digit it printed.
#[test]
fn an_external_field_matches_the_mopac_field_keyword() {
    // MOPAC v23.2.5: `PM3 PRECISE AUX(PRECISION=9) 1SCF NOREOR FIELD=(f,0,0)`, minus the
    // field-free heat of formation of the same geometry.
    const MOPAC: [(f64, f64); 3] = [
        (0.001, 0.005_178_44),
        (0.002, 0.010_355_61),
        (0.004, 0.020_706_08),
    ];

    let params = Pm3Parameters::standard().unwrap();
    let molecule = Molecule::from_xyz_str(WATER, 0.0).unwrap();
    let base = Pm3Options {
        e_tol: 1.0e-12,
        p_tol: 1.0e-11,
        ..Pm3Options::default()
    };
    let zero = run_pm3(&molecule, &params, &base).unwrap();

    for (mopac_field, expected) in MOPAC {
        let field =
            pm3_rs::math::Vec3::new(-mopac_field * pm3_rs::constants::BOHR_TO_ANGSTROM, 0.0, 0.0);
        let options = Pm3Options {
            field: Some(field),
            ..base.clone()
        };
        let shifted = run_pm3(&molecule, &params, &options).unwrap();
        let delta = shifted.heat_of_formation_kcal - zero.heat_of_formation_kcal;
        assert!(
            (delta - expected).abs() < 5.0e-8,
            "FIELD=({mopac_field},0,0): got {delta:.8} kcal, MOPAC gives {expected:.8}"
        );
    }
}

/// Dipole derivatives against MOPAC, element by element.
///
/// MOPAC reports VIB._T_DIP for a FORCE run, but its normal-mode normalization is not documented
/// well enough to compare an intensity against directly -- our |dmu/dQ| and its T_DIP agree only
/// to a few percent, and the residual looks like a reduced-mass convention rather than an error.
/// So the comparison is made one step earlier, where there is no convention at all: MOPAC's own
/// dipole, central-differenced over each Cartesian coordinate.
///
/// That is dmu/dR exactly as we compute it analytically, and it agrees to every digit MOPAC's
/// finite difference resolves. The frequencies at this geometry agree to 0.5 cm^-1
/// (1742.11/3868.52/3988.94 against MOPAC's 1741.99/3867.75/3988.47), so the modes the tensor is
/// projected onto are MOPAC's too.
#[test]
fn dipole_derivatives_match_mopac() {
    // MOPAC v23.2.5, PM3 PRECISE AUX(PRECISION=9) 1SCF NOREOR, dipole central-differenced at
    // +-0.005 Angstrom about the pm3-rs optimum below, converted from Debye to e.
    const MOPAC: [[f64; 9]; 3] = [
        [-0.64523, 0.0, 0.0, 0.32262, 0.0, 0.0, 0.32262, 0.0, 0.0],
        [
            0.0, -0.29722, 0.0, 0.0, 0.14861, 0.12714, 0.0, 0.14861, -0.12714,
        ],
        [
            0.0, 0.0, -0.19191, 0.0, -0.00274, 0.09595, 0.0, 0.00274, 0.09595,
        ],
    ];
    const OPTIMIZED: &str = "3\nwater\nO 0.0 0.0 0.10030517\nH 0.0 0.76783584 -0.46070259\nH 0.0 -0.76783584 -0.46070259\n";

    let params = Pm3Parameters::standard().unwrap();
    let molecule = Molecule::from_xyz_str(OPTIMIZED, 0.0).unwrap();
    let options = Pm3Options {
        e_tol: 1.0e-12,
        p_tol: 1.0e-11,
        max_scf: 500,
        ..Pm3Options::default()
    };
    let analytic = pm3_rs::ir::dipole_derivatives(&molecule, &params, &options).unwrap();

    let mut worst = 0.0_f64;
    for (axis, row) in MOPAC.iter().enumerate() {
        for (dof, expected) in row.iter().enumerate() {
            worst = worst.max((analytic[(axis, dof)] - expected).abs());
        }
    }
    // MOPAC's own central difference is only good to about this, which is what sets the bound.
    assert!(
        worst < 5.0e-5,
        "the analytic dipole derivatives differ from MOPAC's by {worst:.3e} e"
    );
    // And the tensor is not trivially zero, so the agreement means something.
    assert!(analytic[(0, 0)].abs() > 0.5);
}
