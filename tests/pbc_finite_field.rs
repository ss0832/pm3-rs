// SPDX-License-Identifier: GPL-3.0-or-later

//! The Berry-phase finite field, and the polarizability it has to reproduce.
//!
//! [`the_finite_field_reproduces_the_coupled_perturbed_polarizability`] is the point of this file.
//! `α = Ω ∂P/∂𝓔` from a self-consistent electric enthalpy, against the CPHF `α` from
//! `pbc::dielectric` — two formalisms sharing only the SCF. The field operator carries a `J`, a
//! `4π`, a sign, and a Hermitization; each of those is invisible inside its own derivation and
//! visible across the pair. One of them was in fact wrong, and this is what found it.
//!
//! # Why hydrogen
//!
//! The Berry phase places every orbital at its atom's centre, so it carries no intra-atomic
//! `s`–`p` moment `dd`, while the CPHF route's dipole operator does. On a cell containing `dd`
//! the two therefore *should* disagree, and by an amount that is not independently known — on
//! `HF` the ratio sits at 1.455.
//!
//! Hydrogen has no `p` orbitals, so `dd` is identically zero for every atom and both routes carry
//! the same position operator by construction. That turns a comparison needing a caveat into one
//! that can be asserted tightly, with no diagnostic switch and nothing to trust.
//!
//! This is a semiempirical model, so neither number is a prediction of any measurement; what is
//! being checked is that the crate computes its own model's polarizability consistently by two
//! independent routes.

use pm3_rs::pbc::gamma::PeriodicOptions;
use pm3_rs::{Cell, FiniteFieldOptions, Molecule, Pm3Options, Pm3Parameters, Vec3};

const HARTREE_TO_EV: f64 = 27.211386245988;
/// No `p` orbitals anywhere, so `dd = 0` and both routes share the position operator.
const H2: &str = "2\nhydrogen\nH 0.0 0.0 0.0\nH 0.74 0.0 0.0\n";
/// Polar, for the tests that need a polarization that is not zero. `dd` is irrelevant where both
/// sides of the comparison are Berry-phase routes.
const HF: &str = "2\nhydrogen fluoride\nF 0.0 0.0 0.0\nH 0.93 0.0 0.0\n";
const EDGE_BOHR: f64 = 20.0;

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

/// `alpha_xx = Omega dP_x/dE_x` by central differences of the finite field.
fn finite_field_alpha(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    divisions: [usize; 3],
    strength: f64,
) -> f64 {
    let ff = FiniteFieldOptions::default();
    let at = |sign: f64| {
        pm3_rs::run_finite_field(
            molecule,
            params,
            options,
            periodic,
            divisions,
            Vec3::new(sign * strength, 0.0, 0.0),
            &ff,
        )
        .expect("the finite field should converge at this strength")
    };
    let (minus, plus) = (at(-1.0), at(1.0));
    let dp = (plus.polarization - minus.polarization) / (2.0 * strength);
    // `P` is e/Bohr^2, `Omega` Bohr^3, `E` eV/(e Bohr), so this is e^2 Bohr^2/eV. Times Hartree
    // per eV gives e^2 Bohr^2/Hartree = Bohr^3, since e^2/Hartree = Bohr -- the same conversion
    // `dielectric.rs` applies, for the same reason.
    dp.x * molecule.cell.unwrap().measure() * HARTREE_TO_EV
}

/// **The cross-check.** The electric enthalpy gives the CPHF polarizability.
///
/// # What this found
///
/// A factor of two. The field operator is built by projecting the enthalpy gradient onto the
/// occupied manifold, `M = i λ (W₊ − W₋) C†`, and the conventional way to make such a thing
/// Hermitian is `½(M + M†)`. That is wrong here: `M` is one-sided — `M|v⟩ = 0` for a virtual `v`,
/// since `C†` annihilates everything outside the occupied span — so it holds the entire
/// virtual-occupied block and none of the occupied-virtual one. Averaging halves the
/// occupied-virtual block, which is exactly the block a linear response is made of, and the
/// polarizability came out at 0.5001 of the CPHF value. `M + M†` fills the two disjoint blocks
/// once each and gives 1.0001.
///
/// The ratio is asserted rather than the value, because the value is a property of the model and
/// the ratio is a property of the implementation.
#[test]
fn the_finite_field_reproduces_the_coupled_perturbed_polarizability() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let options = options();
    let molecule = boxed(H2, EDGE_BOHR);

    let margin = pm3_rs::run_gamma(&molecule, &params, &options, &periodic)
        .unwrap()
        .gamma_margin;
    assert!(margin > 0.0, "the Gamma margin is {margin:.3} Bohr");

    let cphf = pm3_rs::polarizability(&molecule, &params, &options, &periodic).unwrap();
    // Non-vacuity: two routes that both returned zero would agree perfectly.
    assert!(
        cphf[0][0] > 1.0,
        "the CPHF alpha_xx is {:.4} Bohr^3; there is no response here to compare",
        cphf[0][0]
    );

    let alpha = finite_field_alpha(&molecule, &params, &options, &periodic, [8, 1, 1], 2.0e-4);
    let ratio = alpha / cphf[0][0];
    eprintln!(
        "finite field {alpha:.5} vs CPHF {:.5} Bohr^3, ratio {ratio:.5}",
        cphf[0][0]
    );
    assert!(
        (ratio - 1.0).abs() < 5.0e-3,
        "the finite-field alpha_xx is {alpha:.5} Bohr^3 against the CPHF {:.5} (ratio \
         {ratio:.5}). These share the SCF and nothing else, so a ratio near ½ is the \
         Hermitization of the field operator, near 2 a double count, and a sign is a sign.",
        cphf[0][0]
    );
}

/// The answer is a linear response: independent of the field strength that measured it.
///
/// A finite difference of a self-consistent quantity is only a derivative if the interval is
/// inside the linear regime. If it is not, the number above is a property of the step.
#[test]
fn the_polarizability_does_not_depend_on_the_field_that_measured_it() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let options = options();
    let molecule = boxed(H2, EDGE_BOHR);

    let weak = finite_field_alpha(&molecule, &params, &options, &periodic, [6, 1, 1], 2.0e-4);
    let strong = finite_field_alpha(&molecule, &params, &options, &periodic, [6, 1, 1], 5.0e-4);
    assert!(
        (weak / strong - 1.0).abs() < 1.0e-3,
        "alpha_xx is {weak:.6} at a field of 2e-4 and {strong:.6} at 5e-4; the finite difference \
         is not inside the linear regime"
    );
}

/// Refining the string moves the answer less each time.
#[test]
fn the_answer_converges_in_the_string_length() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let options = options();
    let molecule = boxed(H2, EDGE_BOHR);

    let coarse = finite_field_alpha(&molecule, &params, &options, &periodic, [4, 1, 1], 2.0e-4);
    let medium = finite_field_alpha(&molecule, &params, &options, &periodic, [6, 1, 1], 2.0e-4);
    let fine = finite_field_alpha(&molecule, &params, &options, &periodic, [8, 1, 1], 2.0e-4);

    let first = (medium - coarse).abs();
    let second = (fine - medium).abs();
    assert!(
        second <= first + 1.0e-9,
        "refining 6 -> 8 moved alpha by {second:.3e}, more than 4 -> 6 did ({first:.3e})"
    );
    eprintln!("string convergence: 4->6 {first:.3e}, 6->8 {second:.3e}");
}

/// The reported enthalpy is `E − Ω 𝓔·P`, from the pieces the result also reports.
#[test]
fn the_enthalpy_is_the_quantity_it_says_it_is() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let options = options();
    let molecule = boxed(H2, EDGE_BOHR);
    let volume = molecule.cell.unwrap().measure();
    let field = Vec3::new(1.0e-3, 0.0, 0.0);

    let result = pm3_rs::run_finite_field(
        &molecule,
        &params,
        &options,
        &periodic,
        [6, 1, 1],
        field,
        &FiniteFieldOptions::default(),
    )
    .unwrap();

    assert!(result.converged);
    assert!(result.iterations > 0);
    let expected = result.scf.total_ev - volume * field.dot(result.polarization);
    assert!(
        (result.enthalpy_ev - expected).abs() < 1.0e-9,
        "the enthalpy is {:.9} but E - Omega E.P from the reported pieces is {expected:.9}",
        result.enthalpy_ev
    );
    // And the two halves really do sum to the total that was used.
    let sum = result.electronic_polarization + result.ionic_polarization;
    assert!((sum - result.polarization).norm() < 1.0e-12);
    assert!(
        result.ionic_polarization.norm() > 1.0e-6,
        "the ionic polarization is zero, so none of this was exercised"
    );
}

/// The polarization is a property of the state, not of which axes the field happens to touch.
///
/// # The bug this catches
///
/// The first version computed the Berry phase only along the axes the field coupled to, because
/// those are the axes that need a `ΔH`. Those are not the same set. An axis the field misses
/// still carries polarization, so a zero field — or one orthogonal to every lattice vector —
/// came back with an electronic polarization of exactly zero, which is not its value. The state
/// was right; the number reported about it was not, and nothing in the result said so.
///
/// Asserted against `pbc::berry` at zero field, which is the same quantity by an entirely
/// separate code path, and against `resolved` for the axes a coarse mesh cannot see.
#[test]
fn the_polarization_covers_every_axis_the_mesh_resolves() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let options = options();
    // Polar, so there is a polarization to compare. Both sides here are Berry-phase routes, so
    // the `dd` moment that separates this family from the CPHF one does not enter.
    let molecule = boxed(HF, EDGE_BOHR);
    let ff = FiniteFieldOptions::default();

    // A mesh that resolves all three axes, and no field at all.
    let zero = pm3_rs::run_finite_field(
        &molecule,
        &params,
        &options,
        &periodic,
        [4, 4, 4],
        Vec3::zero(),
        &ff,
    )
    .unwrap();
    assert_eq!(zero.resolved, [true, true, true]);

    // At zero field this must be the ordinary Berry-phase polarization, computed by a module
    // that shares none of this code.
    let reference = pm3_rs::berry_polarization(
        &molecule,
        &params,
        &options,
        &periodic,
        &pm3_rs::KpointOptions::mesh([4, 4, 4]),
        4,
    )
    .unwrap();
    let gap = (zero.polarization - reference.total).norm();
    assert!(
        gap < 1.0e-6,
        "at zero field the enthalpy route gives {:?} and `pbc::berry` gives {:?} (gap {gap:.3e}); \
         these are the same quantity",
        zero.polarization,
        reference.total
    );
    // Non-vacuity: a polarization of zero would satisfy the comparison without testing anything.
    assert!(
        reference.total.norm() > 1.0e-6,
        "the reference polarization is zero, so nothing was compared"
    );

    // And a mesh that resolves only one axis says so rather than reporting a zero.
    let coarse = pm3_rs::run_finite_field(
        &molecule,
        &params,
        &options,
        &periodic,
        [6, 1, 1],
        Vec3::new(1.0e-3, 0.0, 0.0),
        &ff,
    )
    .unwrap();
    assert_eq!(
        coarse.resolved,
        [true, false, false],
        "a 6x1x1 mesh resolves x alone, and the result has to say so"
    );
}

/// What the finite field refuses.
#[test]
fn what_the_finite_field_refuses() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let options = options();
    let ff = FiniteFieldOptions::default();
    let field = Vec3::new(1.0e-3, 0.0, 0.0);
    let molecule = boxed(H2, EDGE_BOHR);

    // Two points cannot resolve a winding, so a string needs three.
    let error = pm3_rs::run_finite_field(
        &molecule,
        &params,
        &options,
        &periodic,
        [2, 1, 1],
        field,
        &ff,
    )
    .expect_err("a two-point string is not a string");
    assert!(error.to_string().contains("at least 3"), "{error}");

    // A chain: the polarization quantum needs a volume.
    let mut chain = Molecule::from_xyz_str(H2, 0.0).unwrap();
    chain.cell = Some(
        Cell::new(
            Vec3::new(EDGE_BOHR, 0.0, 0.0),
            Vec3::new(0.0, 30.0, 0.0),
            Vec3::new(0.0, 0.0, 30.0),
            [true, false, false],
        )
        .unwrap(),
    );
    let error =
        pm3_rs::run_finite_field(&chain, &params, &options, &periodic, [6, 1, 1], field, &ff)
            .expect_err("a chain has no volume");
    assert!(error.to_string().contains("three-dimensional"), "{error}");

    // The same perturbation applied twice.
    let doubled = Pm3Options {
        field: Some(Vec3::new(0.0, 0.0, 1.0e-3)),
        ..options.clone()
    };
    let error = pm3_rs::run_finite_field(
        &molecule,
        &params,
        &doubled,
        &periodic,
        [6, 1, 1],
        field,
        &ff,
    )
    .expect_err("`Pm3Options::field` and this are two treatments of one perturbation");
    assert!(error.to_string().contains("two treatments"), "{error}");

    // An open shell would need each spin manifold's phase separately.
    let open = Pm3Options {
        multiplicity: 3,
        ..options.clone()
    };
    let error =
        pm3_rs::run_finite_field(&molecule, &params, &open, &periodic, [6, 1, 1], field, &ff)
            .expect_err("the Berry-phase field is restricted-only");
    assert!(error.to_string().contains("restricted-only"), "{error}");

    // And a legitimate call still works, so the refusals above are about what they name.
    assert!(pm3_rs::run_finite_field(
        &molecule,
        &params,
        &options,
        &periodic,
        [6, 1, 1],
        field,
        &ff
    )
    .is_ok());
}
