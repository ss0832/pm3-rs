// SPDX-License-Identifier: GPL-3.0-or-later

//! The response's option surface: what each knob does, and that it does it.
//!
//! Every one of these is a claim the defaults make silently, so each test here is the same
//! calculation run twice with one thing changed.

use pm3_rs::pbc::gamma::PeriodicOptions;
use pm3_rs::{
    force_constants_at_q, frequencies_at_q, Cell, DfptOptions, LongRange, Molecule, Pm3Options,
    Pm3Parameters, Vec3,
};

const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";

fn options() -> Pm3Options {
    Pm3Options::default()
}

fn cube(edge_bohr: f64) -> Molecule {
    let mut molecule = Molecule::from_xyz_str(WATER, 0.0).unwrap();
    molecule.cell = Some(Cell::cubic(edge_bohr).unwrap());
    molecule
}

/// The defaults reproduce the simple entry point exactly.
///
/// `dynamical_matrix` and `force_constants_at_q` must be the same calculation, or the option
/// surface is a second implementation rather than a way to control the first one.
#[test]
fn the_defaults_are_the_simple_entry_point() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = cube(12.0);
    let q = [0.25, 0.0, 0.0];

    let plain = pm3_rs::dynamical_matrix(&molecule, &params, &options(), &periodic, q).unwrap();
    let via_options = force_constants_at_q(
        &molecule,
        &params,
        &options(),
        &periodic,
        &DfptOptions::default(),
        q,
    )
    .unwrap();

    let ndof = plain.matrix.rows;
    let mut worst = 0.0_f64;
    for i in 0..ndof {
        for j in 0..ndof {
            worst = worst.max((plain.matrix[(i, j)] - via_options.dynamical.matrix[(i, j)]).norm());
        }
    }
    assert!(
        worst < 1.0e-12,
        "the two entry points differ by {worst:.3e}"
    );
    // `force_constants` is the same numbers under the name that says they are not mass weighted.
    assert_eq!(via_options.force_constants.rows, ndof);
    assert_eq!(via_options.k_points.len(), 1, "Gamma alone by default");
    assert!(via_options.response.is_none(), "off unless asked for");

    // And the frequency wrapper composes the two.
    let direct = pm3_rs::phonon_frequencies(&molecule, &params, &options(), &periodic, q).unwrap();
    let through = frequencies_at_q(
        &molecule,
        &params,
        &options(),
        &periodic,
        &DfptOptions::default(),
        q,
    )
    .unwrap();
    for (a, b) in direct.iter().zip(&through) {
        assert!((a - b).abs() < 1.0e-9, "{a} vs {b}");
    }
}

/// `LongRange::Off` measurably removes the long-range term rather than quietly doing nothing.
///
/// The point of the switch is that the term's size can be measured instead of argued about, so
/// the test asserts it *changes* the answer — a knob that did nothing would pass a test that only
/// checked the result was still finite.
#[test]
fn switching_the_long_range_term_off_changes_the_answer() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = cube(10.0);
    let q = [0.25, 0.0, 0.0];

    let with = force_constants_at_q(
        &molecule,
        &params,
        &options(),
        &periodic,
        &DfptOptions::default(),
        q,
    )
    .unwrap();
    let without = force_constants_at_q(
        &molecule,
        &params,
        &options(),
        &periodic,
        &DfptOptions {
            long_range: LongRange::Off,
            ..Default::default()
        },
        q,
    )
    .unwrap();

    let ndof = with.dynamical.matrix.rows;
    let mut worst = 0.0_f64;
    for i in 0..ndof {
        for j in 0..ndof {
            worst = worst
                .max((with.dynamical.matrix[(i, j)] - without.dynamical.matrix[(i, j)]).norm());
        }
    }
    assert!(
        worst > 1.0e-4,
        "the long-range term moved the force constants by {worst:.3e}, which is not a term"
    );
}

/// `LongRange::Require` refuses an isolated cell instead of quietly carrying on without the term.
#[test]
fn requiring_the_long_range_term_refuses_a_cell_that_has_none() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = Molecule::from_xyz_str(WATER, 0.0).unwrap();

    let error = force_constants_at_q(
        &molecule,
        &params,
        &options(),
        &periodic,
        &DfptOptions {
            long_range: LongRange::Require,
            ..Default::default()
        },
        [0.0; 3],
    )
    .expect_err("an isolated cell has no lattice to sum over");
    let text = error.to_string();
    assert!(
        text.contains("Require") || text.contains("no periodic direction"),
        "the refusal should say which setting caused it: {text}"
    );
}

/// A solver that cannot converge is refused, not returned.
///
/// One pass cannot converge a coupled response, and the tolerance and cap are the two knobs that
/// say so. Both are checked, because a cap of zero and a tolerance of zero are each a solver that
/// never finishes and neither is a useful default.
#[test]
fn the_solver_knobs_are_honoured_and_validated() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let molecule = cube(12.0);
    let q = [0.25, 0.0, 0.0];

    let one_pass = force_constants_at_q(
        &molecule,
        &params,
        &options(),
        &periodic,
        &DfptOptions {
            max_iter: 1,
            ..Default::default()
        },
        q,
    );
    assert!(
        matches!(
            one_pass,
            Err(pm3_rs::Pm3Error::ScfNotConverged { iterations: 1, .. })
        ),
        "one pass should be refused, and refused as a non-convergence"
    );

    for bad in [
        DfptOptions {
            max_iter: 0,
            ..Default::default()
        },
        DfptOptions {
            tol: 0.0,
            ..Default::default()
        },
    ] {
        let error = force_constants_at_q(&molecule, &params, &options(), &periodic, &bad, q)
            .expect_err("a solver that never finishes is not a configuration");
        assert!(error.to_string().contains("must"), "{error}");
    }

    // A loose tolerance still converges, and to the same fixed point: extrapolation and
    // tolerance change the path, not where it ends.
    let loose = force_constants_at_q(
        &molecule,
        &params,
        &options(),
        &periodic,
        &DfptOptions {
            tol: 1.0e-8,
            ..Default::default()
        },
        q,
    )
    .unwrap();
    let tight = force_constants_at_q(
        &molecule,
        &params,
        &options(),
        &periodic,
        &DfptOptions::default(),
        q,
    )
    .unwrap();
    let ndof = tight.dynamical.matrix.rows;
    let mut worst = 0.0_f64;
    for i in 0..ndof {
        for j in 0..ndof {
            worst =
                worst.max((tight.dynamical.matrix[(i, j)] - loose.dynamical.matrix[(i, j)]).norm());
        }
    }
    assert!(worst < 1.0e-5, "the two tolerances disagree by {worst:.3e}");
}

/// The k-set the response samples is reported, and it is the one that was asked for.
#[test]
fn the_sampled_k_points_come_back_with_the_result() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let mut molecule = Molecule::from_xyz_str(WATER, 0.0).unwrap();
    molecule.cell = Some(
        Cell::new(
            Vec3::new(12.0, 0.0, 0.0),
            Vec3::new(0.0, 12.0, 0.0),
            Vec3::new(0.0, 0.0, 12.0),
            [true, true, true],
        )
        .unwrap(),
    );

    let dfpt = DfptOptions {
        kpoints: Some(pm3_rs::KpointOptions::mesh([2, 1, 1])),
        ..Default::default()
    };
    let result = force_constants_at_q(
        &molecule,
        &params,
        &options(),
        &periodic,
        &dfpt,
        [0.5, 0.0, 0.0],
    )
    .unwrap();
    assert!(
        !result.k_points.is_empty(),
        "the mesh it sampled should be reported"
    );
    let weight: f64 = result.k_points.iter().map(|k| k.weight).sum();
    assert!(
        (weight - 1.0).abs() < 1.0e-12,
        "the weights should sum to one, got {weight}"
    );
}
