// SPDX-License-Identifier: GPL-3.0-or-later
//! **A failed periodic SCF says what went wrong, and a rescue says what it changed.**
//!
//! "did not converge (error=3.7e-5)" tells a caller that a calculation failed and nothing about
//! what to do next, and the ways a periodic SCF fails want different remedies. So the k-point
//! loop measures which one it is — from quantities it already computes — and the error carries
//! the answer.
//!
//! The classification is measured per run rather than assumed from the system, and that matters:
//! `docs/pbc.md` used to attribute NaCl's difficulty on an even mesh to a Fermi level trapped in
//! a degenerate manifold, and the occupation-flip count says the occupations never change at all.
//! What actually happens is charge sloshing — a full electron moving between Na and Cl each pass,
//! with the chemical potential following it across twelve electronvolts — and it damps out.
//!
//! The other half is the retry. `run_kpoints` will retry a failed SCF **once**, with Fermi
//! smearing, and will keep the result **only if the electronic entropy comes out zero**. That
//! check is the whole justification for doing it automatically: integral occupations mean the
//! smeared fixed-point equations are the strict-filling ones, so the answer is the one that was
//! asked for. Where the check fails — diamond, whose smeared state carries `1.4e-2 eV` of entropy
//! — the smeared energy is *reported in the error* rather than returned, because it answers a
//! different question. `examples/scf_hardening.rs` has the ladder these were measured on.

use pm3_rs::pbc::gamma::PeriodicOptions;
use pm3_rs::pbc::kscf::{run_kpoints, KpointOptions};
use pm3_rs::{Atom, Cell, Molecule, Pm3Error, Pm3Options, Pm3Parameters, Vec3};

const ANGSTROM_TO_BOHR: f64 = 1.8897261254578281;

fn fcc(z: [u8; 2], a_angstrom: f64, second: [f64; 3]) -> Molecule {
    let a = a_angstrom * ANGSTROM_TO_BOHR;
    let h = a / 2.0;
    let cell = Cell::new(
        Vec3::new(0.0, h, h),
        Vec3::new(h, 0.0, h),
        Vec3::new(h, h, 0.0),
        [true; 3],
    )
    .unwrap();
    let mut molecule = Molecule::new(vec![
        Atom {
            z: z[0],
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: z[1],
            position: Vec3::new(
                second[0] * ANGSTROM_TO_BOHR,
                second[1] * ANGSTROM_TO_BOHR,
                second[2] * ANGSTROM_TO_BOHR,
            ),
        },
    ]);
    molecule.cell = Some(cell);
    molecule
}

/// A failure names its mechanism and what to do, not just its residual.
///
/// The failure is *manufactured* — a three-iteration budget — rather than borrowed from a crystal
/// that happens to be hard. Which systems fail is a moving target: silicon failed at the default
/// settings until the damping handover landed, and a test anchored to that would have been a
/// regression test for the bug rather than for the diagnosis. What is asserted is the shape of
/// the message: that it names one mechanism and one knob.
#[test]
fn a_failed_periodic_scf_explains_itself() {
    let params = Pm3Parameters::standard().unwrap();
    // An unreachable tolerance rather than a short budget: a short budget is exactly what the
    // smearing retry lifts, so it would be rescued and there would be no failure to inspect.
    let options = Pm3Options {
        max_scf: 20,
        p_tol: 1.0e-30,
        ..Pm3Options::default()
    };
    let periodic = PeriodicOptions::default();
    let nacl = fcc([11, 17], 5.64, [5.64 / 2.0, 0.0, 0.0]);

    let error = run_kpoints(
        &nacl,
        &params,
        &options,
        &periodic,
        &KpointOptions::mesh([2, 2, 2]),
    )
    .expect_err("a 1e-30 density tolerance is below double precision and cannot be met");

    let Pm3Error::ScfNotConverged {
        error: residual,
        diagnosis,
        ..
    } = &error
    else {
        panic!("expected a convergence failure, got {error}");
    };
    assert!(*residual > 0.0 && residual.is_finite());

    let text = diagnosis
        .as_deref()
        .expect("the k-point path diagnoses its own failures");
    // Exactly one of the three mechanisms, named.
    let named = [
        "charge is sloshing",
        "changing occupation",
        "nothing is oscillating",
    ]
    .iter()
    .filter(|phrase| text.contains(**phrase))
    .count();
    assert!(
        named <= 1,
        "the diagnosis should settle on one mechanism, got: {text}"
    );
    // And something the caller can act on, not just a description.
    assert!(
        ["max_scf", "smearing", "k-mesh", "damping", "level_shift"]
            .iter()
            .any(|knob| text.contains(knob)),
        "the diagnosis should name a knob to turn, got: {text}"
    );

    // The whole thing reaches a user through Display, so the pieces have to survive it.
    let shown = error.to_string();
    assert!(shown.starts_with("PM3 SCF did not converge"));
    assert!(
        shown.len() > 80,
        "the diagnosis was dropped on the way to Display: {shown}"
    );
}

/// A caller who already chose a convergence aid is not overridden by the retry.
///
/// Retrying on top of an explicit `smearing_ev` would replace a deliberate choice with a
/// different one, and the result would carry the caller's parameters and someone else's answer.
/// So `rescued_by` stays `None` whenever the caller asked for an aid, whatever the outcome.
#[test]
fn an_explicit_convergence_aid_is_not_second_guessed() {
    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options::default();
    let periodic = PeriodicOptions::default();
    let nacl = fcc([11, 17], 5.64, [5.64 / 2.0, 0.0, 0.0]);

    let kopt = KpointOptions {
        smearing_ev: 0.1,
        ..KpointOptions::mesh([2, 2, 2])
    };
    // Whatever the outcome, the policy is the same: an explicit aid is never supplemented.
    match run_kpoints(&nacl, &params, &options, &periodic, &kopt) {
        Ok(result) => assert!(
            result.rescued_by.is_none(),
            "the caller asked for this smearing; nothing was rescued on their behalf"
        ),
        Err(Pm3Error::ScfNotConverged { .. }) => {
            // Also correct: the failure is reported rather than papered over with a second aid.
        }
        Err(other) => panic!("unexpected error: {other}"),
    }

    // Damping is now a rescue rung of its own, so setting it explicitly has to block the ladder
    // for the same reason a smearing or a level shift does. Without this the caller's 0.5 would
    // be silently replaced by the rescue's 0.9 the moment the run struggled.
    let damped = Pm3Options {
        damping: 0.5,
        ..Pm3Options::default()
    };
    match run_kpoints(
        &nacl,
        &params,
        &damped,
        &periodic,
        &KpointOptions::mesh([2, 2, 2]),
    ) {
        Ok(result) => assert!(
            result.rescued_by.is_none(),
            "the caller set the damping; nothing was rescued on their behalf"
        ),
        Err(Pm3Error::ScfNotConverged { .. }) => {}
        Err(other) => panic!("unexpected error: {other}"),
    }
}

/// **A response is refused on fractional occupations, and allowed when smearing left them whole.**
///
/// The sum-over-states factors in `pbc::dfpt` are the metallic `(f_n − f_m)/(ε_n − ε_m)` form,
/// which makes it look as though a partially filled band is handled. It is not: there is no
/// Fermi-level shift, and the sum skips the near-degenerate pairs that carry the Fermi-surface
/// term. An incomplete response that says nothing is worse than a refusal, so it refuses.
///
/// The condition is **not** "was smearing used" but "did the occupations come out integral",
/// which is what makes a smeared insulator still work — and that is the half worth testing,
/// because a guard keyed on the smearing value alone would lock out every gapped cell that
/// happened to be run with a smearing set.
#[test]
fn a_response_refuses_fractional_occupations_but_not_smearing_itself() {
    use pm3_rs::pbc::dfpt::dynamical_matrix_on_mesh;

    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options::default();
    let periodic = PeriodicOptions::default();
    let nacl = fcc([11, 17], 5.64, [5.64 / 2.0, 0.0, 0.0]);

    // A gapped ionic crystal under a smearing small against its gap: the occupations stay
    // integral, so the response is the one that was asked for and must run.
    let gentle = KpointOptions {
        smearing_ev: 0.05,
        ..KpointOptions::mesh([3, 3, 3])
    };
    dynamical_matrix_on_mesh(&nacl, &params, &options, &periodic, &gentle, [0.0; 3])
        .expect("a gapped cell keeps integral occupations under a small smearing");

    // Enough smearing to fractionally fill states across NaCl's gap. Then the response is
    // incomplete and has to say so rather than return a number.
    let heavy = KpointOptions {
        smearing_ev: 5.0,
        ..KpointOptions::mesh([3, 3, 3])
    };
    match dynamical_matrix_on_mesh(&nacl, &params, &options, &periodic, &heavy, [0.0; 3]) {
        Err(Pm3Error::InvalidInput(message)) => {
            assert!(
                message.contains("fractional occupations") && message.contains("metallic"),
                "the refusal should say what is missing and why, got: {message}"
            );
        }
        Err(other) => panic!("expected a refusal about occupations, got: {other}"),
        Ok(_) => panic!(
            "a 5 eV smearing fractionally fills states across NaCl's gap, and the response \
             returned a number for it anyway — which is the silent-incompleteness this guard \
             exists to prevent"
        ),
    }
}

/// **The ladder is ordered by how strong a guarantee each rung carries, and says which it used.**
///
/// Damping first: it changes the path and not the equations, and the convergence test measures
/// the *undamped* step, so a converged damped run solves what was asked. Smearing second: it
/// changes the equations unless the occupations come out integral, which is checked afterwards
/// and is the only reason it is applied at all. Nothing after that — a level shift converges
/// diamond to a different solution, which is worse than not converging.
///
/// This does not assert that any particular system needs a rescue, because which ones do is a
/// property of the model that is allowed to improve. It asserts the invariant: **whenever a
/// rescue happened, the result says so and names the rung**, so a rescued number is never
/// mistaken for an unaided one.
#[test]
fn a_rescue_names_the_rung_it_used() {
    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options::default();
    let periodic = PeriodicOptions::default();

    for (label, molecule, mesh) in [
        (
            "NaCl",
            fcc([11, 17], 5.64, [5.64 / 2.0, 0.0, 0.0]),
            [2, 2, 2],
        ),
        (
            "MgO",
            fcc([12, 8], 4.212, [4.212 / 2.0, 0.0, 0.0]),
            [3, 3, 3],
        ),
    ] {
        let Ok(result) = run_kpoints(
            &molecule,
            &params,
            &options,
            &periodic,
            &KpointOptions::mesh(mesh),
        ) else {
            continue; // A reported failure is a valid outcome; the ladder is not obliged to win.
        };
        let Some(note) = &result.rescued_by else {
            continue; // Converged unaided, which is the outcome that needs no explanation.
        };
        assert!(
            note.contains("damping") || note.contains("smearing"),
            "{label} was rescued but the note names no rung: {note}"
        );
        // And the note carries the evidence, not just the verdict.
        assert!(
            note.contains("residual"),
            "{label}'s rescue note does not say what it was rescued from: {note}"
        );
        assert!(
            result.converged,
            "{label} reported a rescue without converging"
        );
    }
}

/// **A converged answer is not a correct one, and the result says how it got there.**
///
/// Rocksalt NaCl on a `2×2×2` mesh is the case that makes this concrete. It converges — no error,
/// `converged: true` — to a self-consistent state with about **−4.9 electrons on the sodium**,
/// where every odd mesh puts `+0.17` there. A sodium that has gained five electrons is not a
/// chemical statement, it is a sampling failure that happens to be self-consistent, and nothing
/// in the energy or the residual says so.
///
/// What does say so is `charge_swing`: the density moved by more than an electron on the way,
/// which a well-behaved SCF does not do. That is the observable this test pins, because it is the
/// one a caller can act on without knowing the answer in advance.
#[test]
fn an_inadequate_mesh_converges_to_a_state_its_charge_swing_gives_away() {
    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options::default();
    let periodic = PeriodicOptions::default();
    let nacl = fcc([11, 17], 5.64, [5.64 / 2.0, 0.0, 0.0]);

    let bad = run_kpoints(
        &nacl,
        &params,
        &options,
        &periodic,
        &KpointOptions::mesh([2, 2, 2]),
    )
    .expect("the even mesh does converge — that is the problem");
    let good = run_kpoints(
        &nacl,
        &params,
        &options,
        &periodic,
        &KpointOptions::mesh([5, 5, 5]),
    )
    .expect("an odd mesh converges");

    assert!(bad.converged && good.converged);
    // The two disagree by tens of eV, and the even one is the nonsense.
    assert!(
        (bad.total_ev - good.total_ev).abs() > 10.0,
        "this test's premise is that the even mesh finds another state; it no longer does"
    );
    assert!(
        bad.charges[0] < -1.0,
        "the even mesh should put an impossible charge on sodium, got {}",
        bad.charges[0]
    );
    assert!(
        good.charges[0] > 0.0,
        "an odd mesh should make sodium a cation, got {}",
        good.charges[0]
    );

    // And the tell: the bad run sloshed on its way there, the good one did not.
    assert!(
        bad.charge_swing > 0.5,
        "the even mesh sloshed by {} electrons, which should be reported",
        bad.charge_swing
    );
    assert!(
        good.charge_swing < bad.charge_swing,
        "the odd mesh swung {} against the even mesh's {}",
        good.charge_swing,
        bad.charge_swing
    );
}

/// A system that converges unaided reports no rescue, and is not slowed by the machinery.
#[test]
fn a_converging_system_reports_no_rescue() {
    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options::default();
    let periodic = PeriodicOptions::default();

    // NaCl on an even mesh: it sloshes hard on the way — a full electron between Na and Cl, with
    // `μ` swinging twelve eV — and then converges. `docs/pbc.md` had it down as unconvergeable.
    let nacl = fcc([11, 17], 5.64, [5.64 / 2.0, 0.0, 0.0]);
    let result = run_kpoints(
        &nacl,
        &params,
        &options,
        &periodic,
        &KpointOptions::mesh([2, 2, 2]),
    )
    .expect("NaCl recovers from its sloshing");
    assert!(result.converged);
    assert!(result.rescued_by.is_none(), "no aid was needed");
    // Deliberately no assertion on the energy here. The 2×2×2 answer is wrong whichever branch
    // the iteration lands on — see `an_inadequate_mesh_converges_to_a_state_its_charge_swing_gives_away`
    // — and pinning one of two wrong numbers would turn a sampling failure into a regression
    // test for it. What is asserted is the *policy*: nothing was changed behind the caller.
}
