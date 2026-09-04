// SPDX-License-Identifier: GPL-3.0-or-later

//! Real crystals, one per feature the water-in-a-box unit tests cannot reach.
//!
//! The unit tests establish that the lattice sums are counted right and that the derivatives are
//! derivatives. They do it on a molecule in a large cell, which is convenient precisely because it
//! is nearly isolated — and that is also the limit of what they can say. Nothing about them
//! exercises a bond that runs through a cell boundary, an ion, or a metal.
//!
//! | system | tests |
//! |---|---|
//! | NaCl rocksalt | ionic binding carried by the lattice sum |
//! | graphene | bonding through images; the exact 2D Ewald against 3D-with-vacuum |
//! | polyacetylene | the Peierls instability — dimerization lowers the energy and opens a gap |
//!
//! Absolute energies are not asserted against anything. PM3 was parameterized for molecules and
//! there is no oracle for these systems; what is asserted is the qualitative physics each one is
//! known for, which is exactly what a wrong lattice sum or k-point convention destroys.
//!
//! Two limitations are pinned here rather than left latent — see
//! `a_one_dimensional_cell_is_refused` and the note on rocksalt's lattice constant.

use pm3_rs::pbc::gamma::{run_gamma, PeriodicOptions};
use pm3_rs::pbc::kscf::{run_kpoints, KpointOptions};
use pm3_rs::{Atom, Cell, KpointSpec, Molecule, Pm3Options, Pm3Parameters, Vec3};

const ANGSTROM: f64 = 1.8897261254578281;

fn options() -> Pm3Options {
    Pm3Options {
        max_scf: 500,
        ..Pm3Options::default()
    }
}

fn atom(z: u8, x: f64, y: f64, z_coordinate: f64) -> Atom {
    Atom {
        z,
        position: Vec3::new(x, y, z_coordinate) * ANGSTROM,
    }
}

fn mesh(divisions: [usize; 3], smearing_ev: f64) -> KpointOptions {
    KpointOptions {
        spec: KpointSpec::mesh(divisions),
        smearing_ev,
        ..KpointOptions::default()
    }
}

/// Conventional rocksalt cell: four NaCl formula units, lattice constant `a` in Ångström.
fn rocksalt(a: f64) -> Molecule {
    let mut atoms = Vec::new();
    for site in [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ] {
        atoms.push(atom(11, site[0] * a, site[1] * a, site[2] * a));
    }
    for site in [
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.5],
    ] {
        atoms.push(atom(17, site[0] * a, site[1] * a, site[2] * a));
    }
    let mut molecule = Molecule::new(atoms);
    molecule.cell = Some(Cell::cubic(a * ANGSTROM).unwrap());
    molecule
}

/// Graphene: two carbons in a hexagonal cell, `a` in Ångström, with `vacuum` along `z`.
///
/// `three_d` chooses whether that vacuum direction is periodic — which is the whole point of the
/// comparison below.
fn graphene(a: f64, vacuum: f64, three_d: bool) -> Molecule {
    let atoms = vec![
        atom(6, 0.0, 0.0, 0.0),
        atom(6, a / 2.0, a / (2.0 * 3.0_f64.sqrt()), 0.0),
    ];
    let mut molecule = Molecule::new(atoms);
    molecule.cell = Some(
        Cell::new(
            Vec3::new(a, 0.0, 0.0) * ANGSTROM,
            Vec3::new(-a / 2.0, a * 3.0_f64.sqrt() / 2.0, 0.0) * ANGSTROM,
            Vec3::new(0.0, 0.0, vacuum) * ANGSTROM,
            [true, true, three_d],
        )
        .unwrap(),
    );
    molecule
}

/// Trans-polyacetylene, two CH units per cell along `x`, in a 3D cell with vacuum.
///
/// `alternation` is the difference between the long and short C–C bonds; zero is the uniform chain
/// the Peierls theorem says cannot be stable.
fn polyacetylene(alternation: f64, one_dimensional: bool) -> Molecule {
    let mean = 1.40;
    let short = mean - 0.5 * alternation;
    let long = mean + 0.5 * alternation;
    let half_angle = 60.0_f64.to_radians();
    let dx_short = short * half_angle.sin();
    let dx_long = long * half_angle.sin();
    let dy = short * half_angle.cos();
    let period = dx_short + dx_long;

    let atoms = vec![
        atom(6, 0.0, 0.0, 0.0),
        atom(6, dx_short, dy, 0.0),
        atom(1, -0.55, -0.94, 0.0),
        atom(1, dx_short + 0.55, dy + 0.94, 0.0),
    ];
    let mut molecule = Molecule::new(atoms);
    molecule.cell = Some(
        Cell::new(
            Vec3::new(period, 0.0, 0.0) * ANGSTROM,
            Vec3::new(0.0, 18.0, 0.0) * ANGSTROM,
            Vec3::new(0.0, 0.0, 18.0) * ANGSTROM,
            [true, !one_dimensional, !one_dimensional],
        )
        .unwrap(),
    );
    molecule
}

/// An ionic crystal must be bound, and bound by its lattice sum.
///
/// NaCl is the case where the Ewald sum is not a correction but the whole story: take it away and
/// the cell is a collection of atoms with nothing holding them together.
///
/// Run at 8 Å rather than the experimental 5.64 Å, and the reason is physics rather than
/// convergence — see `rocksalt_converges_at_its_experimental_lattice_constant`, which now covers
/// that spacing. At 8 Å PM3 puts ±0.87 electrons on the ions and binds the cell by 15 eV, which
/// is what "ionic, and bound by its lattice sum" is supposed to mean. Compressed to 5.64 Å the
/// same method collapses the separation to ±0.16 while binding *more* strongly, and a test that
/// tried to assert ionicity there would be asserting something PM3 does not say.
#[test]
fn rocksalt_is_bound_and_ionic() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let kopt = mesh([3, 3, 3], 0.1);

    let crystal = run_kpoints(&rocksalt(8.0), &params, &options(), &periodic, &kopt).unwrap();
    assert!(crystal.converged);

    // The same eight atoms nearly three times as far apart: essentially isolated.
    let expanded = run_kpoints(&rocksalt(22.56), &params, &options(), &periodic, &kopt).unwrap();
    assert!(expanded.converged);

    let binding = expanded.total_ev - crystal.total_ev;
    assert!(
        binding > 5.0,
        "rocksalt should be bound by several eV per cell, not {binding:.3} eV"
    );

    // And the binding is ionic: sodium has given its electron up.
    let sodium: f64 = crystal.charges[..4].iter().sum::<f64>() / 4.0;
    let chlorine: f64 = crystal.charges[4..].iter().sum::<f64>() / 4.0;
    assert!(
        sodium > 0.3 && chlorine < -0.3,
        "expected ionic charges, got Na {sodium:.3} e and Cl {chlorine:.3} e"
    );
    // Charge is conserved, as it must be whatever the SCF did.
    assert!((sodium + chlorine).abs() < 1.0e-6);
}

/// The experimental lattice constant converges — with a mesh dense enough for it.
///
/// A `2×2×2` mesh does **not**, and the failure is a sampling one rather than a defect. The
/// chemical potential gives it away: it swings by an electronvolt every iteration as occupations
/// flip inside a degenerate manifold, because an even mesh on a cubic cell lands every one of its
/// points on a zone-boundary symmetry point where bands meet. Damping to 0.95, a thousand
/// iterations, level shifts up to 5 eV and smearing to 1 eV all fail; 2 eV smearing converges to
/// a *different* solution 200 eV away.
///
/// Odd meshes do not sit on those points, and `3×3×3` converges in eighteen iterations. That the
/// answer is the right one is what the mesh convergence below says: energies agree to 0.02 eV
/// and charges to 0.02 electrons across `3×3×3`, `4×4×4` and `5×5×5`.
#[test]
fn rocksalt_converges_at_its_experimental_lattice_constant() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let crystal = rocksalt(5.64);

    let mut energies = Vec::new();
    let mut charges = Vec::new();
    for n in [3usize, 4, 5] {
        let result = run_kpoints(
            &crystal,
            &params,
            &options(),
            &periodic,
            &mesh([n, n, n], 0.1),
        )
        .unwrap_or_else(|e| panic!("{n}x{n}x{n} at 5.64 A: {e}"));
        assert!(result.converged);
        energies.push(result.total_ev);
        charges.push(result.charges[..4].iter().sum::<f64>() / 4.0);
    }

    let spread = energies.iter().fold(f64::NEG_INFINITY, |m, v| m.max(*v))
        - energies.iter().fold(f64::INFINITY, |m, v| m.min(*v));
    assert!(
        spread < 0.05,
        "the energy should be converged with respect to the mesh, spread {spread:.3e} eV: \
         {energies:?}"
    );
    for charge in &charges {
        assert!(
            *charge > 0.05,
            "sodium should still be the positive one, got {charge:.4} e"
        );
    }
}

/// Graphene: the exact 2D Ewald against 3D-with-vacuum, on a system where the electrostatics is
/// not an afterthought.
///
/// These are two genuinely different treatments — Parry's slab sum with an analytic `z` dependence
/// versus the textbook 3D sum with a neutralizing background — and with enough vacuum they must
/// agree. This is the cleanest available check that the 2D branch is right, because it compares it
/// against the 3D branch rather than against itself.
#[test]
fn graphene_agrees_between_two_and_three_dimensions() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let kopt = mesh([4, 4, 1], 0.2);

    let flat = run_kpoints(
        &graphene(2.46, 20.0, false),
        &params,
        &options(),
        &periodic,
        &kopt,
    )
    .unwrap();
    let padded = run_kpoints(
        &graphene(2.46, 20.0, true),
        &params,
        &options(),
        &periodic,
        &kopt,
    )
    .unwrap();
    assert!(flat.converged && padded.converged);

    let difference = (flat.total_ev - padded.total_ev).abs();
    assert!(
        difference < 1.0e-3,
        "the exact 2D sum and 3D-with-vacuum should agree: {} vs {} ({difference:.3e} eV)",
        flat.total_ev,
        padded.total_ev
    );
    // The 2D run samples its plane only; the vacuum direction is not a dimension to sample.
    assert!(flat.kpoints.iter().all(|k| k.frac[2] == 0.0));
}

/// Graphene is a system the Γ point cannot do, and says so.
///
/// A two-atom cell has no bond that does not run through an image, so `P(Γ)` standing in for
/// `P(0, T)` is not an approximation here but a different problem. The margin is permanently
/// negative — no larger cell fixes it, because widening the cell adds atoms rather than separating
/// them — and only a k-mesh helps.
#[test]
fn graphene_reports_that_the_gamma_point_is_not_enough() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let sheet = graphene(2.46, 20.0, false);

    let sampled = run_kpoints(
        &sheet,
        &params,
        &options(),
        &periodic,
        &mesh([4, 4, 1], 0.2),
    )
    .unwrap();
    assert!(
        sampled.gamma_margin < 0.0,
        "a 2.46 Å cell is far narrower than the exchange cutoff; margin came out {:.3}",
        sampled.gamma_margin
    );

    // And the mesh is doing real work: a coarser one is measurably different, so this is not a
    // system where any sampling would have done.
    let coarse = run_kpoints(
        &sheet,
        &params,
        &options(),
        &periodic,
        &mesh([2, 2, 1], 0.2),
    )
    .unwrap();
    assert!(
        (coarse.total_ev - sampled.total_ev).abs() > 1.0e-3,
        "the mesh should matter for a semimetal"
    );
}

/// The Peierls instability: a half-filled one-dimensional chain is unstable to dimerization.
///
/// This is why polyacetylene alternates its bonds instead of sitting uniform, and it is a property
/// of the *band structure* — the uniform chain has a half-filled band, and doubling the period
/// opens a gap at the Fermi level and pushes the occupied states down. None of it survives a
/// k-point convention that is wrong, because the whole effect is in how occupied states are
/// counted across the zone.
#[test]
fn polyacetylene_dimerizes() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let kopt = mesh([8, 1, 1], 0.05);

    let uniform = run_kpoints(
        &polyacetylene(0.0, false),
        &params,
        &options(),
        &periodic,
        &kopt,
    )
    .unwrap();
    let dimerized = run_kpoints(
        &polyacetylene(0.10, false),
        &params,
        &options(),
        &periodic,
        &kopt,
    )
    .unwrap();
    assert!(uniform.converged && dimerized.converged);

    assert!(
        dimerized.total_ev < uniform.total_ev,
        "dimerization should lower the energy: {} vs {}",
        dimerized.total_ev,
        uniform.total_ev
    );
    match (uniform.band_gap_ev, dimerized.band_gap_ev) {
        (Some(closed), Some(open)) => assert!(
            open > closed,
            "dimerization should widen the gap: {open:.3} vs {closed:.3} eV"
        ),
        // The textbook case: uniform is metallic and dimerized is not.
        (None, Some(_)) => {}
        (a, b) => panic!("unexpected gaps: uniform {a:?}, dimerized {b:?}"),
    }
}

/// A genuinely one-dimensional cell gives the same answer as the same chain in a 3D cell with
/// vacuum.
///
/// This is the check that the electrostatic split survives one dimension. The SCF divides the
/// lattice sum into a core half and an electron half so that one can enter `H_core` and the other
/// the Fock matrix; each half carries the full nuclear charge with opposite sign, so each is a
/// *charged* chain on its own — and a charged chain's potential grows logarithmically with
/// transverse distance, with no background available to tame it as there is in 3D and 2D.
///
/// The fix is to neutralize each half with a uniform offset that cancels between them, so the
/// total distribution is untouched and only the division changes. If that were wrong, the two
/// treatments below would disagree — the 3D-with-vacuum path never needs the trick, because its
/// background does the same job.
#[test]
fn a_one_dimensional_chain_matches_the_same_chain_with_vacuum() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();

    let chain = run_gamma(&polyacetylene(0.10, true), &params, &options(), &periodic).unwrap();
    let padded = run_gamma(&polyacetylene(0.10, false), &params, &options(), &periodic).unwrap();
    assert!(chain.converged && padded.converged);

    let difference = (chain.total_ev - padded.total_ev).abs();
    assert!(
        difference < 1.0e-3,
        "1D and 3D-with-vacuum should agree: {} vs {} ({difference:.3e} eV)",
        chain.total_ev,
        padded.total_ev
    );
    // The chain has one periodic direction, so the other two contribute no images at all.
    let cell = polyacetylene(0.10, true).cell.unwrap();
    assert_eq!(cell.n_periodic(), 1);
}

/// A charged chain converges, and its energy does not depend on how far the chain was summed.
///
/// A charged 1D cell diverges without help: shell `n` contributes `Q²/(nL)` once the cell looks
/// like a point from that distance, so the partial sum grows like `ln N`. A uniform neutralizing
/// line charge removes exactly that, leaving a finite energy defined *within a convention* — the
/// cell length as the logarithm's reference — in the same way a charged 3D cell is defined within
/// the jellium convention.
///
/// The test that matters is therefore not the value but its stability: quadrupling the image
/// count must leave the answer alone. If the line charge were missing, or scaled wrongly, the
/// number would still look perfectly reasonable and would move every time the sum got longer.
#[test]
fn a_charged_one_dimensional_cell_is_summed_against_a_neutralizing_line_charge() {
    let params = Pm3Parameters::standard().unwrap();
    let mut charged = polyacetylene(0.10, true);
    charged.charge = 1.0;
    charged.multiplicity = 2;

    let energy_at = |images: usize| {
        let periodic = PeriodicOptions {
            ewald: Some(pm3_rs::pbc::ewald::EwaldParams {
                chain_images: images,
                ..pm3_rs::pbc::ewald::EwaldParams::default()
            }),
            ..PeriodicOptions::default()
        };
        run_gamma(&charged, &params, &options(), &periodic)
            .unwrap_or_else(|e| panic!("a charged chain should converge, got: {e}"))
            .total_ev
    };

    let short = energy_at(100);
    let long = energy_at(400);
    assert!(
        (short - long).abs() < 1.0e-3,
        "the charged chain's energy moved from {short} to {long} on quadrupling the image \
         count, so the sum is still carrying its divergence"
    );

    // And it is a cation: removing an electron from a conjugated chain costs energy.
    let neutral = run_gamma(
        &polyacetylene(0.10, true),
        &params,
        &options(),
        &PeriodicOptions::default(),
    )
    .unwrap()
    .total_ev;
    assert!(
        long > neutral,
        "the cation ({long}) should lie above the neutral chain ({neutral})"
    );
}
