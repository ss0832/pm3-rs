// SPDX-License-Identifier: GPL-3.0-or-later
//! **Two identities the `q = 0` dynamical matrix has to satisfy**, on a cubic ionic crystal.
//!
//! Neither is a comparison against a measured number. Both are statements that must hold whatever
//! PM3 thinks of rocksalt, which is what makes them useful on a model that was never parameterized
//! for ionic solids: they separate "the code is wrong" from "PM3 is wrong", and only the first is
//! a defect of this crate.
//!
//! 1. **Cubic symmetry.** Writing the second atom along `x`, `y` or `z` is the same crystal with
//!    the axis labels permuted, so `D(0)` must satisfy `D_xx = D_yy = D_zz` and the three spectra
//!    must be identical.
//!
//! 2. **DFPT against a finite difference.** `D(0)` is the second derivative of the energy, so it
//!    must equal a central difference of the *analytic forces* at the same geometry and the same
//!    sampling. That difference shares no second-derivative code with DFPT — it needs only the
//!    gradient, which the stress and optimizer tests pin independently.
//!
//! # What these found
//!
//! At **Γ sampling** both hold, and hold well: DFPT and the finite difference agree to
//! `3.2e-4 eV/Å²` out of `4.19` on NaCl, and the diagonal is isotropic to `1.4e-8`. Those are the
//! tests that run.
//!
//! On a **k-mesh** both used to fail badly, and both now pass. The cause was one substitution,
//! made in two places.
//!
//! `P(T) = P(0)` — one density matrix serving every lattice image — *is* Γ sampling: a single
//! k-point cannot tell images apart, so it holds there by construction. On a mesh it is false,
//! because `P(0)` becomes the Brillouin-zone average while `P(T)` decays with `T`. Two places
//! asserted it anyway:
//!
//! * `phased_skeleton` passed one density matrix to `pair_block` for every image. The skeleton is
//!   the larger half of `D(q)`, so a meshed run built its dominant term from the wrong density.
//! * `bare_blocks` read `exchange_scale · P^σ(0)` for the exchange part of each bare
//!   perturbation, where the response needs `P^σ(T)`. `SpinChannel::p_images` already held the
//!   right thing under both samplings; nothing was reading it.
//!
//! On NaCl `3×3×3` that moved `D(0)`'s diagonal from `(−1.290, −0.359, −0.359)` — anisotropic,
//! 300% out, and reporting an imaginary optical mode for a stable crystal — to isotropic and
//! agreeing with the finite difference of the analytic forces.
//!
//! `pbc::dfpt::tests::the_meshed_skeleton_is_where_the_cubic_symmetry_breaks` is the experiment
//! that separated the two halves, and is what made the second one findable: it runs the skeleton
//! *without* the response under both samplings, which no public entry point can do.
//!
//! # What was ruled out on the way
//!
//! Recorded because it is the part that took the time, and because it is where to start if a
//! meshed identity ever fails again:
//!
//! * **Not the ground state.** The converged energy, charges and gap are identical across the
//!   three orientations to `5e-13 eV`.
//! * **Not the forces.** The finite difference is isotropic and mesh-converged, and agrees with a
//!   second difference of the total energy to five digits.
//! * **Not degeneracy or the energy denominators.** Fermi smearing from 0 to 0.5 eV changed the
//!   anisotropy in the fourth digit. Both crystals have gaps above 6 eV.
//! * **Not the long-range monopole term.** `LongRange::Off` left the anisotropy at 49% of the
//!   scale, and `PM3_DFPT_NO_BARE_LONG_RANGE` left it at 49% too.
//! * **Not noise.** It converged smoothly with mesh size — 0% at `1³`, 16% at `2³`, 72% at `3³`,
//!   77% at `7³`. **Zero at `1³` was the tell**, and it sat unused for a long time: a `1×1×1`
//!   mesh *is* Γ, where the substitution is exactly true, and the error then grows as the mesh
//!   resolves `P(0)` away from the Γ value and saturates once it has. That is the signature of
//!   `P(T) = P(0)` and of nothing else on the list above.
//!
//! `examples/cubic_anisotropy.rs` reproduces the bisection.

use pm3_rs::pbc::gamma::PeriodicOptions;
use pm3_rs::pbc::kscf::{kpoint_gradient, KpointOptions};
use pm3_rs::{
    dynamical_matrix, dynamical_matrix_on_mesh, frequencies_of, periodic_gradient, Atom, Cell,
    DynamicalMatrix, Molecule, Pm3Options, Pm3Parameters, Vec3,
};

const ANGSTROM_TO_BOHR: f64 = 1.8897261254578281;
/// An odd mesh. `docs/pbc.md` records why: an even Monkhorst–Pack mesh on a cubic cell puts
/// k-points on zone-boundary symmetry points where bands meet, and the Fermi bisection has no
/// stable filling to find there.
const MESH: [usize; 3] = [3, 3, 3];

/// Rocksalt in its primitive FCC cell, with the second atom placed along `axis`.
fn rocksalt(z: [u8; 2], a_angstrom: f64, axis: usize) -> Molecule {
    let a = a_angstrom * ANGSTROM_TO_BOHR;
    let h = a / 2.0;
    let cell = Cell::new(
        Vec3::new(0.0, h, h),
        Vec3::new(h, 0.0, h),
        Vec3::new(h, h, 0.0),
        [true; 3],
    )
    .unwrap();
    let mut second = [0.0; 3];
    second[axis] = h;
    let mut molecule = Molecule::new(vec![
        Atom {
            z: z[0],
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: z[1],
            position: Vec3::new(second[0], second[1], second[2]),
        },
    ]);
    molecule.cell = Some(cell);
    molecule
}

fn options() -> (Pm3Parameters, Pm3Options, PeriodicOptions) {
    (
        Pm3Parameters::standard().unwrap(),
        Pm3Options {
            max_scf: 500,
            ..Pm3Options::default()
        },
        PeriodicOptions::default(),
    )
}

/// `D(0)` of a cubic crystal is isotropic and does not care which axis the basis was written
/// along.
fn assert_isotropic(name: &str, sampling: &str, made: &[DynamicalMatrix; 3]) {
    for (axis, dynamical) in made.iter().enumerate() {
        let diagonal: Vec<f64> = (0..3).map(|i| dynamical.matrix[(i, i)].re).collect();
        let spread = diagonal
            .iter()
            .map(|v| (v - diagonal[0]).abs())
            .fold(0.0_f64, f64::max);
        let scale = diagonal.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(
            spread <= 1.0e-6 * scale.max(1.0),
            "{name} ({sampling}): D(0) is anisotropic with the second atom along axis {axis}. \
             diag(xx, yy, zz) = {diagonal:?}, spread {spread:.6e} against a scale of {scale:.6e}. \
             A cubic crystal has no axis to prefer, and this one is the axis the input used."
        );
    }
    let reference = frequencies_of(&made[0]).unwrap();
    for (axis, dynamical) in made.iter().enumerate().skip(1) {
        let mut got = frequencies_of(dynamical).unwrap();
        let mut want = reference.clone();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        want.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (mode, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1.0e-3,
                "{name} ({sampling}): mode {mode} is {g} cm^-1 with the basis along axis {axis} \
                 and {w} cm^-1 along axis 0; relabelling the axes changed the physics"
            );
        }
    }
}

/// The identity, at whatever sampling the two closures agree on.
fn assert_matches_finite_difference<D, G>(
    name: &str,
    sampling: &str,
    base: &Molecule,
    dfpt: D,
    gradient_at: G,
) where
    D: Fn(&Molecule) -> DynamicalMatrix,
    G: Fn(&Molecule) -> Vec<Vec3>,
{
    // Bohr. Large enough that the force difference clears the SCF's own noise, small enough that
    // the cubic term stays below the tolerance; the answer is stable between half and twice it.
    const STEP: f64 = 0.02;
    let ndof = 3 * base.atoms.len();

    let mut finite = vec![vec![0.0; ndof]; ndof];
    for (dof, row) in finite.iter_mut().enumerate() {
        let (atom, axis) = (dof / 3, dof % 3);
        let mut sides = Vec::with_capacity(2);
        for sign in [-1.0, 1.0] {
            let mut moved = base.clone();
            let mut p = moved.atoms[atom].position.to_array();
            p[axis] += sign * STEP;
            moved.atoms[atom].position = Vec3::new(p[0], p[1], p[2]);
            sides.push(gradient_at(&moved));
        }
        for (other, slot) in row.iter_mut().enumerate() {
            let (b, beta) = (other / 3, other % 3);
            *slot = (sides[1][b].to_array()[beta] - sides[0][b].to_array()[beta]) / (2.0 * STEP);
        }
    }

    let made = dfpt(base);
    let scale = finite
        .iter()
        .flatten()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    let mut worst = (0.0_f64, 0usize, 0usize);
    for (i, row) in finite.iter().enumerate() {
        for (j, reference) in row.iter().enumerate() {
            let delta = (made.matrix[(i, j)].re - reference).abs();
            if delta > worst.0 {
                worst = (delta, i, j);
            }
        }
    }
    assert!(
        worst.0 <= 0.01 * scale,
        "{name} ({sampling}): D(0) and the finite difference of the analytic forces disagree by \
         {:.5e} eV/Bohr^2 at element ({}, {}), against a largest element of {scale:.5e}. \
         DFPT says {:.6}, the finite difference says {:.6}.",
        worst.0,
        worst.1,
        worst.2,
        made.matrix[(worst.1, worst.2)].re,
        finite[worst.1][worst.2]
    );
}

// ---------------------------------------------------------------------------
// Γ sampling — these hold, and are the coverage that stays green.
// ---------------------------------------------------------------------------

#[test]
fn the_gamma_sampled_dynamical_matrix_is_isotropic_on_a_cubic_crystal() {
    let (params, opts, periodic) = options();
    // NaCl only: MgO's primitive cell has a Γ margin of −9.4 Bohr, so one k-point cannot
    // represent it and the SCF says so rather than converging to something.
    let made: [DynamicalMatrix; 3] = std::array::from_fn(|axis| {
        let molecule = rocksalt([11, 17], 5.64, axis);
        dynamical_matrix(&molecule, &params, &opts, &periodic, [0.0; 3]).unwrap()
    });
    assert_isotropic("NaCl", "Γ", &made);
}

#[test]
fn gamma_sampled_dfpt_matches_a_finite_difference_of_the_analytic_forces() {
    let (params, opts, periodic) = options();
    let base = rocksalt([11, 17], 5.64, 0);
    assert_matches_finite_difference(
        "NaCl",
        "Γ",
        &base,
        |m| dynamical_matrix(m, &params, &opts, &periodic, [0.0; 3]).unwrap(),
        |m| {
            periodic_gradient(m, &params, &opts, &periodic)
                .unwrap()
                .gradient
        },
    );
}

// ---------------------------------------------------------------------------
// k-mesh sampling — the defect. See the module note.
// ---------------------------------------------------------------------------

#[test]
fn the_mesh_sampled_dynamical_matrix_is_isotropic_on_a_cubic_crystal() {
    let (params, opts, periodic) = options();
    let kopt = KpointOptions::mesh(MESH);
    for (name, z, a) in [("NaCl", [11u8, 17u8], 5.64), ("MgO", [12, 8], 4.212)] {
        let made: [DynamicalMatrix; 3] = std::array::from_fn(|axis| {
            let molecule = rocksalt(z, a, axis);
            dynamical_matrix_on_mesh(&molecule, &params, &opts, &periodic, &kopt, [0.0; 3]).unwrap()
        });
        assert_isotropic(name, "3×3×3", &made);
    }
}

#[test]
fn mesh_sampled_dfpt_matches_a_finite_difference_of_the_analytic_forces() {
    let (params, opts, periodic) = options();
    let kopt = KpointOptions::mesh(MESH);
    for (name, z, a) in [("NaCl", [11u8, 17u8], 5.64), ("MgO", [12, 8], 4.212)] {
        let base = rocksalt(z, a, 0);
        assert_matches_finite_difference(
            name,
            "3×3×3",
            &base,
            |m| dynamical_matrix_on_mesh(m, &params, &opts, &periodic, &kopt, [0.0; 3]).unwrap(),
            |m| {
                kpoint_gradient(m, &params, &opts, &periodic, &kopt)
                    .unwrap()
                    .gradient
            },
        );
    }
}
