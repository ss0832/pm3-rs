// SPDX-License-Identifier: GPL-3.0-or-later

//! **Where a periodic run's time actually goes**, measured rather than guessed.
//!
//! `examples/born_profile.rs` is the precedent and the cautionary tale: three plausible
//! optimizations of the Born-charge path each changed the wall clock by nothing, because 85% of
//! the run was in a call nobody had timed. So this decomposes the periodic cost with public
//! entry points that differ in one term each, before anything is changed.
//!
//! Three suspects, from reading:
//!
//! 1. **The k-point loop is serial** (`pbc::kscf`). Each k-point is an independent Bloch
//!    transform and Hermitian diagonalization, so the SCF iteration should scale with the mesh.
//!    Measured here by holding the system fixed and growing the mesh: if the loop dominates,
//!    the time is linear in the number of k-points with a small intercept.
//! 2. **The integral setup is serial** (`pbc::gamma`), while the molecular equivalents use
//!    `par_chunks_mut`. Measured as the intercept above -- the part that does not grow with the
//!    mesh -- and directly, as a single-point run at the Γ point.
//! 3. **The setup is rebuilt per line-search trial** during a relaxation. Measured as the cost
//!    of an optimization step against the cost of the gradient it is built from.
//!
//! ```text
//! cargo run --release --example periodic_profile
//! ```

use std::time::{Duration, Instant};

use pm3_rs::pbc::gamma::{run_gamma, PeriodicOptions};
use pm3_rs::pbc::kpoints::KpointSpec;
use pm3_rs::pbc::kscf::{run_kpoints, KpointOptions};
use pm3_rs::pbc::optimize::{relax, CellRelaxation, PeriodicOptOptions};
use pm3_rs::{periodic_gradient, Cell, Molecule, Pm3Options, Pm3Parameters};

/// A cell wide enough that the Γ-point answer means something.
///
/// The condition is on the *margin*: the cell width less the 14 Bohr exchange cutoff, less the
/// extent of what is in it. 18 Bohr looks generous and is not — three waters stacked six
/// Ångström deep leave the margin at −4.5 Bohr, and the Γ run then spends 143 iterations
/// failing to converge, which is the sort of thing that quietly ruins a benchmark rather than
/// stopping it.
const EDGE_ANGSTROM: f64 = 16.0;

fn cell() -> Cell {
    Cell::from_rows(
        [
            [EDGE_ANGSTROM, 0.0, 0.0],
            [0.0, EDGE_ANGSTROM, 0.0],
            [0.0, 0.0, EDGE_ANGSTROM],
        ],
        [true, true, true],
    )
    .expect("a cubic cell")
}

/// `copies` waters on a cubic grid inside the cell, 3.5 Å apart.
///
/// A grid rather than a stack: stacking them along one axis runs 24 waters out to 72 Å inside
/// a 16 Å box, which is not a denser system but a broken one. The grid keeps every count inside
/// the cell and keeps the density roughly constant, so the size scan measures size.
fn system(copies: usize) -> Molecule {
    const SPACING: f64 = 3.5;
    let per_side = (copies as f64).cbrt().ceil() as usize;
    // Centre the grid in the cell so nothing sits on a face.
    let span = SPACING * (per_side.saturating_sub(1)) as f64;
    let origin = 0.5 * (EDGE_ANGSTROM - span);

    let mut text = format!("{}\nwater grid\n", copies * 3);
    for i in 0..copies {
        let x = origin + SPACING * (i % per_side) as f64;
        let y = origin + SPACING * ((i / per_side) % per_side) as f64;
        let z = origin + SPACING * (i / (per_side * per_side)) as f64;
        text.push_str(&format!("O {x:.4} {y:.4} {z:.4}\n"));
        text.push_str(&format!("H {:.4} {y:.4} {z:.4}\n", x + 0.9584));
        text.push_str(&format!("H {:.4} {:.4} {z:.4}\n", x - 0.24, y + 0.9278));
    }
    let mut molecule = Molecule::from_xyz_str(&text, 0.0).expect("the fixture parses");
    molecule.cell = Some(cell());
    molecule
}

fn time<T>(mut run: impl FnMut() -> T) -> (Duration, T) {
    let start = Instant::now();
    let value = run();
    (start.elapsed(), value)
}

fn main() {
    let params = Pm3Parameters::standard().expect("parameters");
    let options = Pm3Options::default();
    let periodic = PeriodicOptions::default();
    let molecule = system(3);
    println!(
        "system: {} atoms in a {EDGE_ANGSTROM} A cube\n",
        molecule.atoms.len()
    );

    // --- 1 and 2: how the cost splits between the mesh and everything else ------------------
    println!("A k-point SCF against the size of the mesh");
    println!("  Each k-point is an independent Bloch transform and diagonalization. Time is");
    println!("  normalised **per SCF iteration**, because a bigger mesh also changes how many");
    println!("  iterations the SCF takes -- comparing raw wall clock across meshes measures");
    println!("  convergence behaviour and calls it scaling.\n");
    println!(
        "  {:>10}  {:>8}  {:>10}  {:>10}  {:>14}  {:>14}",
        "mesh", "points", "time (s)", "iters", "per iter (ms)", "per k (ms)"
    );

    let mut samples: Vec<(f64, f64)> = Vec::new();
    for divisions in [1usize, 2, 3, 4] {
        let kopt = KpointOptions {
            spec: KpointSpec::mesh([divisions, divisions, divisions]),
            ..KpointOptions::default()
        };
        let (elapsed, result) = time(|| {
            run_kpoints(&molecule, &params, &options, &periodic, &kopt)
                .expect("the k-point SCF converges")
        });
        let points = result.kpoints.len() as f64;
        let iterations = result.iterations.max(1) as f64;
        let per_iteration = elapsed.as_secs_f64() / iterations;
        samples.push((points, per_iteration));
        println!(
            "  {:>10}  {:>8}  {:>10.3}  {:>10}  {:>14.2}  {:>14.2}",
            format!("{d}x{d}x{d}", d = divisions),
            points as usize,
            elapsed.as_secs_f64(),
            result.iterations,
            1000.0 * per_iteration,
            1000.0 * per_iteration / points,
        );
    }

    // A straight line through the smallest and largest meshes separates the two costs.
    if let (Some(first), Some(last)) = (samples.first(), samples.last()) {
        let slope = (last.1 - first.1) / (last.0 - first.0);
        let intercept = first.1 - slope * first.0;
        let parallel_share = 100.0 * slope * last.0 / last.1;
        println!(
            "\n  per iteration: {:.2} ms per k-point + {:.2} ms fixed",
            1000.0 * slope,
            1000.0 * intercept,
        );
        println!(
            "  At the {}-point mesh that is {:.0}% in the k loop, which is the share that",
            last.0 as usize, parallel_share,
        );
        println!(
            "  parallelising it can reach. The rest is serial and threading the loop misses it."
        );
        if parallel_share < 40.0 {
            println!(
                "\n  NOTE: under half the time is in the loop the plan proposed to parallelise.\n  \
                 Amdahl caps the whole-run gain at {:.0}% even with a free, perfect speed-up.",
                parallel_share,
            );
        }
    }

    // --- 2 directly: a Γ-point single point is setup plus one diagonalization per iteration ---
    let (gamma_time, gamma) =
        time(|| run_gamma(&molecule, &params, &options, &periodic).expect("the Γ SCF converges"));
    println!(
        "\nΓ-point single point: {:.3} s over {} iterations ({:.1} ms each)",
        gamma_time.as_secs_f64(),
        gamma.iterations,
        1000.0 * gamma_time.as_secs_f64() / gamma.iterations.max(1) as f64,
    );

    // --- how the fixed cost grows with the system ------------------------------------------
    //
    // The mesh scan above says most of a periodic iteration does not depend on the mesh. What
    // it does depend on is the system, and that is the axis a user cares about: nobody runs a
    // bigger mesh to make a calculation take longer, they run a bigger molecule. An exponent
    // near 2 is pair tables; near 3 is the diagonalization.
    println!("\nThe mesh-independent cost against the system size (Γ point)");
    println!(
        "  {:>8}  {:>8}  {:>10}  {:>8}  {:>14}  {:>10}",
        "waters", "atoms", "time (s)", "iters", "per iter (ms)", "exponent"
    );
    let mut previous: Option<(f64, f64)> = None;
    for copies in [2usize, 4, 8, 16, 24] {
        let big = system(copies);
        let (elapsed, result) = match time(|| run_gamma(&big, &params, &options, &periodic)) {
            (elapsed, Ok(result)) => (elapsed, result),
            (_, Err(error)) => {
                println!(
                    "  {copies:>8}  {:>8}  did not converge: {error}",
                    big.atoms.len()
                );
                continue;
            }
        };
        let atoms = big.atoms.len() as f64;
        let per_iteration = elapsed.as_secs_f64() / result.iterations.max(1) as f64;
        // Local slope in log-log: how the cost grows between this size and the last.
        let exponent = previous.map(|(n0, t0)| (per_iteration / t0).ln() / (atoms / n0).ln());
        println!(
            "  {:>8}  {:>8}  {:>10.3}  {:>8}  {:>14.2}  {:>10}",
            copies,
            atoms as usize,
            elapsed.as_secs_f64(),
            result.iterations,
            1000.0 * per_iteration,
            exponent.map_or("--".to_string(), |e| format!("{e:.2}")),
        );
        previous = Some((atoms, per_iteration));
    }

    // --- 3: what a relaxation step costs against the gradient it is built from ---------------
    let (grad_time, _) =
        time(|| periodic_gradient(&molecule, &params, &options, &periodic).expect("a gradient"));
    println!("\nOne periodic gradient: {:.3} s", grad_time.as_secs_f64());

    let opt = PeriodicOptOptions {
        max_iter: 3,
        cell: CellRelaxation::Fixed,
        ..PeriodicOptOptions::default()
    };
    let (relax_time, relaxed) =
        time(|| relax(&molecule, &params, &options, &periodic, &opt).expect("a relaxation"));
    let steps = relaxed.iterations.max(1);
    println!(
        "Three optimization steps: {:.3} s ({:.3} s per step = {:.1} gradients' worth)",
        relax_time.as_secs_f64(),
        relax_time.as_secs_f64() / steps as f64,
        relax_time.as_secs_f64() / steps as f64 / grad_time.as_secs_f64(),
    );
    println!(
        "\n  A step is one gradient plus the line search's trials. Every trial rebuilds the\n  \
         integral setup from scratch, so the ratio above is how many setups a step pays for."
    );
}
