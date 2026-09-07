// SPDX-License-Identifier: GPL-3.0-or-later

//! **Phonon dispersion of an organometallic crystal**, written out for plotting.
//!
//! PM3's transition metals are Zn, Cd and Hg — the group-12 `d¹⁰` trio — so ferrocene and the
//! other iron-centred textbook organometallics are outside the model. They are not the only
//! organometallics. `Zn(CH₃)₂` is a real, characterised compound with two genuine `Zn–C` σ bonds
//! and methyl ligands: the motif that makes a compound organometallic rather than merely a
//! coordination complex, and something PM3 can actually do.
//!
//! # Why this molecule and this cell
//!
//! Dispersion needs a supercell, and a Γ-sampled supercell needs every width past the 14 Bohr
//! exchange range. Those two pull in opposite directions and the compound decides who wins:
//!
//! | | primitive | supercell for dispersion along one axis | atoms |
//! |---|---|---|---|
//! | `Cd(CN)₂` framework | 6.30 Å = 11.9 Bohr — **fails** the margin | `2×2×2`, forced | 80 |
//! | `Zn(CH₃)₂` molecular | 8.0 Å = 15.1 Bohr — clears it already | `2×1×1` | **18** |
//!
//! The framework has to be replicated in all three directions before *any* of them is legal, and
//! its 80-atom Hessian did not finish. The molecular crystal is legal at `1×1×1`, so one
//! replication buys dispersion along `x` and nothing else has to grow. Same chemistry, twenty
//! times cheaper.
//!
//! What the spectrum should show, and what the experimental comparison is against: a stiff
//! symmetric `Zn–C` stretch near 615 cm⁻¹ and C–H stretches above 2900, over a manifold of
//! low-frequency modes that are the molecules rocking against each other in the lattice. The
//! intramolecular modes should be nearly flat across the zone — a molecular crystal's internal
//! vibrations barely know about `q` — and the lattice modes should disperse. That contrast is the
//! qualitative check.
//!
//! Force constants come from the supercell (`ForceConstants::from_supercell`), Γ-sampled
//! throughout and exact at the commensurate wavevectors, which `tests/pbc_phonon.rs` pins.
//!
//! ```text
//! cargo run --release --example organometallic_bands
//! ```
//!
//! Writes `phonon_bands.csv` — one row per q-point, one column per branch — and prints a summary.

use std::time::Instant;

use pm3_rs::pbc::gamma::PeriodicOptions;
use pm3_rs::pbc::optimize::{relax, CellRelaxation, PeriodicOptOptions};
use pm3_rs::pbc::phonon::{q_path, ForceConstants};
use pm3_rs::{Atom, Cell, Molecule, Pm3Options, Pm3Parameters, Vec3};

const A: f64 = 1.8897261254578281;
/// Cubic cell holding one `Zn(CH₃)₂`. 8 Å is 15.1 Bohr, which clears the 14 Bohr Γ margin
/// without any replication — the whole reason this compound is affordable and the framework was
/// not.
const EDGE_ANGSTROM: f64 = 8.0;
const ZN_C: f64 = 1.930;
const C_H: f64 = 1.09;

/// One linear `CH₃–Zn–CH₃`, centred in the cell, axis along `z`.
fn dimethylzinc() -> Molecule {
    let centre = 0.5 * EDGE_ANGSTROM;
    // Tetrahedral at carbon: the C–H bonds sit 109.5° from C–Zn, so they lean away from the
    // metal by 70.5° off the axis.
    let tilt: f64 = 70.5_f64.to_radians();
    let (sin_t, cos_t) = tilt.sin_cos();

    let mut atoms = vec![Atom {
        z: 30,
        position: Vec3::new(centre, centre, centre) * A,
    }];
    for side in [1.0_f64, -1.0] {
        let c_z = centre + side * ZN_C;
        atoms.push(Atom {
            z: 6,
            position: Vec3::new(centre, centre, c_z) * A,
        });
        for k in 0..3 {
            let phi = std::f64::consts::TAU * k as f64 / 3.0;
            atoms.push(Atom {
                z: 1,
                position: Vec3::new(
                    centre + C_H * sin_t * phi.cos(),
                    centre + C_H * sin_t * phi.sin(),
                    c_z + side * C_H * cos_t,
                ) * A,
            });
        }
    }

    let edge = EDGE_ANGSTROM * A;
    let mut molecule = Molecule::new(atoms);
    molecule.cell = Some(
        Cell::new(
            Vec3::new(edge, 0.0, 0.0),
            Vec3::new(0.0, edge, 0.0),
            Vec3::new(0.0, 0.0, edge),
            [true; 3],
        )
        .expect("a cubic cell"),
    );
    molecule
}

fn main() {
    let params = Pm3Parameters::standard().expect("parameters");
    let options = Pm3Options {
        max_scf: 400,
        ..Pm3Options::default()
    };
    let periodic = PeriodicOptions::default();

    // **Relax before differentiating.** A Hessian is a phonon spectrum only at a stationary
    // point; the idealised bond lengths above are not PM3's, and the negative curvatures that
    // would come back are the geometry being wrong rather than the crystal being unstable.
    // Atoms only: the cell here is vacuum around one molecule, and relaxing it would be a
    // statement about dispersion packing that plain PM3 has no terms for.
    let guess = dimethylzinc();
    println!(
        "Zn(CH3)2: {} atoms in a {EDGE_ANGSTROM} A cell (margin {:.1} Bohr at 1x1x1)",
        guess.atoms.len(),
        EDGE_ANGSTROM * A - 14.0,
    );
    let started = Instant::now();
    let relaxed = match relax(
        &guess,
        &params,
        &options,
        &periodic,
        &PeriodicOptOptions {
            max_iter: 200,
            cell: CellRelaxation::Fixed,
            ..PeriodicOptOptions::default()
        },
    ) {
        Ok(r) => r,
        Err(error) => {
            eprintln!("the relaxation failed: {error}");
            return;
        }
    };
    let zn_c = (relaxed.molecule.atoms[1].position - relaxed.molecule.atoms[0].position).norm() / A;
    println!(
        "relaxed in {:.0} s ({} steps, converged {}); Zn-C {:.3} A from a {:.3} A guess",
        started.elapsed().as_secs_f64(),
        relaxed.iterations,
        relaxed.converged,
        zn_c,
        ZN_C,
    );

    // One replication along x buys dispersion along Γ→X; y and z are already legal, so nothing
    // else has to grow. 18 atoms rather than the 80 the framework forced.
    let reps = [2usize, 1, 1];
    let started = Instant::now();
    let force_constants =
        match ForceConstants::from_supercell(&relaxed.molecule, &params, &options, &periodic, reps)
        {
            Ok(fc) => fc,
            Err(error) => {
                eprintln!("the supercell Hessian failed: {error}");
                return;
            }
        };
    println!(
        "force constants from a {}x{}x{} supercell ({} atoms) in {:.0} s; acoustic sum-rule \
         residual {:.3e}",
        reps[0],
        reps[1],
        reps[2],
        relaxed.molecule.atoms.len() * reps[0] * reps[1] * reps[2],
        started.elapsed().as_secs_f64(),
        force_constants.acoustic_sum_rule_residual(),
    );

    // Γ → X, the only direction this supercell resolves. Saying so is the point: a path drawn
    // through directions the force constants cannot see is interpolation dressed as dispersion.
    let corners = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]];
    let per_segment = 40;
    let path = q_path(&corners, per_segment);

    let bands = match force_constants.band_structure(&path) {
        Ok(b) => b,
        Err(error) => {
            eprintln!("the band structure failed: {error}");
            return;
        }
    };

    let branches = bands.first().map_or(0, |row| row.len());
    let mut csv = String::from("index,qx,label");
    for branch in 0..branches {
        csv.push_str(&format!(",branch{branch}"));
    }
    csv.push('\n');
    for (index, (q, row)) in path.iter().zip(&bands).enumerate() {
        let label = match index {
            0 => "G",
            i if i == path.len() - 1 => "X",
            _ => "",
        };
        csv.push_str(&format!("{index},{:.6},{label}", q[0]));
        for value in row {
            csv.push_str(&format!(",{value:.4}"));
        }
        csv.push('\n');
    }
    std::fs::write("phonon_bands.csv", csv).expect("the csv writes");

    let gamma = &bands[0];
    let highest = bands.iter().flatten().cloned().fold(f64::MIN, f64::max);
    let lowest = bands.iter().flatten().cloned().fold(f64::MAX, f64::min);
    let acoustic = gamma.iter().filter(|f| f.abs() < 1.0e-8).count();
    let imaginary = gamma.iter().filter(|f| **f < -1.0e-8).count();
    println!(
        "{} q-points x {branches} branches; {lowest:.1} to {highest:.1} cm^-1",
        path.len()
    );
    println!("at Γ: {acoustic} acoustic, {imaginary} imaginary");

    // The qualitative check: a molecular crystal's internal modes barely disperse, its lattice
    // modes do. Bandwidth per branch is what separates them.
    let mut widths: Vec<(f64, f64)> = (0..branches)
        .map(|b| {
            let column: Vec<f64> = bands.iter().map(|row| row[b]).collect();
            let hi = column.iter().cloned().fold(f64::MIN, f64::max);
            let lo = column.iter().cloned().fold(f64::MAX, f64::min);
            (gamma[b], hi - lo)
        })
        .collect();
    widths.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    println!("\n  {:>12}  {:>12}", "omega(G) cm^-1", "bandwidth");
    for (omega, width) in &widths {
        println!("  {omega:>12.1}  {width:>12.1}");
    }
    println!(
        "\nexperiment: the symmetric Zn-C stretch of Zn(CH3)2 is near 615 cm^-1, and the C-H\n\
         stretches above 2900 cm^-1. Internal modes should be nearly flat across the zone;\n\
         the modes that disperse are the molecules moving against each other."
    );
    println!("wrote phonon_bands.csv");
}
