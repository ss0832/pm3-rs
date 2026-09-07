// SPDX-License-Identifier: GPL-3.0-or-later

//! Which term breaks cubic symmetry in `D(0)`.
//!
//! ```text
//! cargo run --release --example cubic_anisotropy
//! ```
//!
//! Rocksalt is cubic, so `D(0)` must satisfy `D_xx = D_yy = D_zz` whatever PM3 thinks of the
//! crystal. It does not, and the odd component is the axis the basis atom was written along —
//! which is a property of the input, not of the lattice.
//!
//! This bisects the assembly. Each row switches off one contribution, so the row where the spread
//! collapses is the row that owns the defect:
//!
//! * **rigid ion** drops the electronic response entirely.
//! * **long range off** drops the fixed-charge lattice sum *and* the response's screening of it.
//! * **bare long range off** (`PM3_DFPT_NO_BARE_LONG_RANGE`) drops only the long-range channel of
//!   the bare perturbation, leaving the screening in.

use pm3_rs::pbc::dfpt::{force_constants_at_q, DfptOptions, LongRange};
use pm3_rs::pbc::gamma::PeriodicOptions;
use pm3_rs::pbc::kscf::KpointOptions;
use pm3_rs::{
    dynamical_matrix_on_mesh, rigid_ion_dynamical_matrix, Atom, Cell, Molecule, Pm3Options,
    Pm3Parameters, Vec3,
};

const ANGSTROM_TO_BOHR: f64 = 1.8897261254578281;
const MESH: [usize; 3] = [3, 3, 3];

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

fn report(label: &str, diagonal: [f64; 3]) {
    let spread = diagonal
        .iter()
        .map(|v| (v - diagonal[0]).abs())
        .fold(0.0_f64, f64::max);
    let scale = diagonal.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let relative = if scale > 0.0 { spread / scale } else { 0.0 };
    println!(
        "  {label:<26} [{:+10.6} {:+10.6} {:+10.6}]   spread {spread:.3e}  ({:.2}% of scale)",
        diagonal[0],
        diagonal[1],
        diagonal[2],
        100.0 * relative
    );
}

fn main() {
    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options {
        max_scf: 500,
        ..Pm3Options::default()
    };
    let periodic = PeriodicOptions::default();
    let kopt = KpointOptions::mesh(MESH);

    for (name, z, a) in [("NaCl", [11u8, 17u8], 5.64), ("MgO", [12, 8], 4.212)] {
        println!("=== {name} ===  D(0) diagonal on atom 1, eV/Bohr^2");
        let molecule = rocksalt(z, a, 0);

        let show = |label: &str, made: pm3_rs::Result<pm3_rs::DynamicalMatrix>| match made {
            Ok(d) => report(
                label,
                [
                    d.matrix[(0, 0)].re,
                    d.matrix[(1, 1)].re,
                    d.matrix[(2, 2)].re,
                ],
            ),
            Err(e) => println!(
                "  {label:<26} unavailable: {}",
                &e.to_string()[..60.min(e.to_string().len())]
            ),
        };

        show(
            "rigid ion, Gamma",
            rigid_ion_dynamical_matrix(&molecule, &params, &options, &periodic, [0.0; 3]),
        );
        // The control that separates "the response is wrong" from "the mesh handling is wrong":
        // same response, Gamma sampling instead of a mesh.
        show(
            "full, Gamma",
            pm3_rs::dynamical_matrix(&molecule, &params, &options, &periodic, [0.0; 3]),
        );
        show(
            "full, mesh",
            dynamical_matrix_on_mesh(&molecule, &params, &options, &periodic, &kopt, [0.0; 3]),
        );

        show(
            "mesh, long range off",
            force_constants_at_q(
                &molecule,
                &params,
                &options,
                &periodic,
                &DfptOptions {
                    kpoints: Some(kopt.clone()),
                    long_range: LongRange::Off,
                    ..DfptOptions::default()
                },
                [0.0; 3],
            )
            .map(|r| r.dynamical),
        );
        println!();
    }

    println!("The rigid-ion row is the control: it has no response in it at all. Whichever row");
    println!("first shows a spread is the one that introduced the anisotropy.");
}
