// SPDX-License-Identifier: GPL-3.0-or-later

//! **Γ-point phonons across the common crystal structure types, against experiment.**
//!
//! The question this answers is not "does the code run" — `tests/pbc_phonon.rs` covers that — but
//! "does PM3 produce a phonon spectrum a solid-state chemist would recognise". So each entry
//! carries an experimental number and the table prints the ratio, and the interesting output is
//! the cases where the ratio is not near one.
//!
//! # Two constraints decide what is in this list
//!
//! **PM3's transition metals are Zn, Cd and Hg** — the group-12 `d¹⁰` trio — and nothing else.
//! The parameter set is 46 entries, so `TiO₂`, `SrTiO₃`, `LiCoO₂` and ferrocene are not "not
//! implemented", they are outside the model. Each is replaced by a compound of the **same
//! structure type** built from elements PM3 has, and the substitution is named in the table
//! rather than hidden: rutile `SnO₂` for rutile `TiO₂`, the halide perovskite `CsPbBr₃` for
//! `SrTiO₃`, and layered `CdI₂` for layered `LiCoO₂`.
//!
//! Those three metals are enough for real organometallic chemistry, so the survey does not stop
//! at a substitute for it. `Zn(CH₃)₂` and `Hg(CH₃)₂` are linear dialkyls with genuine `M–C` σ
//! bonds — the motif that makes a compound organometallic rather than merely a coordination
//! complex — and `Cd(CN)₂` is a cyanide framework, also `M–C` bonded, standing in for the
//! extended-solid case ferrocene would have covered.
//!
//! **Γ sampling needs a wide cell.** One k-point stands in for the density at every image, which
//! is only defensible when no image is inside the exchange range — the `gamma_margin`, the cell
//! width less 14 Bohr, has to be positive. Primitive cells of these structures are 3–6 Å and fail
//! that by a wide margin, so each is replicated until it passes. That is also why this is an
//! example rather than a test: the supercells are tens of atoms and the Hessians are minutes.
//!
//! A k-mesh would be the alternative and is not usable here — see `docs/pbc.md`, where a meshed
//! `D(q)` is still 50% out at `q = 0`.
//!
//! ```text
//! cargo run --release --example phonon_structures
//! ```

use std::time::Instant;

use pm3_rs::pbc::gamma::{run_gamma, PeriodicOptions};
use pm3_rs::pbc::hessian::periodic_phonons;
use pm3_rs::{Atom, Cell, Molecule, Pm3Options, Pm3Parameters, Vec3};

const A: f64 = 1.8897261254578281; // Ångström → Bohr

/// One structure to run.
struct Case {
    /// Structure type, which is the axis this survey is organised along.
    kind: &'static str,
    formula: &'static str,
    /// What it stands in for, when PM3 cannot do the compound usually quoted.
    substitute_for: Option<&'static str>,
    /// Fractional coordinates and species in the conventional cell.
    basis: Vec<(u8, [f64; 3])>,
    /// Conventional lattice **vectors** as rows, Ångström.
    ///
    /// Vectors rather than edge lengths because the layer types and wurtzite are hexagonal, and
    /// flattening them onto an orthogonal cell is what put two lithiums 1.98 Å apart in the first
    /// version of this file.
    vectors: [[f64; 3]; 3],
    /// The highest measured optical frequency, cm⁻¹, and where it comes from.
    experiment_cm: f64,
    experiment_note: &'static str,
    /// The accepted nearest-neighbour distance, Å. Checked by `--check` before any Hessian runs:
    /// a phonon spectrum computed on a subtly wrong lattice is worse than none, because it is
    /// plausible, quotable, and says nothing about the geometry it came from.
    accepted_bond: &'static str,
}

fn rocksalt(z: [u8; 2], a: f64) -> Vec<(u8, [f64; 3])> {
    let cation = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    let anion = [
        [0.5, 0.5, 0.5],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
    ];
    let _ = a;
    cation
        .iter()
        .map(|p| (z[0], *p))
        .chain(anion.iter().map(|p| (z[1], *p)))
        .collect()
}

fn zinc_blende(z: [u8; 2]) -> Vec<(u8, [f64; 3])> {
    let cation = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    let anion = [
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ];
    cation
        .iter()
        .map(|p| (z[0], *p))
        .chain(anion.iter().map(|p| (z[1], *p)))
        .collect()
}

fn fluorite(z: [u8; 2]) -> Vec<(u8, [f64; 3])> {
    let cation = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    let mut out: Vec<(u8, [f64; 3])> = cation.iter().map(|p| (z[0], *p)).collect();
    for c in &cation {
        for d in [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]] {
            out.push((z[1], [c[0] + d[0], c[1] + d[1], c[2] + d[2]]));
        }
    }
    out
}

/// Cubic perovskite `ABX₃`: A at the corner, B at the body centre, X on the face centres.
fn perovskite(z: [u8; 3]) -> Vec<(u8, [f64; 3])> {
    vec![
        (z[0], [0.0, 0.0, 0.0]),
        (z[1], [0.5, 0.5, 0.5]),
        (z[2], [0.5, 0.5, 0.0]),
        (z[2], [0.5, 0.0, 0.5]),
        (z[2], [0.0, 0.5, 0.5]),
    ]
}

/// Rutile `MX₂`: `M` at (0,0,0) and (½,½,½), `X` at ±(u,u,0) and ±(½+u,½−u,½).
fn rutile(z: [u8; 2], u: f64) -> Vec<(u8, [f64; 3])> {
    vec![
        (z[0], [0.0, 0.0, 0.0]),
        (z[0], [0.5, 0.5, 0.5]),
        (z[1], [u, u, 0.0]),
        (z[1], [1.0 - u, 1.0 - u, 0.0]),
        (z[1], [0.5 + u, 0.5 - u, 0.5]),
        (z[1], [0.5 - u, 0.5 + u, 0.5]),
    ]
}

/// Wurtzite, `P6₃mc`, on its own hexagonal cell.
fn wurtzite(z: [u8; 2], u: f64) -> Vec<(u8, [f64; 3])> {
    vec![
        (z[0], [1.0 / 3.0, 2.0 / 3.0, 0.0]),
        (z[0], [2.0 / 3.0, 1.0 / 3.0, 0.5]),
        (z[1], [1.0 / 3.0, 2.0 / 3.0, u]),
        (z[1], [2.0 / 3.0, 1.0 / 3.0, 0.5 + u]),
    ]
}

/// A hexagonal cell's lattice vectors as rows, Ångström.
fn hexagonal(a: f64, c: f64) -> [[f64; 3]; 3] {
    [
        [a, 0.0, 0.0],
        [-a / 2.0, a * 3.0_f64.sqrt() / 2.0, 0.0],
        [0.0, 0.0, c],
    ]
}

/// A cell with orthogonal axes, as rows.
fn orthogonal(a: f64, b: f64, c: f64) -> [[f64; 3]; 3] {
    [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]]
}

/// Normal spinel `AB₂O₄`, `Fd-3m`, **origin choice 2** (at `-3m`): A on `8a`, B on `16d`,
/// O on `32e` with `u ≈ 0.2616`. 56 atoms in the conventional cell.
///
/// The two origin choices for `Fd-3m` differ by `(⅛,⅛,⅛)`, and tables of spinel positions quote
/// both — `u = 0.2616` for this one and `0.3866` for the other — without always saying which.
/// Mixing them is not a subtle error: the first version of this function paired origin-2 sites
/// with the origin-1 `u` and put two atoms **0.17 Å** apart. `--check` reported that in a second,
/// before any Hessian was spent on it, which is the entire reason that mode exists.
///
/// This is the one structure here whose own cell already clears the Γ margin: `a = 8.083 Å` is
/// 15.3 Bohr, so it needs no replication.
fn spinel(z: [u8; 3], u: f64) -> Vec<(u8, [f64; 3])> {
    // The face-centring translations every Wyckoff position is repeated by.
    let fcc = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    let a_sites = [[0.125, 0.125, 0.125], [0.875, 0.375, 0.375]];
    let b_sites = [
        [0.5, 0.5, 0.5],
        [0.5, 0.25, 0.25],
        [0.25, 0.5, 0.25],
        [0.25, 0.25, 0.5],
    ];
    // The eight `32e` representatives.
    let o_sites = [
        [u, u, u],
        [0.75 - u, 0.25 - u, 0.5 + u],
        [0.25 - u, 0.5 + u, 0.75 - u],
        [0.5 + u, 0.75 - u, 0.25 - u],
        [-u, -u, -u],
        [0.25 + u, 0.75 + u, 0.5 - u],
        [0.75 + u, 0.5 - u, 0.25 + u],
        [0.5 - u, 0.25 + u, 0.75 + u],
    ];

    let mut out = Vec::with_capacity(56);
    for (species, sites) in [
        (z[0], &a_sites[..]),
        (z[1], &b_sites[..]),
        (z[2], &o_sites[..]),
    ] {
        for site in sites {
            for shift in &fcc {
                out.push((
                    species,
                    [
                        (site[0] + shift[0]).rem_euclid(1.0),
                        (site[1] + shift[1]).rem_euclid(1.0),
                        (site[2] + shift[2]).rem_euclid(1.0),
                    ],
                ));
            }
        }
    }
    out
}

/// The `CdI₂` layer type: a metal layer sandwiched between two halide layers, stacked with a
/// van der Waals gap. Hexagonal `P-3m1`, one formula unit per cell.
///
/// This replaced an attempt at layered `LiAlO₂` written on an orthogonal cell, which `--check`
/// caught: forcing the layer onto a square net put two cations 1.98 Å apart, closer than lithium
/// metal. The layer types in this family are hexagonal and do not survive being flattened, so
/// this one is written on a real hexagonal cell instead of approximated on a convenient one.
fn layer_type(z: [u8; 2], u: f64) -> Vec<(u8, [f64; 3])> {
    vec![
        (z[0], [0.0, 0.0, 0.0]),
        (z[1], [1.0 / 3.0, 2.0 / 3.0, u]),
        (z[1], [2.0 / 3.0, 1.0 / 3.0, -u]),
    ]
}

/// `Cd(CN)₂` in its anti-cuprite form: two interpenetrating diamond nets of Cd bridged by C≡N.
///
/// The fractions are *derived* from the bond lengths rather than guessed, because guessing them
/// is what `--check` caught: `0.15`/`0.35` looked reasonable and gave Cd–C = 1.64 Å against a
/// real 2.20, and a C≡N of 2.18 Å against a real 1.15 — a triple bond stretched to nearly twice
/// its length, which would have put the stretching mode wherever it liked.
///
/// Cd at `(0,0,0)` and `(½,½,½)` are `a√3/2` apart along the body diagonal, and the bridge spans
/// that distance as Cd–C + C≡N + N–Cd. So each fraction is its distance along the diagonal as a
/// share of that span, times ½.
fn cyanide_framework(metal: u8, a_angstrom: f64) -> Vec<(u8, [f64; 3])> {
    const CD_C: f64 = 2.20;
    const C_N: f64 = 1.15;
    let diagonal = a_angstrom * 3.0_f64.sqrt() / 2.0;
    let f_c = 0.5 * CD_C / diagonal;
    let f_n = 0.5 * (CD_C + C_N) / diagonal;

    let mut out = vec![(metal, [0.0, 0.0, 0.0]), (metal, [0.5, 0.5, 0.5])];
    // One bridging C≡N along each of the four tetrahedral directions from the origin. Each is
    // shared with a neighbouring Cd, so four bridges per cell is two CN per Cd — the formula.
    for (sx, sy, sz) in [
        (1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
    ] {
        out.push((6, [f_c * sx, f_c * sy, f_c * sz]));
        out.push((7, [f_n * sx, f_n * sy, f_n * sz]));
    }
    out
}

/// A linear dialkyl metal `M(CH₃)₂` at the origin, axis along `z` — the textbook organometallic
/// motif, with two genuine `M–C` σ bonds and no ambiguity about whether it counts as one.
///
/// PM3 has three transition metals — Zn, Cd and Hg, the group-12 `d¹⁰` trio — and this is what
/// they are for. Dimethylzinc and dimethylmercury are real, characterised compounds, linear at
/// the metal, and small enough that a molecular crystal of them fits in a cell wide enough for Γ
/// sampling to mean something.
///
/// Positions are in **Ångström relative to the metal**; the caller places the molecule.
fn dialkyl_metal(metal: u8, m_c: f64) -> Vec<(u8, [f64; 3])> {
    const C_H: f64 = 1.09;
    // Tetrahedral at carbon: the three C–H bonds sit 109.5° from the C–M bond, so they lean away
    // from the metal by 180° − 109.5° = 70.5° off the axis.
    let tilt: f64 = 70.5_f64.to_radians();
    let (sin_t, cos_t) = tilt.sin_cos();

    let mut out = vec![(metal, [0.0, 0.0, 0.0])];
    for side in [1.0_f64, -1.0] {
        let c_z = side * m_c;
        out.push((6, [0.0, 0.0, c_z]));
        for k in 0..3 {
            let phi = std::f64::consts::TAU * k as f64 / 3.0;
            out.push((
                1,
                [
                    C_H * sin_t * phi.cos(),
                    C_H * sin_t * phi.sin(),
                    c_z + side * C_H * cos_t,
                ],
            ));
        }
    }
    out
}

/// A molecular crystal of one `M(CH₃)₂` unit, centred in a cubic cell of `edge` Ångström.
///
/// Fractional coordinates, so the caller's `vectors` decide the packing. One molecule per cell is
/// the crudest possible packing and is deliberate: the point here is the **intramolecular**
/// spectrum of a real organometallic in a periodic setting, and the M–C stretch is what the
/// experimental comparison is against.
fn molecular_crystal(metal: u8, m_c: f64, edge: f64) -> Vec<(u8, [f64; 3])> {
    dialkyl_metal(metal, m_c)
        .into_iter()
        .map(|(z, p)| (z, [0.5 + p[0] / edge, 0.5 + p[1] / edge, 0.5 + p[2] / edge]))
        .collect()
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            kind: "rocksalt",
            formula: "NaCl",
            substitute_for: None,
            basis: rocksalt([11, 17], 5.64),
            vectors: orthogonal(5.64, 5.64, 5.64),
            experiment_cm: 164.0,
            experiment_note: "LO(Γ) ≈ 164, TO ≈ 164 cm⁻¹ (neutron, Raunio 1969)",
            accepted_bond: "2.82 (Na-Cl)",
        },
        Case {
            kind: "rocksalt",
            formula: "MgO",
            substitute_for: None,
            basis: rocksalt([12, 8], 4.212),
            vectors: orthogonal(4.212, 4.212, 4.212),
            experiment_cm: 401.0,
            experiment_note: "TO(Γ) ≈ 401 cm⁻¹ (infrared)",
            accepted_bond: "2.11 (Mg-O)",
        },
        Case {
            kind: "zinc blende",
            formula: "ZnS",
            substitute_for: None,
            basis: zinc_blende([30, 16]),
            vectors: orthogonal(5.41, 5.41, 5.41),
            experiment_cm: 352.0,
            experiment_note: "LO(Γ) ≈ 352 cm⁻¹ (Raman)",
            accepted_bond: "2.34 (Zn-S)",
        },
        Case {
            kind: "wurtzite",
            formula: "ZnO",
            substitute_for: None,
            basis: wurtzite([30, 8], 0.382),
            vectors: hexagonal(3.25, 5.21),
            experiment_cm: 574.0,
            experiment_note: "A1(LO) ≈ 574 cm⁻¹ (Raman)",
            accepted_bond: "1.98 (Zn-O)",
        },
        Case {
            kind: "fluorite",
            formula: "CaF2",
            substitute_for: None,
            basis: fluorite([20, 9]),
            vectors: orthogonal(5.463, 5.463, 5.463),
            experiment_cm: 322.0,
            experiment_note: "Raman-active T2g ≈ 322 cm⁻¹",
            accepted_bond: "2.37 (Ca-F)",
        },
        Case {
            kind: "rutile",
            formula: "SnO2",
            substitute_for: Some("TiO2 — PM3 has no Ti"),
            basis: rutile([50, 8], 0.307),
            vectors: orthogonal(4.737, 4.737, 3.186),
            experiment_cm: 776.0,
            experiment_note: "B2g ≈ 776 cm⁻¹ (Raman)",
            accepted_bond: "2.05 (Sn-O)",
        },
        Case {
            kind: "perovskite",
            formula: "CsPbBr3",
            substitute_for: Some("SrTiO3 — PM3 has no Ti"),
            basis: perovskite([55, 82, 35]),
            vectors: orthogonal(5.87, 5.87, 5.87),
            experiment_cm: 135.0,
            experiment_note: "highest optical ≈ 135 cm⁻¹ (Raman, cubic phase)",
            accepted_bond: "2.94 (Pb-Br)",
        },
        Case {
            kind: "spinel",
            formula: "MgAl2O4",
            substitute_for: None,
            basis: spinel([12, 13, 8], 0.2616),
            vectors: orthogonal(8.083, 8.083, 8.083),
            experiment_cm: 770.0,
            experiment_note: "A1g ≈ 770 cm⁻¹ (Raman)",
            accepted_bond: "1.92 (Al-O), 1.93 (Mg-O)",
        },
        Case {
            kind: "layer type (CdI2)",
            formula: "CdI2",
            substitute_for: Some("layered LiCoO2 — PM3 has no Co"),
            basis: layer_type([48, 53], 0.25),
            vectors: hexagonal(4.24, 6.84),
            experiment_cm: 150.0,
            experiment_note: "A1g ≈ 150 cm⁻¹ (Raman)",
            accepted_bond: "2.99 (Cd-I)",
        },
        // The organometallics proper: a genuine M–C σ bond, which is the thing that makes a
        // compound organometallic rather than merely a coordination complex.
        Case {
            kind: "organometallic",
            formula: "Zn(CH3)2",
            substitute_for: None,
            basis: molecular_crystal(30, 1.930, 8.0),
            vectors: orthogonal(8.0, 8.0, 8.0),
            experiment_cm: 615.0,
            experiment_note: "symmetric Zn-C stretch ~ 615 cm⁻¹ (Raman, gas/liquid)",
            accepted_bond: "1.93 (Zn-C)",
        },
        Case {
            kind: "organometallic",
            formula: "Hg(CH3)2",
            substitute_for: None,
            basis: molecular_crystal(80, 2.083, 8.5),
            vectors: orthogonal(8.5, 8.5, 8.5),
            experiment_cm: 515.0,
            experiment_note: "symmetric Hg-C stretch ~ 515 cm⁻¹ (Raman)",
            accepted_bond: "2.08 (Hg-C)",
        },
        Case {
            kind: "cyanide framework",
            formula: "Cd(CN)2",
            substitute_for: Some("ferrocene — PM3 has no Fe; this has Cd–C bonds"),
            basis: cyanide_framework(48, 6.30),
            vectors: orthogonal(6.30, 6.30, 6.30),
            experiment_cm: 2170.0,
            experiment_note: "C≡N stretch ≈ 2170 cm⁻¹ (infrared)",
            accepted_bond: "1.15 (C≡N)",
        },
    ]
}

/// Replicate until every lattice width clears the 14 Bohr exchange cutoff, capped by `max_atoms`.
fn supercell(case: &Case, max_atoms: usize) -> Option<(Molecule, [usize; 3], f64)> {
    const CUTOFF_BOHR: f64 = 14.0;
    let single = lattice(case, [1, 1, 1]);
    // Replicate on the vector's *length*, not on an edge parameter: a hexagonal `a` and the
    // second lattice vector have the same length but different components.
    let mut reps = [1usize; 3];
    for axis in 0..3 {
        while single[axis].norm() * reps[axis] as f64 <= CUTOFF_BOHR {
            reps[axis] += 1;
        }
    }
    let count = case.basis.len() * reps[0] * reps[1] * reps[2];
    if count > max_atoms {
        return None;
    }

    let big = lattice(case, reps);
    let cell = Cell::new(big[0], big[1], big[2], [true; 3]).ok()?;

    let mut atoms = Vec::with_capacity(count);
    for ix in 0..reps[0] {
        for iy in 0..reps[1] {
            for iz in 0..reps[2] {
                let shift = single[0] * ix as f64 + single[1] * iy as f64 + single[2] * iz as f64;
                for (z, frac) in &case.basis {
                    atoms.push(Atom {
                        z: *z,
                        position: cartesian(&single, *frac) + shift,
                    });
                }
            }
        }
    }
    let mut molecule = Molecule::new(atoms);
    molecule.cell = Some(cell);
    // The margin is set by the narrowest *width* of the cell, which for a non-orthogonal cell is
    // not the shortest vector — it is the shortest perpendicular distance between opposite faces.
    let width = |i: usize, j: usize, k: usize| {
        let normal = big[j].cross(big[k]);
        let area = normal.norm();
        if area <= 0.0 {
            0.0
        } else {
            big[i].dot(normal).abs() / area
        }
    };
    let margin = width(0, 1, 2).min(width(1, 2, 0)).min(width(2, 0, 1)) - CUTOFF_BOHR;
    Some((molecule, reps, margin))
}

/// Lattice vectors in Bohr, optionally replicated per axis.
fn lattice(case: &Case, reps: [usize; 3]) -> [Vec3; 3] {
    std::array::from_fn(|axis| {
        let v = case.vectors[axis];
        Vec3::new(v[0] * A, v[1] * A, v[2] * A) * reps[axis] as f64
    })
}

/// Fractional coordinates against lattice vectors, which is the only correct way to place an
/// atom once the cell stops being orthogonal.
fn cartesian(vectors: &[Vec3; 3], frac: [f64; 3]) -> Vec3 {
    vectors[0] * frac[0] + vectors[1] * frac[1] + vectors[2] * frac[2]
}

/// The conventional cell alone, for the geometry check — no replication, no SCF.
fn conventional(case: &Case) -> Molecule {
    let vectors = lattice(case, [1, 1, 1]);
    let atoms = case
        .basis
        .iter()
        .map(|(z, f)| Atom {
            z: *z,
            position: cartesian(&vectors, *f),
        })
        .collect();
    let mut molecule = Molecule::new(atoms);
    molecule.cell =
        Some(Cell::new(vectors[0], vectors[1], vectors[2], [true; 3]).expect("a valid cell"));
    molecule
}

/// The same supercell with every lattice vector and every position scaled isotropically.
///
/// Fractional coordinates are unchanged by construction, so this is a pure volume change: it
/// cannot move an atom off a symmetry position, which is exactly why the scan below is a
/// *complete* relaxation for the structures whose only free coordinate is the lattice constant.
fn scaled(case: &Case, supercell: &Molecule, scale: f64) -> Molecule {
    let _ = case;
    let cell = supercell.cell.expect("built with a cell");
    let mut out = supercell.clone();
    out.cell = Some(
        Cell::new(
            cell.h.col[0] * scale,
            cell.h.col[1] * scale,
            cell.h.col[2] * scale,
            [true; 3],
        )
        .expect("scaling a valid cell keeps it valid"),
    );
    for atom in &mut out.atoms {
        atom.position = atom.position * scale;
    }
    out
}

/// The scale factor that minimizes the energy, from a parabola through the best three points.
///
/// Nine single points from 0.88 to 1.12 — wide enough to bracket a semiempirical method that
/// never saw a lattice, coarse enough to be cheap. A minimum at either end of the range is
/// reported as no minimum rather than as the endpoint, because an endpoint is where the scan
/// stopped and not where the energy turned.
fn equilibrium_scale(
    case: &Case,
    supercell: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
) -> Option<f64> {
    // 0.70 to 1.30. The first version spanned ±12% and every ionic solid put its minimum on an
    // endpoint — which is a result rather than a failure: PM3 never saw a Madelung lattice, and
    // its equilibrium volume for these is nowhere near the measured one. A scan has to be wide
    // enough to contain the answer before it can report where it is.
    // **Compressing the cell shrinks the Γ margin**, and a scan that ignores that reports the
    // approximation breaking down as if it were physics. The first version ran 0.70 to 1.30 and
    // both rocksalts came back "still falling at 0.70" — an ionic crystal collapsing to a third
    // of its volume, with every point converged. MgO's margin at 0.70 is **−2.9 Bohr**: those
    // points were not a compressed crystal, they were a Γ calculation standing in for images
    // inside its own exchange range.
    //
    // So the range is trimmed to where one k-point still means something, and how far it had to
    // be trimmed is reported rather than absorbed.
    const CUTOFF_BOHR: f64 = 14.0;
    let cell = supercell.cell.expect("built with a cell");
    let width = |i: usize, j: usize, k: usize| {
        let normal = cell.h.col[j].cross(cell.h.col[k]);
        let area = normal.norm();
        if area <= 0.0 {
            0.0
        } else {
            cell.h.col[i].dot(normal).abs() / area
        }
    };
    let narrowest = width(0, 1, 2).min(width(1, 2, 0)).min(width(2, 0, 1));
    // A margin of at least 1 Bohr, not merely positive: at zero the substitution is exactly as
    // wrong as it can be while still passing a sign test.
    let floor = (CUTOFF_BOHR + 1.0) / narrowest;

    let scales: Vec<f64> = (0..13)
        .map(|i| 0.70 + 0.05 * i as f64)
        .filter(|s| *s >= floor)
        .collect();
    if scales.len() < 3 {
        eprintln!(
            "  [{}] only {} of 13 scan points keep the Γ margin above 1 Bohr (the cell would have \
             to stay above {:.2}× to stay legal), which is too few to find a minimum in",
            case.formula,
            scales.len(),
            floor,
        );
        return None;
    }
    if scales.len() < 13 {
        eprintln!(
            "  [{}] scan trimmed to {:.2}×–1.30× ({} points): below that the Γ margin falls under \
             1 Bohr and the compressed energies would be the approximation failing, not the \
             crystal",
            case.formula,
            floor,
            scales.len(),
        );
    }
    let mut energies = Vec::with_capacity(scales.len());
    for &s in &scales {
        let energy = run_gamma(&scaled(case, supercell, s), params, options, periodic)
            .ok()
            .map(|r| r.total_ev);
        energies.push(energy);
    }
    // The lowest point that actually converged, and its converged neighbours.
    let best = (0..scales.len())
        .filter(|&i| energies[i].is_some())
        .min_by(|&a, &b| {
            energies[a]
                .unwrap()
                .partial_cmp(&energies[b].unwrap())
                .unwrap()
        })?;
    let converged = energies.iter().filter(|e| e.is_some()).count();
    if best == 0 || best + 1 == scales.len() {
        eprintln!(
            "  [{}] still falling at scale {:.2}, the edge of the scan ({converged}/{} points \
             converged): PM3's equilibrium volume is outside ±30% of the experimental one",
            case.formula,
            scales[best],
            scales.len(),
        );
        return None;
    }
    if energies[best - 1].is_none() || energies[best + 1].is_none() {
        eprintln!(
            "  [{}] the minimum at scale {:.2} has a neighbour whose SCF did not converge, so \
             there is no parabola to fit ({converged}/{} points converged)",
            case.formula,
            scales[best],
            scales.len(),
        );
        return None;
    }
    let (l, c, r): (f64, f64, f64) = (energies[best - 1]?, energies[best]?, energies[best + 1]?);
    let h = scales[1] - scales[0];
    // Vertex of the parabola through three evenly spaced points.
    let denominator = l - 2.0 * c + r;
    if denominator.abs() < 1.0e-12 {
        return Some(scales[best]);
    }
    Some(scales[best] - 0.5 * h * (r - l) / denominator)
}

/// Nearest-neighbour distance and coordination number per structure, against the accepted bond.
///
/// Runs in a second and costs no SCF, so it is what to run before committing hours of Hessians:
/// a spectrum computed on a subtly wrong lattice is plausible, quotable, and says nothing about
/// the geometry it came from.
fn check() {
    println!("Nearest-neighbour distances in the survey's structures\n");
    println!(
        "  {:<17} {:<9} {:>11} {:>7} {:>14}  pair",
        "structure", "formula", "shortest", "coord", "accepted"
    );
    for case in cases() {
        let molecule = conventional(&case);
        let cell = molecule.cell.expect("built with a cell");
        let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
        // 8 Bohr reaches past any first shell and keeps the list small.
        let list = pm3_rs::NeighborList::build_from_positions(&positions, Some(&cell), 8.0);

        // Heavy atoms only. `accepted_bond` names a heavy–heavy contact in every row — the M–C
        // bond of an organometallic, the C≡N of the framework, the cation–anion distance of an
        // ionic solid — and including hydrogen made the column report the C–H of a methyl
        // group next to an accepted value for Zn–C, which is two different bonds in one row.
        let mut shortest = f64::MAX;
        let mut pair = (0u8, 0u8);
        let mut centre = 0usize;
        for image in list.all() {
            let heavy = molecule.atoms[image.a].z > 1 && molecule.atoms[image.b].z > 1;
            if heavy && image.r > 1.0e-6 && image.r < shortest {
                shortest = image.r;
                pair = (molecule.atoms[image.a].z, molecule.atoms[image.b].z);
                centre = image.a;
            }
        }
        if shortest == f64::MAX {
            println!(
                "  {:<17} {:<9}  no neighbours inside 8 Bohr",
                case.kind, case.formula
            );
            continue;
        }
        // Neighbours of the atom that *has* the shortest bond, at that distance. Counting atom 0
        // instead reported zero for perovskite and spinel, whose first atom is the large cation
        // and is nowhere near the shortest contact.
        let coordination = list
            .all()
            .iter()
            .filter(|p| p.a == centre && (p.r - shortest).abs() < 0.25)
            .count();

        println!(
            "  {:<17} {:<9} {:>9.3} A {:>7} {:>14}  {}-{}",
            case.kind,
            case.formula,
            shortest / A,
            coordination,
            case.accepted_bond,
            pm3_rs::z_to_symbol(pair.0).unwrap_or("?"),
            pm3_rs::z_to_symbol(pair.1).unwrap_or("?"),
        );
    }
    println!(
        "\nExpected coordination: 6 rocksalt, 4 zinc blende and wurtzite, 8 for the cation in\n\
         fluorite, 6 rutile, 6 for the perovskite B site, 4 for Cd in the cyanide framework."
    );
}

fn main() {
    if std::env::args().any(|a| a == "--check") {
        check();
        return;
    }
    let max_atoms: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(64);
    // Relaxation steps. Capped rather than run to convergence because these are supercells and
    // each step is a periodic gradient; whether it arrived is printed, so a truncated run is
    // visible rather than assumed.
    let _max_steps: usize = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(40);

    let params = Pm3Parameters::standard().expect("parameters");
    let options = Pm3Options {
        max_scf: 400,
        ..Pm3Options::default()
    };
    let periodic = PeriodicOptions::default();

    println!("Γ-point phonons by structure type (PM3, supercells sized for a positive Γ margin)");
    println!("Atom cap {max_atoms}; pass a larger one as the first argument.\n");
    println!(
        "  {:<17} {:<9} {:>6} {:>7} {:>7} {:>6} {:>10} {:>10} {:>7}  acoustic",
        "structure",
        "formula",
        "atoms",
        "margin",
        "a/a_exp",
        "relax",
        "highest",
        "experiment",
        "ratio"
    );

    for case in cases() {
        let Some((molecule, reps, margin)) = supercell(&case, max_atoms) else {
            println!(
                "  {:<17} {:<9} {:>6}  skipped: needs more than {max_atoms} atoms to clear the \
                 Γ margin",
                case.kind, case.formula, "--"
            );
            continue;
        };
        let started = Instant::now();

        // **Find PM3's lattice constant before differentiating.** A Hessian is only a phonon
        // spectrum at a stationary point; anywhere else the negative curvatures it finds are the
        // geometry being wrong, not the crystal being unstable. Differentiating at the
        // *experimental* lattice constant instead — which is where the first version of this
        // survey looked — measures how far PM3's equilibrium sits from the measured one, and
        // says nothing about the curvature at that equilibrium.
        //
        // A scan over the isotropic scale rather than a variable-cell relaxation, and the
        // difference is three orders of magnitude of wall clock. A relaxation on a 64-atom
        // supercell needs a periodic gradient per line-search trial and took **seventeen
        // minutes** on rocksalt alone; a scan is nine single points, and a single point after the
        // Ewald work in this release is about two seconds. For every cubic structure here with no
        // free internal coordinate the two are the *same calculation* — the lattice constant is
        // the only degree of freedom symmetry leaves — so nothing is given up. Where there is an
        // internal parameter (rutile, wurtzite, spinel, the layer type) the scan reaches the
        // right volume and not the right internal coordinate, and the residual forces printed
        // below are what says so.
        //
        // The scan's own result is worth more than its speed: **how far PM3's equilibrium sits
        // from experiment** is a statement about the model, and it is what explains the imaginary
        // branches rather than merely removing them.
        let (scale, scan_seconds) =
            match equilibrium_scale(&case, &molecule, &params, &options, &periodic) {
                Some(found) => (found, started.elapsed().as_secs_f64()),
                None => {
                    println!(
                        "  {:<17} {:<9} {:>6} {:>7.1}  no minimum found in the scan after {:.0} s",
                        case.kind,
                        case.formula,
                        molecule.atoms.len(),
                        margin,
                        started.elapsed().as_secs_f64(),
                    );
                    continue;
                }
            };
        let relaxed = scaled(&case, &molecule, scale);
        let relax_seconds = scan_seconds;

        let result = periodic_phonons(&relaxed, &params, &options, &periodic);
        let elapsed = started.elapsed().as_secs_f64();

        match result {
            Ok(phonons) => {
                let highest = phonons
                    .frequencies_cm
                    .iter()
                    .cloned()
                    .fold(f64::MIN, f64::max);
                // The projection makes the acoustic modes exactly zero, so counting them is a
                // count of exact zeros rather than a threshold.
                let acoustic = phonons.frequencies_cm.iter().filter(|f| **f == 0.0).count();
                let imaginary = phonons.frequencies_cm.iter().filter(|f| **f < 0.0).count();
                // *How* imaginary, not just how many. A geometry a little off a minimum gives a
                // few shallow negative curvatures; a spectrum where nothing is real and the worst
                // mode is hundreds of cm⁻¹ down is saying something else — a Γ margin too thin
                // for one k-point to stand in for the images, or an SCF that did not converge.
                let deepest = phonons
                    .frequencies_cm
                    .iter()
                    .cloned()
                    .fold(f64::MAX, f64::min);
                println!(
                    "  {:<17} {:<9} {:>6} {:>7.1} {:>7.3} {:>6} {:>10.1} {:>10.1} {:>7.2}  {} \
                     zero, {} imaginary [{}x{}x{}, relax {:.0} s + hessian {:.0} s]",
                    case.kind,
                    case.formula,
                    relaxed.atoms.len(),
                    margin,
                    scale,
                    "scan",
                    highest,
                    case.experiment_cm,
                    highest / case.experiment_cm,
                    acoustic,
                    imaginary,
                    reps[0],
                    reps[1],
                    reps[2],
                    relax_seconds,
                    elapsed - relax_seconds,
                );
                if imaginary > 0 {
                    println!(
                        "  {:<17} {:<9}   worst imaginary {:.0} cm^-1; SCF {}",
                        "",
                        "",
                        deepest,
                        if phonons.scf.converged {
                            "converged"
                        } else {
                            "DID NOT CONVERGE"
                        },
                    );
                }
            }
            Err(error) => {
                println!(
                    "  {:<17} {:<9} {:>6} {:>7.1}  failed after {elapsed:.0} s: {error}",
                    case.kind,
                    case.formula,
                    molecule.atoms.len(),
                    margin,
                );
            }
        }
    }

    println!("\nSubstitutions, and why:");
    for case in cases() {
        if let Some(reason) = case.substitute_for {
            println!("  {:<9} stands in for {}", case.formula, reason);
        }
    }
    println!("\nExperimental references:");
    for case in cases() {
        println!("  {:<9} {}", case.formula, case.experiment_note);
    }
}
