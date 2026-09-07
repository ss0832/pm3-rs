// SPDX-License-Identifier: GPL-3.0-or-later

//! What actually goes wrong when a periodic SCF does not converge, and what rescues it.
//!
//! ```text
//! cargo run --release --example scf_hardening
//! PM3_KSCF_TRACE=1 cargo run --release --example scf_hardening   # per-iteration detail
//! ```
//!
//! Three systems that fail, and they do not fail the same way. The trace
//! (`PM3_KSCF_TRACE=1`) reports two things beyond the residual, because the two failure modes
//! look identical in the residual alone and want opposite remedies:
//!
//! * `flips` — states whose occupation changed since the last pass. Non-zero means **band
//!   crossing**: the iteration is choosing a different filling each time, and no amount of
//!   damping the density can help, because the discontinuity is in the occupation rule.
//! * `n_e` — electrons per atom. A swing here with the energy barely moving is **charge
//!   sloshing**: a long-wavelength mode the iteration overshoots, cheap in energy because moving
//!   charge between well-separated sites costs little.
//!
//! Measured, all three at their default settings:
//!
//! | system | mesh | `flips` | `n_e` swing | outcome |
//! |---|---|---|---|---|
//! | NaCl | `2×2×2` | **0** | **1.1 electrons** | sloshes for ~30 passes, then converges |
//! | diamond | `3×3×3` | **0** | 0.0000 | slow tail, runs out |
//! | silicon | `3×3×3` | **0** | 0.0000 | flat residual, runs out |
//!
//! **Band crossing is not the mechanism in any of them.** `flips` is zero at every iteration of
//! all three, so the occupations never change sides. That matters because `docs/pbc.md` used to
//! attribute NaCl's difficulty to a Fermi level trapped in a degenerate manifold — the charge
//! moves first and `μ` follows it across twelve electronvolts, not the other way round. NaCl also
//! *recovers*: the sloshing damps out and it converges to `−341.786 eV`.
//!
//! Diamond and silicon are the real failures, and neither oscillates. Diamond's residual falls at
//! about `0.993` per step — `1.1e-4` at iteration 40 to `3.7e-5` at 200, against a `1e-7`
//! tolerance — which is a stiff iteration rather than an unstable one.
//!
//! # What rescues them, and what only appears to
//!
//! | rung | diamond | silicon | NaCl |
//! |---|---|---|---|
//! | default | fails | fails | `−341.786494` |
//! | 2000 iterations | fails `2.6e-5` | fails, *worse* | `−341.786494` |
//! | damping 0.9 | fails | **`−145.536996`** | `−341.786466` |
//! | level shift 5 eV | fails `8.5e-8` | **`−145.536996`** | `−379.223901` |
//! | smearing 0.1 eV | `−249.266817` | `−144.847720` | `−341.786495` |
//! | smearing 0.5 eV | `−249.266817` | fails | `−379.223901` |
//!
//! Read the energies, not the words. Damping and the level shift agree with each other on silicon
//! — independent routes to `−145.536996` — which is what says they found the intended solution.
//! Everywhere a cell is **bold** the answer is corroborated; everywhere it is not, the rung
//! converged to something and cannot show it is the right something. NaCl under a level shift or
//! heavy smearing lands 37 eV away from the answer it reaches unaided.
//!
//! This is why [`pm3_rs::pbc::kscf::run_kpoints`] retries with **smearing only**, and keeps the
//! result **only when the electronic entropy comes out zero** — integral occupations mean the
//! smeared fixed-point equations are the strict-filling ones, so the answer is provably the one
//! that was asked for. Diamond fails that check (entropy `1.4e-2 eV`), so its smeared energy is
//! reported in the error rather than returned. Damping and level shift have no such certificate
//! and are suggested to the caller instead of applied on their behalf.

use pm3_rs::pbc::gamma::PeriodicOptions;
use pm3_rs::pbc::kscf::{run_kpoints, KpointOptions};
use pm3_rs::{Atom, Cell, Molecule, Pm3Options, Pm3Parameters, Vec3};

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

struct Case {
    name: &'static str,
    molecule: Molecule,
    mesh: [usize; 3],
}

fn cases() -> Vec<Case> {
    let q = 3.567 / 4.0;
    let s = 5.431 / 4.0;
    vec![
        Case {
            name: "diamond",
            molecule: fcc([6, 6], 3.567, [q, q, q]),
            mesh: [3, 3, 3],
        },
        Case {
            name: "silicon",
            molecule: fcc([14, 14], 5.431, [s, s, s]),
            mesh: [3, 3, 3],
        },
        Case {
            name: "NaCl",
            molecule: fcc([11, 17], 5.64, [5.64 / 2.0, 0.0, 0.0]),
            mesh: [2, 2, 2],
        },
    ]
}

/// One rung of the ladder.
struct Rung {
    label: &'static str,
    max_scf: usize,
    damping: f64,
    level_shift_ev: f64,
    smearing_ev: f64,
}

const LADDER: &[Rung] = &[
    Rung {
        label: "default",
        max_scf: 200,
        damping: 0.0,
        level_shift_ev: 0.0,
        smearing_ev: 0.0,
    },
    Rung {
        label: "2000 iterations",
        max_scf: 2000,
        damping: 0.0,
        level_shift_ev: 0.0,
        smearing_ev: 0.0,
    },
    Rung {
        label: "damping 0.7",
        max_scf: 2000,
        damping: 0.7,
        level_shift_ev: 0.0,
        smearing_ev: 0.0,
    },
    Rung {
        label: "damping 0.9",
        max_scf: 2000,
        damping: 0.9,
        level_shift_ev: 0.0,
        smearing_ev: 0.0,
    },
    Rung {
        label: "level shift 5 eV",
        max_scf: 2000,
        damping: 0.0,
        level_shift_ev: 5.0,
        smearing_ev: 0.0,
    },
    Rung {
        label: "smearing 0.1 eV",
        max_scf: 2000,
        damping: 0.0,
        level_shift_ev: 0.0,
        smearing_ev: 0.1,
    },
    Rung {
        label: "smearing 0.5 eV",
        max_scf: 2000,
        damping: 0.0,
        level_shift_ev: 0.0,
        smearing_ev: 0.5,
    },
];

fn main() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();

    for case in cases() {
        println!(
            "=== {} on a {}x{}x{} mesh ===",
            case.name, case.mesh[0], case.mesh[1], case.mesh[2]
        );
        for rung in LADDER {
            let options = Pm3Options {
                max_scf: rung.max_scf,
                damping: rung.damping,
                level_shift_ev: rung.level_shift_ev,
                ..Pm3Options::default()
            };
            let kopt = KpointOptions {
                smearing_ev: rung.smearing_ev,
                ..KpointOptions::mesh(case.mesh)
            };
            let started = std::time::Instant::now();
            match run_kpoints(&case.molecule, &params, &options, &periodic, &kopt) {
                Ok(scf) => println!(
                    "  {:<18} converged, E = {:>14.6} eV   [{:.1} s]",
                    rung.label,
                    scf.electronic_ev + scf.core_ev,
                    started.elapsed().as_secs_f64()
                ),
                Err(e) => {
                    let text = e.to_string();
                    let short = text.split(',').next().unwrap_or(&text);
                    println!(
                        "  {:<18} {}   [{:.1} s]",
                        rung.label,
                        short.trim(),
                        started.elapsed().as_secs_f64()
                    );
                }
            }
        }
        println!();
    }

    println!("A rung that converges to a *different* energy has not rescued the calculation --");
    println!("it has answered a different question. Compare the energies, not just the words.");
}
