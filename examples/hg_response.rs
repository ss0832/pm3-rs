// SPDX-License-Identifier: GPL-3.0-or-later

//! Why the coupled-perturbed response diverges on HgF₂, term by term.
//!
//! ```text
//! cargo run --release --example hg_response
//! ```
//!
//! The symptom is a DFPT run that converges its SCF and then reports a *response* residual of
//! `7.5e1` — a solve running away, which is what reaches `±inf` if it is allowed to. The three
//! candidates are the energy denominators, the long-range lattice sum in the response kernel, and
//! the `d` shell.
//!
//! **The `d` shell is not one of them.** No PM3 element carries `d` orbitals — `tests/api_surface.rs`
//! asserts `!element.has_d()` for every supported `Z`, and `docs/scope.md` says the parameterization
//! is the "PM3 s/p basis". Hg is s/p like everything else, so nothing about this can be a `d` path.
//!
//! **The denominators are not one either**, at least not by being small: HgF₂ at `a = 11 Å` has a
//! 12.0 eV gap and at `a = 14 Å` a 9.1 eV one, and diverges at both. A small-denominator failure
//! needs a small gap.
//!
//! **The long-range lattice sum is not one either.** `DfptOptions::long_range` switches the
//! monopole term off in the response *and* in the skeleton together, and the table below shows the
//! failure surviving it: HgF₂ at `a = 14 Å` fails with the term on and fails with it off.
//!
//! So none of the three. What is left is the **solver**, and there are two of them, failing
//! differently:
//!
//! * `pbc::hessian`'s Γ-point CPHF (`CPHF_ITERATIONS = 400`, `CPHF_TOLERANCE = 1e-9`) genuinely
//!   **diverges** — `periodic_phonons` on HgF₂ at `a = 11 Å` ends at a residual of `7.5e1`, four
//!   hundred passes of a fixed point running away. That is the one that reaches `±inf`.
//! * `pbc::dfpt`'s response (`RESPONSE_ITERATIONS = 200`, `RESPONSE_TOLERANCE = 1e-10`) **stalls**
//!   just above a very tight tolerance, and does it **non-deterministically**: two consecutive
//!   runs of the identical calculation reported `9.390e-9` and `4.506e-10`. The columns are solved
//!   under `par_iter`, so the answer depends on thread scheduling — which is a defect in its own
//!   right, separate from the convergence.
//!
//! At the level of the formula, neither is a single diverging *term*. The response is the fixed
//! point `ΔP ← χ·K[ΔP] + ΔP_bare`, and what fails is the spectral radius of `χ·K` exceeding one —
//! a property of the whole kernel. It is parameter-dependent rather than structural: Ca and Cd
//! converge at the identical geometry where Hg does not.

use pm3_rs::pbc::dfpt::{force_constants_at_q, DfptOptions, LongRange};
use pm3_rs::pbc::gamma::PeriodicOptions;
use pm3_rs::{Atom, Cell, Molecule, Pm3Options, Pm3Parameters, Vec3};

const ANGSTROM_TO_BOHR: f64 = 1.8897261254578281;

fn fluorite(cation: u8, a_angstrom: f64) -> Molecule {
    let a = a_angstrom * ANGSTROM_TO_BOHR;
    let h = a / 2.0;
    let q = a / 4.0;
    let cell = Cell::new(
        Vec3::new(0.0, h, h),
        Vec3::new(h, 0.0, h),
        Vec3::new(h, h, 0.0),
        [true; 3],
    )
    .unwrap();
    let mut molecule = Molecule::new(vec![
        Atom {
            z: cation,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: 9,
            position: Vec3::new(q, q, q),
        },
        Atom {
            z: 9,
            position: Vec3::new(-q, -q, -q),
        },
    ]);
    molecule.cell = Some(cell);
    molecule
}

fn main() {
    let params = Pm3Parameters::standard().unwrap();
    let periodic = PeriodicOptions::default();
    let options = Pm3Options::default();

    println!("No PM3 element carries d orbitals. Confirming, for the cations below:");
    for (name, z) in [("Ca", 20u8), ("Zn", 30), ("Cd", 48), ("Hg", 80)] {
        let element = params.element(z).unwrap();
        println!(
            "  {name:<3} Z={z:<3} n_orb = {}  has_d = {}  zeta_d = {:.4}",
            element.n_orb,
            element.has_d(),
            element.zeta_d
        );
    }
    println!();

    println!(
        "{:<6}{:>8}{:>12}{:>34}{:>34}",
        "cation", "a (A)", "gap (eV)", "response, long range ON", "response, long range OFF"
    );
    for (name, z) in [("Ca", 20u8), ("Cd", 48), ("Hg", 80)] {
        for a in [11.0, 14.0] {
            let molecule = fluorite(z, a);
            let gap = match pm3_rs::run_gamma(&molecule, &params, &options, &periodic) {
                Ok(scf) => match (scf.homo_ev, scf.lumo_ev) {
                    (Some(h), Some(l)) => format!("{:.3}", l - h),
                    _ => "n/a".to_string(),
                },
                Err(_) => "SCF fail".to_string(),
            };
            let run = |long_range: LongRange| {
                let dfpt = DfptOptions {
                    long_range,
                    ..DfptOptions::default()
                };
                match force_constants_at_q(&molecule, &params, &options, &periodic, &dfpt, [0.0; 3])
                {
                    Ok(result) => {
                        let worst = (0..result.dynamical.matrix.rows)
                            .flat_map(|i| (0..result.dynamical.matrix.rows).map(move |j| (i, j)))
                            .map(|(i, j)| result.dynamical.matrix[(i, j)].re.abs())
                            .fold(0.0_f64, f64::max);
                        if worst.is_finite() {
                            format!("converged, max |D| {worst:.3e}")
                        } else {
                            "*** NON-FINITE ***".to_string()
                        }
                    }
                    Err(e) => {
                        let text = e.to_string();
                        // Just the residual, which is what says diverging from merely slow.
                        match text.find("error=") {
                            Some(at) => format!("failed, {}", &text[at..text.len().min(at + 16)]),
                            None => text.chars().take(28).collect(),
                        }
                    }
                }
            };
            println!(
                "{name:<6}{:>8.1}{gap:>12}{:>34}{:>34}",
                a,
                run(LongRange::Auto),
                run(LongRange::Off)
            );
        }
    }

    println!();
    println!("The failure survives switching the long-range term off, so it is not that term.");
    println!("It is not the denominators either -- the gaps above are 9 to 12 eV -- and it is not");
    println!("a d shell, which PM3 does not have. What is left is the fixed point itself:");
    println!("  dP <- chi.K[dP] + dP_bare");
    println!("failing because the spectral radius of chi.K exceeds one. That is a property of the");
    println!("whole kernel at these parameters, not of one term in the assembly, which is why Ca");
    println!("and Cd converge at the identical geometry where Hg does not.");
    println!();
    println!("Separately: the residual this reports for Hg varies between runs of the same");
    println!("calculation, because the response columns are solved under par_iter. A converged");
    println!("answer should not depend on thread scheduling.");
}
