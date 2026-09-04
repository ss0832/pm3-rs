// SPDX-License-Identifier: GPL-3.0-or-later

//! The scaling order of the analytic correction Hessian and gradient.
//!
//! ```text
//! cargo run --release --example correction_scaling
//! ```
//!
//! What is *measured* here is `correction_energy` itself, which is the thing the cutoff changed.
//! The derivative orders follow from it arithmetically and are stated rather than timed: the
//! analytic Hessian evaluates this once for each of `N²/2` atom pairs (times nine mixed
//! derivatives), and the gradient once per atom. So an evaluation that falls from `O(N²)` to
//! `O(N)` takes the Hessian from `O(N⁴)` to `O(N³)` and the gradient from `O(N³)` to `O(N²)`.
//!
//! Timing the Hessian directly would be the better measurement and is not done here, because the
//! correction Hessian is an internal of `crate::hessian` and exposing it to time it would be
//! adding API for a benchmark.

use std::time::Instant;

use pm3_rs::{Molecule, Pm3Options, Pm3Parameters, Variant};

fn water_block(n: usize) -> Molecule {
    let mut xyz = String::new();
    let mut count = 0;
    let mut body = String::new();
    let spacing = 3.0; // Angstrom
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let (x, y, z) = (i as f64 * spacing, j as f64 * spacing, k as f64 * spacing);
                body.push_str(&format!("O {x} {y} {z}\n"));
                body.push_str(&format!("H {} {y} {z}\n", x + 0.96));
                body.push_str(&format!("H {} {} {z}\n", x - 0.24, y + 0.93));
                count += 3;
            }
        }
    }
    xyz.push_str(&format!("{count}\nwater block\n"));
    xyz.push_str(&body);
    Molecule::from_xyz_str(&xyz, 0.0).unwrap()
}

fn slope(sizes: &[f64], times: &[f64]) -> f64 {
    let n = sizes.len() as f64;
    let lx: Vec<f64> = sizes.iter().map(|v| v.ln()).collect();
    let ly: Vec<f64> = times.iter().map(|v| v.ln()).collect();
    let mx = lx.iter().sum::<f64>() / n;
    let my = ly.iter().sum::<f64>() / n;
    let num: f64 = lx.iter().zip(&ly).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = lx.iter().map(|x| (x - mx) * (x - mx)).sum();
    num / den
}

fn main() {
    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options {
        variant: Variant::Pm3D3H4,
        ..Pm3Options::default()
    };

    let mut sizes = Vec::new();
    let mut times = Vec::new();

    println!(
        "{:>8} {:>16} {:>16}",
        "atoms", "energy (ms)", "per atom (us)"
    );
    for n in [3, 4, 5, 6, 7] {
        let molecule = water_block(n);
        let nat = molecule.atoms.len();

        // Repeated, because one call at the small sizes is under the clock's resolution.
        let repeats = if nat < 200 { 200 } else { 20 };
        let start = Instant::now();
        for _ in 0..repeats {
            let _ = pm3_rs::corrections::correction_energy(&molecule, options.variant);
        }
        let each = start.elapsed().as_secs_f64() / repeats as f64;

        println!(
            "{nat:>8} {:>16.4} {:>16.3}",
            each * 1.0e3,
            each * 1.0e6 / nat as f64
        );
        sizes.push(nat as f64);
        times.push(each);
        let _ = &params;
    }

    println!();
    println!(
        "correction_energy log-log slope: {:.3}",
        slope(&sizes, &times)
    );
    println!("(1.0 is linear; this was 2.0 before the sums were bounded)");
}
