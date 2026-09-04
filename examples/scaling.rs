// SPDX-License-Identifier: GPL-3.0-or-later

//! Wall-clock scaling of the divide-and-conquer path against full diagonalization.
//!
//! Prints a table and fits a log-log slope. The claim divide-and-conquer makes is that its slope
//! is 1; full diagonalization's is 3 in the limit. Run with:
//!
//! ```text
//! cargo run --release --example scaling
//! ```
//!
//! The systems are chains of water molecules, which grow in one dimension — the case where
//! near-sightedness bites hardest and a partitioning has the most to gain.

use std::time::Instant;

use pm3_rs::dc::{run_dc, DcOptions};
use pm3_rs::params::Pm3Parameters;
use pm3_rs::scf::{run_pm3, Pm3Options};
use pm3_rs::system::Molecule;

fn water_chain(n: usize) -> Molecule {
    let mut lines = format!("{}\nchain\n", 3 * n);
    for i in 0..n {
        let x = 3.2 * i as f64;
        lines.push_str(&format!("O {:.4} 0.0 0.0\n", x));
        lines.push_str(&format!("H {:.4} 0.0 0.0\n", x + 0.9584));
        lines.push_str(&format!("H {:.4} 0.9278 0.0\n", x - 0.24));
    }
    Molecule::from_xyz_str(&lines, 0.0).unwrap()
}

/// Least-squares slope of `log(time)` against `log(n)`.
fn slope(sizes: &[f64], times: &[f64]) -> f64 {
    let points: Vec<(f64, f64)> = sizes
        .iter()
        .zip(times)
        .filter(|(_, t)| **t > 0.0)
        .map(|(n, t)| (n.ln(), t.ln()))
        .collect();
    if points.len() < 2 {
        return f64::NAN;
    }
    let count = points.len() as f64;
    let mean_x = points.iter().map(|(x, _)| x).sum::<f64>() / count;
    let mean_y = points.iter().map(|(_, y)| y).sum::<f64>() / count;
    let numerator: f64 = points
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let denominator: f64 = points.iter().map(|(x, _)| (x - mean_x).powi(2)).sum();
    numerator / denominator
}

fn main() {
    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options {
        max_scf: 400,
        ..Pm3Options::default()
    };
    let dc = DcOptions {
        core_radius: 4.0,
        buffer_radius: 9.0,
        ..DcOptions::default()
    };

    let counts: Vec<usize> = std::env::args()
        .nth(1)
        .map(|arg| {
            arg.split(',')
                .filter_map(|value| value.trim().parse().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![10, 20, 40, 80, 160]);

    // The same partitioning with the long-range Coulomb handed to the point-charge model, so
    // the near field comes from a neighbour list rather than a dense O(N^2) pair cache.
    let linear = DcOptions {
        long_range_cutoff: Some(22.0),
        ..dc
    };

    println!(
        "{:>6} {:>6} {:>10} {:>10} {:>10} {:>7} {:>7} {:>11} {:>10}",
        "waters",
        "atoms",
        "full (s)",
        "dc (s)",
        "linear (s)",
        "dc x",
        "lin x",
        "dE (eV)",
        "ueV/atom"
    );
    let mut sizes = Vec::new();
    let mut full_times = Vec::new();
    let mut dc_times = Vec::new();
    let mut linear_times = Vec::new();

    for n in &counts {
        let molecule = water_chain(*n);
        let atoms = molecule.atoms.len();

        let clock = Instant::now();
        let full = run_pm3(&molecule, &params, &options).ok();
        let full_time = clock.elapsed().as_secs_f64();

        let clock = Instant::now();
        let attempt = run_dc(&molecule, &params, &options, &dc);
        if let Err(e) = &attempt {
            eprintln!("  dc n={n}: {e}");
        }
        let partitioned = attempt.ok();
        let dc_time = clock.elapsed().as_secs_f64();

        let clock = Instant::now();
        let scaled = run_dc(&molecule, &params, &options, &linear);
        if let Err(e) = &scaled {
            eprintln!("  linear n={n}: {e}");
        }
        let scaled = scaled.ok();
        let linear_time = clock.elapsed().as_secs_f64();

        let difference = match (&full, &partitioned) {
            (Some(a), Some(b)) => {
                format!(
                    "{:.3e} ({} its)",
                    (a.total_ev - b.total_ev).abs(),
                    b.iterations
                )
            }
            (Some(_), None) => "DC FAILED".to_string(),
            _ => "-".to_string(),
        };
        let ratio = |t: f64| if t > 0.0 { full_time / t } else { f64::NAN };
        let switch_cost = match (&partitioned, &scaled) {
            (Some(a), Some(b)) => {
                format!(
                    "{:.1}",
                    1.0e6 * (a.total_ev - b.total_ev).abs() / atoms as f64
                )
            }
            _ => "-".to_string(),
        };
        println!(
            "{n:>6} {atoms:>6} {full_time:>10.3} {dc_time:>10.3} {linear_time:>10.3} \
             {:>7.2} {:>7.2} {difference:>11} {switch_cost:>10}",
            ratio(dc_time),
            ratio(linear_time)
        );

        sizes.push(atoms as f64);
        full_times.push(full_time);
        dc_times.push(dc_time);
        linear_times.push(linear_time);
    }

    println!();
    println!(
        "log-log slope, full diagonalization: {:.2}",
        slope(&sizes, &full_times)
    );
    println!(
        "log-log slope, divide and conquer:   {:.2}",
        slope(&sizes, &dc_times)
    );
    println!(
        "log-log slope, linear-scaling DC:    {:.2}",
        slope(&sizes, &linear_times)
    );

    // Per-component costs, which is what says *which* term is setting the slope. A whole-run
    // slope alone cannot distinguish "the diagonalization is cubic" from "the Fock build is
    // quadratic", and the two want completely different fixes.
    println!();
    println!(
        "{:>6} {:>12} {:>12} {:>12}",
        "atoms", "core (s)", "fock (s)", "eigen (s)"
    );
    let mut core_times = Vec::new();
    let mut fock_times = Vec::new();
    let mut eigen_times = Vec::new();
    for n in &counts {
        let molecule = water_chain(*n);
        let basis = pm3_rs::basis::Basis::build(&molecule, &params).unwrap();

        let clock = Instant::now();
        let core = pm3_rs::hamiltonian::build_core(&molecule, &basis, &params).unwrap();
        let core_time = clock.elapsed().as_secs_f64();

        let density = pm3_rs::pbc::gamma::initial_density(&molecule, &params, &basis).unwrap();
        let clock = Instant::now();
        let fock = pm3_rs::fock::build_fock(&molecule, &basis, &params, &core, &density).unwrap();
        let fock_time = clock.elapsed().as_secs_f64();

        let clock = Instant::now();
        let _ = pm3_rs::linalg::symmetric_eigen(&fock).unwrap();
        let eigen_time = clock.elapsed().as_secs_f64();

        println!(
            "{:>6} {core_time:>12.4} {fock_time:>12.4} {eigen_time:>12.4}",
            molecule.atoms.len()
        );
        core_times.push(core_time);
        fock_times.push(fock_time);
        eigen_times.push(eigen_time);
    }
    println!();
    println!(
        "component slopes: core {:.2}, fock {:.2}, eigen {:.2}",
        slope(&sizes, &core_times),
        slope(&sizes, &fock_times),
        slope(&sizes, &eigen_times)
    );

    // The tail is what matters: early points are dominated by fixed costs.
    if sizes.len() >= 3 {
        let tail = sizes.len() - 3;
        println!(
            "log-log slope over the last three points: full {:.2}, dc {:.2}, linear {:.2}",
            slope(&sizes[tail..], &full_times[tail..]),
            slope(&sizes[tail..], &dc_times[tail..]),
            slope(&sizes[tail..], &linear_times[tail..])
        );
    }
}
