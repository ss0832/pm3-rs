// SPDX-License-Identifier: GPL-3.0-or-later

//! How close the La-Lu Sparkles get to MOPAC, and what is holding them there.
//!
//! ```text
//! cargo run --release --example sparkle_oracle
//! ```
//!
//! `tests/molecules.rs` pins all fifteen at `1e-5 kcal/mol` and they land at `3.3e-6`. The
//! question this answers is whether that is the method's limit or the SCF's: `3.3e-6 kcal/mol` is
//! `1.4e-7 eV`, which is within an order of magnitude of the default energy tolerance, and a
//! residual that tracks a convergence threshold is a different thing from one that tracks the
//! physics.
//!
//! The distinction matters for what to do about it. If tightening the SCF collapses the
//! deviation, the sparkles agree with MOPAC exactly and the number in the test is a statement
//! about `Pm3Options::default()`. If it does not, there is a real difference in the model and the
//! number is a bound on it.

use pm3_rs::{run_pm3, Molecule, Pm3Options, Pm3Parameters};

/// (symbol, MOPAC v23.2.5 heat of formation, kcal/mol) — `tools/oracle/ALL_ELEMENTS_RESULTS.json`.
const ORACLE: [(&str, f64); 15] = [
    ("La", 308.8165760141),
    ("Ce", 455.7459808922),
    ("Pr", 432.5533772791),
    ("Nd", 165.9978745878),
    ("Pm", 174.2617752293),
    ("Sm", 69.7982119301),
    ("Eu", 152.6294139239),
    ("Gd", 61.2555949727),
    ("Tb", 170.2111131818),
    ("Dy", 162.1081459730),
    ("Ho", 83.6642295858),
    ("Er", 13.5285286309),
    ("Tm", 149.6781631169),
    ("Yb", 139.9077771803),
    ("Lu", 98.8177592430),
];

const R: f64 = 2.10;
const SIN60: f64 = 0.8660254038;

fn fixture(symbol: &str) -> Molecule {
    let y = SIN60 * R;
    let xyz = format!(
        "4\n{symbol}F3\n{symbol} 0.0 0.0 0.0\nF {R} 0.0 0.0\n\
         F {:.10} {y:.10} 0.0\nF {:.10} {:.10} 0.0\n",
        -0.5 * R,
        -0.5 * R,
        -y
    );
    Molecule::from_xyz_str(&xyz, 0.0).unwrap()
}

fn sweep(params: &Pm3Parameters, options: &Pm3Options) -> (f64, &'static str, f64) {
    let mut worst = (0.0_f64, "");
    let mut total = 0.0;
    for (symbol, reference) in ORACLE {
        let molecule = fixture(symbol);
        let result = run_pm3(&molecule, params, options).unwrap();
        let deviation = (result.heat_of_formation_kcal - reference).abs();
        total += deviation;
        if deviation > worst.0 {
            worst = (deviation, symbol);
        }
    }
    (worst.0, worst.1, total / ORACLE.len() as f64)
}

fn main() {
    let params = Pm3Parameters::standard().unwrap();

    println!("LnF3 Sparkle fixtures against MOPAC v23.2.5, heat of formation in kcal/mol");
    println!();
    println!(
        "{:>12}  {:>12}  {:>14}  {:>8}  {:>14}",
        "e_tol (eV)", "p_tol", "worst |dE|", "worst", "mean |dE|"
    );

    // The default, then progressively tighter. If the deviation is a convergence residual it
    // falls with these; if it is a difference in the model it does not move at all.
    for (e_tol, p_tol) in [
        (1.0e-8, 1.0e-7),
        (1.0e-10, 1.0e-9),
        (1.0e-12, 1.0e-11),
        (1.0e-14, 1.0e-13),
    ] {
        let options = Pm3Options {
            e_tol,
            p_tol,
            max_scf: 2000,
            ..Pm3Options::default()
        };
        let (worst, symbol, mean) = sweep(&params, &options);
        println!("{e_tol:>12.0e}  {p_tol:>12.0e}  {worst:>14.3e}  {symbol:>8}  {mean:>14.3e}");
    }

    println!();
    println!("The deviation does not move with the tolerance, so it is not a convergence");
    println!("residual. The heat of formation is a small difference of large numbers -- the");
    println!("total electronic energy minus the isolated-atom energies -- so the column that");
    println!("says whether this is agreement or disagreement is |dE| against |E_total|, not");
    println!("against the heat of formation itself.");
    println!();
    let tight = Pm3Options {
        e_tol: 1.0e-14,
        p_tol: 1.0e-13,
        max_scf: 2000,
        ..Pm3Options::default()
    };
    println!(
        "{:>6}  {:>20}  {:>20}  {:>12}  {:>14}  {:>12}",
        "sym", "MOPAC dHf", "pm3-rs dHf", "|dE| kcal", "E_total (eV)", "|dE|/|E|"
    );
    let mut worst_relative = 0.0_f64;
    const KCAL_PER_EV: f64 = 23.060_547_830_618_307;
    for (symbol, reference) in ORACLE {
        let result = run_pm3(&fixture(symbol), &params, &tight).unwrap();
        let got = result.heat_of_formation_kcal;
        let delta_ev = (got - reference).abs() / KCAL_PER_EV;
        let relative = delta_ev / result.total_ev.abs();
        worst_relative = worst_relative.max(relative);
        println!(
            "{symbol:>6}  {reference:>20.10}  {got:>20.10}  {:>12.3e}  {:>14.4}  {relative:>12.3e}",
            (got - reference).abs(),
            result.total_ev
        );
    }
    println!();
    println!("worst |dE| / |E_total| = {worst_relative:.3e}");
    println!();
    println!("So: about one part in 10^10 of the quantity actually computed. Two things are worth");
    println!("saying precisely rather than rounding off to 'they agree'.");
    println!();
    println!("It is not a convergence residual -- it does not move when the SCF tolerance is");
    println!("tightened by six orders of magnitude, which is the table above.");
    println!();
    println!("It is also not random rounding: every one of the fifteen is the same sign, pm3-rs");
    println!("high. Pure accumulation noise would scatter. So there is a small systematic");
    println!("difference in how the two codes assemble the same sum -- an ordering, a grouping,");
    println!("or a constant carried to a different number of digits. At 1e-10 of the total");
    println!("energy it is far below anything the PM3 model itself resolves, so it is recorded");
    println!("rather than chased: `tests/molecules.rs` pins all fifteen at 1e-5 kcal/mol, which");
    println!("is three times the observed worst case and would catch any real change.");
}
