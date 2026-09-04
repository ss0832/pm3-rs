// SPDX-License-Identifier: GPL-3.0-or-later

//! What the correction cutoff costs in accuracy, and what it buys in scaling.
//!
//! ```text
//! cargo run --release --example correction_cutoff
//! ```
//!
//! `correction_energy_g` bounds its sums at `DEFAULT_D3_CUTOFF` / `DEFAULT_CN_CUTOFF` rather than
//! running them over every pair. That turns an `O(N²)` evaluation into an `O(N)` one, and every
//! derivative built on it drops an order with it -- but it also changes the answer for anything
//! wider than the radius. This prints both sides of that trade at a range of sizes, so the choice
//! rests on measurement.

use pm3_rs::corrections::{correction_energy_cluster_g, periodic, Variant};

/// A cubic-ish block of water molecules, `n` of them along each axis.
fn water_block(n: usize) -> (Vec<u8>, Vec<[f64; 3]>) {
    let mut numbers = Vec::new();
    let mut positions = Vec::new();
    let spacing = 5.7; // Bohr, roughly liquid density
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let origin = [i as f64 * spacing, j as f64 * spacing, k as f64 * spacing];
                numbers.push(8);
                positions.push(origin);
                numbers.push(1);
                positions.push([origin[0] + 1.81, origin[1], origin[2]]);
                numbers.push(1);
                positions.push([origin[0] - 0.45, origin[1] + 1.75, origin[2]]);
            }
        }
    }
    (numbers, positions)
}

fn main() {
    let variant = Variant::Pm3D3H4;
    println!(
        "{:>7} {:>8} {:>20} {:>20} {:>14}",
        "blocks", "atoms", "uncut (eV)", "cut (eV)", "difference"
    );
    for n in [2, 3, 4, 5] {
        let (numbers, positions) = water_block(n);
        let nat = numbers.len();

        let uncut = correction_energy_cluster_g::<f64>(
            &numbers, &positions, nat, None, None, None, None, variant,
        );
        let cut = correction_energy_cluster_g::<f64>(
            &numbers,
            &positions,
            nat,
            None,
            None,
            Some(periodic::DEFAULT_D3_CUTOFF),
            Some(periodic::DEFAULT_CN_CUTOFF),
            variant,
        );
        println!(
            "{n:>7} {nat:>8} {uncut:>20.9} {cut:>20.9} {:>14.3e}",
            (uncut - cut).abs()
        );
    }
}
