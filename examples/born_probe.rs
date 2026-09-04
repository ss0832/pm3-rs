// SPDX-License-Identifier: GPL-3.0-or-later

//! Print Born effective charges beside the static charges they would be if the electrons did not
//! respond, so the response's own size is visible.
//!
//! ```text
//! cargo run --release --example born_probe
//! ```
//!
//! The point of the comparison: `Z*_{a,αβ} = Q_a δ_αβ + (response)`. An implementation with a
//! broken or absent response still satisfies the acoustic sum rule, because the net charges of a
//! neutral cell already sum to zero. Only seeing how far `Z*` sits from `Q δ` says whether the
//! coupled-perturbed part arrived.

use pm3_rs::pbc::born::{born_charge_sum_rule_residual, born_charges};
use pm3_rs::pbc::gamma::{run_gamma, PeriodicOptions};
use pm3_rs::{Cell, Molecule, Pm3Options, Pm3Parameters, Vec3};

fn main() {
    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options::default();
    let periodic = PeriodicOptions::default();

    let cases: Vec<(&str, Molecule)> = vec![
        ("water in a 14 Bohr cube", {
            let mut m = Molecule::from_xyz_str(
                "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n",
                0.0,
            )
            .unwrap();
            m.cell = Some(Cell::cubic(14.0).unwrap());
            m
        }),
        ("water chain, 6 Bohr repeat", {
            let mut m = Molecule::from_xyz_str(
                "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n",
                0.0,
            )
            .unwrap();
            m.cell = Some(
                Cell::new(
                    Vec3::new(6.0, 0.0, 0.0),
                    Vec3::new(0.0, 40.0, 0.0),
                    Vec3::new(0.0, 0.0, 40.0),
                    [true, false, false],
                )
                .unwrap(),
            );
            m
        }),
        ("H2 chain, 5 Bohr repeat", {
            let mut m =
                Molecule::from_xyz_str("2\nH2\nH 0.0 0.0 0.0\nH 0.7414 0.0 0.0\n", 0.0).unwrap();
            m.cell = Some(
                Cell::new(
                    Vec3::new(5.0, 0.0, 0.0),
                    Vec3::new(0.0, 40.0, 0.0),
                    Vec3::new(0.0, 0.0, 40.0),
                    [true, false, false],
                )
                .unwrap(),
            );
            m
        }),
    ];

    for (label, molecule) in cases {
        println!("=== {label} ===");
        let scf = run_gamma(&molecule, &params, &options, &periodic).unwrap();
        let born = born_charges(&molecule, &params, &options, &periodic).unwrap();

        println!(
            "{:>5}  {:>9}   {:>9} {:>9} {:>9}   {:>10}",
            "atom", "Mulliken", "Z*xx", "Z*yy", "Z*zz", "|Z*-Qd|"
        );
        for (index, atom) in molecule.atoms.iter().enumerate() {
            let q = scf.charges[index];
            let z = born[index];
            // How far the diagonal sits from the static charge: the response's own contribution.
            let departure = (0..3).map(|a| (z[a][a] - q).abs()).fold(0.0_f64, f64::max);
            println!(
                "{:>5}  {:>9.5}   {:>9.5} {:>9.5} {:>9.5}   {:>10.5}",
                pm3_rs::z_to_symbol(atom.z).unwrap_or("X"),
                q,
                z[0][0],
                z[1][1],
                z[2][2],
                departure
            );
        }
        println!(
            "sum-rule residual: {:.3e}",
            born_charge_sum_rule_residual(&born)
        );

        match pm3_rs::pbc::dielectric::polarizability(&molecule, &params, &options, &periodic) {
            Ok(alpha) => {
                println!(
                    "polarizability (Bohr^3): xx {:.4}  yy {:.4}  zz {:.4}",
                    alpha[0][0], alpha[1][1], alpha[2][2]
                );
                if let Ok(t) = pm3_rs::pbc::dielectric::dielectric_tensor(
                    &molecule, &params, &options, &periodic,
                ) {
                    println!(
                        "epsilon_infinity:        xx {:.4}  yy {:.4}  zz {:.4}",
                        t.epsilon[0][0], t.epsilon[1][1], t.epsilon[2][2]
                    );
                }
            }
            Err(e) => println!("polarizability: {e}"),
        }
        println!();
    }

    // The comparison the units rest on: a molecule in a big box against its own finite field.
    println!("=== water: periodic response vs isolated finite field ===");
    let mut big = Molecule::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n",
        0.0,
    )
    .unwrap();
    big.cell = Some(Cell::cubic(20.0).unwrap());
    let periodic_alpha =
        pm3_rs::pbc::dielectric::polarizability(&big, &params, &options, &periodic).unwrap();

    let bare = Molecule::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n",
        0.0,
    )
    .unwrap();
    let step = 1.0e-4;
    println!(
        "{:>5} {:>14} {:>14} {:>10}",
        "axis", "periodic", "finite field", "ratio"
    );
    for beta in 0..3 {
        let dipole_at = |strength: f64| -> Vec3 {
            let mut f = [0.0; 3];
            f[beta] = strength;
            let opts = Pm3Options {
                field: Some(Vec3::new(f[0], f[1], f[2])),
                ..Pm3Options::default()
            };
            pm3_rs::run_pm3(&bare, &params, &opts).unwrap().dipole_debye / 2.541_746_473
        };
        let d = (dipole_at(step).to_array()[beta] - dipole_at(-step).to_array()[beta])
            / (2.0 * step)
            * 27.211_386_245_988;
        println!(
            "{:>5} {:>14.5} {:>14.5} {:>10.4}",
            ["x", "y", "z"][beta],
            periodic_alpha[beta][beta],
            d,
            periodic_alpha[beta][beta] / d
        );
    }
}
