// SPDX-License-Identifier: GPL-3.0-or-later

//! PM3 core-core repulsion, matching the PM3 branches of MOPAC `ccrep.F90`.

use crate::constants::{PM3_A0, PM3_EV};
use crate::dual::{Dual, Scalar};
use crate::error::Result;
use crate::math::Vec3;
use crate::params::{Pm3Element, Pm3Parameters};
use crate::system::Molecule;

/// PM3 core-core energy in eV.
pub fn core_core_energy(molecule: &Molecule, params: &Pm3Parameters) -> Result<f64> {
    let mut energy = 0.0;
    for i in 0..molecule.atoms.len() {
        for j in (i + 1)..molecule.atoms.len() {
            energy += pair_core_energy(
                molecule.atoms[i].z,
                molecule.atoms[j].z,
                molecule.atoms[i].position,
                molecule.atoms[j].position,
                params,
            )?;
        }
    }
    Ok(energy)
}

/// Analytic Cartesian gradient of the PM3 core-core energy (eV/Bohr).
pub fn core_core_gradient(molecule: &Molecule, params: &Pm3Parameters) -> Result<Vec<Vec3>> {
    let mut gradient = vec![Vec3::zero(); molecule.atoms.len()];
    for i in 0..molecule.atoms.len() {
        for j in (i + 1)..molecule.atoms.len() {
            let zi = molecule.atoms[i].z;
            let zj = molecule.atoms[j].z;
            let ei = params.element(zi)?;
            let ej = params.element(zj)?;
            let displacement = molecule.atoms[j].position - molecule.atoms[i].position;
            let distance = displacement.norm();
            let energy =
                pair_core_energy_scalar::<Dual>(params, ei, ej, zi, zj, Dual::var(distance, 0));
            let force_direction = displacement / distance;
            let contribution = force_direction * (-energy.d[0]);
            gradient[i] += contribution;
            gradient[j] -= contribution;
        }
    }
    Ok(gradient)
}

/// PM3 pair energy for a generic distance scalar in Bohr.
///
/// Pair-fitted interactions use `1 + 2 x exp(-alpha R)`. Other interactions
/// use the two per-element exponentials and PM3 Gaussian corrections. PM6's
/// sixth-power damping, special H-X branches, VdW term and repulsive wall are
/// intentionally absent.
pub fn pair_core_energy_scalar<S: Scalar>(
    params: &Pm3Parameters,
    ei: &Pm3Element,
    ej: &Pm3Element,
    zi: u8,
    zj: u8,
    distance_bohr: S,
) -> S {
    let distance_angstrom = distance_bohr * PM3_A0;
    // MOPAC `reppd` uses the dedicated core monopoles `po(9)` for G_AB.
    // This differs from the electron-electron `am` path for Cb and from
    // `po(1)` for elements carrying a core additive override.
    let rho = ei.po[9] + ej.po[9];
    let gab = (distance_bohr * distance_bohr + rho * rho).sqrt().recip() * PM3_EV;
    let core_product = ei.core_charge * ej.core_charge;
    let bare_energy = gab * core_product;

    // MOPAC reads xfac only when both atomic-number codes are below 101.
    let pair = if zi < 101 && zj < 101 && ei.n_orb != 0 && ej.n_orb != 0 {
        params
            .pair
            .get(&(zi.max(zj), zi.min(zj)))
            .copied()
            .filter(|entry| entry.x.abs() > 1.0e-5)
    } else {
        None
    };

    let mut energy = if let Some(entry) = pair {
        let alpha = if entry.alpha < 1.0e-6 {
            1.2
        } else {
            entry.alpha
        };
        let scale = S::cst(1.0) + (distance_angstrom * -alpha).exp() * (2.0 * entry.x);
        bare_energy * scale
    } else {
        let exp_i = (distance_angstrom * -ei.alpha).exp();
        let exp_j = (distance_angstrom * -ej.alpha).exp();
        let mut scale = exp_i + exp_j;
        // Legacy MNDO N-H/O-H adjustment retained by MOPAC's PM3 path.
        if zi as u16 + zj as u16 == 8 || zi as u16 + zj as u16 == 9 {
            if matches!(zi, 7 | 8) {
                scale = scale + (distance_angstrom - 1.0) * exp_i;
            }
            if matches!(zj, 7 | 8) {
                scale = scale + (distance_angstrom - 1.0) * exp_j;
            }
        }
        (scale * bare_energy).abs() + bare_energy
    };

    // For an unparameterized PM3 pair, MOPAC adds all per-element Gaussians.
    if pair.is_none() {
        for &(height, exponent, center) in ei.gauss.iter().chain(ej.gauss.iter()) {
            let delta = distance_angstrom - center;
            let ax = delta * delta * exponent;
            if ax.val() <= 25.0 {
                energy = energy + (-ax).exp() * (core_product * height) / distance_angstrom;
            }
        }
    }
    energy
}

/// PM3 core-core energy for one pair (eV).
pub fn pair_core_energy(
    zi: u8,
    zj: u8,
    position_i: Vec3,
    position_j: Vec3,
    params: &Pm3Parameters,
) -> Result<f64> {
    let ei = params.element(zi)?;
    let ej = params.element(zj)?;
    Ok(pair_core_energy_scalar::<f64>(
        params,
        ei,
        ej,
        zi,
        zj,
        (position_j - position_i).norm(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual2::Dual2;

    #[test]
    fn analytic_core_core_gradient_matches_finite_difference() {
        let molecule = Molecule::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 1.02 0.05 0.0\nH -0.28 0.96 0.10\n",
            0.0,
        )
        .unwrap();
        let params = Pm3Parameters::standard().unwrap();
        let analytic = core_core_gradient(&molecule, &params).unwrap();
        let step = 1.0e-5;
        let mut maximum_error = 0.0_f64;
        for (atom, analytic_atom) in analytic.iter().enumerate() {
            for coordinate in 0..3 {
                let mut plus = molecule.clone();
                let mut minus = molecule.clone();
                match coordinate {
                    0 => {
                        plus.atoms[atom].position.x += step;
                        minus.atoms[atom].position.x -= step;
                    }
                    1 => {
                        plus.atoms[atom].position.y += step;
                        minus.atoms[atom].position.y -= step;
                    }
                    _ => {
                        plus.atoms[atom].position.z += step;
                        minus.atoms[atom].position.z -= step;
                    }
                }
                let finite_difference = (core_core_energy(&plus, &params).unwrap()
                    - core_core_energy(&minus, &params).unwrap())
                    / (2.0 * step);
                maximum_error =
                    maximum_error.max((analytic_atom.get(coordinate) - finite_difference).abs());
            }
        }
        assert!(maximum_error < 1.0e-6, "maximum error {maximum_error:.3e}");
    }

    #[test]
    fn generic_scalar_second_derivative_matches_finite_difference() {
        let params = Pm3Parameters::standard().unwrap();
        let (zi, zj) = (8_u8, 1_u8);
        let (ei, ej) = (params.element(zi).unwrap(), params.element(zj).unwrap());
        let distance = 1.9;
        let dual =
            pair_core_energy_scalar::<Dual2>(&params, ei, ej, zi, zj, Dual2::var(distance, 0));
        let step = 1.0e-5;
        let center = pair_core_energy_scalar::<f64>(&params, ei, ej, zi, zj, distance);
        let plus = pair_core_energy_scalar::<f64>(&params, ei, ej, zi, zj, distance + step);
        let minus = pair_core_energy_scalar::<f64>(&params, ei, ej, zi, zj, distance - step);
        let finite_difference = (plus - 2.0 * center + minus) / (step * step);
        assert!((dual.h[0][0] - finite_difference).abs() < 1.0e-3);
    }
}
