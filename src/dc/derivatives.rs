// SPDX-License-Identifier: GPL-3.0-or-later

//! Forces and stress from a divide-and-conquer density.
//!
//! # Why this is short
//!
//! ZDO removes the Pulay term, so the nuclear gradient is `Tr[P ∂H]` with no overlap-constraint
//! piece and no energy-weighted density matrix. What that buys here is that the gradient
//! expression never asks whether `P` came from a full diagonalization or from a partitioned one —
//! it only asks for a density. Both the molecular and the periodic gradients already take one as
//! an argument, so this module is a change of density and nothing else.
//!
//! # The approximation this inherits
//!
//! A divide-and-conquer density is not variational: it minimizes nothing, it is assembled. So the
//! usual argument that the first-order energy error vanishes at the SCF solution does not apply,
//! and the gradient carries the density's own truncation error rather than its square. In
//! practice that is the same error the energy carries and converges the same way with the buffer,
//! which the tests check directly rather than assume.

use crate::basis::Basis;
use crate::dc::partition::DcOptions;
use crate::dc::scf::{run_dc, run_dc_gamma, DcPeriodicResult, DcResult};
use crate::error::Result;
use crate::math::{Mat3, Vec3};
use crate::params::Pm3Parameters;
use crate::pbc::gamma::PeriodicOptions;
use crate::pbc::gradient::GammaDensity;
use crate::scf::Pm3Options;
use crate::system::Molecule;

/// Molecular forces from a divide-and-conquer density.
#[derive(Clone, Debug)]
pub struct DcGradient {
    pub scf: DcResult,
    pub energy_ev: f64,
    pub gradient: Vec<Vec3>,
    pub forces: Vec<Vec3>,
    pub max_gradient: f64,
}

/// Periodic forces and stress from a divide-and-conquer density.
#[derive(Clone, Debug)]
pub struct DcPeriodicGradient {
    pub scf: DcPeriodicResult,
    pub energy_ev: f64,
    pub gradient: Vec<Vec3>,
    pub forces: Vec<Vec3>,
    pub virial: Option<Mat3>,
    pub stress: Option<Mat3>,
    pub max_gradient: f64,
}

/// Converge a molecular divide-and-conquer SCF and take its gradient.
pub fn dc_gradient(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    dc: &DcOptions,
) -> Result<DcGradient> {
    let scf = run_dc(molecule, params, options, dc)?;
    let mut gradient = if scf.unrestricted {
        // Split the total density back into spin channels; the exchange term is the only part
        // that distinguishes them.
        let spin = scf
            .spin_density
            .as_ref()
            .expect("an unrestricted result carries a spin density");
        let mut alpha = scf.density.clone();
        let mut beta = scf.density.clone();
        for (index, difference) in spin.as_slice().iter().enumerate() {
            let total = scf.density.as_slice()[index];
            alpha.as_mut_slice()[index] = 0.5 * (total + difference);
            beta.as_mut_slice()[index] = 0.5 * (total - difference);
        }
        let mut out = crate::repulsion::core_core_gradient(molecule, params)?;
        let electronic = crate::gradient::electronic_gradient_fixed_density_spin(
            molecule,
            params,
            &Basis::build(molecule, params)?,
            &scf.density,
            &alpha,
            &beta,
        )?;
        for (slot, value) in out.iter_mut().zip(&electronic) {
            *slot += *value;
        }
        out
    } else {
        crate::gradient::fixed_density_gradient(molecule, params, &scf.density)?
    };
    crate::gradient::add_correction_gradient(molecule, options.variant, &mut gradient);
    // `−q_A f`, the force a net charge feels in the field. `fixed_density_gradient` above builds
    // its core Hamiltonian without a field, so nothing else here would supply it, and the energy
    // `run_dc` reported *does* include the field — a gradient missing this term is inconsistent
    // with the energy it accompanies rather than merely incomplete.
    if let Some(field) = options.field {
        for (slot, charge) in gradient.iter_mut().zip(&scf.charges) {
            *slot -= field * *charge;
        }
    }

    let forces: Vec<Vec3> = gradient.iter().map(|g| *g * -1.0).collect();
    let max_gradient = gradient
        .iter()
        .flat_map(|g| g.to_array())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    Ok(DcGradient {
        energy_ev: scf.total_ev,
        scf,
        gradient,
        forces,
        max_gradient,
    })
}

/// Converge a Γ-point periodic divide-and-conquer SCF and take its forces and stress.
pub fn dc_periodic_gradient(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    dc: &DcOptions,
) -> Result<DcPeriodicGradient> {
    let scf = run_dc_gamma(molecule, params, options, periodic, dc)?;
    let basis = Basis::build(molecule, params)?;
    let density = GammaDensity {
        density: &scf.density,
        spin: scf.spin_density.as_ref(),
        offsets: &basis.atom_offset,
    };
    let (gradient, virial, stress) = crate::pbc::gradient::forces_and_stress(
        molecule,
        params,
        options,
        periodic,
        &density,
        &scf.density,
    )?;
    let forces: Vec<Vec3> = gradient.iter().map(|g| *g * -1.0).collect();
    let max_gradient = gradient
        .iter()
        .flat_map(|g| g.to_array())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    Ok(DcPeriodicGradient {
        energy_ev: scf.total_ev,
        scf,
        gradient,
        forces,
        virial,
        stress,
        max_gradient,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;

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

    fn options() -> Pm3Options {
        Pm3Options {
            max_scf: 400,
            ..Pm3Options::default()
        }
    }

    fn full_coverage() -> DcOptions {
        DcOptions {
            core_radius: 3.0,
            buffer_radius: 500.0,
            smearing_ev: 1.0e-4,
            ..DcOptions::default()
        }
    }

    /// At full coverage the divide-and-conquer density *is* the full density, so its gradient has
    /// to be the full gradient — exactly, not approximately. This separates the gradient
    /// machinery from the truncation: anything that fails here is a wiring error.
    #[test]
    fn a_reaching_buffer_reproduces_the_full_gradient() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = water_chain(3);
        let reference = crate::gradient::closed_form_gradient(&molecule, &params, &options())
            .unwrap()
            .gradient;
        let result = dc_gradient(&molecule, &params, &options(), &full_coverage()).unwrap();
        for (atom, (a, b)) in result.gradient.iter().zip(&reference).enumerate() {
            for axis in 0..3 {
                let (x, y) = (a.to_array()[axis], b.to_array()[axis]);
                assert!(
                    (x - y).abs() < 1.0e-6,
                    "atom {atom} axis {axis}: DC {x} vs full {y}"
                );
            }
        }
    }

    /// Truncating the buffer must converge the forces, atom by atom, the way it converges the
    /// energy.
    #[test]
    fn the_forces_converge_with_the_buffer() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = water_chain(5);
        let reference = crate::gradient::closed_form_gradient(&molecule, &params, &options())
            .unwrap()
            .gradient;
        let worst_at = |buffer: f64| -> f64 {
            let result = dc_gradient(
                &molecule,
                &params,
                &options(),
                &DcOptions {
                    core_radius: 3.0,
                    buffer_radius: buffer,
                    ..DcOptions::default()
                },
            )
            .unwrap();
            result
                .gradient
                .iter()
                .zip(&reference)
                .flat_map(|(a, b)| (0..3).map(move |k| (a.to_array()[k] - b.to_array()[k]).abs()))
                .fold(0.0_f64, f64::max)
        };
        let narrow = worst_at(4.0);
        let wide = worst_at(14.0);
        assert!(wide < narrow, "widening did not help: {wide} vs {narrow}");
        assert!(
            wide < 1.0e-4,
            "the widest buffer still misses the forces by {wide:.3e} eV/Bohr"
        );
    }

    /// The periodic path, forces and stress alike.
    #[test]
    fn the_periodic_gradient_reproduces_the_full_periodic_one() {
        let params = Pm3Parameters::standard().unwrap();
        let mut molecule = water_chain(3);
        molecule.cell = Some(Cell::cubic(30.0).unwrap());
        let periodic = PeriodicOptions::default();
        let reference =
            crate::pbc::gradient::periodic_gradient(&molecule, &params, &options(), &periodic)
                .unwrap();
        let result =
            dc_periodic_gradient(&molecule, &params, &options(), &periodic, &full_coverage())
                .unwrap();

        for (atom, (a, b)) in result.gradient.iter().zip(&reference.gradient).enumerate() {
            for axis in 0..3 {
                let (x, y) = (a.to_array()[axis], b.to_array()[axis]);
                assert!(
                    (x - y).abs() < 1.0e-6,
                    "atom {atom} axis {axis}: DC {x} vs full {y}"
                );
            }
        }
        let (sa, sb) = (
            result.stress.expect("3D reports a stress"),
            reference.stress.expect("3D reports a stress"),
        );
        for beta in 0..3 {
            for alpha in 0..3 {
                let (x, y) = (
                    sa.col[beta].to_array()[alpha],
                    sb.col[beta].to_array()[alpha],
                );
                assert!(
                    (x - y).abs() < 1.0e-9,
                    "stress ({alpha},{beta}): DC {x} vs full {y}"
                );
            }
        }
    }

    /// Forces from a divide-and-conquer density must still sum to zero: translating everything
    /// changes no interatomic distance, whatever the partitioning did.
    #[test]
    fn the_forces_sum_to_zero() {
        let params = Pm3Parameters::standard().unwrap();
        let result = dc_gradient(
            &water_chain(4),
            &params,
            &options(),
            &DcOptions {
                core_radius: 3.0,
                buffer_radius: 8.0,
                ..DcOptions::default()
            },
        )
        .unwrap();
        let mut total = Vec3::zero();
        for force in &result.forces {
            total += *force;
        }
        let largest = total.to_array().iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(largest < 1.0e-8, "net force {largest:.3e}");
    }
}
