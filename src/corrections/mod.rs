// SPDX-License-Identifier: GPL-3.0-or-later

//! Post-SCF classical corrections for the PM3 derived methods:
//! **D3** dispersion, **H4** hydrogen-bond, and **X** halogen-bond terms,
//! composable into PM3-D3, PM3-D3H4 and PM3-D3H4X.
//!
//! All three are classical functions of the geometry only (no SCF coupling), so they are added
//! to the total energy after the SCF converges. They are written **generic over
//! [`crate::dual::Scalar`]**, so the same source yields the energy (`f64`), the gradient
//! ([`crate::dual::Dual`]) and the exact **analytic** Hessian ([`crate::dual2::Dual2`]) — the D3
//! coordination-number coupling included, since the coordination numbers are computed in the same
//! generic arithmetic.

pub mod d3;
pub mod h4;
pub mod hx;

use crate::dual::Scalar;
use crate::system::Molecule;

/// Extract `(atomic numbers, Bohr positions)` from a molecule for the generic energy kernels.
pub(crate) fn geometry_f64(mol: &Molecule) -> (Vec<u8>, Vec<[f64; 3]>) {
    let numbers = mol.atoms.iter().map(|a| a.z).collect();
    let pos = mol
        .atoms
        .iter()
        .map(|a| [a.position.x, a.position.y, a.position.z])
        .collect();
    (numbers, pos)
}

/// Squared interatomic distance for a generic geometry.
#[inline]
pub(crate) fn dist2_g<S: Scalar>(a: &[S; 3], b: &[S; 3]) -> S {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    dx * dx + dy * dy + dz * dz
}
/// Interatomic distance for a generic geometry.
#[inline]
pub(crate) fn dist_g<S: Scalar>(a: &[S; 3], b: &[S; 3]) -> S {
    dist2_g(a, b).sqrt()
}

/// Total correction energy (eV), **generic over the scalar** — the single source used for the
/// energy (`f64`), gradient (`Dual`) and analytic Hessian (`Dual2`). `pos` is the Bohr geometry.
pub fn correction_energy_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]], variant: Variant) -> S {
    let mut e = S::cst(0.0);
    match variant {
        Variant::Pm3 => {}
        Variant::Pm3D3 => {
            e = e + d3::d3_energy_g(numbers, pos, &d3::D3Params::pm3_d3());
        }
        Variant::Pm3D3H4 | Variant::Pm3D3H4X => {
            e = e + d3::d3_energy_g(numbers, pos, &d3::D3Params::pm3_d3h4());
            e = e + h4::h4_energy_g(numbers, pos);
            e = e + h4::hh_rep_energy_g(numbers, pos);
        }
    }
    if variant.wants_x() {
        e = e + hx::hx_energy_g(numbers, pos);
    }
    e
}

/// PM3 method variant selecting which post-SCF corrections are applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Variant {
    /// Plain PM3 (no corrections).
    #[default]
    Pm3,
    /// PM3-D3: Grimme D3 dispersion.
    Pm3D3,
    /// PM3-D3H4: D3 dispersion + H4 hydrogen-bond correction.
    Pm3D3H4,
    /// PM3-D3H4X: D3 + H4 + halogen-bond correction.
    Pm3D3H4X,
}

impl Variant {
    /// Parse a method string ("PM3", "PM3-D3", "PM3-D3H4", "PM3-D3H4X").
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_uppercase().replace([' ', '_'], "-").as_str() {
            "PM3" => Some(Self::Pm3),
            "PM3-D3" => Some(Self::Pm3D3),
            "PM3-D3H4" => Some(Self::Pm3D3H4),
            "PM3-D3H4X" => Some(Self::Pm3D3H4X),
            _ => None,
        }
    }

    fn wants_x(self) -> bool {
        matches!(self, Self::Pm3D3H4X)
    }
}

/// Individual correction energies (eV).
#[derive(Clone, Copy, Debug, Default)]
pub struct CorrectionEnergies {
    pub d3_ev: f64,
    /// H4 hydrogen-bond term, used by the D3H4 family.
    pub h4_ev: f64,
    /// H–H repulsion (`energy_corr_hh_rep`), added for the D3H4 family.
    pub hh_ev: f64,
    pub hx_ev: f64,
}

impl CorrectionEnergies {
    pub fn total(&self) -> f64 {
        self.d3_ev + self.h4_ev + self.hh_ev + self.hx_ev
    }
}

/// Compute post-SCF correction energies (eV). PM3-D3 is the historical PM3 D3
/// zero-damping set. The D3H4 family uses its refitted D3 set (no C8), the H4
/// hydrogen-bond and H-H terms, and optionally X for D3H4X.
pub fn correction_energies(mol: &Molecule, variant: Variant) -> CorrectionEnergies {
    let mut c = CorrectionEnergies::default();
    match variant {
        Variant::Pm3 => {}
        Variant::Pm3D3 => {
            c.d3_ev = d3::d3_energy(mol, &d3::D3Params::pm3_d3());
        }
        Variant::Pm3D3H4 | Variant::Pm3D3H4X => {
            c.d3_ev = d3::d3_energy(mol, &d3::D3Params::pm3_d3h4());
            c.h4_ev = h4::h4_energy(mol);
            c.hh_ev = h4::hh_rep_energy(mol);
        }
    }
    if variant.wants_x() {
        c.hx_ev = hx::hx_energy(mol);
    }
    c
}

/// Total correction energy (eV) for the variant.
pub fn correction_energy(mol: &Molecule, variant: Variant) -> f64 {
    correction_energies(mol, variant).total()
}
