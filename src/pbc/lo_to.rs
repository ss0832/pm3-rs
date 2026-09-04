// SPDX-License-Identifier: GPL-3.0-or-later

//! The non-analytic term: LO–TO splitting at `q → 0`.
//!
//! ```text
//! D_NA(q̂)_{aα,bβ} = (4π/Ω) · (q̂·Z*_a)_α (q̂·Z*_b)_β / (q̂·ε∞·q̂)
//! ```
//!
//! # What it is for
//!
//! In a polar crystal the `q → 0` limit of the dynamical matrix is **direction dependent**: a
//! longitudinal optical mode carries a macroscopic electric field that a transverse one does not,
//! and the two are split by an amount that is not small. `D(0)` — evaluated *at* `q = 0`, where
//! the background/sheet convention removes the divergent Coulomb term — is the transverse limit.
//! The term above restores the longitudinal one, given a direction to approach along.
//!
//! # Where to add it, and where not to
//!
//! Add it to a dynamical matrix that **cannot** carry the long-range field itself:
//!
//! * a `D(0)` from [`crate::pbc::hessian::periodic_hessian`], the Γ-point Hessian;
//! * a `D(q)` interpolated from truncated real-space force constants (a supercell), where the
//!   truncation structurally removes the `1/r` tail.
//!
//! Do **not** add it to [`crate::pbc::dfpt::dynamical_matrix`] at finite `q`. That one sums the
//! phased Ewald kernel over every `G + q`, so it already carries the long-range channel — and
//! its `q → 0` limit is what this term is *checked against*, in `tests/pbc_lo_to.rs`.
//!
//! That check is the reason the prefactor here can be stated rather than hoped for. A
//! Fourier-interpolating implementation has no independent handle on the coefficient: the only
//! thing to compare against is the same closed form written twice. This crate computes the full
//! `D(q)` at any wavevector, so the coefficient is measured against a calculation that shares
//! none of its algebra.
//!
//! # Units
//!
//! `Z*` is in elementary charges and `Ω` in Bohr³, so `4πZ*²/(Ωε)` is `e²/Bohr³` — which is
//! `Hartree/Bohr²`, not `eV/Bohr²`. The force constants this is added to are in eV/Bohr², so the
//! conversion is not optional. `tests/pbc_lo_to.rs` measures the coefficient against the finite-`q`
//! response rather than re-deriving it, which is what makes a missing factor here visible.

use crate::constants::HARTREE_TO_EV;
use crate::error::{Pm3Error, Result};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::system::Molecule;

/// The non-analytic contribution to the force constants, in eV/Bohr², **not** mass weighted.
///
/// `direction` is the Cartesian direction `q` is approached from; it is normalized here, so its
/// length does not matter, but it must not be zero — the limit does not exist without one.
///
/// Three-dimensional only: the `4π/Ω` prefactor is the 3D Coulomb kernel's. In 2D the
/// `q²·v(q)` product goes to zero linearly and there is no splitting at `Γ`, only a kink; in 1D
/// it vanishes as `q² ln(1/q)`. Applying this form to a slab or a chain would manufacture a
/// splitting that the physics does not have.
pub fn non_analytic_term(
    molecule: &Molecule,
    direction: Vec3,
    born: &[[[f64; 3]; 3]],
    epsilon: [[f64; 3]; 3],
) -> Result<Matrix> {
    let cell = molecule
        .cell
        .ok_or_else(|| Pm3Error::InvalidInput("the non-analytic term needs a cell".to_string()))?;
    if cell.n_periodic() != 3 {
        return Err(Pm3Error::InvalidInput(
            "LO-TO splitting is a three-dimensional effect: the `4π/Ω` prefactor is the 3D \
             Coulomb kernel's, and in 2D the macroscopic field vanishes linearly in `q` (a kink \
             at Γ, not a splitting) while in 1D it vanishes as `q² ln(1/q)`. Applying it to a \
             slab or a chain would manufacture a splitting the model does not have."
                .to_string(),
        ));
    }
    let nat = molecule.atoms.len();
    if born.len() != nat {
        return Err(Pm3Error::InvalidInput(format!(
            "expected {nat} Born-charge tensors, got {}",
            born.len()
        )));
    }
    let norm = direction.norm();
    if norm < 1.0e-12 {
        return Err(Pm3Error::InvalidInput(
            "the non-analytic term is the limit along a direction, and the zero vector is not \
             one: `q → 0` in a polar crystal depends on which way it is approached from"
                .to_string(),
        ));
    }
    let qhat = direction / norm;
    let q = qhat.to_array();

    // `q̂ · ε∞ · q̂`, the screening the macroscopic field sees along this direction.
    let mut denominator = 0.0;
    for a in 0..3 {
        for b in 0..3 {
            denominator += q[a] * epsilon[a][b] * q[b];
        }
    }
    if denominator.abs() < 1.0e-12 {
        return Err(Pm3Error::InvalidInput(
            "the dielectric tensor screens this direction to nothing, so the non-analytic term \
             is singular. That is a statement about the tensor rather than about the direction."
                .to_string(),
        ));
    }

    // `(q̂ · Z*_a)_α`, contracting over the **first** index of the tensor — the polarization
    // direction, which is the one the macroscopic field couples to.
    let projected: Vec<[f64; 3]> = born
        .iter()
        .map(|z| {
            let mut row = [0.0; 3];
            for (alpha, slot) in row.iter_mut().enumerate() {
                *slot = (0..3).map(|gamma| q[gamma] * z[gamma][alpha]).sum();
            }
            row
        })
        .collect();

    let prefactor = 4.0 * std::f64::consts::PI / cell.measure() / denominator * HARTREE_TO_EV;
    let ndof = 3 * nat;
    let mut out = Matrix::zeros(ndof, ndof);
    for a in 0..nat {
        for b in 0..nat {
            for alpha in 0..3 {
                for beta in 0..3 {
                    out[(3 * a + alpha, 3 * b + beta)] =
                        prefactor * projected[a][alpha] * projected[b][beta];
                }
            }
        }
    }
    Ok(out)
}

/// Add the non-analytic term to a dynamical matrix in place.
///
/// The term is real and symmetric, so it lands entirely on the real part. See the module note for
/// which matrices this belongs on — adding it to a finite-`q` [`crate::pbc::dfpt::dynamical_matrix`]
/// would count the long-range channel twice.
pub fn add_non_analytic(
    matrix: &mut crate::pbc::dfpt::DynamicalMatrix,
    molecule: &Molecule,
    direction: Vec3,
    born: &[[[f64; 3]; 3]],
    epsilon: [[f64; 3]; 3],
) -> Result<()> {
    let term = non_analytic_term(molecule, direction, born, epsilon)?;
    let ndof = matrix.matrix.rows;
    if term.rows != ndof {
        return Err(Pm3Error::InvalidInput(format!(
            "the non-analytic term is {}×{} and the matrix is {ndof}×{ndof}",
            term.rows, term.cols
        )));
    }
    for i in 0..ndof {
        for j in 0..ndof {
            matrix.matrix[(i, j)] += faer::c64::new(term[(i, j)], 0.0);
        }
    }
    Ok(())
}
