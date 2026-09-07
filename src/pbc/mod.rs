// SPDX-License-Identifier: GPL-3.0-or-later

//! Periodic boundary conditions for the NDDO model: 1D chains, 2D slabs and 3D crystals,
//! at the Γ point or over a k-point mesh.
//!
//! # How the electrostatics are split
//!
//! Every two-center Coulomb term in PM3 — electron–electron, electron–core and core–core — is
//! built from the Dewar–Thiel multipole model, which places a handful of point charges at
//! fixed offsets from each nucleus and interacts them through the Klopman–Ohno screened form
//! `q_i q_j / √(r² + a)` rather than `q_i q_j / r`. Under a lattice sum this matters twice
//! over: the `1/r` part is only conditionally convergent, and the screening leaves a spurious
//! `−a/(2r³)` tail whose 3D lattice sum diverges logarithmically.
//!
//! Both are handled by one split:
//!
//! ```text
//! E = Ewald[ Σ_conf q_i q_j / r_ij ]  +  Σ_{|r| < r_off} [ W_KO(r) − W_point(r) ] · f(r)
//! ```
//!
//! * The **Ewald** part sees only the point limit of the very same multipole configurations,
//!   so the split is exact by construction rather than by approximation, and it inherits the
//!   absolute convergence and well-defined boundary conventions of a point-charge Ewald sum.
//! * The **short-range correction** carries the Klopman–Ohno screening, smoothly switched off
//!   between `r_on` and `r_off` by a C² quintic. That switch removes the `−a/(2r³)` artefact
//!   rather than summing it: real spherical charge distributions interact exactly as `q_i q_j / r`
//!   once they stop overlapping, so the tail is an artefact of the interpolation formula, not
//!   physics. It is a documented approximation, and its `r_off` convergence is a test.
//!
//! Everything else — resonance `β·S`, exchange, the exponential and Gaussian core-core terms,
//! and the classical D3/H4/X corrections — decays exponentially or as `1/r⁶` and needs only a
//! real-space cutoff.
//!
//! # Dimensionality
//!
//! Only the reciprocal-space half of the Ewald sum depends on how many directions are
//! periodic; see [`ewald`]. Everything else consumes [`crate::neighbor::NeighborList`], which
//! is dimension-agnostic.

pub mod berry;
pub mod born;
pub mod dfpt;
pub mod dielectric;
pub mod ewald;
pub mod ewald_hessian;
pub mod finite_field;
pub mod gamma;
pub mod gradient;
pub mod hessian;
pub mod kernel;
pub mod kpoints;
pub mod kscf;
pub mod lo_to;
pub mod multipole;
pub mod optimize;
pub mod phased;
pub mod phonon;
pub mod screen;

#[cfg(test)]
pub mod ewald_reference;

/// Refuse a uniform external electric field under periodic boundary conditions.
///
/// `−f·r` is unbounded and not lattice-periodic, so there is no periodic Hamiltonian to add it
/// to: the potential drops without limit across every cell, and the "energy per cell" a
/// calculation would report depends on which cell was chosen. The physical treatment is a Berry
/// phase, which is a different calculation rather than a bigger one.
///
/// An isolated cell would be a legitimate place for a field — `−f·r` is well-defined with no
/// lattice to be non-periodic against — but this machinery does not carry one: nothing under
/// `pbc` reads `Pm3Options::field`, so allowing it there would drop the field silently rather
/// than honour it. It is refused too, and the message says which of the two reasons applies.
/// `run_dc_screened` reaches this path with `Cell::isolated`, so this is also what keeps a
/// screened divide-and-conquer run from ignoring a field it was handed.
pub(crate) fn refuse_field(
    molecule: &crate::system::Molecule,
    options: &crate::scf::Pm3Options,
) -> crate::error::Result<()> {
    if options.field.is_none() {
        return Ok(());
    }
    let periodic = molecule.cell.map(|c| c.n_periodic()).unwrap_or(0);
    if periodic > 0 {
        return Err(crate::error::Pm3Error::InvalidInput(
            "a uniform electric field is not compatible with periodic boundary conditions: \
             `−f·r` is not lattice-periodic, so the energy per cell would depend on which cell \
             was chosen. Use an isolated cell, or a Berry-phase treatment, which is not \
             implemented."
                .to_string(),
        ));
    }
    Err(crate::error::Pm3Error::InvalidInput(
        "the periodic machinery does not carry a uniform electric field, not even for an \
         isolated cell where one would be well-defined. Run the molecular path, which does."
            .to_string(),
    ))
}
