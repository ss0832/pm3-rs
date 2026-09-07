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
pub mod mmok;
pub mod periodic;

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
///
/// # Why there is no cutoff here, though the periodic path has one
///
/// Passing [`periodic::DEFAULT_D3_CUTOFF`] and [`periodic::DEFAULT_CN_CUTOFF`] here was tried and
/// reverted, because it was measured to cost accuracy and buy nothing.
///
/// The cost, from `examples/correction_cutoff.rs` on blocks of water at liquid density: `1.4e-6`
/// eV at 81 atoms, `2.8e-5` at 192, `2.5e-4` at 375. Extensive, since it is the dispersion of
/// the pairs beyond the radius and a bigger system has more of them.
///
/// The gain was nil. `examples/correction_scaling.rs` puts the log-log slope at **1.959 with the
/// cutoff applied** — still quadratic. Two reasons, and both have to be fixed before a radius is
/// worth anything:
///
/// 1. The sums in [`d3`] and [`h4`] are `for i { for j { if r > cutoff { continue } } }`. That is
///    an `O(N²)` traversal with a cheap body, not an `O(N)` one. Bounding the *range* does not
///    bound the *work* until the inner loop comes from a neighbour list — `crate::neighbor` has
///    the cell grid for it.
/// 2. Even then, 30 Bohr at liquid water density encloses roughly 2000 atoms, so the D3 sphere is
///    larger than every system in the measurement above. Below a couple of thousand atoms a D3
///    cutoff excludes nothing and can only add error. The 15 Bohr coordination radius and H4's
///    5.5 Å radius enclose far less and are where a grid would first pay.
///
/// So the order reduction for the correction Hessian — `O(N⁴)`, this crate's worst — is a real
/// piece of work in the loop structure rather than a parameter, and it is **not done**. The two
/// examples above are the measurements a future attempt should start from.
pub fn correction_energy_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]], variant: Variant) -> S {
    correction_energy_cluster_g(numbers, pos, numbers.len(), None, None, None, None, variant)
}

/// [`correction_energy_g`] plus MOPAC's `MMOK` amide term when `mmok` is set.
///
/// Kept separate from `Variant` because it is orthogonal to it: `MMOK` is not a dispersion
/// or hydrogen-bond correction and composes with any of them, and it is not part of PM3 at
/// all — see [`mmok`]. Going through the same generic scalar means the gradient and Hessian
/// pick it up from the same expression.
pub fn correction_energy_with_mmok_g<S: Scalar>(
    numbers: &[u8],
    pos: &[[S; 3]],
    variant: Variant,
    mmok: bool,
) -> S {
    let base = correction_energy_g(numbers, pos, variant);
    if mmok {
        base + mmok::amide_correction_g(numbers, pos)
    } else {
        base
    }
}

/// Total correction energy **per unit cell** (eV) over an image-expanded cluster.
///
/// The corrections are classical and finite-ranged, so the whole of periodicity for them is a
/// question of counting: every distinct tuple in the crystal must contribute exactly once per
/// cell. Each term uses the assignment that makes that automatic rather than a fractional
/// weight, which is both cheaper and harder to get wrong:
///
/// | term | tuple | assigned to the cell containing |
/// |---|---|---|
/// | D3, H–H repulsion | pair | *(symmetric — summed cell × cluster with weight ½)* |
/// | H4 | donor–hydrogen–acceptor | the **hydrogen**, of which every triple has exactly one |
/// | X | halogen → acceptor | the **halogen**; the term is directed and its table asymmetric |
///
/// `n_cell` is how many leading entries of `numbers`/`pos` belong to the reference cell, and
/// `parent` maps each image back to the cell atom it copies (used only by D3, whose coordination
/// numbers would otherwise be truncated at the cluster boundary). Passing `n_cell =
/// numbers.len()` and `parent = None` reproduces the molecular result exactly, which is what
/// keeps the two paths from drifting apart.
#[allow(clippy::too_many_arguments)] // parent and exact travel together and describe one cluster
pub fn correction_energy_cluster_g<S: Scalar>(
    numbers: &[u8],
    pos: &[[S; 3]],
    n_cell: usize,
    parent: Option<&[usize]>,
    // Per entry, whether the cluster is wide enough around it for its own coordination number to
    // come out right. See [`d3::d3_energy_cluster_g`].
    exact: Option<&[bool]>,
    cutoff: Option<f64>,
    cn_cutoff: Option<f64>,
    variant: Variant,
) -> S {
    let mut e = S::cst(0.0);
    match variant {
        Variant::Pm3 => {}
        Variant::Pm3D3 => {
            e = e + d3::d3_energy_cluster_g(
                numbers,
                pos,
                n_cell,
                parent,
                exact,
                cutoff,
                cn_cutoff,
                &d3::D3Params::pm3_d3(),
            );
        }
        Variant::Pm3D3H4 | Variant::Pm3D3H4X => {
            e = e + d3::d3_energy_cluster_g(
                numbers,
                pos,
                n_cell,
                parent,
                exact,
                cutoff,
                cn_cutoff,
                &d3::D3Params::pm3_d3h4(),
            );
            e = e + h4::h4_energy_cluster_g(numbers, pos, n_cell);
            e = e + h4::hh_rep_energy_cluster_g(numbers, pos, n_cell);
        }
    }
    if variant.wants_x() {
        e = e + hx::hx_energy_cluster_g(numbers, pos, n_cell);
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

/// [`correction_energy`] plus the `MMOK` amide term when `mmok` is set.
pub fn correction_energy_with_mmok(mol: &Molecule, variant: Variant, mmok: bool) -> f64 {
    let mut total = correction_energy(mol, variant);
    if mmok {
        // `geometry_f64` is the same Bohr-valued view the gradient path differentiates, so
        // the energy here and the gradient there cannot drift apart in units.
        let (numbers, pos) = geometry_f64(mol);
        total += mmok::amide_correction_g(&numbers, &pos);
    }
    total
}
