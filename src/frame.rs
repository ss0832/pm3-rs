// SPDX-License-Identifier: GPL-3.0-or-later

//! Frame-rotation robustness for the analytic derivatives.
//!
//! The d/sparkle two-center rotation ([`crate::rotations::Rotation`]) parametrizes the diatomic
//! frame by a polar/azimuthal angle pair that is **singular** when a bond lies on the global
//! z-axis (`sqb = √(x²+y²) → 0`, azimuth undefined). The contracted integral *values* stay
//! correct (and, thanks to exact cancellation in the AD arithmetic, so do the derivatives down to
//! `sb ≈ 1e-5`), but exactly on the axis the degenerate branch of `Rotation::build` returns the
//! correct value with **zeroed derivatives** — so the analytic gradient and Hessian of a pair
//! whose bond sits on `+z` in an asymmetric environment are wrong.
//!
//! Because the PM3 energy is rotationally invariant, the cure is purely geometric: detect a
//! near-z d/sparkle pair, rotate the whole molecule into a generic frame (every such bond then
//! well off the z-axis, where the analytic path is correct to ~`1e-7`), evaluate there, and
//! rotate the resulting **vector**/**tensor** back exactly. Molecules with no near-axis d/sparkle
//! pair skip all of this and are bit-identical to the unrotated path.

use crate::error::Result;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::params::Pm3Parameters;
use crate::system::Molecule;

/// Below this `sin(polar angle from +z)` a d/sparkle bond is treated as on-axis and the molecule
/// is evaluated in a rotated frame. The analytic path stays ≤`1e-7` accurate for `sb ≥ 1e-5`, so
/// `1e-3` is a wide safety margin that still only triggers for essentially axis-aligned bonds.
pub(crate) const Z_AXIS_TOL: f64 = 1.0e-3;

/// For each pair routed through the d/sparkle diatomic rotation, `sin` of the bond's polar angle
/// from `+z` (`√(dx²+dy²)/|d|`); returns the minimum over such pairs (`1.0` if there are none).
pub(crate) fn min_daxis_sb(molecule: &Molecule, params: &Pm3Parameters) -> Result<f64> {
    let nat = molecule.atoms.len();
    let mut min_sb = 1.0_f64;
    for i in 0..nat {
        let ei = params.element(molecule.atoms[i].z)?;
        for j in (i + 1)..nat {
            let ej = params.element(molecule.atoms[j].z)?;
            if !(ei.has_d() || ej.has_d() || ei.n_orb == 0 || ej.n_orb == 0) {
                continue;
            }
            let d = molecule.atoms[j].position - molecule.atoms[i].position;
            let r = d.norm();
            if r <= 0.0 {
                continue;
            }
            min_sb = min_sb.min((d.x * d.x + d.y * d.y).sqrt() / r);
        }
    }
    Ok(min_sb)
}

/// `true` if any d/sparkle bond is within [`Z_AXIS_TOL`] of the global z-axis (needs a rotated
/// frame for correct analytic derivatives).
pub(crate) fn needs_generic_frame(molecule: &Molecule, params: &Pm3Parameters) -> Result<bool> {
    Ok(min_daxis_sb(molecule, params)? < Z_AXIS_TOL)
}

/// `Rz(g)·Ry(b)·Rx(a)` rotation matrix (rows = output axes).
pub(crate) fn euler_rotation(a: f64, b: f64, g: f64) -> [[f64; 3]; 3] {
    let (ca, sa) = (a.cos(), a.sin());
    let (cb, sb) = (b.cos(), b.sin());
    let (cg, sg) = (g.cos(), g.sin());
    [
        [cg * cb, cg * sb * sa - sg * ca, cg * sb * ca + sg * sa],
        [sg * cb, sg * sb * sa + cg * ca, sg * sb * ca - cg * sa],
        [-sb, cb * sa, cb * ca],
    ]
}

/// Apply a 3×3 rotation to a vector.
pub(crate) fn rot_apply(r0: &[[f64; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        r0[0][0] * v.x + r0[0][1] * v.y + r0[0][2] * v.z,
        r0[1][0] * v.x + r0[1][1] * v.y + r0[1][2] * v.z,
        r0[2][0] * v.x + r0[2][1] * v.y + r0[2][2] * v.z,
    )
}

/// A copy of `molecule` with every atomic position rotated by `r0`.
pub(crate) fn rotate_molecule(molecule: &Molecule, r0: &[[f64; 3]; 3]) -> Molecule {
    let mut m = molecule.clone();
    for atom in &mut m.atoms {
        atom.position = rot_apply(r0, atom.position);
    }
    m
}

/// Choose a deterministic generic rotation that lifts every d/sparkle bond well off the z-axis
/// (`min sb > 0.1`). Returns `None` if none of the candidates succeeds.
pub(crate) fn pick_generic_frame(
    molecule: &Molecule,
    params: &Pm3Parameters,
) -> Result<Option<[[f64; 3]; 3]>> {
    // Deterministic, mutually "generic" Euler triples (radians): irregular angles so at least one
    // avoids returning any bond to the z-axis for realistic geometries.
    const CAND: [(f64, f64, f64); 5] = [
        (0.6, 0.7, 0.9),
        (1.1, 0.5, 0.3),
        (0.37, 1.05, 1.7),
        (1.3, 0.85, 0.15),
        (0.9, 1.25, 2.1),
    ];
    let mut best: Option<([[f64; 3]; 3], f64)> = None;
    for (a, b, g) in CAND {
        let r0 = euler_rotation(a, b, g);
        let sb = min_daxis_sb(&rotate_molecule(molecule, &r0), params)?;
        if best.is_none_or(|(_, s)| sb > s) {
            best = Some((r0, sb));
        }
    }
    Ok(best.and_then(|(r0, sb)| (sb > 0.1).then_some(r0)))
}

/// Map per-atom gradient vectors computed in a rotated frame back to the original frame:
/// `g_i = R0ᵀ g'_i` (a gradient transforms as a covector under the rigid rotation).
pub(crate) fn back_rotate_gradient(grad: &[Vec3], r0: &[[f64; 3]; 3]) -> Vec<Vec3> {
    grad.iter()
        .map(|g| {
            Vec3::new(
                r0[0][0] * g.x + r0[1][0] * g.y + r0[2][0] * g.z,
                r0[0][1] * g.x + r0[1][1] * g.y + r0[2][1] * g.z,
                r0[0][2] * g.x + r0[1][2] * g.y + r0[2][2] * g.z,
            )
        })
        .collect()
}

/// Map a Hessian computed in a rotated frame back to the original Cartesian frame:
/// `H[iα,jβ] = Σ_{γδ} R0[γ,α] H_rot[iγ,jδ] R0[δ,β]` (rigid-rotation covariance of the energy).
pub(crate) fn back_rotate_hessian(h_rot: &Matrix, r0: &[[f64; 3]; 3]) -> Matrix {
    let n = h_rot.rows;
    let nat = n / 3;
    let mut h = Matrix::zeros(n, n);
    for ia in 0..nat {
        for ja in 0..nat {
            let mut blk = [[0.0f64; 3]; 3];
            for (g, brow) in blk.iter_mut().enumerate() {
                for (d, bv) in brow.iter_mut().enumerate() {
                    *bv = h_rot[(3 * ia + g, 3 * ja + d)];
                }
            }
            for al in 0..3 {
                for be in 0..3 {
                    let mut acc = 0.0;
                    for g in 0..3 {
                        for d in 0..3 {
                            acc += r0[g][al] * blk[g][d] * r0[d][be];
                        }
                    }
                    h[(3 * ia + al, 3 * ja + be)] = acc;
                }
            }
        }
    }
    h
}
