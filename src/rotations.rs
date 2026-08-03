// SPDX-License-Identifier: GPL-3.0-or-later

//! Local→global rotation matrices for p and d orbitals (MOPAC `rotmat`,
//! mndod.F90:1382). For an atom pair the diatomic axis defines a local frame;
//! integrals and overlaps are evaluated there and rotated to the molecular
//! frame with `R[global][local]` = the p (3×3) and d (5×5) blocks below.
//!
//! Orbital order (MOPAC): global AOs are `s, px, py, pz, d(x²−y²), d(xz),
//! d(z²), d(yz), d(xy)`; the local shell order is the same layout with the
//! diatomic axis as the local z. The rotation is written generically over
//! [`crate::dual::Scalar`] so the analytic gradient/Hessian differentiate
//! through it.
//!
//! PROVENANCE: openmopac/mopac v23.2.5 (Apache-2.0). See THIRD_PARTY_NOTICES.md.

use crate::dual::Scalar;

/// `sqrt(3)/2`, MOPAC `pt5sq3`.
const PT5SQ3: f64 = 0.866_025_403_784_1;

/// Block-diagonal AO rotation `R[global][local]` (9×9): `s` (1), `p` (3×3),
/// `d` (5×5). Built from the interatomic displacement `d = R_j − R_i` (the
/// local +z axis points i→j).
pub struct Rotation<S: Scalar> {
    /// `p[gi][li]` — global p-orbital `gi` in terms of local p-orbital `li`.
    pub p: [[S; 3]; 3],
    /// `d[gi][li]` — global d-orbital `gi` in terms of local d-orbital `li`.
    pub d: [[S; 5]; 5],
}

impl<S: Scalar> Rotation<S> {
    /// Build the rotation from the displacement vector (Bohr). `has_d` toggles
    /// the (more expensive) 5×5 d block.
    pub fn build(dvec: [S; 3], has_d: bool) -> Self {
        let (x11, x22, x33) = (dvec[0], dvec[1], dvec[2]);
        let b = x11 * x11 + x22 * x22;
        let r = (b + x33 * x33).sqrt();
        let sqb = b.sqrt();
        let sb = sqb / r;

        let (ca, sa, cb);
        if sb.val() > 1.0e-7 {
            ca = x11 / sqb;
            sa = x22 / sqb;
            cb = x33 / r;
            Self::assemble(ca, sa, cb, sb, has_d)
        } else {
            // Both atoms on the local z axis (collinear special case).
            let sa = S::cst(0.0);
            let sb = S::cst(0.0);
            let (ca, cb) = if x33.val() < 0.0 {
                (S::cst(-1.0), S::cst(-1.0))
            } else if x33.val() > 0.0 {
                (S::cst(1.0), S::cst(1.0))
            } else {
                (S::cst(0.0), S::cst(0.0))
            };
            Self::assemble(ca, sa, cb, sb, has_d)
        }
    }

    fn assemble(ca: S, sa: S, cb: S, sb: S, has_d: bool) -> Self {
        // p[global][local] (MOPAC rotmat p(i,j)).
        let p = [
            [ca * sb, sa * sb, cb],
            [ca * cb, sa * cb, -sb],
            [-sa, ca, S::cst(0.0)],
        ];
        let mut d = [[S::cst(0.0); 5]; 5];
        if has_d {
            let c2a = ca * ca * 2.0 - 1.0;
            let c2b = cb * cb * 2.0 - 1.0;
            let s2a = sa * ca * 2.0;
            let s2b = sb * cb * 2.0;
            // d[global i][local j] (MOPAC rotmat d(i,j)).
            d[0][0] = c2a * sb * sb * PT5SQ3;
            d[1][0] = c2a * s2b * 0.5;
            d[2][0] = -(s2a * sb);
            d[3][0] = c2a * (cb * cb + sb * sb * 0.5);
            d[4][0] = -(s2a * cb);
            d[0][1] = ca * s2b * PT5SQ3;
            d[1][1] = ca * c2b;
            d[2][1] = -(sa * cb);
            d[3][1] = ca * s2b * -0.5;
            d[4][1] = sa * sb;
            d[0][2] = cb * cb - sb * sb * 0.5;
            d[1][2] = s2b * -PT5SQ3;
            d[2][2] = S::cst(0.0);
            d[3][2] = sb * sb * PT5SQ3;
            d[4][2] = S::cst(0.0);
            d[0][3] = sa * s2b * PT5SQ3;
            d[1][3] = sa * c2b;
            d[2][3] = ca * cb;
            d[3][3] = sa * s2b * -0.5;
            d[4][3] = -(ca * sb);
            d[0][4] = s2a * sb * sb * PT5SQ3;
            d[1][4] = s2a * s2b * 0.5;
            d[2][4] = c2a * sb;
            d[3][4] = s2a * (cb * cb + sb * sb * 0.5);
            d[4][4] = c2a * cb;
        }
        Self { p, d }
    }

    /// Rotation coefficient `R[global][local] = ⟨global | local⟩` for orbital
    /// indices 0..9 (0 = s, 1..4 = p, 4..9 = d). Zero across shells.
    ///
    /// MOPAC `rotmat` stores `p(local, global)` / `d(local, global)`, so the
    /// local→global map used to rotate integrals is the transpose.
    #[inline]
    pub fn coeff(&self, global: usize, local: usize) -> S {
        match (global, local) {
            (0, 0) => S::cst(1.0),
            (0, _) | (_, 0) => S::cst(0.0),
            (g, l) if (1..4).contains(&g) && (1..4).contains(&l) => self.p[l - 1][g - 1],
            (g, l) if (4..9).contains(&g) && (4..9).contains(&l) => self.d[l - 4][g - 4],
            _ => S::cst(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p_rotation_is_orthogonal_along_z() {
        // Bond along +z: local == global (identity p block up to MOPAC's frame).
        let rot = Rotation::<f64>::build([0.0, 0.0, 2.0], true);
        // R must be orthogonal: rows orthonormal.
        for i in 0..3 {
            let mut norm = 0.0;
            for j in 0..3 {
                norm += rot.p[i][j] * rot.p[i][j];
            }
            assert!((norm - 1.0).abs() < 1e-10, "p row {i} not unit: {norm}");
        }
    }

    #[test]
    fn d_rotation_is_orthonormal() {
        let rot = Rotation::<f64>::build([0.7, -0.4, 1.1], true);
        for i in 0..5 {
            let mut norm = 0.0;
            for j in 0..5 {
                norm += rot.d[i][j] * rot.d[i][j];
            }
            assert!((norm - 1.0).abs() < 1e-9, "d row {i} not unit: {norm}");
        }
        // Orthogonality of two distinct d rows.
        let mut dot = 0.0;
        for j in 0..5 {
            dot += rot.d[0][j] * rot.d[1][j];
        }
        assert!(dot.abs() < 1e-9, "d rows not orthogonal: {dot}");
    }
}
