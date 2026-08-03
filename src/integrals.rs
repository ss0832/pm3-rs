// SPDX-License-Identifier: GPL-3.0-or-later

//! NDDO two-center two-electron integrals (Dewar–Sabelli–Klopman multipole model) and the
//! electron–core attraction integrals.
//!
//! The kernels are written generically over [`crate::dual::Scalar`]: instantiating them at
//! `f64` gives the (validated) energy path; instantiating at [`crate::dual::Dual`] gives the
//! exact derivatives with respect to an interatomic displacement, used by the fully analytic
//! gradient. Distances are in Bohr, energies in eV (`PM3_EV`); orbital order is `s,px,py,pz`.

use crate::constants::PM3_EV;
use crate::dual::{Dual, Scalar};
use crate::math::Vec3;
use crate::params::Pm3Element;

/// Lower-triangle pack index of an orbital pair `(a, b)` within a 4-orbital atom block.
#[inline]
pub fn pack(a: usize, b: usize) -> usize {
    let (h, l) = if a >= b { (a, b) } else { (b, a) };
    h * (h + 1) / 2 + l
}

/// Rotation matrix (rows) that rotates the unit vector `v` onto +x (`R·v = (1,0,0)`),
/// generic over the scalar type. Port of PySEQM `rotate_with_quaternion`.
pub fn rotation_to_x_g<S: Scalar>(vx: S, vy: S, vz: S) -> [[S; 3]; 3] {
    // A normalized quaternion is ill-conditioned at -x: its vector and
    // scalar parts both vanish there, losing Dual derivatives.  Use the
    // Householder reflection in that hemisphere instead.
    if vx.val() < 0.0 {
        let one = S::cst(1.0);
        let ux = vx - one;
        let uy = vy;
        let uz = vz;
        let factor = (ux * ux + uy * uy + uz * uz).recip() * (-2.0);
        [
            [one + factor * ux * ux, factor * ux * uy, factor * ux * uz],
            [factor * uy * ux, one + factor * uy * uy, factor * uy * uz],
            [factor * uz * ux, factor * uz * uy, one + factor * uz * uz],
        ]
    } else {
        // q = cross(v, x), 1 + dot(v, x) = (0, vz, -vy, vx + 1).
        let qy = vz;
        let qz = -vy;
        let qw = vx + 1.0;
        let inv = (qy * qy + qz * qz + qw * qw).sqrt().recip();
        let qy = qy * inv;
        let qz = qz * inv;
        let qw = qw * inv;
        [
            [
                (qy * qy + qz * qz) * (-2.0) + 1.0,
                qz * qw * (-2.0),
                qy * qw * 2.0,
            ],
            [qz * qw * 2.0, qz * qz * (-2.0) + 1.0, qy * qz * 2.0],
            [qy * qw * (-2.0), qy * qz * 2.0, qy * qy * (-2.0) + 1.0],
        ]
    }
}

/// Rotation matrix for an `f64` unit vector (used by the overlap f64 path).
pub fn rotation_to_x(v: Vec3) -> [[f64; 3]; 3] {
    rotation_to_x_g(v.x, v.y, v.z)
}

/// Rotated two-electron integrals + electron–core attractions for one ordered atom pair,
/// generic over the scalar type.
///
/// `w` is a **single flat row-major buffer** rather than a `Vec<Vec<_>>`: the
/// `f64` instantiation of this table is cached for every one of the `O(N²)`
/// atom pairs and is the innermost operand of every Fock build, so the nested
/// form cost one heap allocation per packed row (10 per sp pair) and scattered
/// the 10-element rows across the heap. Flattening removes those allocations
/// and makes each packed row a contiguous slice ([`Self::w_row`]).
pub struct PairTwoElecG<S: Scalar> {
    pub norb_i: usize,
    pub norb_j: usize,
    /// Packed bra/ket dimensions, `n(n+1)/2` for `n` AOs.
    pub npack_i: usize,
    pub npack_j: usize,
    /// `w[pack_i(a,b) * npack_j + pack_j(c,d)] = (a_i b_i | c_j d_j)` (eV).
    pub w: Vec<S>,
    /// Electron–core attraction blocks (up to 9 AOs; sp pairs use the top-left 4×4).
    pub e1b: [[S; 9]; 9],
    pub e2a: [[S; 9]; 9],
}

impl<S: Scalar> PairTwoElecG<S> {
    #[inline]
    pub fn two_e(&self, a: usize, b: usize, c: usize, d: usize) -> S {
        self.w[pack(a, b) * self.npack_j + pack(c, d)]
    }

    /// Single element by packed indices (`p = pack(a,b)`, `q = pack(c,d)`).
    #[inline]
    pub fn w_at(&self, p: usize, q: usize) -> S {
        self.w[p * self.npack_j + q]
    }

    /// Packed bra row `p = pack(a,b)` as a contiguous slice over the ket pairs.
    #[inline]
    pub fn w_row(&self, p: usize) -> &[S] {
        &self.w[p * self.npack_j..(p + 1) * self.npack_j]
    }
}

/// f64 alias used by the SCF/Fock.
pub type PairTwoElec = PairTwoElecG<f64>;

/// The part of [`PairTwoElec`] that stays resident for the whole calculation:
/// the packed two-electron table only.
///
/// The electron–core attraction blocks `e1b`/`e2a` (2 × 81 `f64` = 1296 bytes
/// per pair) are consumed **once**, while `H_core` is assembled, and are not
/// referenced again. Keeping them alive for all `N(N−1)/2` pairs cost more
/// memory than the integrals themselves — for a 2400-atom system, 3.7 GiB of
/// pure ballast — so the cache stores this slimmed form instead.
#[derive(Clone, Debug)]
pub struct PackedTwoElec {
    pub norb_i: usize,
    pub norb_j: usize,
    pub npack_i: usize,
    pub npack_j: usize,
    /// Flat row-major `npack_i × npack_j`, indexed as in [`PairTwoElecG::w`].
    pub w: Vec<f64>,
}

impl PackedTwoElec {
    #[inline]
    pub fn two_e(&self, a: usize, b: usize, c: usize, d: usize) -> f64 {
        self.w[pack(a, b) * self.npack_j + pack(c, d)]
    }

    /// Packed bra row `p = pack(a,b)` as a contiguous slice over the ket pairs.
    #[inline]
    pub fn w_row(&self, p: usize) -> &[f64] {
        &self.w[p * self.npack_j..(p + 1) * self.npack_j]
    }

    /// Heap + inline bytes one cached pair of these packed dimensions occupies.
    /// Used by the pre-flight cache-size estimate in [`crate::hamiltonian`].
    pub const fn bytes_for(npack_i: usize, npack_j: usize) -> usize {
        npack_i * npack_j * std::mem::size_of::<f64>() + std::mem::size_of::<Self>()
    }
}

impl From<PairTwoElecG<f64>> for PackedTwoElec {
    fn from(te: PairTwoElecG<f64>) -> Self {
        Self {
            norb_i: te.norb_i,
            norb_j: te.norb_j,
            npack_i: te.npack_i,
            npack_j: te.npack_j,
            w: te.w,
        }
    }
}

/// Compute the rotated two-electron integrals for the ordered pair (i, j) with `xij` the unit
/// vector from i to j and `r` the distance in Bohr (f64 energy path).
pub fn pair_two_electron(ei: &Pm3Element, ej: &Pm3Element, xij: Vec3, r: f64) -> PairTwoElec {
    pair_two_electron_g(ei, ej, [xij.x * r, xij.y * r, xij.z * r])
}

/// Generic driver: `dvec` is the interatomic displacement `R_j − R_i`. Seeding `dvec` with
/// [`Dual`] variables yields the integral derivatives.
pub fn pair_two_electron_g<S: Scalar>(
    ei: &Pm3Element,
    ej: &Pm3Element,
    dvec: [S; 3],
) -> PairTwoElecG<S> {
    let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
    let inv = r.recip();
    let xij = [dvec[0] * inv, dvec[1] * inv, dvec[2] * inv];

    let heavy_i = ei.has_p();
    let heavy_j = ej.has_p();
    // Electron-core attraction uses MOPAC's separate `spcore` multipoles.
    // They intentionally differ from the electron-electron rho terms for Cb;
    // `spcore` also treats Z < 3 as s-only even when He has formal p AOs.
    let e1b = core_attraction_g(ei, ej, dvec);
    let e2a = core_attraction_g(ej, ei, [-dvec[0], -dvec[1], -dvec[2]]);

    // Two-electron frame uses v = -xij (PySEQM convention).
    let rot = rotation_to_x_g(-xij[0], -xij[1], -xij[2]);
    let r0 = rot[0];
    let r1 = rot[1];
    let r2 = rot[2];

    if !heavy_i && !heavy_j {
        let aee = (ei.rho0 + ej.rho0).powi(2);
        let ee = (r * r + aee).sqrt().recip() * PM3_EV;
        return PairTwoElecG {
            norb_i: 1,
            norb_j: 1,
            npack_i: 1,
            npack_j: 1,
            w: vec![ee],
            e1b,
            e2a,
        };
    }

    if heavy_i && !heavy_j {
        let ri = local_xh_g(ei, ej, r);
        let mut wxh = [S::cst(0.0); 10];
        build_wxh_g(&ri, &r0, &r1, &r2, &mut wxh);
        return PairTwoElecG {
            norb_i: 4,
            norb_j: 1,
            npack_i: 10,
            npack_j: 1,
            w: wxh.to_vec(),
            e1b,
            e2a,
        };
    }

    let ri = local_xx_g(ei, ej, r);
    let w100 = rotate_xx_g(&ri, &r0, &r1, &r2);
    PairTwoElecG {
        norb_i: 4,
        norb_j: 4,
        npack_i: 10,
        npack_j: 10,
        w: w100.to_vec(),
        e1b,
        e2a,
    }
}

/// MOPAC `spcore`: attraction of the AOs on `orbital` to `core`.
/// `dvec` points from the orbital center to the core center.
fn core_attraction_g<S: Scalar>(
    orbital: &Pm3Element,
    core: &Pm3Element,
    dvec: [S; 3],
) -> [[S; 9]; 9] {
    let mut block = [[S::cst(0.0); 9]; 9];
    if orbital.n_orb == 0 {
        return block;
    }
    let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
    let core_rho = core.po[9];
    block[0][0] =
        (r * r + (orbital.po[1] + core_rho).powi(2)).sqrt().recip() * (-core.core_charge * PM3_EV);
    if orbital.z < 3 || !orbital.has_p() {
        return block;
    }

    let inv = r.recip();
    let xij = [dvec[0] * inv, dvec[1] * inv, dvec[2] * inv];
    let rot = rotation_to_x_g(-xij[0], -xij[1], -xij[2]);
    let mut orbital_core = orbital.clone();
    orbital_core.rho0 = orbital.po[1];
    orbital_core.rho1 = orbital.po[2];
    orbital_core.rho2 = orbital.po[3];
    orbital_core.dd = orbital.ddp[2];
    orbital_core.qq = orbital.ddp[3] / 2.0_f64.sqrt();
    let mut point_core = core.clone();
    point_core.rho0 = core_rho;
    let local = local_xh_g(&orbital_core, &point_core, r);
    let mut packed = [S::cst(0.0); 10];
    build_wxh_g(&local, &rot[0], &rot[1], &rot[2], &mut packed);
    for a in 0..4 {
        for b in 0..4 {
            block[a][b] = packed[pack(a, b)] * (-core.core_charge);
        }
    }
    block
}

/// PM3 s/p integrals when `point` is a zero-orbital Sparkle or MOPAC point
/// charge. The standard s/p kernel supplies the electron-core attraction on
/// `orbital`; the point partner contributes no AO block or two-electron terms.
pub fn pair_with_point_core_g<S: Scalar>(
    orbital: &Pm3Element,
    point: &Pm3Element,
    dvec: [S; 3],
) -> PairTwoElecG<S> {
    debug_assert!(orbital.n_orb > 0);
    debug_assert_eq!(point.n_orb, 0);
    let mut result = pair_two_electron_g(orbital, point, dvec);
    result.norb_i = orbital.n_orb;
    result.norb_j = 0;
    result
}

fn local_xh_g<S: Scalar>(ei: &Pm3Element, ej: &Pm3Element, r: S) -> [S; 4] {
    let ev1 = PM3_EV / 2.0;
    let ev2 = PM3_EV / 4.0;
    let da = S::cst(ei.dd);
    let qa = S::cst(ei.qq * 2.0);
    let aee = (ei.rho0 + ej.rho0).powi(2);
    let ade = (ei.rho1 + ej.rho0).powi(2);
    let aqe = (ei.rho2 + ej.rho0).powi(2);
    let ee = (r * r + aee).sqrt().recip() * PM3_EV;
    let ev1dsqr6 = (r * r + aqe).sqrt().recip() * ev1;
    let mut ri = [S::cst(0.0); 4];
    ri[0] = ee;
    ri[1] = ((r + da) * (r + da) + ade).sqrt().recip() * ev1
        - ((r - da) * (r - da) + ade).sqrt().recip() * ev1;
    ri[2] = ee
        + ((r + qa) * (r + qa) + aqe).sqrt().recip() * ev2
        + ((r - qa) * (r - qa) + aqe).sqrt().recip() * ev2
        - ev1dsqr6;
    ri[3] = ee + (r * r + qa * qa + aqe).sqrt().recip() * ev1 - ev1dsqr6;
    ri
}

fn local_xx_g<S: Scalar>(ei: &Pm3Element, ej: &Pm3Element, r: S) -> [S; 22] {
    let ev1 = PM3_EV / 2.0;
    let ev2 = PM3_EV / 4.0;
    let ev3 = PM3_EV / 8.0;
    let ev4 = PM3_EV / 16.0;
    let da = S::cst(ei.dd);
    let db = S::cst(ej.dd);
    let qa = S::cst(ei.qq * 2.0);
    let qb = S::cst(ej.qq * 2.0);
    let qa1 = S::cst(ei.qq);
    let qb1 = S::cst(ej.qq);
    let aee = (ei.rho0 + ej.rho0).powi(2);
    let ade = (ei.rho1 + ej.rho0).powi(2);
    let aqe = (ei.rho2 + ej.rho0).powi(2);
    let aed = (ei.rho0 + ej.rho1).powi(2);
    let aeq = (ei.rho0 + ej.rho2).powi(2);
    let axx = (ei.rho1 + ej.rho1).powi(2);
    let adq = (ei.rho1 + ej.rho2).powi(2);
    let aqd = (ei.rho2 + ej.rho1).powi(2);
    let aqq = (ei.rho2 + ej.rho2).powi(2);

    // 1/sqrt(x + c) helper.
    let g = |x: S, c: f64| (x + c).sqrt().recip();
    let sq2 = |a: S, c: f64| (a * a + c).sqrt().recip(); // 1/sqrt(a^2 + c)

    let ee = (r * r + aee).sqrt().recip() * PM3_EV;
    let dze = sq2(r - da, ade) * ev1 - sq2(r + da, ade) * ev1;
    let ev1dsqr6 = (r * r + aqe).sqrt().recip() * ev1;
    let qzze = sq2(r - qa, aqe) * ev2 + sq2(r + qa, aqe) * ev2 - ev1dsqr6;
    let qxxe = g(r * r + qa * qa, aqe) * ev1 - ev1dsqr6;
    let edz = sq2(r + db, aed) * ev1 - sq2(r - db, aed) * ev1;
    let ev1dsqr12 = (r * r + aeq).sqrt().recip() * ev1;
    let eqzz = sq2(r - qb, aeq) * ev2 + sq2(r + qb, aeq) * ev2 - ev1dsqr12;
    let eqxx = g(r * r + qb * qb, aeq) * ev1 - ev1dsqr12;
    let ev2dsqr20 = sq2(r + da, adq) * ev2;
    let ev2dsqr22 = sq2(r - da, adq) * ev2;
    let ev2dsqr24 = sq2(r - db, aqd) * ev2;
    let ev2dsqr26 = sq2(r + db, aqd) * ev2;
    let ev2dsqr36 = (r * r + aqq).sqrt().recip() * ev2;
    let ev2dsqr39 = g(r * r + qa * qa, aqq) * ev2;
    let ev2dsqr40 = g(r * r + qb * qb, aqq) * ev2;
    let ev3dsqr42 = sq2(r - qb, aqq) * ev3;
    let ev3dsqr44 = sq2(r + qb, aqq) * ev3;
    let ev3dsqr46 = sq2(r + qa, aqq) * ev3;
    let ev3dsqr48 = sq2(r - qa, aqq) * ev3;

    let mut ri = [S::cst(0.0); 22];
    ri[0] = ee;
    ri[1] = -dze;
    ri[2] = ee + qzze;
    ri[3] = ee + qxxe;
    ri[4] = -edz;
    ri[5] = sq2(r + da - db, axx) * ev2 + sq2(r - da + db, axx) * ev2
        - sq2(r - da - db, axx) * ev2
        - sq2(r + da + db, axx) * ev2;
    ri[6] =
        g(r * r + (da - db) * (da - db), axx) * ev1 - g(r * r + (da + db) * (da + db), axx) * ev1;
    ri[7] = -edz + sq2(r + qa - db, aqd) * ev3 - sq2(r + qa + db, aqd) * ev3
        + sq2(r - qa - db, aqd) * ev3
        - sq2(r - qa + db, aqd) * ev3
        - ev2dsqr24
        + ev2dsqr26;
    ri[8] = -edz - ev2dsqr24 + g((r - db) * (r - db) + qa * qa, aqd) * ev2 + ev2dsqr26
        - g((r + db) * (r + db) + qa * qa, aqd) * ev2;
    ri[9] = g((qa1 - db) * (qa1 - db) + (r + qa1) * (r + qa1), aqd) * ev2
        - g((qa1 - db) * (qa1 - db) + (r - qa1) * (r - qa1), aqd) * ev2
        - g((qa1 + db) * (qa1 + db) + (r + qa1) * (r + qa1), aqd) * ev2
        + g((qa1 + db) * (qa1 + db) + (r - qa1) * (r - qa1), aqd) * ev2;
    ri[10] = ee + eqzz;
    ri[11] = ee + eqxx;
    ri[12] = -dze + sq2(r + da - qb, adq) * ev3 - sq2(r - da - qb, adq) * ev3
        + sq2(r + da + qb, adq) * ev3
        - sq2(r - da + qb, adq) * ev3
        + ev2dsqr22
        - ev2dsqr20;
    ri[13] = -dze - ev2dsqr20 + g((r + da) * (r + da) + qb * qb, adq) * ev2 + ev2dsqr22
        - g((r - da) * (r - da) + qb * qb, adq) * ev2;
    ri[14] = g((da - qb1) * (da - qb1) + (r - qb1) * (r - qb1), adq) * ev2
        - g((da - qb1) * (da - qb1) + (r + qb1) * (r + qb1), adq) * ev2
        - g((da + qb1) * (da + qb1) + (r - qb1) * (r - qb1), adq) * ev2
        + g((da + qb1) * (da + qb1) + (r + qb1) * (r + qb1), adq) * ev2;
    ri[15] = ee
        + eqzz
        + qzze
        + sq2(r + qa - qb, aqq) * ev4
        + sq2(r + qa + qb, aqq) * ev4
        + sq2(r - qa - qb, aqq) * ev4
        + sq2(r - qa + qb, aqq) * ev4
        - ev3dsqr48
        - ev3dsqr46
        - ev3dsqr42
        - ev3dsqr44
        + ev2dsqr36;
    ri[16] = ee
        + eqzz
        + qxxe
        + g((r - qb) * (r - qb) + qa * qa, aqq) * ev3
        + g((r + qb) * (r + qb) + qa * qa, aqq) * ev3
        - ev3dsqr42
        - ev3dsqr44
        - ev2dsqr39
        + ev2dsqr36;
    ri[17] = ee
        + eqxx
        + qzze
        + g((r + qa) * (r + qa) + qb * qb, aqq) * ev3
        + g((r - qa) * (r - qa) + qb * qb, aqq) * ev3
        - ev3dsqr46
        - ev3dsqr48
        - ev2dsqr40
        + ev2dsqr36;
    let qxxqxx = g(r * r + (qa - qb) * (qa - qb), aqq) * ev3
        + g(r * r + (qa + qb) * (qa + qb), aqq) * ev3
        - ev2dsqr39
        - ev2dsqr40
        + ev2dsqr36;
    ri[18] = ee + eqxx + qxxe + qxxqxx;
    ri[19] = g(
        (r + qa1 - qb1) * (r + qa1 - qb1) + (qa1 - qb1) * (qa1 - qb1),
        aqq,
    ) * ev3
        - g(
            (r + qa1 + qb1) * (r + qa1 + qb1) + (qa1 - qb1) * (qa1 - qb1),
            aqq,
        ) * ev3
        - g(
            (r - qa1 - qb1) * (r - qa1 - qb1) + (qa1 - qb1) * (qa1 - qb1),
            aqq,
        ) * ev3
        + g(
            (r - qa1 + qb1) * (r - qa1 + qb1) + (qa1 - qb1) * (qa1 - qb1),
            aqq,
        ) * ev3
        - g(
            (r + qa1 - qb1) * (r + qa1 - qb1) + (qa1 + qb1) * (qa1 + qb1),
            aqq,
        ) * ev3
        + g(
            (r + qa1 + qb1) * (r + qa1 + qb1) + (qa1 + qb1) * (qa1 + qb1),
            aqq,
        ) * ev3
        + g(
            (r - qa1 - qb1) * (r - qa1 - qb1) + (qa1 + qb1) * (qa1 + qb1),
            aqq,
        ) * ev3
        - g(
            (r - qa1 + qb1) * (r - qa1 + qb1) + (qa1 + qb1) * (qa1 + qb1),
            aqq,
        ) * ev3;
    let qxxqyy = g(r * r + qa * qa + qb * qb, aqq) * ev2 - ev2dsqr39 - ev2dsqr40 + ev2dsqr36;
    ri[20] = ee + eqxx + qxxe + qxxqyy;
    ri[21] = (qxxqxx - qxxqyy) * 0.5;
    ri
}

fn build_wxh_g<S: Scalar>(ri: &[S; 4], r0: &[S; 3], r1: &[S; 3], r2: &[S; 3], wxh: &mut [S; 10]) {
    wxh[pack(0, 0)] = ri[0];
    for k in 0..3 {
        wxh[pack(k + 1, 0)] = ri[1] * r0[k];
    }
    for k in 0..3 {
        for l in 0..=k {
            let t0 = r0[k] * r0[l];
            let t1 = r1[k] * r1[l] + r2[k] * r2[l];
            wxh[pack(k + 1, l + 1)] = ri[2] * t0 + ri[3] * t1;
        }
    }
}

fn rotate_xx_g<S: Scalar>(ri: &[S; 22], r0: &[S; 3], r1: &[S; 3], r2: &[S; 3]) -> [S; 100] {
    let mut w = [S::cst(0.0); 100];
    let mut idx = 0usize;
    for kk in 0..4usize {
        for ll in 0..=kk {
            for mm in 0..4usize {
                for nn in 0..=mm {
                    let (k, l, m, n) = (
                        kk.wrapping_sub(1),
                        ll.wrapping_sub(1),
                        mm.wrapping_sub(1),
                        nn.wrapping_sub(1),
                    );
                    let val = if kk == 0 {
                        if mm == 0 {
                            ri[0]
                        } else if nn == 0 {
                            ri[4] * r0[m]
                        } else {
                            ri[10] * (r0[m] * r0[n]) + ri[11] * (r1[m] * r1[n] + r2[m] * r2[n])
                        }
                    } else if ll == 0 {
                        if mm == 0 {
                            ri[1] * r0[k]
                        } else if nn == 0 {
                            ri[5] * (r0[k] * r0[m]) + ri[6] * (r1[k] * r1[m] + r2[k] * r2[m])
                        } else {
                            let t0 = r0[k] * r0[m] * r0[n];
                            let t1 = (r1[m] * r1[n] + r2[m] * r2[n]) * r0[k];
                            let mix = r1[k] * (r1[n] * r0[m] + r1[m] * r0[n])
                                + r2[k] * (r2[m] * r0[n] + r2[n] * r0[m]);
                            ri[12] * t0 + ri[13] * t1 + ri[14] * mix
                        }
                    } else if mm == 0 {
                        let t0 = r0[k] * r0[l];
                        let t1 = r1[k] * r1[l] + r2[k] * r2[l];
                        ri[2] * t0 + ri[3] * t1
                    } else if nn == 0 {
                        let t0 = r0[k] * r0[l] * r0[m];
                        let t1 = (r1[k] * r1[l] + r2[k] * r2[l]) * r0[m];
                        let t2 = r1[l] * r1[m] + r2[l] * r2[m];
                        ri[7] * t0
                            + ri[8] * t1
                            + ri[9] * (r0[k] * t2 + r0[l] * (r1[k] * r1[m] + r2[k] * r2[m]))
                    } else {
                        let t0 = r0[k] * r0[l] * r0[m] * r0[n];
                        let mut v = ri[15] * t0;
                        v = v + ri[16] * ((r1[k] * r1[l] + r2[k] * r2[l]) * r0[m] * r0[n]);
                        v = v + ri[17] * ((r1[m] * r1[n] + r2[m] * r2[n]) * (r0[k] * r0[l]));
                        v = v + ri[18]
                            * (r1[k] * r1[l] * r1[m] * r1[n] + r2[k] * r2[l] * r2[m] * r2[n]);
                        let mix1 = r0[m] * (r1[l] * r1[n] + r2[l] * r2[n]);
                        let mix2 = r0[n] * (r1[l] * r1[m] + r2[l] * r2[m]);
                        let val5 = r0[k] * (mix1 + mix2)
                            + r0[l]
                                * (r0[m] * (r1[k] * r1[n] + r2[k] * r2[n])
                                    + r0[n] * (r1[k] * r1[m] + r2[k] * r2[m]));
                        v = v + ri[19] * val5;
                        v = v + ri[20]
                            * (r1[k] * r1[l] * r2[m] * r2[n] + r2[k] * r2[l] * r1[m] * r1[n]);
                        let cross =
                            (r1[k] * r2[l] + r2[k] * r1[l]) * (r1[m] * r2[n] + r2[m] * r1[n]);
                        v = v + ri[21] * cross;
                        v
                    };
                    w[idx] = val;
                    idx += 1;
                }
            }
        }
    }
    w
}

/// Dual-valued two-electron integrals for a pair, seeded on the displacement `R_j − R_i`.
pub fn pair_two_electron_dual(ei: &Pm3Element, ej: &Pm3Element, dvec: Vec3) -> PairTwoElecG<Dual> {
    pair_two_electron_g(
        ei,
        ej,
        [
            Dual::var(dvec.x, 0),
            Dual::var(dvec.y, 1),
            Dual::var(dvec.z, 2),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Pm3Parameters;

    #[test]
    fn rotation_is_orthonormal() {
        let v = Vec3::new(0.3, -0.5, 0.8).normalized();
        let r = rotation_to_x(v);
        for row in &r {
            let ni: f64 = row.iter().map(|value| value * value).sum();
            assert!((ni - 1.0).abs() < 1e-10);
        }
        let rv0: f64 = (0..3).map(|k| r[0][k] * v.get(k)).sum();
        assert!((rv0 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ssss_is_rotation_invariant() {
        let p = Pm3Parameters::standard().unwrap();
        let c = p.element(6).unwrap();
        let r = 2.6;
        let a = pair_two_electron(c, c, Vec3::new(1.0, 0.0, 0.0), r);
        let b = pair_two_electron(c, c, Vec3::new(0.3, -0.5, 0.8).normalized(), r);
        assert!((a.w_at(0, 0) - b.w_at(0, 0)).abs() < 1e-9);
        let expect = PM3_EV / (r * r + (2.0 * c.rho0).powi(2)).sqrt();
        assert!((a.w_at(0, 0) - expect).abs() < 1e-9);
    }

    #[test]
    fn dual_two_electron_matches_fd() {
        // The dual derivative of a two-electron integral must match a finite difference.
        let p = Pm3Parameters::standard().unwrap();
        let (c, o) = (p.element(6).unwrap(), p.element(8).unwrap());
        let d = Vec3::new(1.4, -0.9, 0.7);
        let dual = pair_two_electron_dual(c, o, d);
        let h = 1e-6;
        let mut max_delta = 0.0_f64;
        for axis in 0..3 {
            let mut dp = d;
            let mut dm = d;
            match axis {
                0 => {
                    dp.x += h;
                    dm.x -= h;
                }
                1 => {
                    dp.y += h;
                    dm.y -= h;
                }
                _ => {
                    dp.z += h;
                    dm.z -= h;
                }
            }
            let wp = pair_two_electron_g::<f64>(c, o, [dp.x, dp.y, dp.z]);
            let wm = pair_two_electron_g::<f64>(c, o, [dm.x, dm.y, dm.z]);
            for a in 0..10 {
                for b in 0..10 {
                    let fd = (wp.w_at(a, b) - wm.w_at(a, b)) / (2.0 * h);
                    max_delta = max_delta.max((dual.w_at(a, b).d[axis] - fd).abs());
                }
            }
        }
        eprintln!("dual 2e derivative max delta = {max_delta:.2e}");
        assert!(
            max_delta < 1e-6,
            "dual 2e derivative mismatch {max_delta:.3e}"
        );
    }

    #[test]
    fn axial_dual_two_electron_matches_fd() {
        // This exact x-axis geometry exercises the former quaternion singularity
        // in the two-electron rotation frame.
        let p = Pm3Parameters::standard().unwrap();
        let (o, h_atom) = (p.element(8).unwrap(), p.element(1).unwrap());
        let d = Vec3::new(1.8, 0.0, 0.0);
        let dual = pair_two_electron_dual(o, h_atom, d);
        let h = 1e-6;
        let mut max_delta = 0.0_f64;
        for axis in 1..3 {
            let mut dp = d;
            let mut dm = d;
            if axis == 1 {
                dp.y += h;
                dm.y -= h;
            } else {
                dp.z += h;
                dm.z -= h;
            }
            let wp = pair_two_electron_g::<f64>(o, h_atom, [dp.x, dp.y, dp.z]);
            let wm = pair_two_electron_g::<f64>(o, h_atom, [dm.x, dm.y, dm.z]);
            for a in 0..10 {
                let fd = (wp.w_at(a, 0) - wm.w_at(a, 0)) / (2.0 * h);
                max_delta = max_delta.max((dual.w_at(a, 0).d[axis] - fd).abs());
            }
        }
        assert!(
            max_delta < 1e-6,
            "axial dual 2e derivative mismatch {max_delta:.3e}"
        );
    }
}
