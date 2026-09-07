// SPDX-License-Identifier: GPL-3.0-or-later

// Explicit orbital indices follow MOPAC's published/tabled AO ordering.
#![allow(clippy::needless_range_loop)]

//! Two-center two-electron integrals and electron–core attraction for atom
//! pairs where at least one atom carries d orbitals (MNDO-d / PM3).
//!
//! Each local-frame integral `(ij|kl)` is evaluated directly with the
//! Dewar–Sabelli–Ohno point-charge multipole kernel `charg` via `rijkl`
//! (MOPAC `mndod.F90`), then all AO indices are rotated from the diatomic
//! local frame to the molecular frame with the p/d rotation
//! ([`crate::rotations`]). This bypasses MOPAC's `ind2`/`isym`/`rep(491)`
//! symmetry compression (a performance optimization) while reproducing the
//! same integrals.
//!
//! Local orbital order (diatomic frame): `s, pσ, pπ, pπ', dσ, dπ, dπ', dδ,
//! dδ'` (indices 1..9). Global order matches [`crate::rotations`]:
//! `s, px, py, pz, d(x²−y²), d(xz), d(z²), d(yz), d(xy)`.
//!
//! Generic over [`crate::dual::Scalar`] in the interatomic displacement, so the
//! analytic gradient/Hessian differentiate through both the kernel and the
//! rotation.
//!
//! PROVENANCE: openmopac/mopac v23.2.5 (Apache-2.0). See THIRD_PARTY_NOTICES.md.

use crate::constants::PM3_EV;
use crate::dual::{Dual, Scalar};
use crate::integrals::{pack, PairTwoElecG};
use crate::math::Vec3;
use crate::params::Pm3Element;
use crate::rotations::Rotation;

/// Two-center spd integrals seeded with [`Dual`] variables on the displacement
/// `R_j − R_i`, giving the exact derivatives for the analytic d-block gradient.
pub fn pair_two_electron_spd_dual(
    ei: &Pm3Element,
    ej: &Pm3Element,
    dvec: Vec3,
) -> PairTwoElecG<Dual> {
    pair_two_electron_spd(
        ei,
        ej,
        [
            Dual::var(dvec.x, 0),
            Dual::var(dvec.y, 1),
            Dual::var(dvec.z, 2),
        ],
    )
}

/// Angular momentum of each local orbital index (0-based 0..9).
pub(crate) const LORB: [usize; 9] = [0, 1, 1, 1, 2, 2, 2, 2, 2];

/// Standard packed pair index `indx(a,b) = a(a-1)/2 + b` (1-based, a≥b).
#[inline]
pub(crate) fn indx(a: usize, b: usize) -> usize {
    let (h, l) = if a >= b { (a, b) } else { (b, a) };
    h * (h - 1) / 2 + l
}

/// Column-major lower-triangle pair index `indexd(i,j)` (1-based, 9 rows).
#[inline]
pub(crate) fn indexd(i: usize, j: usize) -> usize {
    let (hi, lo) = if i >= j { (i, j) } else { (j, i) };
    // -(lo*(lo-1))/2 + hi + 9*(lo-1)
    (hi + 9 * (lo - 1)) - (lo * (lo - 1)) / 2
}

/// Multipole coefficient `ch(pair, l, m)` (MOPAC `fordd`, mndod.F90:3355).
/// `pair` is the 1-based `indexd` value, `l ∈ {0,1,2}`, `m ∈ {-2..2}`.
pub(crate) fn ch(pair: usize, l: usize, m: i32) -> f64 {
    let s = |v: f64| v;
    let two_over_sqrt3 = 2.0 / 3.0_f64.sqrt(); // 1.15470054
    let one_over_sqrt3 = 1.0 / 3.0_f64.sqrt(); // 0.57735027
    match (pair, l, m) {
        (1, 0, 0) => 1.0,
        (2, 1, 0) => 1.0,
        (3, 1, 1) => 1.0,
        (4, 1, -1) => 1.0,
        (5, 2, 0) => two_over_sqrt3,
        (6, 2, 1) => 1.0,
        (7, 2, -1) => 1.0,
        (8, 2, 2) => 1.0,
        (9, 2, -2) => 1.0,
        (10, 0, 0) => 1.0,
        (10, 2, 0) => 4.0 / 3.0,
        (11, 2, 1) => 1.0,
        (12, 2, -1) => 1.0,
        (13, 1, 0) => two_over_sqrt3,
        (14, 1, 1) => 1.0,
        (15, 1, -1) => 1.0,
        (18, 0, 0) => 1.0,
        (18, 2, 0) => -2.0 / 3.0,
        (18, 2, 2) => 1.0,
        (19, 2, -2) => 1.0,
        (20, 1, 1) => -one_over_sqrt3,
        (21, 1, 0) => 1.0,
        (23, 1, 1) => 1.0,
        (24, 1, -1) => 1.0,
        (25, 0, 0) => 1.0,
        (25, 2, 0) => -2.0 / 3.0,
        (25, 2, 2) => -1.0,
        (26, 1, -1) => -one_over_sqrt3,
        (28, 1, 0) => 1.0,
        (29, 1, -1) => -1.0,
        (30, 1, 1) => 1.0,
        (31, 0, 0) => 1.0,
        (31, 2, 0) => 4.0 / 3.0,
        (32, 2, 1) => one_over_sqrt3,
        (33, 2, -1) => one_over_sqrt3,
        (34, 2, 2) => -two_over_sqrt3,
        (35, 2, -2) => -two_over_sqrt3,
        (36, 0, 0) => 1.0,
        (36, 2, 0) => 2.0 / 3.0,
        (36, 2, 2) => 1.0,
        (37, 2, -2) => 1.0,
        (38, 2, 1) => 1.0,
        (39, 2, -1) => 1.0,
        (40, 0, 0) => 1.0,
        (40, 2, 0) => 2.0 / 3.0,
        (40, 2, 2) => -1.0,
        (41, 2, -1) => -1.0,
        (42, 2, 1) => 1.0,
        (43, 0, 0) => 1.0,
        (43, 2, 0) => -4.0 / 3.0,
        (45, 0, 0) => 1.0,
        (45, 2, 0) => -4.0 / 3.0,
        _ => s(0.0),
    }
}

/// Point-charge multipole kernel `charg` (MOPAC mndod.F90:2160). `r`, `da`,
/// `db` in Bohr; `add = (ρ_a+ρ_b)²`. Returns atomic units (1/Bohr).
fn charg<S: Scalar>(r: S, l1: usize, l2: usize, m: usize, da: f64, db: f64, add: f64) -> S {
    // 1/sqrt((r + shift)^2 + extra + add), shift/extra as f64.
    let f = |shift: f64, extra: f64| -> S {
        ((r + shift) * (r + shift) + (extra + add)).sqrt().recip()
    };
    // 1/sqrt(r^2 + extra + add) with no r shift.
    let g = |extra: f64| -> S { (r * r + (extra + add)).sqrt().recip() };
    let sqrt2 = 2.0_f64.sqrt();
    match (l1, l2, m) {
        (0, 0, _) => g(0.0),
        (1, 0, _) => (-f(da, 0.0) + f(-da, 0.0)) * 0.5,
        (0, 1, _) => (f(db, 0.0) - f(-db, 0.0)) * 0.5,
        (1, 1, 0) => {
            (f(da - db, 0.0) + f(-da + db, 0.0) - f(-da - db, 0.0) - f(da + db, 0.0)) * 0.25
        }
        (1, 1, 1) => (g((da - db) * (da - db)) * 2.0 - g((da + db) * (da + db)) * 2.0) * 0.25,
        (0, 2, _) => (f(-db, 0.0) - g(db * db) * 2.0 + f(db, 0.0)) * 0.25,
        (2, 0, _) => (f(-da, 0.0) - g(da * da) * 2.0 + f(da, 0.0)) * 0.25,
        (1, 2, 0) => {
            (f(-da - db, 0.0) - f(-da, db * db) * 2.0 + f(db - da, 0.0) - f(-db + da, 0.0)
                + f(da, db * db) * 2.0
                - f(da + db, 0.0))
                * 0.125
        }
        (2, 1, 0) => {
            (-f(-da - db, 0.0) + f(-db, da * da) * 2.0 - f(da - db, 0.0) + f(-da + db, 0.0)
                - f(db, da * da) * 2.0
                + f(da + db, 0.0))
                * 0.125
        }
        (2, 2, 0) => {
            let zzzz = f(-da - db, 0.0) + f(da + db, 0.0) + f(-da + db, 0.0) + f(da - db, 0.0)
                - f(-da, db * db) * 2.0
                - f(-db, da * da) * 2.0
                - f(da, db * db) * 2.0
                - f(db, da * da) * 2.0
                + g((da - db) * (da - db)) * 2.0
                + g((da + db) * (da + db)) * 2.0;
            let xyxy = g((da - db) * (da - db)) * 4.0 + g((da + db) * (da + db)) * 4.0
                - g(da * da + db * db) * 8.0;
            zzzz * (1.0 / 16.0) - xyxy * (1.0 / 64.0)
        }
        (1, 2, 1) => {
            let ab = db / sqrt2;
            (-f2(r, -ab, da - ab, add) * 2.0
                + f2(r, ab, da - ab, add) * 2.0
                + f2(r, -ab, da + ab, add) * 2.0
                - f2(r, ab, da + ab, add) * 2.0)
                * 0.125
        }
        (2, 1, 1) => {
            let aa = da / sqrt2;
            (-f2(r, aa, aa - db, add) * 2.0
                + f2(r, -aa, aa - db, add) * 2.0
                + f2(r, aa, aa + db, add) * 2.0
                - f2(r, -aa, aa + db, add) * 2.0)
                * 0.125
        }
        (2, 2, 1) => {
            let aa = da / sqrt2;
            let ab = db / sqrt2;
            (f2(r, aa - ab, aa - ab, add) * 2.0
                - f2(r, aa + ab, aa - ab, add) * 2.0
                - f2(r, -aa - ab, aa - ab, add) * 2.0
                + f2(r, -aa + ab, aa - ab, add) * 2.0
                - f2(r, aa - ab, aa + ab, add) * 2.0
                + f2(r, aa + ab, aa + ab, add) * 2.0
                + f2(r, -aa - ab, aa + ab, add) * 2.0
                - f2(r, -aa + ab, aa + ab, add) * 2.0)
                * (1.0 / 16.0)
        }
        (2, 2, 2) => {
            let xyxy = g((da - db) * (da - db)) * 4.0 + g((da + db) * (da + db)) * 4.0
                - g(da * da + db * db) * 8.0;
            xyxy * (1.0 / 16.0)
        }
        _ => S::cst(0.0),
    }
}

/// `1/sqrt((r + rshift)^2 + perp^2 + add)` for the π/δ cross terms.
#[inline]
fn f2<S: Scalar>(r: S, rshift: f64, perp: f64, add: f64) -> S {
    ((r + rshift) * (r + rshift) + (perp * perp + add))
        .sqrt()
        .recip()
}

/// Local-frame two-center integral `(ij|kl)` (atomic units), MOPAC `rijkl`.
/// `ij`/`kl` are `indexd` pair indices; `li..ll` are AO angular momenta; `ic`
/// selects the un-polarized core additive term (`po[9]`) for a core monopole.
#[allow(clippy::too_many_arguments)]
fn rijkl<S: Scalar>(
    ea: &Pm3Element,
    eb: &Pm3Element,
    ij: usize,
    kl: usize,
    li: usize,
    lj: usize,
    lk: usize,
    ll: usize,
    ic: u8,
    r: S,
) -> S {
    let l1min = li.abs_diff(lj).min(2);
    let l1max = (li + lj).min(2);
    let lij = indx(li + 1, lj + 1);
    let l2min = lk.abs_diff(ll).min(2);
    let l2max = (lk + ll).min(2);
    let lkl = indx(lk + 1, ll + 1);

    let mut sum = S::cst(0.0);
    for l1 in l1min..=l1max {
        let (pij, dij) = if l1 == 0 {
            let p = match lij {
                1 => {
                    if ic == 1 {
                        ea.po[9]
                    } else {
                        ea.po[1]
                    }
                }
                3 => ea.po[7],
                6 => ea.po[8],
                _ => 0.0,
            };
            (p, 0.0)
        } else {
            (ea.po[lij], ea.ddp[lij])
        };
        for l2 in l2min..=l2max {
            let (pkl, dkl) = if l2 == 0 {
                let p = match lkl {
                    1 => {
                        if ic == 2 {
                            eb.po[9]
                        } else {
                            eb.po[1]
                        }
                    }
                    3 => eb.po[7],
                    6 => eb.po[8],
                    _ => 0.0,
                };
                (p, 0.0)
            } else {
                (eb.po[lkl], eb.ddp[lkl])
            };
            let add = (pij + pkl) * (pij + pkl);
            let lmin = l1.min(l2) as i32;
            let mut s1 = S::cst(0.0);
            for m in -lmin..=lmin {
                let ccc = ch(ij, l1, m) * ch(kl, l2, m);
                if ccc == 0.0 {
                    continue;
                }
                let mm = m.unsigned_abs() as usize;
                s1 = s1 + charg(r, l1, l2, mm, dij, dkl, add) * ccc;
            }
            sum = sum + s1;
        }
    }
    sum
}

/// Two-center two-electron integrals + electron–core attraction for one ordered
/// atom pair (atom i first). Handles any (sp or d) pair; distances in Bohr,
/// energies in eV. `dvec = R_j − R_i`.
pub fn pair_two_electron_spd<S: Scalar>(
    ei: &Pm3Element,
    ej: &Pm3Element,
    dvec: [S; 3],
) -> PairTwoElecG<S> {
    let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
    let has_d = ei.has_d() || ej.has_d();
    let rot = Rotation::build(dvec, has_d);
    let na = ei.n_orb;
    let nb = ej.n_orb;

    // Local-frame two-electron integrals (local orbital indices 0-based).
    // rep_local[a][b][c][d] = (a_i b_i | c_j d_j)_local, eV.
    let mut rep = vec![vec![vec![vec![S::cst(0.0); nb]; nb]; na]; na];
    for a in 0..na {
        for b in 0..=a {
            let ij = indexd(a + 1, b + 1);
            for c in 0..nb {
                for d in 0..=c {
                    let kl = indexd(c + 1, d + 1);
                    let v =
                        rijkl(ei, ej, ij, kl, LORB[a], LORB[b], LORB[c], LORB[d], 0, r) * PM3_EV;
                    rep[a][b][c][d] = v;
                    rep[a][b][d][c] = v;
                    rep[b][a][c][d] = v;
                    rep[b][a][d][c] = v;
                }
            }
        }
    }

    // Rotate the two-electron block to the molecular frame, index by index.
    // W[μν|λσ] = Σ Rot[μ,a] Rot[ν,b] Rot[λ,c] Rot[σ,d] rep_local[a][b][c][d].
    let npair_i = na * (na + 1) / 2;
    let npair_j = nb * (nb + 1) / 2;
    // Flat row-major packed table (see `PairTwoElecG`); `w[p * npair_j + q]`.
    let mut w = vec![S::cst(0.0); npair_i * npair_j];
    // Precompute rotation coefficient tables.
    let ra: Vec<Vec<S>> = (0..na)
        .map(|g| (0..na).map(|l| rot.coeff(g, l)).collect())
        .collect();
    let rb: Vec<Vec<S>> = (0..nb)
        .map(|g| (0..nb).map(|l| rot.coeff(g, l)).collect())
        .collect();

    // Stepwise rotation: rotate index a→μ, b→ν (atom i), then c→λ, d→σ (atom j).
    // T_i[μ][ν][c][d].
    let mut ti = vec![vec![vec![vec![S::cst(0.0); nb]; nb]; na]; na];
    for mu in 0..na {
        for nu in 0..na {
            for c in 0..nb {
                for d in 0..nb {
                    let mut acc = S::cst(0.0);
                    for a in 0..na {
                        // No `ca.val() == 0` skip: a zero-valued rotation coefficient
                        // may still have nonzero derivatives at special orientations.
                        let ca = ra[mu][a];
                        for b in 0..na {
                            let cb = rb_or(&ra, nu, b);
                            acc = acc + ca * cb * rep[a][b][c][d];
                        }
                    }
                    ti[mu][nu][c][d] = acc;
                }
            }
        }
    }
    for mu in 0..na {
        for nu in 0..na {
            let pij = pack(mu, nu);
            for lam in 0..nb {
                for sig in 0..nb {
                    let pkl = pack(lam, sig);
                    let mut acc = S::cst(0.0);
                    for c in 0..nb {
                        // No `cc.val() == 0` skip (derivatives of a zero-valued coeff).
                        let cc = rb[lam][c];
                        for d in 0..nb {
                            acc = acc + cc * rb[sig][d] * ti[mu][nu][c][d];
                        }
                    }
                    w[pij * npair_j + pkl] = acc;
                }
            }
        }
    }

    // Electron–core attraction (rotated), eV. e1b: A electrons ↔ B core (ic=2);
    // e2a: B electrons ↔ A core (ic=1).
    let mut e1b_local = [[S::cst(0.0); 9]; 9];
    let mut e2a_local = [[S::cst(0.0); 9]; 9];
    let zb = ej.core_charge;
    let za = ei.core_charge;
    for a in 0..na {
        for b in 0..=a {
            let ij = indexd(a + 1, b + 1);
            let v = rijkl(ei, ej, ij, 1, LORB[a], LORB[b], 0, 0, 2, r) * PM3_EV * (-zb);
            e1b_local[a][b] = v;
            e1b_local[b][a] = v;
        }
    }
    for c in 0..nb {
        for d in 0..=c {
            let kl = indexd(c + 1, d + 1);
            let v = rijkl(ei, ej, 1, kl, 0, 0, LORB[c], LORB[d], 1, r) * PM3_EV * (-za);
            e2a_local[c][d] = v;
            e2a_local[d][c] = v;
        }
    }
    let mut e1b = rotate2(&e1b_local, &ra, na);
    let mut e2a = rotate2(&e2a_local, &rb, nb);

    // The pure s/p-block integrals use the classic MNDO (`reppd`) quadrupole
    // model, which differs from the MNDO-d point-charge model; MOPAC computes
    // that block via `reppd` and uses the d-multipole kernel only for
    // d-involving integrals. Seed the s/p block from the validated s/p path.
    // (Skip for a zero-orbital partner, e.g. a lanthanide sparkle.)
    if na == 0 || nb == 0 {
        return PairTwoElecG {
            norb_i: na,
            norb_j: nb,
            npack_i: npair_i,
            npack_j: npair_j,
            w,
            e1b,
            e2a,
        };
    }
    let sp = crate::integrals::pair_two_electron_g::<S>(ei, ej, dvec);
    let spi = na.min(4);
    let spj = nb.min(4);
    for a in 0..spi {
        for b in 0..=a {
            let pij = pack(a, b);
            for c in 0..spj {
                for d in 0..=c {
                    let pkl = pack(c, d);
                    w[pij * npair_j + pkl] = sp.two_e(a, b, c, d);
                }
            }
        }
    }
    for a in 0..spi {
        for b in 0..spi {
            e1b[a][b] = sp.e1b[a][b];
        }
    }
    for a in 0..spj {
        for b in 0..spj {
            e2a[a][b] = sp.e2a[a][b];
        }
    }

    PairTwoElecG {
        norb_i: na,
        norb_j: nb,
        npack_i: npair_i,
        npack_j: npair_j,
        w,
        e1b,
        e2a,
    }
}

#[inline]
fn rb_or<S: Scalar>(ra: &[Vec<S>], nu: usize, b: usize) -> S {
    ra[nu][b]
}

/// Rotate a symmetric one-atom block `m_local[a][b]` to the molecular frame:
/// `M[μ][ν] = Σ R[μ,a] R[ν,b] m_local[a][b]`.
fn rotate2<S: Scalar>(m_local: &[[S; 9]; 9], rot: &[Vec<S>], n: usize) -> [[S; 9]; 9] {
    let mut out = [[S::cst(0.0); 9]; 9];
    for mu in 0..n {
        for nu in 0..n {
            let mut acc = S::cst(0.0);
            for a in 0..n {
                // No `ca.val() == 0` skip (derivatives of a zero-valued coeff).
                let ca = rot[mu][a];
                for b in 0..n {
                    acc = acc + ca * rot[nu][b] * m_local[a][b];
                }
            }
            out[mu][nu] = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Pm3Parameters;

    #[test]
    fn indexd_matches_mopac() {
        assert_eq!(indexd(1, 1), 1);
        assert_eq!(indexd(2, 1), 2);
        assert_eq!(indexd(9, 1), 9);
        assert_eq!(indexd(2, 2), 10);
        assert_eq!(indexd(5, 5), 31);
        assert_eq!(indexd(9, 9), 45);
    }

    #[test]
    fn spd_matches_sp_path_for_sp_pair() {
        // O (sp) - H: integrals_d must reproduce the validated sp two-electron
        // integrals and electron-core attraction on the s/p sub-block.
        let p = Pm3Parameters::standard().unwrap();
        let o = p.element(8).unwrap();
        let h = p.element(1).unwrap();
        let dvec = [0.7, 0.5, 1.1];
        let sp = crate::integrals::pair_two_electron_g::<f64>(o, h, dvec);
        let spd = pair_two_electron_spd::<f64>(o, h, dvec);
        let mut maxw = 0.0f64;
        for a in 0..4 {
            for b in 0..=a {
                let d = (sp.two_e(a, b, 0, 0) - spd.two_e(a, b, 0, 0)).abs();
                maxw = maxw.max(d);
                if d > 1e-5 {
                    println!(
                        "w[{a}{b}|00]: sp={:+.6} spd={:+.6}",
                        sp.two_e(a, b, 0, 0),
                        spd.two_e(a, b, 0, 0)
                    );
                }
            }
        }
        let mut maxe = 0.0f64;
        for a in 0..4 {
            for b in 0..4 {
                maxe = maxe.max((sp.e1b[a][b] - spd.e1b[a][b]).abs());
            }
        }
        println!("O-H two-electron: max w diff={maxw:.3e}, max e1b diff={maxe:.3e}");
        println!("e2a: sp={:+.6} spd={:+.6}", sp.e2a[0][0], spd.e2a[0][0]);
        assert!(maxw < 1e-4, "two-electron w mismatch {maxw:.3e}");
        assert!(maxe < 1e-4, "e1b mismatch {maxe:.3e}");
    }

    #[test]
    fn spd_wrapper_accepts_pm3_sp_pairs() {
        let p = Pm3Parameters::standard().unwrap();
        let s = p.element(16).unwrap();
        let h = p.element(1).unwrap();
        let te = pair_two_electron_spd::<f64>(s, h, [0.0, 0.0, 2.5]);
        assert_eq!(te.norb_i, 4);
        assert_eq!(te.norb_j, 1);
        for &v in &te.w {
            assert!(v.is_finite());
        }
        // (ss|ss) must be positive and O(gamma).
        assert!(te.w_at(pack(0, 0), pack(0, 0)) > 0.0);
    }
}
