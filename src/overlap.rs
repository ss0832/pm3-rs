// SPDX-License-Identifier: GPL-3.0-or-later

//! Analytic Slater-type diatomic overlap integrals. The `n ≤ 3` closed-form
//! kernel (H through Ar) covers the core organic + main-group s/p elements and
//! returns the 4×4 block `S[a_i][b_j]` in the global frame (orbital order
//! `s, px, py, pz`); heavier shells fall back to the numerical quadrature
//! ([`crate::overlap_numeric`]). The full spd 9×9 diatomic overlap for d
//! elements is [`diatom_overlap_spd`] (local σ/π/δ overlaps + p/d rotation).

use crate::dual::{Dual, Scalar};
use crate::dual2::Dual2;
use crate::error::Result;
use crate::integrals::rotation_to_x_g;
use crate::math::Vec3;
use crate::overlap_numeric::overlap_sto;
use crate::params::Pm3Element;
use crate::rotations::Rotation;

/// Local diatomic-frame orbital layout used by the spd overlap: angular
/// momentum, signed magnetic number, and shell selector (0=s,1=p,2=d) for
/// indices `s, pσ, pπ, pπ', dσ, dπ, dπ', dδ, dδ'`.
const SPD_L: [u8; 9] = [0, 1, 1, 1, 2, 2, 2, 2, 2];
const SPD_M: [i8; 9] = [0, 0, 1, -1, 0, 1, -1, 2, -2];

/// Diatomic overlap block `S[μ_i][ν_j]` (9×9, global frame) for a pair where at
/// least one atom carries d orbitals. Local σ/π/δ Slater overlaps are evaluated
/// by numerical quadrature ([`overlap_sto`]) and rotated to the molecular frame
/// with the p/d rotation ([`Rotation`]). Generic over the scalar type so the
/// analytic gradient/Hessian differentiate through it.
pub fn diatom_overlap_spd<S: Scalar>(
    ei: &Pm3Element,
    ej: &Pm3Element,
    dvec: [S; 3],
) -> [[S; 9]; 9] {
    let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
    let na = ei.n_orb;
    let nb = ej.n_orb;
    let has_d = ei.has_d() || ej.has_d();
    let rot = Rotation::build(dvec, has_d);

    let exp_of = |e: &Pm3Element, l: u8| match l {
        0 => e.zeta_s,
        1 => e.zeta_p,
        _ => e.zeta_d,
    };
    let n_of = |e: &Pm3Element, l: u8| match l {
        0 => e.n_s.max(1),
        1 => e.n_p.max(2),
        _ => e.n_d.max(3),
    };

    // Local-frame overlaps (nonzero only for matching signed m).
    let mut s_local = [[S::cst(0.0); 9]; 9];
    for a in 0..na {
        for b in 0..nb {
            if SPD_M[a] != SPD_M[b] {
                continue;
            }
            let la = SPD_L[a];
            let lb = SPD_L[b];
            let m = SPD_M[a].unsigned_abs();
            s_local[a][b] = overlap_sto(
                n_of(ei, la),
                exp_of(ei, la),
                la,
                m,
                n_of(ej, lb),
                exp_of(ej, lb),
                lb,
                r,
            );
        }
    }

    // Rotate both AO indices to the molecular frame: S[μ][ν] = Σ R[μ,a] R[ν,b] s_local[a][b].
    let ra: Vec<Vec<S>> = (0..na)
        .map(|g| (0..na).map(|l| rot.coeff(g, l)).collect())
        .collect();
    let rb: Vec<Vec<S>> = (0..nb)
        .map(|g| (0..nb).map(|l| rot.coeff(g, l)).collect())
        .collect();
    let mut out = [[S::cst(0.0); 9]; 9];
    for mu in 0..na {
        for nu in 0..nb {
            let mut acc = S::cst(0.0);
            for a in 0..na {
                // NB: do not skip on `ca.val() == 0` — at in-plane/collinear
                // geometries a rotation coefficient can be zero-valued yet carry a
                // nonzero first/second derivative (e.g. cb = z/r at z = 0), and
                // dropping it corrupts the analytic gradient/Hessian.
                let ca = ra[mu][a];
                for b in 0..nb {
                    acc = acc + ca * rb[nu][b] * s_local[a][b];
                }
            }
            out[mu][nu] = acc;
        }
    }
    out
}

/// Overlap block between atom `i` (rows) and atom `j` (cols), both in the global frame.
pub fn diatom_overlap(
    ei: &Pm3Element,
    pos_i: Vec3,
    ej: &Pm3Element,
    pos_j: Vec3,
) -> Result<[[f64; 4]; 4]> {
    let d = pos_j - pos_i;
    let r = d.norm();
    if r < 1.0e-8 {
        let mut m = [[0.0; 4]; 4];
        for (k, row) in m.iter_mut().enumerate() {
            row[k] = 1.0;
        }
        return Ok(m);
    }
    let xij = d / r;
    // The kernel requires the first atom to have the higher principal quantum number.
    let (locals, dir, swap) = if ei.n >= ej.n {
        (overlap_locals(ei, ej, r)?, xij, false)
    } else {
        (overlap_locals(ej, ei, r)?, xij * -1.0, true)
    };
    let di = build_di_g::<f64>(locals, [dir.x, dir.y, dir.z]);
    Ok(if swap { transpose4(di) } else { di })
}

/// Dual-valued overlap block, seeded on the displacement `R_j − R_i`, giving the exact
/// derivative `∂S/∂R_j`. Both the radial and angular dependence differentiate in **closed form**
/// for all valence shells — `n ≤ 3` uses the analytic Slater kernel, `n ≥ 4` uses forward-mode
/// AD through the numerical Slater quadrature. No finite differences.
pub fn diatom_overlap_dual(
    ei: &Pm3Element,
    pos_i: Vec3,
    ej: &Pm3Element,
    pos_j: Vec3,
) -> Result<[[Dual; 4]; 4]> {
    let d = pos_j - pos_i; // i→j displacement; all derivatives are w.r.t. this vector
    let swap = ei.n < ej.n;
    let (ea, eb) = if swap { (ej, ei) } else { (ei, ej) };

    // a→b direction: +dvec/r if not swapped, else −dvec/r (j→i). Seed on the raw displacement.
    let dvec = [Dual::var(d.x, 0), Dual::var(d.y, 1), Dual::var(d.z, 2)];
    let r_dual = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
    let inv_r = r_dual.recip();
    let s = if swap { -1.0 } else { 1.0 };
    let dir = [
        dvec[0] * inv_r * s,
        dvec[1] * inv_r * s,
        dvec[2] * inv_r * s,
    ];

    // Radial local scalars, differentiated in closed form for all n.
    let locals = overlap_locals::<Dual>(ea, eb, r_dual)?;

    let di = build_di_g::<Dual>(locals, dir);
    Ok(if swap { transpose4(di) } else { di })
}

/// Second-derivative overlap block: `[[Dual2;4];4]` seeded on the displacement `R_j − R_i`,
/// giving the exact `∂²S/∂R∂R` (value + gradient + 3×3 Hessian per orbital pair). The radial
/// *and* angular dependence are both differentiated in closed form (no finite differences), so
/// this feeds the fully analytic skeleton Hessian — for all valence shells (`n ≤ 3` analytic
/// kernel, `n ≥ 4` via second-order AD through the numerical Slater quadrature).
pub fn diatom_overlap_dual2(
    ei: &Pm3Element,
    pos_i: Vec3,
    ej: &Pm3Element,
    pos_j: Vec3,
) -> Result<[[Dual2; 4]; 4]> {
    let d = pos_j - pos_i; // i→j displacement; all derivatives are w.r.t. this vector
    let swap = ei.n < ej.n;
    let (ea, eb) = if swap { (ej, ei) } else { (ei, ej) };

    let dvec = [Dual2::var(d.x, 0), Dual2::var(d.y, 1), Dual2::var(d.z, 2)];
    let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
    // Radial locals, exact first+second derivatives w.r.t. the displacement (via `r`).
    let locals = overlap_locals::<Dual2>(ea, eb, r)?;
    // a→b unit direction: +dvec/r if not swapped, else −dvec/r (j→i).
    let inv_r = r.recip();
    let sgn = if swap { -1.0 } else { 1.0 };
    let dir = [
        dvec[0] * inv_r * sgn,
        dvec[1] * inv_r * sgn,
        dvec[2] * inv_r * sgn,
    ];

    let di = build_di_g::<Dual2>(locals, dir);
    Ok(if swap { transpose4(di) } else { di })
}

fn transpose4<T: Copy>(m: [[T; 4]; 4]) -> [[T; 4]; 4] {
    let mut out = m;
    for a in 0..4 {
        for b in 0..4 {
            out[a][b] = m[b][a];
        }
    }
    out
}

/// Diatomic spd overlap (9×9) seeded with [`Dual`] variables on the
/// displacement `R_j − R_i`, for the analytic d-block gradient.
pub fn diatom_overlap_spd_dual(ei: &Pm3Element, ej: &Pm3Element, dvec: Vec3) -> [[Dual; 9]; 9] {
    diatom_overlap_spd(
        ei,
        ej,
        [
            Dual::var(dvec.x, 0),
            Dual::var(dvec.y, 1),
            Dual::var(dvec.z, 2),
        ],
    )
}

/// Embed a 4×4 `Dual` sp overlap into a 9×9 block (for uniform d-aware contraction).
pub fn embed4_dual(s: [[Dual; 4]; 4]) -> [[Dual; 9]; 9] {
    embed4_g(s)
}

/// Embed a 4×4 s/p overlap block into the top-left of a 9×9 spd block (rest 0),
/// generic over the scalar so the same helper serves the gradient (`Dual`) and the
/// Hessian skeleton (`Dual2`).
pub fn embed4_g<S: Scalar>(s: [[S; 4]; 4]) -> [[S; 9]; 9] {
    let mut out = [[S::cst(0.0); 9]; 9];
    for (i, row) in s.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[i][j] = v;
        }
    }
    out
}

#[cfg(test)]
mod spd_overlap_tests {
    use super::*;

    #[test]
    fn spd_overlap_matches_sp_path_for_sp_pair() {
        // O (sp) - H: the spd overlap builder must reproduce the validated
        // analytic sp overlap on the 4×4 s/p sub-block.
        let p = crate::params::Pm3Parameters::standard().unwrap();
        let o = p.element(8).unwrap();
        let hh = p.element(1).unwrap();
        let pos_o = Vec3::new(0.0, 0.0, 0.0);
        let pos_h = Vec3::new(0.7, 0.5, 1.1);
        let dvec = [pos_h.x - pos_o.x, pos_h.y - pos_o.y, pos_h.z - pos_o.z];
        let sp = diatom_overlap(o, pos_o, hh, pos_h).unwrap();
        let spd = diatom_overlap_spd::<f64>(o, hh, dvec);
        let mut maxdiff = 0.0f64;
        for i in 0..4 {
            // H has only 1 orbital, so only column 0 is meaningful.
            let d = (sp[i][0] - spd[i][0]).abs();
            maxdiff = maxdiff.max(d);
            println!("row {i}: sp={:+.6} spd={:+.6}", sp[i][0], spd[i][0]);
        }
        assert!(maxdiff < 1e-4, "spd vs sp overlap mismatch {maxdiff:.3e}");
    }
}

/// `ea` is the higher-(or equal-)quantum-number atom; `r` is the interatomic distance (Bohr).
/// Generic over the scalar type: instantiated at `f64` for the energy/overlap path, and at
/// [`Dual`]/[`crate::dual2::Dual2`] (seeding `r` on the displacement) for the exact first/second
/// radial derivatives. Products are ordered to keep the scalar on the left of any `f64` factor.
fn overlap_locals<S: Scalar>(ea: &Pm3Element, eb: &Pm3Element, r: S) -> Result<[S; 5]> {
    // MOPAC `diat.F90` enables `diat2` only for Z <= 17 with fewer than five
    // orbitals, and explicitly routes He and Ne through the general `ss` path.
    let use_diat2 = |e: &Pm3Element| e.z <= 17 && e.z != 2 && e.z != 10 && e.n_orb < 5;
    let jcall = if use_diat2(ea) && use_diat2(eb) {
        match (ea.n_s, eb.n_s) {
            (1, 1) => 2,
            (2, 1) => 3,
            (2, 2) => 4,
            (3, 1) => 431,
            (3, 2) => 5,
            (3, 3) => 6,
            _ => 0,
        }
    } else {
        0
    };

    let za = [ea.zeta_s, ea.zeta_p];
    let zb = [eb.zeta_s, eb.zeta_p];

    // A/B auxiliary integrals for each (zeta_a-shell, zeta_b-shell) pair.
    let a111 = aintgs(r * (0.5 * (za[0] + zb[0])));
    let b111 = bintgs(r * (0.5 * (za[0] - zb[0])));
    let a211 = aintgs(r * (0.5 * (za[1] + zb[0])));
    let b211 = bintgs(r * (0.5 * (za[1] - zb[0])));
    let a121 = aintgs(r * (0.5 * (za[0] + zb[1])));
    let b121 = bintgs(r * (0.5 * (za[0] - zb[1])));
    let a22 = aintgs(r * (0.5 * (za[1] + zb[1])));
    let b22 = bintgs(r * (0.5 * (za[1] - zb[1])));

    // `s111` is set by every (reachable) arm; the others default to 0 for arms that only
    // populate the sigma channels.
    let s111;
    let (mut s211, mut s121, mut s221, mut s222) =
        (S::cst(0.0), S::cst(0.0), S::cst(0.0), S::cst(0.0));
    let sq3 = 3.0_f64.sqrt();

    match jcall {
        0 => {
            // Heavy elements (n >= 4): the general numerical Slater overlap, now differentiated
            // analytically through the quadrature (r is carried as the scalar type S).
            let (a, b, c, d, e) = crate::overlap_numeric::slater_locals_numeric::<S>(
                ea.n_s.max(1),
                ea.n_p.max(2),
                ea.zeta_s,
                ea.zeta_p,
                eb.n_s.max(1),
                eb.n_p.max(2),
                eb.zeta_s,
                eb.zeta_p,
                r,
            );
            s111 = a;
            s211 = b;
            s121 = c;
            s221 = d;
            s222 = e;
        }
        2 => {
            s111 =
                (r * r * (za[0] * zb[0])).powf(1.5) * (a111[2] * b111[0] - b111[2] * a111[0]) / 4.0;
        }
        3 => {
            s111 = r.powi(4)
                * (zb[0].powf(1.5) * za[0].powf(2.5))
                * (a111[3] * b111[0] - b111[3] * a111[0] + a111[2] * b111[1] - b111[2] * a111[1])
                / (sq3 * 8.0);
            s211 = r.powi(4)
                * (zb[0].powf(1.5) * za[1].powf(2.5))
                * (a211[2] * b211[0] - b211[2] * a211[0] + a211[3] * b211[1] - b211[3] * a211[1])
                / 8.0;
        }
        4 => {
            s111 = r.powi(5)
                * (zb[0] * za[0]).powf(2.5)
                * (a111[4] * b111[0] + b111[4] * a111[0] - a111[2] * b111[2] * 2.0)
                / 48.0;
            s211 = r.powi(5)
                * (zb[0] * za[1]).powf(2.5)
                * (a211[3] * (b211[0] - b211[2]) - a211[1] * (b211[2] - b211[4])
                    + b211[3] * (a211[0] - a211[2])
                    - b211[1] * (a211[2] - a211[4]))
                / (16.0 * sq3);
            s121 = r.powi(5)
                * (zb[1] * za[0]).powf(2.5)
                * (a121[3] * (b121[0] - b121[2])
                    - a121[1] * (b121[2] - b121[4])
                    - b121[3] * (a121[0] - a121[2])
                    + b121[1] * (a121[2] - a121[4]))
                / (16.0 * sq3);
            let w = r.powi(5) * (zb[1] * za[1]).powf(2.5) / 16.0;
            s221 = -(w * (b22[2] * (a22[4] + a22[0]) - a22[2] * (b22[4] + b22[0])));
            s222 = w
                * (a22[4] * (b22[0] - b22[2]) - b22[4] * (a22[0] - a22[2]) - a22[2] * b22[0]
                    + b22[2] * a22[0])
                * 0.5;
        }
        431 => {
            s111 = r.powi(5)
                * (zb[0].powf(1.5) * za[0].powf(3.5))
                * (a111[4] * b111[0] + b111[1] * a111[3] * 2.0
                    - a111[1] * b111[3] * 2.0
                    - b111[4] * a111[0])
                / (10.0_f64.sqrt() * 24.0);
            s211 = r.powi(5)
                * (zb[0].powf(1.5) * za[1].powf(3.5))
                * (a211[3] * (b211[0] + b211[2]) - a211[1] * (b211[4] + b211[2])
                    + b211[1] * (a211[2] + a211[4])
                    - b211[3] * (a211[2] + a211[0]))
                / (8.0 * 30.0_f64.sqrt());
        }
        5 => {
            s111 = r.powi(6)
                * (zb[0].powf(2.5) * za[0].powf(3.5))
                * (a111[5] * b111[0] + b111[1] * a111[4]
                    - b111[2] * a111[3] * 2.0
                    - a111[2] * b111[3] * 2.0
                    + b111[4] * a111[1]
                    + b111[5] * a111[0])
                / (30.0_f64.sqrt() * 48.0);
            s211 = r.powi(6)
                * (zb[0].powf(2.5) * za[1].powf(3.5))
                * (a211[4] * b211[0] + b211[1] * a211[5]
                    - b211[3] * a211[3] * 2.0
                    - a211[2] * b211[2] * 2.0
                    + a211[1] * b211[5]
                    + a211[0] * b211[4])
                / (48.0 * 10.0_f64.sqrt());
            s121 = r.powi(6)
                * (zb[1].powf(2.5) * za[0].powf(3.5))
                * ((a121[4] * b121[0] - a121[5] * b121[1])
                    + (a121[3] * b121[1] - a121[4] * b121[2]) * 2.0
                    - (a121[1] * b121[3] - a121[2] * b121[4]) * 2.0
                    - (a121[0] * b121[4] - a121[1] * b121[5]))
                / (48.0 * 10.0_f64.sqrt());
            s221 = r.powi(6)
                * (zb[1].powf(2.5) * za[1].powf(3.5))
                * ((a22[3] * b22[0] - a22[5] * b22[2]) + (a22[2] * b22[1] - a22[4] * b22[3])
                    - (a22[1] * b22[2] - a22[3] * b22[4])
                    - (a22[0] * b22[3] - a22[2] * b22[5]))
                / (16.0 * 30.0_f64.sqrt());
            s222 = r.powi(6)
                * (zb[1].powf(2.5) * za[1].powf(3.5))
                * ((a22[5] - a22[3]) * (b22[0] - b22[2]) + (a22[4] - a22[2]) * (b22[1] - b22[3])
                    - (a22[3] - a22[1]) * (b22[2] - b22[4])
                    - (a22[2] - a22[0]) * (b22[3] - b22[5]))
                / (32.0 * 30.0_f64.sqrt());
        }
        6 => {
            s111 = r.powi(7)
                * (zb[0] * za[0]).powf(3.5)
                * (a111[6] * b111[0] - b111[2] * a111[4] * 3.0 + a111[2] * b111[4] * 3.0
                    - a111[0] * b111[6])
                / 1440.0;
            s211 = r.powi(7)
                * (zb[0] * za[1]).powf(3.5)
                * ((a211[5] * b211[0] + a211[6] * b211[1])
                    + (-a211[4] * b211[1] - a211[5] * b211[2])
                    - (a211[3] * b211[2] + a211[4] * b211[3]) * 2.0
                    - (-a211[2] * b211[3] - a211[3] * b211[4]) * 2.0
                    + (a211[1] * b211[4] + a211[2] * b211[5])
                    + (-a211[0] * b211[5] - a211[1] * b211[6]))
                / (480.0 * sq3);
            s121 = r.powi(7)
                * (zb[1] * za[0]).powf(3.5)
                * ((a121[5] * b121[0] - a121[6] * b121[1])
                    + (a121[4] * b121[1] - a121[5] * b121[2])
                    - (a121[3] * b121[2] - a121[4] * b121[3]) * 2.0
                    - (a121[2] * b121[3] - a121[3] * b121[4]) * 2.0
                    + (a121[1] * b121[4] - a121[2] * b121[5])
                    + (a121[0] * b121[5] - a121[1] * b121[6]))
                / (480.0 * sq3);
            s221 = r.powi(7)
                * (zb[1].powf(3.5) * za[1].powf(3.5))
                * ((a22[4] * b22[0] - a22[6] * b22[2]) - (a22[2] * b22[2] - a22[4] * b22[4]) * 2.0
                    + (a22[0] * b22[4] - a22[2] * b22[6]))
                / 480.0;
            s222 = r.powi(7)
                * (zb[1].powf(3.5) * za[1].powf(3.5))
                * ((a22[6] - a22[4]) * (b22[0] - b22[2])
                    - (a22[4] - a22[2]) * (b22[2] - b22[4]) * 2.0
                    + (a22[2] - a22[0]) * (b22[4] - b22[6]))
                / 960.0;
        }
        _ => unreachable!(),
    }

    Ok([s111, s211, s121, s221, s222])
}

/// Assemble the molecular-frame 4×4 overlap block from the local-frame quantities and the
/// a→b direction, generic over the scalar type (so the direction dependence differentiates
/// exactly). Shared by the analytic (n ≤ 3) and numerical (n ≥ 4) overlap paths.
fn build_di_g<S: Scalar>(s: [S; 5], dir: [S; 3]) -> [[S; 4]; 4] {
    let (s111, s211, s121, s221, s222) = (s[0], s[1], s[2], s[3], s[4]);
    // rot_ov = R(dir)^T ; its first column is the a→b direction cosines.
    let rr = rotation_to_x_g(dir[0], dir[1], dir[2]);
    let mut rot_ov = [[S::cst(0.0); 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            rot_ov[a][b] = rr[b][a];
        }
    }
    let mut di = [[S::cst(0.0); 4]; 4];
    di[0][0] = s111;
    for a in 0..3 {
        di[1 + a][0] = s211 * rot_ov[a][0];
        di[0][1 + a] = -s121 * rot_ov[a][0];
    }
    let mm = [-s221, s222, s222];
    for a in 0..3 {
        for b in 0..3 {
            let mut v = S::cst(0.0);
            for k in 0..3 {
                v = v + rot_ov[a][k] * mm[k] * rot_ov[b][k];
            }
            di[1 + a][1 + b] = v;
        }
    }
    di
}

/// Test-only: overlap block built with the *numeric* local overlaps (forces the general
/// path even for n ≤ 3), used to validate the numeric kernel against the analytic one.
#[cfg(test)]
pub(crate) fn diatom_overlap_forced_numeric(
    ei: &Pm3Element,
    pos_i: Vec3,
    ej: &Pm3Element,
    pos_j: Vec3,
) -> [[f64; 4]; 4] {
    let d = pos_j - pos_i;
    let r = d.norm();
    let dir = d / r;
    let (ea, eb, direction, swap) = if ei.n >= ej.n {
        (ei, ej, dir, false)
    } else {
        (ej, ei, dir * -1.0, true)
    };
    let (s111, s211, s121, s221, s222) = crate::overlap_numeric::slater_locals_numeric(
        ea.n_s.max(1),
        ea.n_p.max(2),
        ea.zeta_s,
        ea.zeta_p,
        eb.n_s.max(1),
        eb.n_p.max(2),
        eb.zeta_s,
        eb.zeta_p,
        r,
    );
    let di = build_di_g::<f64>(
        [s111, s211, s121, s221, s222],
        [direction.x, direction.y, direction.z],
    );
    if !swap {
        di
    } else {
        transpose4(di)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Pm3Parameters;

    #[test]
    fn numeric_reproduces_analytic_overlap() {
        // The general numerical overlap must match the MOPAC-validated analytic kernel for
        // n <= 3 pairs, across several orientations, for every AO of the block.
        let p = Pm3Parameters::standard().unwrap();
        let a0 = crate::constants::ANGSTROM_TO_BOHR;
        let cases = [(6u8, 1u8), (6, 6), (6, 8), (7, 8), (1, 8), (16, 1), (17, 6)];
        let dirs = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.3, -0.5, 0.8),
            Vec3::new(-0.6, 0.2, -0.1),
        ];
        let mut max_delta = 0.0_f64;
        for (za, zb) in cases {
            let ea = p.element(za).unwrap();
            let eb = p.element(zb).unwrap();
            for d in dirs {
                let r_ang = 1.5;
                let pos_i = Vec3::zero();
                let pos_j = d.normalized() * (r_ang * a0);
                let ana = diatom_overlap(ea, pos_i, eb, pos_j).unwrap();
                let num = diatom_overlap_forced_numeric(ea, pos_i, eb, pos_j);
                for a in 0..4 {
                    for b in 0..4 {
                        max_delta = max_delta.max((ana[a][b] - num[a][b]).abs());
                    }
                }
            }
        }
        eprintln!("numeric-vs-analytic overlap max delta = {max_delta:.2e}");
        assert!(max_delta < 5e-5, "overlap mismatch {max_delta:.3e}");
    }

    #[test]
    fn pm3_oh_overlap_regression() {
        // MOPAC v23.2.5 PM3/AUX, O at the origin and H at (1.05, 0, 0) Angstrom.
        // The values are the O(s,px,py,pz)-H(s) block of the printed AO overlap matrix.
        let params = Pm3Parameters::standard().unwrap();
        let o = params.element(8).unwrap();
        let h = params.element(1).unwrap();
        let oh = diatom_overlap(
            o,
            Vec3::zero(),
            h,
            Vec3::new(1.05 * crate::constants::ANGSTROM_TO_BOHR, 0.0, 0.0),
        )
        .unwrap();
        assert!((oh[0][0] - 0.250_298_238_693_134_8).abs() < 1.0e-10);
        assert!(oh[1][0].abs() > 0.1);
        assert!(oh[2][0].abs() < 1.0e-14);
        assert!(oh[3][0].abs() < 1.0e-14);
    }

    #[test]
    fn pm3_carbon_capped_bond_overlap_regression() {
        // MOPAC v23.2.5 PM3/AUX at the capped-methyl validation geometry.
        // Rows are carbon (s,px,py,pz), columns are Cb (s,px,py,pz).
        let params = Pm3Parameters::standard().unwrap();
        let carbon = params.element(6).unwrap();
        let capped = params.element(102).unwrap();
        assert_eq!(capped.n_orb, 4);
        let got = diatom_overlap(
            carbon,
            Vec3::zero(),
            capped,
            Vec3::new(0.6276, 0.6276, 0.6276) * crate::constants::ANGSTROM_TO_BOHR,
        )
        .unwrap();
        let expected = [
            [
                0.3312551235018,
                -0.1374987354165,
                -0.1374987354165,
                -0.1374987354165,
            ],
            [
                0.2426324433572,
                0.1154938725667,
                -0.0159181769920,
                -0.0159181769920,
            ],
            [
                0.2426324433572,
                -0.0159181769920,
                0.1154938725667,
                -0.0159181769920,
            ],
            [
                0.2426324433572,
                -0.0159181769920,
                -0.0159181769920,
                0.1154938725667,
            ],
        ];
        let mut max_delta = 0.0_f64;
        for i in 0..4 {
            for j in 0..4 {
                max_delta = max_delta.max((got[i][j] - expected[i][j]).abs());
            }
        }
        assert!(
            max_delta < 5.0e-5,
            "C-Cb overlap mismatch {max_delta:.6e}: {got:?}"
        );
    }
}

/// A auxiliary integrals `A_k(x) = ∫₁^∞ t^k e^{-xt} dt`, returned as `[A_0 … A_12]`.
/// Generic over the scalar type so the radial (`x ∝ r`) dependence differentiates exactly.
fn aintgs<S: Scalar>(x: S) -> [S; 13] {
    let mut a = [S::cst(0.0); 13];
    if x.val().abs() < 1.0e-12 {
        return a; // vanishing exponent (e.g. zeta_p of H); those channels are unused
    }
    let inv = x.recip();
    a[0] = (-x).exp() * inv;
    for i in 1..13 {
        a[i] = a[0] + a[i - 1] * inv * (i as f64);
    }
    a
}

/// B auxiliary integrals `B_k(x) = ∫₋₁^¹ t^k e^{-xt} dt`, returned as `[B_0 … B_12]`.
/// Generic over the scalar type so the radial (`x ∝ r`) dependence differentiates exactly.
fn bintgs<S: Scalar>(x: S) -> [S; 13] {
    let mut b = [S::cst(0.0); 13];
    let absx = x.val().abs();
    if absx > 0.5 {
        let inv = x.recip();
        let tx = x.exp() * inv;
        let tmx = (-x).exp() * inv * (-1.0);
        b[0] = tx + tmx;
        for i in 1..13 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            b[i] = tx * sign + tmx + b[i - 1] * inv * (i as f64);
        }
    } else if absx > 1.0e-6 {
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x2 * x2;
        let x5 = x4 * x;
        let x6 = x4 * x2;
        // even index (b1,b3,...): power series in x²
        b[0] = x2 * (1.0 / 3.0) + x4 * (1.0 / 60.0) + x6 * (1.0 / 2520.0) + 2.0;
        b[2] = x2 * (1.0 / 5.0) + x4 * (1.0 / 84.0) + x6 * (1.0 / 3240.0) + 2.0 / 3.0;
        b[4] = x2 * (1.0 / 7.0) + x4 * (1.0 / 108.0) + x6 * (1.0 / 3960.0) + 2.0 / 5.0;
        b[6] = x2 * (1.0 / 9.0) + x4 * (1.0 / 132.0) + x6 * (1.0 / 4680.0) + 2.0 / 7.0;
        b[8] = x2 * (1.0 / 11.0) + x4 * (1.0 / 156.0) + x6 * (1.0 / 5400.0) + 2.0 / 9.0;
        b[10] = x2 * (1.0 / 13.0) + x4 * (1.0 / 180.0) + x6 * (1.0 / 6120.0) + 2.0 / 11.0;
        b[12] = x2 * (1.0 / 15.0) + x4 * (1.0 / 204.0) + x6 * (1.0 / 6840.0) + 2.0 / 13.0;
        // odd index (b2,b4,...): power series in x
        b[1] = x * (-2.0 / 3.0) - x3 * (1.0 / 15.0) - x5 * (1.0 / 420.0);
        b[3] = x * (-2.0 / 5.0) - x3 * (1.0 / 21.0) - x5 * (1.0 / 540.0);
        b[5] = x * (-2.0 / 7.0) - x3 * (1.0 / 27.0) - x5 * (1.0 / 660.0);
        b[7] = x * (-2.0 / 9.0) - x3 * (1.0 / 33.0) - x5 * (1.0 / 780.0);
        b[9] = x * (-2.0 / 11.0) - x3 * (1.0 / 39.0) - x5 * (1.0 / 900.0);
        b[11] = x * (-2.0 / 13.0) - x3 * (1.0 / 45.0) - x5 * (1.0 / 1020.0);
    } else {
        // x ≈ 0
        for (i, bi) in b.iter_mut().enumerate() {
            *bi = S::cst(if i % 2 == 0 {
                2.0 / (i as f64 + 1.0)
            } else {
                0.0
            });
        }
    }
    b
}
