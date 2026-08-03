// SPDX-License-Identifier: GPL-3.0-or-later

// Slater–Condon radial integrals conventionally expose four (n, exponent)
// pairs; grouping them would obscure the source equation and MOPAC mapping.
#![allow(clippy::too_many_arguments)]

//! One-center two-electron integrals for spd elements (MNDO-d / PM3).
//!
//! Ported from MOPAC v23.2.5 `integrals/mndod.F90` (`rsc`, `scprm`, `inighd`,
//! `wstore`, `eiscor`) and `mndod_C.F90` (the `intij`/`intkl`/`intrep` index
//! tables). For a transition-metal (or hypervalent main-group) element the
//! one-center Coulomb/exchange integrals `(ab|cd)` over the 9 valence AOs are
//! built from Slater–Condon radial integrals of the internal exponents
//! `zsn/zpn/zdn`; the `s`/`p` block additionally uses the fitted `Gss…Hsp`.
//!
//! Orbital order (MOPAC): `s, px, py, pz, d(x²−y²), d(xz), d(z²), d(yz), d(xy)`.
//! Orbital-pair index (1-based in MOPAC) `pack(a,b) = a(a-1)/2 + b`; here we use
//! the 0-based [`crate::integrals::pack`].
//!
//! PROVENANCE: openmopac/mopac v23.2.5 (Apache-2.0). See THIRD_PARTY_NOTICES.md.

use crate::constants::PM3_EV;
use crate::integrals::pack;
use crate::params::Pm3Element;

/// Number of one-center orbital pairs for a 9-orbital atom.
pub const NPAIR: usize = 45;

/// 1-based factorial / Pascal tables (`fx(i) = (i-1)!`, `b(i,j) = C(i-1,j-1)`),
/// exactly as MOPAC's `fbx` builds them.
struct Combinatorics {
    fx: [f64; 31],
    b: [[f64; 31]; 31],
}

impl Combinatorics {
    fn new() -> Self {
        let mut fx = [0.0; 31];
        fx[1] = 1.0;
        for i in 2..=30 {
            fx[i] = fx[i - 1] * (i as f64 - 1.0);
        }
        let mut b = [[0.0; 31]; 31];
        for row in b.iter_mut() {
            row[1] = 1.0;
        }
        for i in 2..=30 {
            for j in 2..=i {
                b[i][j] = b[i - 1][j - 1] + b[i - 1][j];
            }
        }
        Self { fx, b }
    }

    /// Slater–Condon radial integral `R^k(ab,cd)` in eV (`rsc`, mndod.F90:1570).
    fn rsc(
        &self,
        k: i32,
        na: i32,
        ea: f64,
        nb: i32,
        eb: f64,
        nc: i32,
        ec: f64,
        nd: i32,
        ed: f64,
    ) -> f64 {
        let (fx, b) = (&self.fx, &self.b);
        let (aea, aeb, aec, aed) = (ea.ln(), eb.ln(), ec.ln(), ed.ln());
        let nab = na + nb;
        let ncd = nc + nd;
        let ecd = ec + ed;
        let eab = ea + eb;
        let e = ecd + eab;
        let n = nab + ncd;
        let ae = e.ln();
        let a2 = 2.0_f64.ln();
        let acd = ecd.ln();
        let aab = eab.ln();
        let ff = fx[n as usize]
            / (fx[(2 * na + 1) as usize]
                * fx[(2 * nb + 1) as usize]
                * fx[(2 * nc + 1) as usize]
                * fx[(2 * nd + 1) as usize])
                .sqrt();
        let c = PM3_EV
            * ff
            * (na as f64 * aea
                + nb as f64 * aeb
                + nc as f64 * aec
                + nd as f64 * aed
                + 0.5 * (aea + aeb + aec + aed)
                + a2 * (n as f64 + 2.0)
                - ae * n as f64)
                .exp();
        let mut s0 = 1.0 / e;
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        let m = ncd - k;
        for i in 1..=m {
            s0 *= e / ecd;
            s1 += s0 * (b[(ncd - k) as usize][i as usize] - b[(ncd + k + 1) as usize][i as usize])
                / b[n as usize][i as usize];
        }
        let m1 = m + 1;
        let m2 = ncd + k + 1;
        for i in m1..=m2 {
            s0 *= e / ecd;
            s2 += s0 * b[m2 as usize][i as usize] / b[n as usize][i as usize];
        }
        let s3 = (ae * n as f64 - acd * m2 as f64 - aab * (nab - k) as f64).exp()
            / b[n as usize][m2 as usize];
        c * (s1 - s2 + s3)
    }
}

/// Slater–Condon radial integral `R^k(ab,cd)` in eV, from principal quantum
/// numbers and exponents (Bohr⁻¹). MOPAC `rsc` (mndod.F90:1570); used by the
/// transition-metal one-center Coulomb/exchange recompute (`sp_two_electron`).
pub fn slater_rsc(
    k: i32,
    na: i32,
    ea: f64,
    nb: i32,
    eb: f64,
    nc: i32,
    ec: f64,
    nd: i32,
    ed: f64,
) -> f64 {
    Combinatorics::new().rsc(k, na, ea, nb, eb, nc, ec, nd, ed)
}

/// One-center spd two-electron integral matrix for one element.
#[derive(Clone, Debug)]
pub struct OneCenterSpd {
    /// `w[pack(a,b)][pack(c,d)] = (a b | c d)` in eV (symmetric in ab↔cd).
    pub w: Vec<Vec<f64>>,
    /// d-shell contribution to the isolated-atom electronic energy (eV).
    pub eisol_d: f64,
    /// Final one-center d integrals actually used (after f0sd/g2sd override).
    pub f0sd: f64,
    pub g2sd: f64,
    /// The `repd(1..=52)` Slater–Condon combinations (index 0 unused); the
    /// d-multipole additive-term solver `ddpo` reads slots 19,23,29,30,31,35,44,52.
    pub repd: [f64; 53],
}

/// The 243 `(intij, intkl, intrep)` entries mapping d-block one-center pair
/// integrals to `repd` slots (MOPAC `mndod_C.F90`).
const INTIJ: [usize; 243] = [
    1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12,
    12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16,
    16, 17, 17, 17, 17, 17, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21,
    21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24,
    24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28,
    28, 28, 28, 29, 29, 29, 29, 29, 30, 30, 30, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 33, 33, 33,
    33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 37,
    37, 37, 37, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 40, 40, 40, 41, 42, 42, 42, 42, 42, 43, 43,
    43, 43, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
];
const INTKL: [usize; 243] = [
    15, 21, 28, 36, 45, 12, 19, 23, 39, 11, 15, 21, 22, 26, 28, 36, 45, 13, 24, 32, 38, 34, 37, 43,
    11, 15, 21, 22, 26, 28, 36, 45, 17, 25, 31, 16, 20, 27, 44, 29, 33, 35, 42, 15, 21, 22, 28, 36,
    45, 3, 6, 11, 21, 26, 36, 2, 12, 19, 23, 39, 4, 13, 24, 32, 38, 14, 17, 31, 1, 3, 6, 10, 15,
    21, 22, 28, 36, 45, 8, 16, 20, 27, 44, 7, 14, 17, 25, 31, 18, 30, 40, 2, 12, 19, 23, 39, 8, 16,
    20, 27, 44, 1, 3, 6, 10, 11, 15, 21, 22, 26, 28, 36, 45, 3, 6, 10, 15, 21, 22, 28, 36, 45, 2,
    12, 19, 23, 39, 4, 13, 24, 32, 38, 7, 17, 25, 31, 3, 6, 11, 21, 26, 36, 8, 16, 20, 27, 44, 1,
    3, 6, 10, 15, 21, 22, 28, 36, 45, 9, 29, 33, 35, 42, 18, 30, 40, 7, 14, 17, 25, 31, 4, 13, 24,
    32, 38, 9, 29, 33, 35, 42, 5, 34, 37, 43, 9, 29, 33, 35, 42, 1, 3, 6, 10, 11, 15, 21, 22, 26,
    28, 36, 45, 5, 34, 37, 43, 4, 13, 24, 32, 38, 2, 12, 19, 23, 39, 18, 30, 40, 41, 9, 29, 33, 35,
    42, 5, 34, 37, 43, 8, 16, 20, 27, 44, 1, 3, 6, 10, 15, 21, 22, 28, 36, 45,
];
const INTREP: [usize; 243] = [
    1, 1, 1, 1, 1, 3, 3, 8, 3, 9, 6, 6, 12, 14, 13, 7, 6, 15, 8, 3, 3, 11, 9, 14, 17, 6, 7, 12, 18,
    13, 6, 6, 3, 2, 3, 9, 11, 10, 11, 9, 16, 10, 11, 7, 6, 4, 5, 6, 7, 9, 17, 19, 32, 22, 40, 3,
    33, 34, 27, 46, 15, 33, 28, 41, 47, 35, 35, 42, 1, 6, 6, 7, 29, 38, 22, 31, 38, 51, 9, 19, 32,
    21, 32, 3, 35, 33, 24, 34, 35, 35, 35, 3, 34, 33, 26, 34, 11, 32, 44, 37, 49, 1, 6, 7, 6, 32,
    38, 29, 21, 39, 30, 38, 38, 12, 12, 4, 22, 21, 19, 20, 21, 22, 8, 27, 26, 25, 27, 8, 28, 25,
    26, 27, 2, 24, 23, 24, 14, 18, 22, 39, 48, 45, 10, 21, 37, 36, 37, 1, 13, 13, 5, 31, 30, 20,
    29, 30, 31, 9, 19, 40, 21, 32, 35, 35, 35, 3, 42, 34, 24, 33, 3, 41, 26, 33, 34, 16, 40, 44,
    43, 50, 11, 44, 32, 39, 10, 21, 43, 36, 37, 1, 7, 6, 6, 40, 38, 38, 21, 45, 30, 29, 38, 9, 32,
    19, 22, 3, 47, 27, 34, 33, 3, 46, 34, 27, 33, 35, 35, 35, 52, 11, 32, 50, 37, 44, 14, 39, 22,
    48, 11, 32, 49, 37, 44, 1, 6, 6, 7, 51, 38, 22, 31, 38, 29,
];

/// Build the `repd(1..52)` table for one element from Slater–Condon parameters
/// (MOPAC `inighd`/`scprm`, mndod.F90:62-163). `f0sd`/`g2sd` are overridden by
/// the fitted values when > 0.001. Returns `(repd[1..=52], f0sd, g2sd, eisol_d)`.
fn build_repd(elem: &Pm3Element, cmb: &Combinatorics) -> ([f64; 53], f64, f64, f64) {
    let ns = elem.n_s as i32;
    let nd = elem.n_d as i32;
    let (es, ep, ed) = (elem.zsn, elem.zpn, elem.zdn);
    let rsc = |k, na, ea, nb, eb, nc, ec, ndd, edd| cmb.rsc(k, na, ea, nb, eb, nc, ec, ndd, edd);

    // scprm: the 12 Slater–Condon parameters (mndod.F90:1641).
    let r016 = rsc(0, ns, es, ns, es, nd, ed, nd, ed);
    let r036 = rsc(0, elem.n_p as i32, ep, elem.n_p as i32, ep, nd, ed, nd, ed);
    let r066 = rsc(0, nd, ed, nd, ed, nd, ed, nd, ed);
    let r155 = rsc(1, elem.n_p as i32, ep, nd, ed, elem.n_p as i32, ep, nd, ed);
    let r125 = rsc(1, ns, es, elem.n_p as i32, ep, elem.n_p as i32, ep, nd, ed);
    let mut r244 = rsc(2, ns, es, nd, ed, ns, es, nd, ed);
    let r236 = rsc(2, elem.n_p as i32, ep, elem.n_p as i32, ep, nd, ed, nd, ed);
    let r266 = rsc(2, nd, ed, nd, ed, nd, ed, nd, ed);
    let r234 = rsc(2, elem.n_p as i32, ep, elem.n_p as i32, ep, ns, es, nd, ed);
    let r246 = rsc(2, ns, es, nd, ed, nd, ed, nd, ed);
    let r355 = rsc(3, elem.n_p as i32, ep, nd, ed, elem.n_p as i32, ep, nd, ed);
    let r466 = rsc(4, nd, ed, nd, ed, nd, ed, nd, ed);

    // f0sd/g2sd: explicit override wins.
    let mut r016o = r016;
    if elem.f0sd > 0.001 {
        r016o = elem.f0sd;
    }
    if elem.g2sd > 0.001 {
        r244 = elem.g2sd;
    }

    let s3 = 3.0_f64.sqrt();
    let s5 = 5.0_f64.sqrt();
    let s15 = 15.0_f64.sqrt();
    let mut d = [0.0f64; 53];
    d[1] = r016o;
    d[2] = 2.0 / (3.0 * s5) * r125;
    d[3] = 1.0 / s15 * r125;
    d[4] = 2.0 / (5.0 * s5) * r234;
    d[5] = r036 + 4.0 / 35.0 * r236;
    d[6] = r036 + 2.0 / 35.0 * r236;
    d[7] = r036 - 4.0 / 35.0 * r236;
    d[8] = -1.0 / (3.0 * s5) * r125;
    d[9] = (3.0f64 / 125.0).sqrt() * r234;
    d[10] = s3 / 35.0 * r236;
    d[11] = 3.0 / 35.0 * r236;
    d[12] = -1.0 / (5.0 * s5) * r234;
    d[13] = r036 - 2.0 / 35.0 * r236;
    d[14] = -2.0 * s3 / 35.0 * r236;
    d[15] = -d[3];
    d[16] = -d[11];
    d[17] = -d[9];
    d[18] = -d[14];
    d[19] = 1.0 / 5.0 * r244;
    d[20] = 2.0 / (7.0 * s5) * r246;
    d[21] = d[20] / 2.0;
    d[22] = -d[20];
    d[23] = 4.0 / 15.0 * r155 + 27.0 / 245.0 * r355;
    d[24] = 2.0 * s3 / 15.0 * r155 - 9.0 * s3 / 245.0 * r355;
    d[25] = 1.0 / 15.0 * r155 + 18.0 / 245.0 * r355;
    d[26] = -s3 / 15.0 * r155 + 12.0 * s3 / 245.0 * r355;
    d[27] = -s3 / 15.0 * r155 - 3.0 * s3 / 245.0 * r355;
    d[28] = -d[27];
    d[29] = r066 + 4.0 / 49.0 * r266 + 4.0 / 49.0 * r466;
    d[30] = r066 + 2.0 / 49.0 * r266 - 24.0 / 441.0 * r466;
    d[31] = r066 - 4.0 / 49.0 * r266 + 6.0 / 441.0 * r466;
    d[32] = (3.0f64 / 245.0).sqrt() * r246;
    d[33] = 1.0 / 5.0 * r155 + 24.0 / 245.0 * r355;
    d[34] = 1.0 / 5.0 * r155 - 6.0 / 245.0 * r355;
    d[35] = 3.0 / 49.0 * r355;
    d[36] = 1.0 / 49.0 * r266 + 30.0 / 441.0 * r466;
    d[37] = s3 / 49.0 * r266 - 5.0 * s3 / 441.0 * r466;
    d[38] = r066 - 2.0 / 49.0 * r266 - 4.0 / 441.0 * r466;
    d[39] = -2.0 * s3 / 49.0 * r266 + 10.0 * s3 / 441.0 * r466;
    d[40] = -d[32];
    d[41] = -d[34];
    d[42] = -d[35];
    d[43] = -d[37];
    d[44] = 3.0 / 49.0 * r266 + 20.0 / 441.0 * r466;
    d[45] = -d[39];
    d[46] = 1.0 / 5.0 * r155 - 3.0 / 35.0 * r355;
    d[47] = -d[46];
    d[48] = 4.0 / 49.0 * r266 + 15.0 / 441.0 * r466;
    d[49] = 3.0 / 49.0 * r266 - 5.0 / 147.0 * r466;
    d[50] = -d[49];
    d[51] = r066 + 4.0 / 49.0 * r266 - 34.0 / 441.0 * r466;
    d[52] = 35.0 / 441.0 * r466;

    // eiscor: d-shell contribution to the isolated-atom energy (mndod.F90:1777).
    let (ir016, ir066, ir244, ir266, ir466) = eiscor_coeffs(elem.z);
    let eisol_d = ir016 * r016o + ir066 * r066
        - ir244 * r244 / 5.0
        - ir266 * r266 / 49.0
        - ir466 * r466 / 49.0;

    (d, r016o, r244, eisol_d)
}

/// Average-of-configuration d-shell occupancy coefficients for `eisol`
/// (MOPAC `eiscor`, per element Z; zero outside the tabulated d-block ranges).
fn eiscor_coeffs(z: u8) -> (f64, f64, f64, f64, f64) {
    // (ir016, ir066, ir244, ir266, ir466)
    let t: Option<[f64; 5]> = match z {
        21 => Some([2.0, 0.0, 1.0, 0.0, 0.0]),
        22 => Some([4.0, 1.0, 2.0, 8.0, 1.0]),
        23 => Some([6.0, 3.0, 3.0, 15.0, 8.0]),
        24 => Some([5.0, 10.0, 5.0, 35.0, 35.0]),
        25 => Some([10.0, 10.0, 5.0, 35.0, 35.0]),
        26 => Some([12.0, 15.0, 6.0, 35.0, 35.0]),
        27 => Some([14.0, 21.0, 7.0, 43.0, 36.0]),
        28 => Some([16.0, 28.0, 8.0, 50.0, 43.0]),
        29 => Some([10.0, 45.0, 5.0, 70.0, 70.0]),
        39 => Some([2.0, 0.0, 1.0, 0.0, 0.0]),
        40 => Some([4.0, 1.0, 2.0, 8.0, 1.0]),
        41 => Some([4.0, 6.0, 4.0, 21.0, 21.0]),
        42 => Some([5.0, 10.0, 5.0, 35.0, 35.0]),
        43 => Some([10.0, 10.0, 5.0, 35.0, 35.0]),
        44 => Some([7.0, 21.0, 5.0, 43.0, 36.0]),
        45 => Some([8.0, 28.0, 5.0, 50.0, 43.0]),
        46 => Some([0.0, 45.0, 0.0, 70.0, 70.0]),
        47 => Some([10.0, 45.0, 5.0, 70.0, 70.0]),
        57 => Some([2.0, 0.0, 1.0, 0.0, 0.0]),
        71 => Some([2.0, 0.0, 1.0, 0.0, 0.0]),
        72 => Some([4.0, 1.0, 2.0, 8.0, 1.0]),
        73 => Some([6.0, 3.0, 3.0, 15.0, 8.0]),
        74 => Some([5.0, 10.0, 5.0, 35.0, 35.0]),
        75 => Some([10.0, 10.0, 5.0, 35.0, 35.0]),
        76 => Some([12.0, 15.0, 6.0, 35.0, 35.0]),
        77 => Some([14.0, 21.0, 7.0, 43.0, 36.0]),
        78 => Some([9.0, 36.0, 5.0, 56.0, 56.0]),
        79 => Some([10.0, 45.0, 5.0, 70.0, 70.0]),
        80 => Some([0.0, 0.0, 0.0, 0.0, 0.0]),
        _ => None,
    };
    match t {
        Some([a, b, c, d, e]) => (a, b, c, d, e),
        None => (0.0, 0.0, 0.0, 0.0, 0.0),
    }
}

impl OneCenterSpd {
    /// Build the one-center two-electron matrix for a 9-orbital element.
    pub fn build(elem: &Pm3Element) -> Self {
        let cmb = Combinatorics::new();
        let (repd, f0sd, g2sd, eisol_d) = build_repd(elem, &cmb);

        let mut w = vec![vec![0.0f64; NPAIR]; NPAIR];
        let set = |w: &mut Vec<Vec<f64>>, i: usize, j: usize, v: f64| {
            // MOPAC pair index is 1-based; convert to 0-based.
            w[i - 1][j - 1] = v;
            w[j - 1][i - 1] = v;
        };
        let (gss, gsp, gpp, gp2, hsp) = (elem.g_ss, elem.g_sp, elem.g_pp, elem.g_p2, elem.h_sp);
        // sp block (wstore, mndod.F90:2057-2083). ip=1 (1-based).
        set(&mut w, 1, 1, gss);
        let (ipx, ipy, ipz) = (3usize, 6usize, 10usize); // pair(px,px),(py,py),(pz,pz)
        for &ip_ in &[ipx, ipy, ipz] {
            set(&mut w, ip_, 1, gsp);
        }
        for &a in &[ipx, ipy, ipz] {
            for &b in &[ipx, ipy, ipz] {
                if a == b {
                    w[a - 1][b - 1] = gpp;
                } else {
                    w[a - 1][b - 1] = gp2;
                }
            }
        }
        // (sp|sp) = hsp at pair indices 2,4,7; (pp'|pp') = 0.5(gpp-gp2) at 5,8,9.
        for &p in &[2usize, 4, 7] {
            w[p - 1][p - 1] = hsp;
        }
        for &p in &[5usize, 8, 9] {
            w[p - 1][p - 1] = 0.5 * (gpp - gp2);
        }
        // d block: 243 (intij,intkl,intrep) entries.
        for i in 0..243 {
            set(&mut w, INTIJ[i], INTKL[i], repd[INTREP[i]]);
        }

        Self {
            w,
            eisol_d,
            f0sd,
            g2sd,
            repd,
        }
    }

    /// One-center integral `(a b | c d)` in eV over the 9 valence AOs.
    #[inline]
    pub fn get(&self, a: usize, b: usize, c: usize, d: usize) -> f64 {
        self.w[pack(a, b)][pack(c, d)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combinatorics_factorials() {
        let c = Combinatorics::new();
        assert_eq!(c.fx[1], 1.0); // 0!
        assert_eq!(c.fx[2], 1.0); // 1!
        assert_eq!(c.fx[4], 6.0); // 3!
        assert_eq!(c.b[5][2], 4.0); // C(4,1)
        assert_eq!(c.b[5][3], 6.0); // C(4,2)
    }

    #[test]
    fn one_center_sp_block_matches_parameters() {
        // A d-element still reproduces its Gss/Gsp/Gpp/Gp2/Hsp in the sp block.
        let params = crate::params::Pm3Parameters::standard().unwrap();
        if let Ok(zn) = params.element(30) {
            if zn.has_d() {
                let oc = OneCenterSpd::build(zn);
                assert!((oc.get(0, 0, 0, 0) - zn.g_ss).abs() < 1e-9); // (ss|ss)=Gss
                assert!((oc.get(1, 1, 0, 0) - zn.g_sp).abs() < 1e-9); // (px px|ss)=Gsp
                assert!((oc.get(1, 0, 1, 0) - zn.h_sp).abs() < 1e-9); // (px s|px s)=Hsp
                                                                      // All one-center d integrals finite.
                for a in 0..9 {
                    for b in 0..9 {
                        assert!(oc.get(a, a, b, b).is_finite());
                    }
                }
            }
        }
    }
}
