// SPDX-License-Identifier: GPL-3.0-or-later

// The STO overlap kernel signature mirrors its two shells' quantum numbers
// and exponents, which is clearer here than an artificial parameter object.
#![allow(clippy::too_many_arguments)]

//! General Slater-type diatomic overlap by numerical quadrature in prolate-spheroidal
//! coordinates. This works for any valence principal quantum number and is used for the
//! heavy PM3 elements (`n >= 4`), for which the closed-form analytic kernel in
//! [`crate::overlap`] (n <= 3) is not tabulated, and (extended to `l = 2`) for
//! the σ/π/δ d-shell overlaps of the spd path.
//!
//! The routine returns the same five local-frame quantities `S111, S211, S121, S221, S222`
//! the analytic kernel produces (with matching sign conventions), so the downstream
//! rotation/assembly is shared. It is validated to reproduce the analytic kernel for n <= 3.
//!
//! The quadrature is written generically over [`crate::dual::Scalar`], so instantiating at
//! `f64` gives the overlap value and at [`crate::dual::Dual`]/[`crate::dual2::Dual2`] (seeding
//! `r` on the interatomic displacement) gives its **exact closed-form** first/second radial
//! derivatives — the integrand is differentiated analytically point-by-point. The integration
//! domain (`xi_max`, the Gauss–Legendre nodes) is fixed from `r`'s *value*, which is exact up to
//! the boundary term `f(xi_max)·dxi_max/dr`; that term is negligible because `xi_max` is chosen
//! so the exponential integrand has already decayed to ~0 there. No finite differences.

use crate::dual::Scalar;

/// Local-frame Slater overlaps `(S111, S211, S121, S221, S222)` between shells on two atoms
/// at distance `r` (Bohr). `na`/`nb` are valence principal quantum numbers; `zsa`/`zpa` and
/// `zsb`/`zpb` are the s/p Slater exponents on atoms a and b. Generic over the scalar type.
pub fn slater_locals_numeric<S: Scalar>(
    nsa: u8,
    npa: u8,
    zsa: f64,
    zpa: f64,
    nsb: u8,
    npb: u8,
    zsb: f64,
    zpb: f64,
    r: S,
) -> (S, S, S, S, S) {
    // Sigma s-s, p_sigma-s, s-p_sigma, p_sigma-p_sigma; pi p-p.
    let zero = S::cst(0.0);
    let s_ss = overlap_sto(nsa, zsa, 0, 0, nsb, zsb, 0, r);
    let i_pss = if npa > 0 && zpa > 0.0 {
        overlap_sto(npa, zpa, 1, 0, nsb, zsb, 0, r)
    } else {
        zero
    };
    let i_sps = if npb > 0 && zpb > 0.0 {
        overlap_sto(nsa, zsa, 0, 0, npb, zpb, 1, r)
    } else {
        zero
    };
    let (i_pps, i_ppp) = if npa > 0 && npb > 0 && zpa > 0.0 && zpb > 0.0 {
        (
            overlap_sto(npa, zpa, 1, 0, npb, zpb, 1, r),
            overlap_sto(npa, zpa, 1, 1, npb, zpb, 1, r),
        )
    } else {
        (zero, zero)
    };
    // Match the analytic kernel's sign convention.
    (s_ss, i_pss, -i_sps, -i_pps, i_ppp)
}

/// Overlap of two Slater orbitals in the local diatomic frame (b at +z, distance `r`):
/// `<chi(na, la, m; za)_a | chi(nb, lb, m; zb)_b>` for magnetic number `m in {0,1,2}`.
/// `m_a` selects σ (0), π (1) or δ (2). Generic over the scalar type; only the
/// radial factor depends on `r`, so the angular parts and quadrature nodes are `f64` constants.
pub(crate) fn overlap_sto<S: Scalar>(
    na: u8,
    za: f64,
    la: u8,
    m_a: u8,
    nb: u8,
    zb: f64,
    lb: u8,
    r: S,
) -> S {
    if na == 0 || nb == 0 || za <= 0.0 || zb <= 0.0 {
        return S::cst(0.0);
    }
    let na_i = na as i32;
    let nb_i = nb as i32;
    let norm_a = slater_norm(na, za);
    let norm_b = slater_norm(nb, zb);

    // Angular normalization × φ-integral constant, c = k(la,m)·k(lb,m).
    // k values reproduce the real-spherical-harmonic normalization so that the
    // s/p entries match the analytic kernel and the d entries follow the same
    // convention (MNDO-d). Zero unless m ≤ min(la, lb).
    fn k(l: u8, m: u8) -> f64 {
        match (l, m) {
            (0, 0) => 0.5_f64.sqrt(),
            (1, 0) => 1.5_f64.sqrt(),
            (1, 1) => 0.75_f64.sqrt(),
            (2, 0) => 2.5_f64.sqrt(),
            (2, 1) => 3.75_f64.sqrt(),
            (2, 2) => (15.0_f64 / 16.0).sqrt(),
            _ => 0.0,
        }
    }
    let c = if m_a <= la.min(lb) {
        k(la, m_a) * k(lb, m_a)
    } else {
        0.0
    };
    if c == 0.0 {
        return S::cst(0.0);
    }

    let r_val = r.val();
    let half_r = r * 0.5; // scalar (carries derivatives)
    let half_r_val = 0.5 * r_val; // f64, for the (fixed) integration domain
                                  // Integration ranges: eta in [-1, 1]; xi in [1, xi_max] with the exponential decay
                                  // e^{-(za+zb) r xi / 2} captured generously. The domain is fixed from r's value.
    let p = (za + zb) * half_r_val;
    let xi_max = 1.0 + 60.0 / p.max(0.05);
    let (xn, xw) = gauss_legendre_mapped(48, 1.0, xi_max);
    let (en, ew) = gauss_legendre_mapped(40, -1.0, 1.0);

    let mut sum = S::cst(0.0);
    for (i, &xi) in xn.iter().enumerate() {
        for (j, &eta) in en.iter().enumerate() {
            let ra_val = half_r_val * (xi + eta);
            let rb_val = half_r_val * (xi - eta);
            if ra_val <= 0.0 || rb_val <= 0.0 {
                continue;
            }
            let cos_a = (1.0 + xi * eta) / (xi + eta);
            let cos_b = (xi * eta - 1.0) / (xi - eta);
            let ang = |cos: f64, l: u8, m: u8| -> f64 {
                let sin = (1.0 - cos * cos).max(0.0).sqrt();
                match (l, m) {
                    (0, _) => 1.0,
                    (1, 0) => cos,
                    (1, 1) => sin,
                    (2, 0) => (3.0 * cos * cos - 1.0) / 2.0,
                    (2, 1) => sin * cos,
                    (2, 2) => sin * sin,
                    _ => 0.0,
                }
            };
            let ang_a = ang(cos_a, la, m_a);
            let ang_b = ang(cos_b, lb, m_a);
            // Radial factor carries the r-dependence: ra, rb ∝ r.
            let ra = half_r * (xi + eta);
            let rb = half_r * (xi - eta);
            let radial = ra.powi(na_i - 1) * rb.powi(nb_i - 1) * (ra * (-za) - rb * zb).exp();
            let jac = xi * xi - eta * eta;
            let wcoef = xw[i] * ew[j] * ang_a * ang_b * jac; // f64
            sum = sum + radial * wcoef;
        }
    }
    half_r.powi(3) * sum * (norm_a * norm_b * c)
}

/// Radial normalization of a Slater orbital `N = (2 zeta)^{n+1/2} / sqrt((2n)!)`.
fn slater_norm(n: u8, zeta: f64) -> f64 {
    (2.0 * zeta).powf(n as f64 + 0.5) / factorial(2 * n as u64).sqrt()
}

fn factorial(n: u64) -> f64 {
    (1..=n).map(|k| k as f64).product::<f64>().max(1.0)
}

/// Gauss–Legendre nodes and weights on `[a, b]` (nodes computed by Newton iteration).
fn gauss_legendre_mapped(n: usize, a: f64, b: f64) -> (Vec<f64>, Vec<f64>) {
    let (nodes, weights) = gauss_legendre(n);
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let x = nodes.iter().map(|&t| mid + half * t).collect();
    let w = weights.iter().map(|&w| half * w).collect();
    (x, w)
}

fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let m = n.div_ceil(2);
    for i in 0..m {
        // Initial guess (Chebyshev).
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (p, dp) = legendre_p_dp(n, x);
            let dx = -p / dp;
            x += dx;
            if dx.abs() < 1.0e-15 {
                break;
            }
        }
        let (_, dp) = legendre_p_dp(n, x);
        let w = 2.0 / ((1.0 - x * x) * dp * dp);
        nodes[i] = -x;
        nodes[n - 1 - i] = x;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    (nodes, weights)
}

/// Legendre polynomial `P_n(x)` and derivative `P_n'(x)` via the recurrence.
fn legendre_p_dp(n: usize, x: f64) -> (f64, f64) {
    let mut p0 = 1.0;
    let mut p1 = x;
    if n == 0 {
        return (1.0, 0.0);
    }
    for k in 2..=n {
        let kf = k as f64;
        let p2 = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
        p0 = p1;
        p1 = p2;
    }
    let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
    (p1, dp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_legendre_integrates_polynomial() {
        // ∫_0^1 x^4 dx = 1/5.
        let (x, w) = gauss_legendre_mapped(8, 0.0, 1.0);
        let s: f64 = x.iter().zip(&w).map(|(&xi, &wi)| wi * xi.powi(4)).sum();
        assert!((s - 0.2).abs() < 1e-10);
    }

    #[test]
    fn ss_overlap_matches_closed_form() {
        // 1s|1s Slater overlap, equal exponents zeta=1: S = e^{-R}(1 + R + R^2/3).
        for &rr in &[1.0_f64, 2.0, 3.5] {
            let (s111, _, _, _, _) = slater_locals_numeric(1, 1, 1.0, 1.0, 1, 1, 1.0, 1.0, rr);
            let exact = (-rr).exp() * (1.0 + rr + rr * rr / 3.0);
            assert!((s111 - exact).abs() < 1e-7, "R={rr}: {s111} vs {exact}");
        }
    }

    #[test]
    fn twos_overlap_matches_closed_form() {
        // 2s|2s Slater overlap, equal exponents, p = zeta*R (Mulliken 1949):
        // S = e^{-p}(1 + p + (4/9)p^2 + (1/9)p^3 + (1/45)p^4).
        for &rr in &[1.5_f64, 3.0] {
            let z = 1.3_f64;
            let (s111, _, _, _, _) = slater_locals_numeric(2, 2, z, z, 2, 2, z, z, rr);
            let p = z * rr;
            let exact =
                (-p).exp() * (1.0 + p + 4.0 / 9.0 * p * p + p * p * p / 9.0 + p * p * p * p / 45.0);
            assert!((s111 - exact).abs() < 5e-4, "R={rr}: {s111} vs {exact}");
        }
    }
}
