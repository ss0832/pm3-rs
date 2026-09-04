// SPDX-License-Identifier: GPL-3.0-or-later

//! Direct real-space lattice sums, for **testing only**.
//!
//! An Ewald implementation cannot be validated against its own algebra: the split into real
//! and reciprocal parts, the self-term and the neutralizing background all have to cancel
//! against each other exactly, and a sign or factor error in one is invisible until it is
//! compared with something that shares none of that structure. This module is that something.
//! It evaluates the same lattice sums by brute force, one image at a time, with no splitting.
//!
//! Brute force is only *conditionally* convergent, which is the whole reason Ewald exists, so
//! each dimensionality gets the summation order that makes it converge:
//!
//! * **3D** — expanding cubic shells of whole cells. For a charge-neutral cell the monopole
//!   term cancels within each shell and the remainder falls off fast enough to converge; the
//!   result is the "spherical, vacuum boundary" limit, which differs from the tinfoil
//!   convention Ewald computes by the surface-dipole term added explicitly in
//!   [`surface_dipole_energy`].
//! * **2D** — expanding rings of cells in the periodic plane. Absolutely convergent for a
//!   neutral cell with no net dipole normal to the plane; the normal-dipole term is again
//!   explicit.
//! * **1D** — expanding pairs of cells along the axis. The per-cell-group interaction decays
//!   as `1/n²` once the monopole cancels, so this converges absolutely.
//!
//! These sums are `O(N² · images)` and are never used outside tests.

use crate::cell::Cell;
use crate::constants::PM3_EV;
use crate::math::Vec3;

/// One point charge in the reference cell.
#[derive(Clone, Copy, Debug)]
pub struct RefSite {
    pub position: Vec3,
    pub charge: f64,
}

/// Direct lattice sum of the electrostatic energy per cell, in eV.
///
/// `shells` is how many cells to include in each periodic direction; the caller raises it
/// until the value stops moving. Returns the **spherical/ring/segment-truncated** sum, i.e.
/// the vacuum-boundary convention. Add [`surface_dipole_energy`] to compare against a tinfoil
/// Ewald result.
pub fn direct_energy(cell: &Cell, sites: &[RefSite], shells: i32) -> f64 {
    let periodic = cell.periodic_indices();
    let mut energy = 0.0;
    for translation in enumerate_shells(cell, &periodic, shells) {
        let zero = translation.norm2() == 0.0;
        for (i, a) in sites.iter().enumerate() {
            for (j, b) in sites.iter().enumerate() {
                if zero && i == j {
                    continue;
                }
                let d = b.position + translation - a.position;
                let r = d.norm();
                if r <= 0.0 {
                    continue;
                }
                energy += 0.5 * a.charge * b.charge / r;
            }
        }
    }
    energy * PM3_EV
}

/// The potential `∂E/∂q_i` at each site (eV per elementary charge), by the same direct sum.
pub fn direct_potentials(cell: &Cell, sites: &[RefSite], shells: i32) -> Vec<f64> {
    let periodic = cell.periodic_indices();
    let mut potentials = vec![0.0; sites.len()];
    for translation in enumerate_shells(cell, &periodic, shells) {
        let zero = translation.norm2() == 0.0;
        for (i, a) in sites.iter().enumerate() {
            for (j, b) in sites.iter().enumerate() {
                if zero && i == j {
                    continue;
                }
                let d = b.position + translation - a.position;
                let r = d.norm();
                if r > 0.0 {
                    potentials[i] += b.charge / r;
                }
            }
        }
    }
    for potential in &mut potentials {
        *potential *= PM3_EV;
    }
    potentials
}

/// Surface-dipole ("shape") term relating the two boundary conventions, in eV.
///
/// A conditionally convergent lattice sum has no single answer: the value depends on the shape
/// of the region summed and on what surrounds it. Ewald's reciprocal sum with the `G = 0` term
/// dropped is the **tinfoil** limit (the crystal embedded in a perfect conductor); the direct
/// sums here build a finite cluster in vacuum. The two differ by the depolarizing energy of
/// the cluster's own surface charge,
///
/// ```text
/// E_surface = (2π / 3V) |Σ_i q_i r_i|²        (3D, spherical cluster)
/// E_surface = (2π / A)  (Σ_i q_i z_i)²        (2D, slab; z normal to the plane)
/// ```
///
/// For 1D the corresponding term vanishes for the segment-truncated sum. Adding this to a
/// tinfoil Ewald energy gives the vacuum-cluster value the direct sums converge to.
pub fn surface_dipole_energy(cell: &Cell, sites: &[RefSite]) -> f64 {
    let dipole = sites
        .iter()
        .fold(Vec3::zero(), |acc, s| acc + s.position * s.charge);
    let periodic = cell.periodic_indices();
    let measure = cell.measure();
    let value = match periodic.as_slice() {
        [_, _, _] => std::f64::consts::TAU / 3.0 * dipole.norm2() / measure,
        [i, j] => {
            // Only the component normal to the periodic plane contributes.
            let normal = cell.vector(*i).cross(cell.vector(*j)).normalized();
            std::f64::consts::TAU * dipole.dot(normal).powi(2) / measure
        }
        _ => 0.0,
    };
    value * PM3_EV
}

/// The charge-weighted **phased** lattice sum and its first two derivatives in the site
/// positions, by direct summation over whole cells.
///
/// This is the oracle for [`crate::pbc::phased::ewald_phased`], which until now had no
/// independent check: its tests confirm that it reduces to the unphased sum at a reciprocal
/// lattice vector, that its derivatives are the derivatives of its own value, and that reversing
/// `q` conjugates it — all true of a sum with a wrong prefactor.
///
/// # Why this sums cells and not pairs
///
/// The per-displacement object `Σ_T e^{iq·T}/|d + T|` **cannot be summed directly in 3D**. A
/// spherical truncation at radius `R` gives `∫_{|r|<R} e^{iq·r}/r d³r = (4π/q²)(1 − cos qR)`: the
/// partial sums oscillate with amplitude `4π/(Vq²)` and never settle. That is not slow
/// convergence, it is no convergence, and averaging a divergent quantity would hide it.
///
/// Summed as whole cells the monopole cancels within each cell, the leading cell-to-cell term is
/// dipole–dipole `1/T³`, and the phase supplies the rest: a shell at radius `R` contributes
/// `R²·(1/R³)·O(1/qR) = O(1/qR²)`, which converges absolutely. So the summation unit here is the
/// cell bundle, and there is deliberately no per-displacement entry point to misuse.
///
/// This also means the oracle is **weakest where `q` is smallest** — the `1/q` in that estimate
/// is real, and it is worst in exactly the long-wavelength region where the dynamical matrix is
/// most delicate. Compare at a `q` well away from `Γ` and from the zone boundary, and let the
/// exact identities (`D(q + G) = D(q)`, `D(−q) = D(q)*`, the acoustic sum rule) carry the rest.
///
/// # Why requiring a neutral cell costs no coverage
///
/// A charged cell has no per-`T` monopole cancellation, so its bundle falls off as `Q²/T` and the
/// sum is only conditionally summable. Only neutral site sets are accepted here.
///
/// That restricts the *oracle*, not what the oracle covers.
/// [`crate::pbc::phased::ewald_phased`] takes displacements and returns kernels; **charges never
/// enter it**. The caller weights the kernels afterwards, so a charged cell and a neutral one ask
/// it for exactly the same numbers. Validating the kernel through a neutral contraction validates
/// it for every contraction — there is no charge-dependent branch to leave untested.
#[derive(Clone, Debug)]
pub struct PhasedSum {
    /// `Σ_{s,t} q_s q_t Σ'_T e^{iq·T}/|r_t + T − r_s|`, as `[real, imaginary]`.
    pub value: [f64; 2],
    /// `∂(value)/∂r_s`, per site.
    pub gradient: Vec<[[f64; 2]; 3]>,
    /// `∂²(value)/∂r_s∂r_u`, per ordered site pair.
    pub hessian: Vec<Vec<[[[f64; 2]; 3]; 3]>>,
}

impl PhasedSum {
    fn zeros(n: usize) -> Self {
        Self {
            value: [0.0; 2],
            gradient: vec![[[0.0; 2]; 3]; n],
            hessian: vec![vec![[[[0.0; 2]; 3]; 3]; n]; n],
        }
    }
}

/// # How far it reaches, measured
///
/// On a 6 Bohr cube with four charges, against `ewald_phased`:
///
/// | `q` (fractional) | thirty-two shells | with [`extrapolate_shells`] |
/// |---|---|---|
/// | `(0.31, −0.23, 0.17)` — generic interior | `3·10⁻⁸` | not needed |
/// | `(0.5, 0, 0)` — zone boundary | `4·10⁻³`, still drifting | `8·10⁻⁵` |
/// | `(0.05, 0.03, −0.02)` — near `Γ` | `1·10⁻⁴`, erratic | — |
///
/// So a generic interior `q` is where this is a sharp instrument, and that is where the tests
/// use it. A Gaussian convergence factor with `η → 0` extrapolation was tried for the hard
/// regions and is **not** provided: it was five orders of magnitude worse than the plain sum at a
/// generic `q` and gave the wrong sign near `Γ`, because the damping length and the `1/q`
/// correlation length collide there. Extrapolating in the shell count works and is what is here.
///
/// See [`PhasedSum`]. `shells` is how many cells to include in each periodic direction; the
/// caller raises it until the value stops moving. Neutral cells only.
#[allow(clippy::needless_range_loop)] // Cartesian tensor indices, as elsewhere in this crate
pub fn direct_phased_sum(cell: &Cell, sites: &[RefSite], q: Vec3, shells: i32) -> PhasedSum {
    let total: f64 = sites.iter().map(|s| s.charge).sum();
    assert!(
        total.abs() < 1.0e-12,
        "the phased direct sum needs a neutral cell; this one carries {total}. Without the \
         per-cell monopole cancellation the bundle falls off as Q²/T and the series is only \
         conditionally summable. The kernel being validated does not depend on charges, so this \
         costs no coverage — see the type note."
    );

    let periodic = cell.periodic_indices();
    let n = sites.len();
    let mut out = PhasedSum::zeros(n);

    for translation in enumerate_shells(cell, &periodic, shells) {
        // The phase is a property of the cell, so it multiplies the finished bundle rather than
        // each pair — which is the whole point of grouping this way.
        let angle = q.dot(translation);
        let (sin, cos) = angle.sin_cos();
        let phase = [cos, sin];

        let mut bundle = 0.0;
        let mut bundle_gradient = vec![[0.0; 3]; n];
        let mut bundle_hessian = vec![vec![[[0.0; 3]; 3]; n]; n];

        for (s, site_s) in sites.iter().enumerate() {
            for (t, site_t) in sites.iter().enumerate() {
                let d = site_t.position + translation - site_s.position;
                let r = d.norm();
                if r < 1.0e-12 {
                    continue;
                }
                let w = site_s.charge * site_t.charge;
                let (i1, i3, i5) = (1.0 / r, 1.0 / (r * r * r), 1.0 / r.powi(5));
                bundle += w * i1;

                // d = r_t + T − r_s, so ∂/∂r_s = −∂/∂d and ∂/∂r_t = +∂/∂d.
                // ∂(1/r)/∂d_a = −d_a/r³, hence ∂/∂r_s = +w d/r³.
                let v = [d.x, d.y, d.z];
                for a in 0..3 {
                    bundle_gradient[s][a] += w * v[a] * i3;
                    bundle_gradient[t][a] -= w * v[a] * i3;
                }
                // ∂²(1/r)/∂d_a∂d_b = 3 d_a d_b/r⁵ − δ_ab/r³. The r_s and r_t derivatives each
                // carry a sign from ∂d/∂r, so the diagonal blocks take it and the mixed ones
                // take its negative.
                for a in 0..3 {
                    for b in 0..3 {
                        let delta = if a == b { 1.0 } else { 0.0 };
                        let h = w * (3.0 * v[a] * v[b] * i5 - delta * i3);
                        bundle_hessian[s][s][a][b] += h;
                        bundle_hessian[t][t][a][b] += h;
                        bundle_hessian[s][t][a][b] -= h;
                        bundle_hessian[t][s][a][b] -= h;
                    }
                }
            }
        }

        for (slot, term) in out.value.iter_mut().zip(phase) {
            *slot += term * bundle;
        }
        for (s, row) in bundle_gradient.iter().enumerate() {
            for (a, value) in row.iter().enumerate() {
                out.gradient[s][a][0] += phase[0] * value;
                out.gradient[s][a][1] += phase[1] * value;
            }
        }
        for (s, row) in bundle_hessian.iter().enumerate() {
            for (u, block) in row.iter().enumerate() {
                for a in 0..3 {
                    for b in 0..3 {
                        out.hessian[s][u][a][b][0] += phase[0] * block[a][b];
                        out.hessian[s][u][a][b][1] += phase[1] * block[a][b];
                    }
                }
            }
        }
    }
    out
}

/// Richardson extrapolation of [`direct_phased_sum`]'s value to an infinite shell count.
///
/// Near the zone boundary the truncation error is a smooth `A/n` rather than an oscillation —
/// the partial sums approach from one side — so one Richardson step in `1/n` removes most of it.
/// Measured at `q = b₁/2` on the 6 Bohr cube: `4·10⁻³` raw at thirty-two shells, `8·10⁻⁵` after.
///
/// Only the value is extrapolated. The gradient and Hessian bundles fall off two and three powers
/// faster (`1/T⁴` and `1/T⁵` against the value's `1/T³`), so they converge without help.
pub fn extrapolate_shells(
    cell: &Cell,
    sites: &[RefSite],
    q: Vec3,
    coarse: i32,
    fine: i32,
) -> [f64; 2] {
    assert!(fine > coarse, "the fine shell count must be the larger one");
    let a = direct_phased_sum(cell, sites, q, coarse).value;
    let b = direct_phased_sum(cell, sites, q, fine).value;
    // v(n) = v∞ − A/n  ⇒  v∞ = v(f) + (v(f) − v(c)) / (f/c − 1).
    let weight = 1.0 / (fine as f64 / coarse as f64 - 1.0);
    [b[0] + (b[0] - a[0]) * weight, b[1] + (b[1] - a[1]) * weight]
}

/// Lattice translations out to `shells` cells in each periodic direction, in an order that
/// groups whole shells together so the monopole cancellation happens shell by shell.
fn enumerate_shells(cell: &Cell, periodic: &[usize], shells: i32) -> Vec<Vec3> {
    let mut out = Vec::new();
    let ranges: Vec<i32> = (0..3)
        .map(|i| if periodic.contains(&i) { shells } else { 0 })
        .collect();
    for n0 in -ranges[0]..=ranges[0] {
        for n1 in -ranges[1]..=ranges[1] {
            for n2 in -ranges[2]..=ranges[2] {
                out.push(
                    cell.vector(0) * n0 as f64
                        + cell.vector(1) * n1 as f64
                        + cell.vector(2) * n2 as f64,
                );
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rocksalt Madelung constant is the classic closed-form check on any lattice sum:
    /// the electrostatic energy per ion pair of an alternating-charge cubic lattice is
    /// `-M q²/a` with `M = 1.747564594633...` and `a` the nearest-neighbour distance.
    ///
    /// Summing whole neutral cells in expanding cubic shells is what makes this converge; the
    /// same sum done ion-by-ion does not converge at all.
    #[test]
    fn rocksalt_direct_sum_approaches_the_madelung_constant() {
        const MADELUNG: f64 = 1.747_564_594_633_2;
        let a = 1.0; // nearest-neighbour distance; the conventional cube edge is 2a
        let cell = Cell::cubic(2.0 * a).unwrap();
        // Conventional 8-ion NaCl cell: charge = (-1)^(i+j+k) at each corner of the half-cube.
        let mut sites = Vec::new();
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    sites.push(RefSite {
                        position: Vec3::new(i as f64 * a, j as f64 * a, k as f64 * a),
                        charge: if (i + j + k) % 2 == 0 { 1.0 } else { -1.0 },
                    });
                }
            }
        }
        // Energy per ion = -M/(2a) in Hartree-with-Bohr units; the cell holds 8 ions.
        let expected = -MADELUNG / a * PM3_EV * 8.0 / 2.0;
        let mut previous = f64::NAN;
        for shells in [4, 8, 16] {
            let energy = direct_energy(&cell, &sites, shells);
            previous = energy;
            let relative = (energy - expected).abs() / expected.abs();
            if shells == 16 {
                assert!(
                    relative < 2.0e-3,
                    "shells={shells}: {energy} vs {expected} (rel {relative:.2e})"
                );
            }
        }
        assert!(previous.is_finite());
    }

    /// `E = ½ Σ q_i φ_i` must hold for the direct sums, which is what lets the Ewald
    /// potentials be validated against `direct_potentials` independently of the energy.
    #[test]
    fn energy_is_half_the_charge_potential_product() {
        let cell = Cell::new(
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(0.7, 5.4, 0.0),
            Vec3::new(0.0, 0.4, 6.3),
            [true, true, true],
        )
        .unwrap();
        let sites = vec![
            RefSite {
                position: Vec3::new(0.3, 0.4, 0.5),
                charge: 0.7,
            },
            RefSite {
                position: Vec3::new(2.9, 1.6, 3.1),
                charge: -0.45,
            },
            RefSite {
                position: Vec3::new(4.4, 3.8, 1.2),
                charge: -0.25,
            },
        ];
        let energy = direct_energy(&cell, &sites, 10);
        let potentials = direct_potentials(&cell, &sites, 10);
        let from_potentials: f64 = 0.5
            * sites
                .iter()
                .zip(&potentials)
                .map(|(s, p)| s.charge * p)
                .sum::<f64>();
        assert!(
            (energy - from_potentials).abs() < 1.0e-9 * energy.abs().max(1.0),
            "{energy} vs {from_potentials}"
        );
    }
}
