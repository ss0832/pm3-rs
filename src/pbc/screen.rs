// SPDX-License-Identifier: GPL-3.0-or-later

//! The short-range half of the periodic Coulomb split: what is left of PM3's two-center
//! integrals once the point-charge lattice sum has taken the long-range part.
//!
//! [`crate::pbc::ewald`] sums the point-charge model of the cell over every image. That model
//! carries the correct `1/R`, `1/R²`, `1/R³` multipole tail but it is not PM3: the real
//! integrals are Klopman–Ohno screened, `1/√(R² + a)` rather than `1/R`, and they are evaluated
//! in the local diatomic frame rather than as a global point-charge sum. This module supplies
//! the difference,
//!
//! ```text
//! ΔW(R) = f(R) · [ W_NDDO(R) − W_point(R) ]
//! ```
//!
//! for every two-center quantity at once — the two-electron table, both electron–core blocks,
//! and the core–core monopole — so the existing Fock machinery can consume corrected tables and
//! never has to know that a lattice sum is happening underneath.
//!
//! # Why the switch is there, and what it removes
//!
//! `f` is the same `C²` quintic switch the CPHF cutoff uses, `1` below `r_on` and `0` above
//! `r_off`. It is not a convenience: without it the correction has a `−a/2R³` tail from the
//! Klopman–Ohno form whose 3D lattice sum **diverges logarithmically**.
//!
//! That tail is an artefact rather than physics. Two spherical charge distributions interact
//! exactly as `q_A q_B / R` once they stop overlapping — Gauss's law — while
//! `1/√(R² + a)` is an interpolation formula fitted at short range that never quite reaches
//! `1/R`. Switching the correction off therefore *restores* the correct long-range behaviour
//! rather than approximating it away.
//!
//! What it costs is a documented departure from molecular PM3 at long range. For a carbon–oxygen
//! monopole pair, `a = (ρ₀ᶜ + ρ₀ᴼ)² ≈ 4.3 Bohr²`, so the discarded term is `−a/2R³ · e²` — about
//! 5.5 meV at the default 22 Bohr cutoff, falling as `R⁻³`. That exponent is measured rather
//! than assumed (`the_truncated_correction_is_small_and_decaying`), because it is the entire
//! justification for switching at all.
//!
//! Pushing `r_off` out is the obvious lever but an expensive one: the correction costs a full
//! pair table per neighbour and the neighbour count grows as `r_off³`. The cheap version is to
//! notice that beyond ~15 Bohr only the *monopole–monopole* difference survives — every higher
//! multipole difference decays as `R⁻⁴` or faster — so a second, much longer cutoff carrying a
//! single scalar per pair buys most of the accuracy for a fraction of the work. That is a
//! performance refinement, not a correctness one, and is left for the optimization milestone.
//!
//! The frame difference — global point charges versus local-frame multipoles — is `O(1/R⁴)`,
//! absolutely convergent, and is carried by the same correction with nothing further needed.

use crate::constants::PM3_EV;
use crate::integrals::{pack, pair_two_electron_g, PairTwoElec};
use crate::math::Vec3;
use crate::params::Pm3Element;
use crate::pbc::multipole::{packed_index, AtomSites};

/// Default distance (Bohr) at which the Klopman–Ohno correction starts being switched off.
pub const DEFAULT_SWITCH_ON: f64 = 18.0;

/// Default distance (Bohr) beyond which only the point-charge lattice sum survives.
pub const DEFAULT_SWITCH_OFF: f64 = 22.0;

/// How wide the fade is, for a caller that names only where the correction should end.
///
/// A hard cut would put a step in the energy and a delta function in the force, so the two radii
/// always travel together; this is the gap the defaults keep between them.
pub const DEFAULT_SWITCH_WIDTH: f64 = DEFAULT_SWITCH_OFF - DEFAULT_SWITCH_ON;

/// The distance window over which the Klopman–Ohno correction is handed back to the lattice sum.
///
/// The two radii always travel together — a correction switched off at one radius and evaluated
/// out to another is neither one model nor the other — so they are one value rather than two
/// arguments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwitchRange {
    /// Below this the full NDDO integral is used (Bohr).
    pub on: f64,
    /// Above this only the point-charge lattice sum remains (Bohr).
    pub off: f64,
}

impl Default for SwitchRange {
    fn default() -> Self {
        Self {
            on: DEFAULT_SWITCH_ON,
            off: DEFAULT_SWITCH_OFF,
        }
    }
}

impl SwitchRange {
    /// The switch value at distance `r`.
    #[inline]
    pub fn at(&self, r: f64) -> f64 {
        switch(r, self.on, self.off)
    }

    /// The switch value in a generic scalar, so its **derivative** travels with it.
    ///
    /// The switch multiplies the short-range correction, so `f'(r)` is part of the force
    /// wherever the switch is active. Evaluating it in `f64` and multiplying afterwards would
    /// silently drop that term, leaving a force that is wrong only in the switching shell — a
    /// discrepancy small enough to look like noise and large enough to break a finite-difference
    /// check.
    #[inline]
    pub fn at_g<S: crate::dual::Scalar>(&self, r: S) -> S {
        if r.val() <= self.on {
            S::cst(1.0)
        } else if r.val() >= self.off {
            S::cst(0.0)
        } else {
            let x = (r - self.on) / (self.off - self.on);
            let x3 = x * x * x;
            S::cst(1.0) - (x3 * 10.0 - x3 * x * 15.0 + x3 * x * x * 6.0)
        }
    }
}

/// `C²` quintic switch: `1` at and below `r_on`, `0` at and above `r_off`, with vanishing first
/// and second derivatives at both ends so neither the force nor the Hessian sees a kink.
#[inline]
pub fn switch(r: f64, r_on: f64, r_off: f64) -> f64 {
    if r <= r_on {
        1.0
    } else if r >= r_off {
        0.0
    } else {
        let x = (r - r_on) / (r_off - r_on);
        1.0 - (10.0 * x.powi(3) - 15.0 * x.powi(4) + 6.0 * x.powi(5))
    }
}

/// The point-charge model of one atom pair's two-center Coulomb terms, in the same layout the
/// NDDO kernels use so the two can be subtracted term by term.
pub type PointPair = PointPairG<f64>;

#[derive(Clone, Debug)]
pub struct PointPairG<S: crate::dual::Scalar> {
    /// Packed two-electron table, `npack_i × npack_j`, matching [`PairTwoElec::w`].
    pub w: Vec<S>,
    /// Electron(A)–core(B) attraction, `n_orb_a × n_orb_a`.
    pub e1b: Vec<S>,
    /// Electron(B)–core(A) attraction, `n_orb_b × n_orb_b`.
    pub e2a: Vec<S>,
    /// Core(A)–core(B) monopole repulsion (eV).
    pub core_core: S,
    pub norb_i: usize,
    pub norb_j: usize,
    pub npack_i: usize,
    pub npack_j: usize,
}

/// Evaluate the point-charge model for the pair `(a, b)` separated by `dvec = r_b − r_a`.
///
/// This is the quantity the Ewald sum is responsible for, computed here for one pair so it can
/// be removed from the short-range correction. Getting it from the same [`AtomSites`] the Ewald
/// uses — rather than from an independent formula — is what makes the split exact rather than
/// approximate: whatever this returns is precisely what the lattice sum already counted.
///
/// **Atom order matters.** [`crate::integrals::pair_two_electron_g`] only has a `heavy/light`
/// branch, not a `light/heavy` one: given an `s`-only atom first and an `sp` atom second it
/// falls into the `sp/sp` path and returns a 10×10 table with zero rows. `crate::hamiltonian`
/// therefore orders every pair heavy-first, and so must callers here, or the two tables will not
/// even have the same shape.
pub fn point_pair(
    sites_a: &AtomSites,
    core_a: f64,
    sites_b: &AtomSites,
    core_b: f64,
    dvec: Vec3,
) -> PointPair {
    point_pair_g::<f64>(sites_a, core_a, sites_b, core_b, [dvec.x, dvec.y, dvec.z])
}

/// [`point_pair`] in a generic scalar, so seeding `dvec` yields the model's exact derivatives.
///
/// The site offsets are fixed geometry, so every distance is `|dvec + Δ|` with `Δ` constant —
/// which is why the whole point model differentiates with no extra machinery.
pub fn point_pair_g<S: crate::dual::Scalar>(
    sites_a: &AtomSites,
    core_a: f64,
    sites_b: &AtomSites,
    core_b: f64,
    dvec: [S; 3],
) -> PointPairG<S> {
    debug_assert!(
        sites_a.n_orb >= sites_b.n_orb,
        "point_pair needs the heavier atom first, matching the integral kernel's branches"
    );
    let (na, nb) = (sites_a.n_orb, sites_b.n_orb);
    let (npack_i, npack_j) = (na * (na + 1) / 2, nb * (nb + 1) / 2);

    // Distance between site `i` of A (at the origin) and site `j` of B (at `dvec`).
    let separation = |i: usize, j: usize| -> S {
        let delta = sites_b.offsets[j] - sites_a.offsets[i];
        let (x, y, z) = (dvec[0] + delta.x, dvec[1] + delta.y, dvec[2] + delta.z);
        (x * x + y * y + z * z).sqrt()
    };

    let mut w = vec![S::cst(0.0); npack_i.max(1) * npack_j.max(1)];
    for mu in 0..na {
        for nu in 0..=mu {
            let terms_a = &sites_a.pair_weights[packed_index(nu, mu)];
            for la in 0..nb {
                for si in 0..=la {
                    let terms_b = &sites_b.pair_weights[packed_index(si, la)];
                    let mut sum = S::cst(0.0);
                    for ta in terms_a {
                        for tb in terms_b {
                            sum = sum
                                + separation(ta.site, tb.site).recip() * (ta.weight * tb.weight);
                        }
                    }
                    w[pack(mu, nu) * npack_j + pack(la, si)] = sum * PM3_EV;
                }
            }
        }
    }

    // Electron–core: atom A's orbital-pair configuration against atom B's bare core monopole.
    // The minus sign is the electron's, matching `crate::integrals::core_attraction_g`.
    let mut e1b = vec![S::cst(0.0); na * na];
    for mu in 0..na {
        for nu in 0..na {
            let mut sum = S::cst(0.0);
            for ta in &sites_a.pair_weights[packed_index(nu, mu)] {
                sum = sum + separation(ta.site, 0).recip() * ta.weight;
            }
            e1b[mu * na + nu] = sum * (-core_b * PM3_EV);
        }
    }
    let mut e2a = vec![S::cst(0.0); nb * nb];
    for la in 0..nb {
        for si in 0..nb {
            let mut sum = S::cst(0.0);
            for tb in &sites_b.pair_weights[packed_index(si, la)] {
                sum = sum + separation(0, tb.site).recip() * tb.weight;
            }
            e2a[la * nb + si] = sum * (-core_a * PM3_EV);
        }
    }

    PointPairG {
        w,
        e1b,
        e2a,
        core_core: separation(0, 0).recip() * (core_a * core_b * PM3_EV),
        norb_i: na,
        norb_j: nb,
        npack_i,
        npack_j,
    }
}

/// The screened two-center tables with the point-charge model removed and the switch applied.
///
/// What remains is short-ranged by construction and is summed directly over lattice images; the
/// part that was removed is what [`crate::pbc::ewald`] sums over all of them.
#[derive(Clone, Debug)]
pub struct ScreenedPair {
    /// `f · (W_NDDO − W_point)`, packed like [`PairTwoElec::w`].
    pub w: Vec<f64>,
    /// `f · (e1b_NDDO − e1b_point)`, `n_orb_a × n_orb_a`.
    pub e1b: Vec<f64>,
    /// `f · (e2a_NDDO − e2a_point)`, `n_orb_b × n_orb_b`.
    pub e2a: Vec<f64>,
    pub norb_i: usize,
    pub norb_j: usize,
    pub npack_i: usize,
    pub npack_j: usize,
}

/// Build the screened correction for one atom pair.
///
/// `te` is the ordinary NDDO pair integral for the same `dvec`, so callers that already have it
/// (every SCF iteration does) do not pay for it twice.
pub fn screened_pair(
    te: &PairTwoElec,
    sites_a: &AtomSites,
    core_a: f64,
    sites_b: &AtomSites,
    core_b: f64,
    dvec: Vec3,
    range: SwitchRange,
) -> ScreenedPair {
    let f = range.at(dvec.norm());
    let point = point_pair(sites_a, core_a, sites_b, core_b, dvec);
    let (na, nb) = (te.norb_i, te.norb_j);
    let mut w = vec![0.0; te.w.len()];
    for (index, value) in w.iter_mut().enumerate() {
        *value = f * (te.w[index] - point.w[index]);
    }
    let mut e1b = vec![0.0; na * na];
    for mu in 0..na {
        for nu in 0..na {
            e1b[mu * na + nu] = f * (te.e1b[mu][nu] - point.e1b[mu * na + nu]);
        }
    }
    let mut e2a = vec![0.0; nb * nb];
    for la in 0..nb {
        for si in 0..nb {
            e2a[la * nb + si] = f * (te.e2a[la][si] - point.e2a[la * nb + si]);
        }
    }
    ScreenedPair {
        w,
        e1b,
        e2a,
        norb_i: na,
        norb_j: nb,
        npack_i: te.npack_i,
        npack_j: te.npack_j,
    }
}

/// The core–core monopole correction: PM3's `G_AB Z_A Z_B` minus the bare `Z_A Z_B / R` the
/// lattice sum already has, switched off with everything else.
///
/// PM3's exponential and Gaussian core–core terms are **not** included here: they decay
/// exponentially, are not part of the point-charge model, and are summed directly over images.
pub fn screened_core_core(ei: &Pm3Element, ej: &Pm3Element, r: f64, range: SwitchRange) -> f64 {
    let rho = ei.po[9] + ej.po[9];
    let screened = PM3_EV / (r * r + rho * rho).sqrt();
    let point = PM3_EV / r;
    range.at(r) * (screened - point) * ei.core_charge * ej.core_charge
}

/// Convenience: the NDDO pair integral for `dvec`, in the same convention the periodic code uses
/// (atom `a` first, atom `b` displaced by `dvec`).
pub fn nddo_pair(ei: &Pm3Element, ej: &Pm3Element, dvec: Vec3) -> PairTwoElec {
    pair_two_electron_g::<f64>(ei, ej, [dvec.x, dvec.y, dvec.z])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Pm3Parameters;

    fn unscreened(elem: &Pm3Element) -> Pm3Element {
        let mut out = elem.clone();
        out.rho0 = 0.0;
        out.rho1 = 0.0;
        out.rho2 = 0.0;
        out.po = [0.0; 10];
        out
    }

    /// Where one partner is a bare monopole the point model must reproduce the **unscreened**
    /// kernel exactly, whatever the other atom's configuration is. That covers the electron–core
    /// blocks and the core–core term for every pair, and the two-electron table whenever one
    /// atom is `s`-only.
    ///
    /// The electron–core check matters on its own: MOPAC's `spcore` uses a different additive
    /// set (`po`, `ddp`) from the electron–electron path, and only because `derive_multipoles`
    /// makes `ddp[2] = dd` and `ddp[3]/√2 = qq` for a main-group sp element do the two share one
    /// geometry. If that ever stopped holding, this is where it would show.
    ///
    /// Two `sp` atoms are a different matter — see
    /// `sp_pair_quadrupole_terms_agree_asymptotically` — because even on the interatomic axis
    /// the local frame is only fixed up to a rotation about it, and a four-corner point set is
    /// not invariant under that rotation beyond its quadrupole moment.
    #[test]
    fn point_model_is_exact_against_a_monopole_partner() {
        let params = Pm3Parameters::standard().unwrap();
        let mut failures = Vec::new();
        for (za, zb) in [(6u8, 1u8), (8, 1), (16, 1), (17, 1), (1, 1)] {
            let ea = params.element(za).unwrap();
            let eb = params.element(zb).unwrap();
            let sites_a = AtomSites::build(ea).unwrap();
            let sites_b = AtomSites::build(eb).unwrap();
            for axis in 0..3 {
                for distance in [4.0_f64, 6.5] {
                    let mut components = [0.0; 3];
                    components[axis] = distance;
                    let dvec = Vec3::new(components[0], components[1], components[2]);
                    let te = nddo_pair(&unscreened(ea), &unscreened(eb), dvec);
                    let point =
                        point_pair(&sites_a, ea.core_charge, &sites_b, eb.core_charge, dvec);

                    for (index, kernel) in te.w.iter().enumerate() {
                        if (kernel - point.w[index]).abs() > 1.0e-9 * kernel.abs().max(1.0) {
                            failures.push(format!(
                                "Z{za}/Z{zb} axis {axis} r={distance}: w[{index}] kernel \
                                 {kernel:.10} vs point {:.10}",
                                point.w[index]
                            ));
                        }
                    }
                    for mu in 0..te.norb_i {
                        for nu in 0..te.norb_i {
                            let kernel = te.e1b[mu][nu];
                            let modelled = point.e1b[mu * te.norb_i + nu];
                            if (kernel - modelled).abs() > 1.0e-9 * kernel.abs().max(1.0) {
                                failures.push(format!(
                                    "Z{za}/Z{zb} axis {axis} r={distance}: e1b[{mu}][{nu}] \
                                     kernel {kernel:.10} vs point {modelled:.10}"
                                ));
                            }
                        }
                    }
                    for la in 0..te.norb_j {
                        for si in 0..te.norb_j {
                            let kernel = te.e2a[la][si];
                            let modelled = point.e2a[la * te.norb_j + si];
                            if (kernel - modelled).abs() > 1.0e-9 * kernel.abs().max(1.0) {
                                failures.push(format!(
                                    "Z{za}/Z{zb} axis {axis} r={distance}: e2a[{la}][{si}] \
                                     kernel {kernel:.10} vs point {modelled:.10}"
                                ));
                            }
                        }
                    }
                    let expected_core = ea.core_charge * eb.core_charge / distance * PM3_EV;
                    if (point.core_core - expected_core).abs() > 1.0e-9 * expected_core.abs() {
                        failures.push(format!(
                            "Z{za}/Z{zb}: core-core {} vs {expected_core}",
                            point.core_core
                        ));
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} point-model mismatch(es) against a monopole partner:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    /// Between two `sp` atoms, the off-diagonal quadrupole terms transverse to the interatomic
    /// axis are the one place the point model is not exact even on axis, and it is worth being
    /// precise about why and about what survives.
    ///
    /// The local diatomic frame is fixed only up to a rotation about the bond. A four-corner
    /// point set has the right quadrupole moment but also carries hexadecapole content, and
    /// that content is *not* invariant under the rotation, so the global set and the rotated
    /// local one differ. Measured for C/O at 4 and 6.5 Bohr, the `(p_y p_z | p_y p_z)` term
    /// differs by 20% and 8%.
    ///
    /// What has to be true is that the **quadrupole moments agree**, so the long-range tail
    /// Ewald sums is right, and the residue is a faster-decaying multipole. It is: the leading
    /// term falls off about as `R⁻⁴·⁵` over this range while the difference falls off about as
    /// `R⁻⁶·⁴`, two powers steeper. This test measures both exponents rather than assuming them.
    #[test]
    fn sp_pair_quadrupole_terms_agree_asymptotically() {
        let params = Pm3Parameters::standard().unwrap();
        let carbon = params.element(6).unwrap();
        let oxygen = params.element(8).unwrap();
        let sites_c = AtomSites::build(carbon).unwrap();
        let sites_o = AtomSites::build(oxygen).unwrap();
        // (p_y, p_z) on both atoms, with the atoms separated along x: the transverse
        // off-diagonal quadrupole pair that the monopole-partner test cannot reach.
        let index = pack(2, 3) * 10 + pack(2, 3);

        let sample = |distance: f64| -> (f64, f64) {
            let dvec = Vec3::new(distance, 0.0, 0.0);
            let te = nddo_pair(&unscreened(carbon), &unscreened(oxygen), dvec);
            let point = point_pair(
                &sites_c,
                carbon.core_charge,
                &sites_o,
                oxygen.core_charge,
                dvec,
            );
            (te.w[index], (te.w[index] - point.w[index]).abs())
        };

        let (near_value, near_difference) = sample(5.0);
        let (far_value, far_difference) = sample(10.0);
        let decay_exponent = |near: f64, far: f64| (near / far).ln() / 2.0_f64.ln();
        let leading = decay_exponent(near_value.abs(), far_value.abs());
        let residue = decay_exponent(near_difference, far_difference);
        assert!(
            residue > leading + 1.5,
            "the point model's quadrupole residue decays as R^-{residue:.2}, only \
             {:.2} powers steeper than the R^-{leading:.2} term it sits on; the quadrupole \
             moments must not match",
            residue - leading
        );
    }

    /// Below `r_on` the correction restores the NDDO integral exactly: the point model that the
    /// lattice sum will add back is subtracted here in full.
    #[test]
    fn inside_the_switch_the_correction_restores_the_nddo_integral() {
        let params = Pm3Parameters::standard().unwrap();
        let carbon = params.element(6).unwrap();
        let oxygen = params.element(8).unwrap();
        let sites_c = AtomSites::build(carbon).unwrap();
        let sites_o = AtomSites::build(oxygen).unwrap();
        let dvec = Vec3::new(0.61, -0.47, 0.64).normalized() * 2.6;
        let te = nddo_pair(carbon, oxygen, dvec);
        let point = point_pair(
            &sites_c,
            carbon.core_charge,
            &sites_o,
            oxygen.core_charge,
            dvec,
        );
        let screened = screened_pair(
            &te,
            &sites_c,
            carbon.core_charge,
            &sites_o,
            oxygen.core_charge,
            dvec,
            SwitchRange::default(),
        );
        for index in 0..te.w.len() {
            let restored = screened.w[index] + point.w[index];
            assert!(
                (restored - te.w[index]).abs() < 1.0e-12 * te.w[index].abs().max(1.0),
                "w[{index}]: correction + point = {restored} != NDDO {}",
                te.w[index]
            );
        }
    }

    /// Beyond `r_off` nothing is left: only the lattice sum speaks there.
    #[test]
    fn beyond_the_switch_the_correction_vanishes() {
        let params = Pm3Parameters::standard().unwrap();
        let carbon = params.element(6).unwrap();
        let sites = AtomSites::build(carbon).unwrap();
        let dvec = Vec3::new(1.0, 0.0, 0.0) * (DEFAULT_SWITCH_OFF + 1.0);
        let te = nddo_pair(carbon, carbon, dvec);
        let screened = screened_pair(
            &te,
            &sites,
            carbon.core_charge,
            &sites,
            carbon.core_charge,
            dvec,
            SwitchRange::default(),
        );
        assert!(screened.w.iter().all(|v| *v == 0.0));
        assert!(screened.e1b.iter().all(|v| *v == 0.0));
        assert!(screened.e2a.iter().all(|v| *v == 0.0));
        assert_eq!(
            screened_core_core(carbon, carbon, dvec.norm(), SwitchRange::default()),
            0.0
        );
    }

    /// The whole reason the switch exists: what it truncates has to be small and getting
    /// smaller. The correction's magnitude at the switch-off radius bounds the error the
    /// truncation introduces, and it must fall off steeply with distance.
    #[test]
    fn the_truncated_correction_is_small_and_decaying() {
        let params = Pm3Parameters::standard().unwrap();
        let carbon = params.element(6).unwrap();
        let oxygen = params.element(8).unwrap();
        let sites_c = AtomSites::build(carbon).unwrap();
        let sites_o = AtomSites::build(oxygen).unwrap();
        let direction = Vec3::new(0.61, -0.47, 0.64).normalized();

        let magnitude = |distance: f64| -> f64 {
            let dvec = direction * distance;
            let te = nddo_pair(carbon, oxygen, dvec);
            let point = point_pair(
                &sites_c,
                carbon.core_charge,
                &sites_o,
                oxygen.core_charge,
                dvec,
            );
            te.w.iter()
                .zip(&point.w)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };

        // At a typical bond length the correction is a large fraction of an eV; by the
        // switch-off radius it has to have collapsed by orders of magnitude.
        let bonded = magnitude(2.6);
        let far = magnitude(DEFAULT_SWITCH_OFF);
        assert!(
            bonded > 1.0e-2,
            "the correction is suspiciously small at 2.6 Bohr: {bonded:.3e}"
        );
        // Measured: about 1.3e-3 eV per orbital-pair term at 22 Bohr. That is the scale of what
        // the switch discards, and it is a modelling choice rather than an error to drive to
        // zero — beyond the switch the point form is the *physically correct* one, since real
        // spherical distributions interact exactly as q_A q_B / R once they stop overlapping.
        // What matters is that it is small and shrinking, which the next two assertions pin.
        assert!(
            far < 1.0e-2,
            "the correction is still {far:.3e} eV at the switch-off radius"
        );
        assert!(
            far < 5.0e-3 * bonded,
            "the correction has only fallen from {bonded:.3e} to {far:.3e} eV by the cutoff"
        );

        // The decisive check is not the size but the *shape*. The whole justification for the
        // switch is that what remains out there is the Klopman–Ohno interpolation's spurious
        // `−a/2R³` tail, so the correction must decay as `R⁻³` once it is past the region where
        // the higher multipoles still matter. Measuring the exponent says that directly; a
        // steeper or shallower law would mean something else is being truncated.
        let exponent = (magnitude(15.0) / magnitude(30.0)).ln() / 2.0_f64.ln();
        assert!(
            (exponent - 3.0).abs() < 0.3,
            "the truncated correction decays as R^-{exponent:.2}, not the R^-3 expected of the \
             Klopman–Ohno tail"
        );
    }

    /// The switch itself is `C²`: value, slope and curvature all reach the ends smoothly, which
    /// is what keeps the force and the Hessian free of a kink at the cutoff.
    #[test]
    fn the_switch_is_c2() {
        let (r_on, r_off) = (18.0, 22.0);
        assert_eq!(switch(17.0, r_on, r_off), 1.0);
        assert_eq!(switch(23.0, r_on, r_off), 0.0);
        assert_eq!(switch(r_on, r_on, r_off), 1.0);
        assert_eq!(switch(r_off, r_on, r_off), 0.0);
        let h = 1.0e-4;
        for edge in [r_on, r_off] {
            for side in [-1.0, 1.0] {
                let x = edge + side * 3.0 * h;
                let first = (switch(x + h, r_on, r_off) - switch(x - h, r_on, r_off)) / (2.0 * h);
                let second = (switch(x + h, r_on, r_off) - 2.0 * switch(x, r_on, r_off)
                    + switch(x - h, r_on, r_off))
                    / (h * h);
                assert!(first.abs() < 0.05, "slope {first} near {edge}");
                assert!(second.abs() < 5.0, "curvature {second} near {edge}");
            }
        }
    }
}
