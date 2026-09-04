// SPDX-License-Identifier: GPL-3.0-or-later

//! The lattice sum carrying a phase: `Σ_T e^{iq·T} / |d + T|`, and its first two derivatives.
//!
//! # What needs this
//!
//! A phonon at wavevector `q` displaces the atom in cell `T` by `u e^{iq·T}`. Every quantity that
//! couples one cell to another therefore picks up that phase, and the Coulomb term — the one that
//! reaches every cell rather than a handful of neighbours — needs its lattice sum computed *with*
//! the phase rather than after it. That is the whole reason a dynamical matrix at finite `q`
//! cannot be assembled from the `q = 0` machinery.
//!
//! # Why `q ≠ 0` is the easy case
//!
//! At `q = 0` the reciprocal sum has a `G = 0` term that diverges, and a neutralizing background
//! exists to cancel it. Shifted by `q`, the reciprocal sum runs over `|G + q|`, which never
//! vanishes — so there is no divergent term, no background, and no special case at all.
//!
//! The price is that `|G + q| ≠ |−G + q|`. The unphased sum folds `±G` onto a cosine and does half
//! the work; this one cannot, and runs the full set.
//!
//! # The test that is not `q == 0`
//!
//! Every `q` in the **reciprocal lattice** phases the sum by `e^{i2πn} = 1`, so `Φ_q` must reduce
//! to the unphased sum at all of them, not just at the origin — background, `G = 0` term and all.
//! Checking `q == 0` instead is the silent version of the same test, and it matters in practice:
//! a phonon commensurate with an `n × n × n` k-mesh sits at a *supercell* reciprocal lattice
//! vector, and those are exactly the `q` a supercell comparison uses.
//!
//! # Scope
//!
//! Every dimensionality, each the `q`-shifted form of the machinery its unphased sum in
//! [`crate::pbc::ewald`] uses:
//!
//! * **3D** — the textbook split, over the full shifted reciprocal set.
//! * **2D** — Parry's slab sum, shifted: the full `G + q` set with prefactor `π/(A|G+q|)`. The
//!   ±G folding and its factor of two are a `q = 0` symmetry, and the `G = 0` sheet term exists
//!   only where `q` is a reciprocal lattice vector — everywhere else the shifted set has no
//!   `G + q = 0` member, and the `g = |q|` member is an ordinary term of the sum.
//! * **1D** — direct summation, phased, keeping the chain path's no-splitting character. The
//!   neutralizing line charge and the `1/n³` dipole tail are `q = 0` artefacts and appear only
//!   where the phase is one; at `q ≠ 0` the oscillation itself is what converges the sum, and
//!   the truncated tail is summed by repeated Abel transformation — see [`abel_tail`] for why
//!   the naive truncation would be `O(1/N)` there rather than `O(1/N³)`.
//! * **0D** — the single `T = 0` term, with any `q` trivially phasing it by one.
//!
//! `q` must lie in the periodic subspace: a slab has no perpendicular wavevector to phase its
//! images with, and handing one in is refused rather than silently projected away.

use std::f64::consts::{PI, TAU};

use crate::cell::Cell;
use crate::error::{Pm3Error, Result};
use crate::math::Vec3;
use crate::pbc::ewald::{slab_kernel, EwaldParams, CHAIN_REFERENCE_LENGTH};
use crate::special::{erf, erfc};

/// `Σ_T e^{iq·T} v(d + T)` at one displacement, with its gradient and Hessian in `d`.
///
/// Each entry is `[real, imaginary]`. In atomic units — the caller multiplies by charges and by
/// [`crate::constants::PM3_EV`], exactly as the unphased sum's users do.
#[derive(Clone, Copy, Debug, Default)]
pub struct PhasedKernel {
    pub value: [f64; 2],
    pub gradient: [[f64; 2]; 3],
    pub hessian: [[[f64; 2]; 3]; 3],
}

impl PhasedKernel {
    #[inline]
    fn add(&mut self, phase: [f64; 2], value: f64, first: f64, second: f64, d: Vec3, r: f64) {
        let v = [d.x, d.y, d.z];
        accumulate(&mut self.value, phase, value);
        for (alpha, va) in v.iter().enumerate() {
            accumulate(&mut self.gradient[alpha], phase, first * va / r);
            for (beta, vb) in v.iter().enumerate() {
                let radial = va * vb / (r * r);
                let delta = if alpha == beta { 1.0 } else { 0.0 };
                accumulate(
                    &mut self.hessian[alpha][beta],
                    phase,
                    second * radial + first * (delta - radial) / r,
                );
            }
        }
    }

    /// The bare `1/r` kernel of one image, with its first two derivatives.
    #[inline]
    fn add_bare(&mut self, phase: [f64; 2], v: Vec3) {
        let r = v.norm();
        if r < 1.0e-12 {
            return;
        }
        let f = 1.0 / r;
        self.add(phase, f, -f * f, 2.0 * f * f * f, v, r);
    }
}

#[inline]
fn accumulate(slot: &mut [f64; 2], phase: [f64; 2], real: f64) {
    slot[0] += phase[0] * real;
    slot[1] += phase[1] * real;
}

#[inline]
fn complex_mul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

/// Whether `q` is a reciprocal lattice vector of `cell`, so every `e^{iq·T}` is exactly one.
///
/// Compared against the shortest reciprocal vector, so the test is scale-free — and so that a `q`
/// merely *near* a lattice vector is correctly rejected. The phased sum really is discontinuous
/// there, and rounding a nearby `q` into place would hide physics rather than noise.
fn is_reciprocal_lattice_vector(cell: &Cell, q: Vec3) -> bool {
    let basis = cell.reciprocal_basis();
    if basis.is_empty() {
        return true;
    }
    let mut lattice = Vec3::zero();
    let mut shortest = f64::INFINITY;
    for (index, b) in &basis {
        let a = cell.vector(*index);
        lattice += *b * (q.dot(a) / TAU).round();
        shortest = shortest.min(b.norm());
    }
    (q - lattice).norm() < 1.0e-9 * shortest
}

/// The component of `q` outside the periodic subspace, which must vanish for the phase
/// `e^{iq·T}` to mean anything: every translation `T` lies *in* that subspace, so a
/// perpendicular component would be silently unread — accepted, and then ignored.
fn off_subspace_component(cell: &Cell, q: Vec3) -> f64 {
    let periodic = cell.periodic_indices();
    match periodic.as_slice() {
        [i, j] => {
            let normal = cell.vector(*i).cross(cell.vector(*j)).normalized();
            q.dot(normal).abs()
        }
        [i] => {
            let axis = cell.vector(*i).normalized();
            (q - axis * q.dot(axis)).norm()
        }
        // 3D spans everything; 0D reads no component of `q` at all, so none is wrong.
        _ => 0.0,
    }
}

/// `Σ'_T e^{iq·T} / |d + T|` and its first two derivatives, at each displacement (atomic units).
///
/// `q` is Cartesian and must lie in the periodic subspace — in-plane for a slab, axial for a
/// chain — or an error is returned. The `T = 0` term is included except where `d + T` vanishes,
/// which is the caller's self-interaction to own — the same convention [`crate::pbc::ewald`]
/// uses.
///
/// In 1D at a reciprocal lattice vector the value carries the same neutralizing line-charge
/// convention as [`crate::pbc::ewald`]'s chain path: `−2(H_N + ln(L/L₀))/L` per displacement,
/// with `L₀` = [`CHAIN_REFERENCE_LENGTH`]. A charged contraction is then defined only within
/// that convention, exactly as its energy is; a neutral contraction never sees the constant.
/// Away from the reciprocal lattice no convention enters and the value is absolute.
pub fn ewald_phased(
    cell: &Cell,
    displacements: &[Vec3],
    q: Vec3,
    params: &EwaldParams,
) -> Result<Vec<PhasedKernel>> {
    ewald_phased_with(cell, displacements, q, params, Macroscopic::Include)
}

/// Whether the `G = 0` member of the shifted reciprocal sum is kept.
///
/// At `q = 0` the question does not arise: `G = 0` gives `|G + q| = 0`, the term is divergent, and
/// the neutralizing background removes it — which every path here already does by skipping the
/// zero-length vector. At finite `q` the same member is `|q|`, an ordinary finite term, and it is
/// the **macroscopic field**: the `4π/(Ωq²)` that a charge-density wave of wavevector `q` sets up
/// across the whole crystal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Macroscopic {
    /// Keep it. The right choice for a *fixed-charge* lattice sum, where charge neutrality gives
    /// the term a finite `q → 0` limit, and for any absolute energy.
    Include,
    /// Leave it out, so the kernel carries only the microscopic (`G ≠ 0`) field.
    ///
    /// The right choice inside a **self-consistent response**. Keeping it there makes the bare
    /// perturbation `O(1/q)` per atom and the induced density `O(1/q)` in reply, so their
    /// contraction is `O(1/q²)` — and the per-atom factors do not factorize out of the product,
    /// so the `Σ_a Q_a = 0` that saves the fixed-charge sum cannot save this one. Measured on
    /// water: the acoustic sum rule of `D(q)` grows as `1/q²` with it and falls linearly in `q`
    /// without it.
    ///
    /// What is dropped is not lost: the macroscopic field's contribution to the dynamical matrix
    /// is exactly the non-analytic term [`crate::pbc::lo_to::non_analytic_term`] computes from the
    /// Born charges and `ε∞`, which is the standard decomposition and the only one in which the
    /// `q → 0` limit is direction dependent rather than divergent.
    Exclude,
}

/// [`ewald_phased`] with control over the macroscopic term. See [`Macroscopic`].
pub fn ewald_phased_with(
    cell: &Cell,
    displacements: &[Vec3],
    q: Vec3,
    params: &EwaldParams,
    macroscopic: Macroscopic,
) -> Result<Vec<PhasedKernel>> {
    let residual = off_subspace_component(cell, q);
    let scale = cell
        .reciprocal_basis()
        .iter()
        .map(|(_, b)| b.norm())
        .fold(q.norm(), f64::max);
    if residual > 1.0e-9 * scale.max(1.0e-300) {
        return Err(Pm3Error::InvalidInput(format!(
            "the wavevector ({}, {}, {}) has a component of {residual:.3e} outside the periodic \
             subspace. A phase e^{{iq·T}} only ever reads the in-subspace part of q, so a \
             perpendicular component would be accepted and then silently ignored — pass a q that \
             lies in the periodic plane (2D) or along the axis (1D).",
            q.x, q.y, q.z
        )));
    }

    // Not `q == 0`: every reciprocal lattice vector phases the sum by one and needs the same
    // treatment. See [`is_reciprocal_lattice_vector`].
    let unphased = is_reciprocal_lattice_vector(cell, q);
    // At a reciprocal lattice vector the shifted set contains `G + q = 0` and every path already
    // skips it as the background does, so there is nothing left for `Exclude` to remove and the
    // two settings agree exactly. Only away from the reciprocal lattice do they differ.
    let mut out = vec![PhasedKernel::default(); displacements.len()];
    match cell.n_periodic() {
        3 => phased_3d(
            cell,
            displacements,
            q,
            params,
            unphased,
            macroscopic,
            &mut out,
        ),
        2 => phased_2d(cell, displacements, q, params, unphased, &mut out),
        1 => phased_1d(cell, displacements, q, params, unphased, &mut out),
        _ => phased_0d(displacements, &mut out),
    }
    Ok(out)
}

/// The real-space half of the split, shared by 3D and 2D: `f(r) = erfc(αr)/r`, differentiated
/// twice in `r`, over every image inside the cutoff, with the image's phase.
fn real_space_half(
    cell: &Cell,
    displacements: &[Vec3],
    q: Vec3,
    params: &EwaldParams,
    unphased: bool,
    out: &mut [PhasedKernel],
) {
    let alpha = params.alpha;
    let two_alpha_over_sqrt_pi = 2.0 * alpha / PI.sqrt();
    let translations = cell.translations_within(params.real_cutoff + longest(displacements));
    for (slot, d) in out.iter_mut().zip(displacements) {
        for (_, shift) in &translations {
            let v = *d + *shift;
            let r = v.norm();
            if r < 1.0e-12 || r > params.real_cutoff {
                continue;
            }
            // At a reciprocal lattice vector the phase is one *exactly*. Taking the cosine of
            // `2πn` instead leaves an imaginary residue of order 1e-16 per image, and the
            // reduction to the unphased sum would then hold only to within that.
            let phase = if unphased {
                [1.0, 0.0]
            } else {
                let angle = q.dot(*shift);
                [angle.cos(), angle.sin()]
            };
            let f = erfc(alpha * r) / r;
            let gauss = two_alpha_over_sqrt_pi * (-(alpha * r).powi(2)).exp();
            let first = -(f + gauss) / r;
            let second = -2.0 * first / r + 2.0 * alpha * alpha * gauss;
            slot.add(phase, f, first, second, v, r);
        }
    }
}

/// A site against its own images. The real-space sum skipped `T = 0` because `r` was zero,
/// but the reciprocal sum included it — that is what the Ewald self term exists to remove,
/// and it applies to both split formulations (3D and 2D) for exactly the same reason. Only the
/// value carries it: a constant has no gradient, which is why the derivative tests never
/// noticed its absence. The 1D path has no split and therefore no self term to remove.
fn remove_gaussian_self_term(displacements: &[Vec3], alpha: f64, out: &mut [PhasedKernel]) {
    let two_alpha_over_sqrt_pi = 2.0 * alpha / PI.sqrt();
    for (slot, d) in out.iter_mut().zip(displacements) {
        if d.norm() < 1.0e-12 {
            accumulate(&mut slot.value, [1.0, 0.0], -two_alpha_over_sqrt_pi);
        }
    }
}

/// The three-dimensional phased sum: real-space half, full shifted reciprocal set, and the
/// background restored only where the phase is one.
#[allow(clippy::too_many_arguments)]
fn phased_3d(
    cell: &Cell,
    displacements: &[Vec3],
    q: Vec3,
    params: &EwaldParams,
    unphased: bool,
    macroscopic: Macroscopic,
    out: &mut [PhasedKernel],
) {
    let alpha = params.alpha;
    real_space_half(cell, displacements, q, params, unphased, out);

    // Reciprocal space, over the **full** shifted set: `|G + q| ≠ |−G + q|`, so there is no ±G
    // folding to exploit.
    //
    // The phase is `e^{−i(G+q)·d}`, minus sign and all: Poisson-summing
    // `Σ_T e^{iq·T} f(d + T)` gives `(1/V) Σ_G f̂(|G+q|) e^{−i(G+q)·d}`, and the sign is pinned
    // by the exact re-indexing identity `Φ_q(d + T₀) = e^{−iq·T₀} Φ_q(d)`, which only the minus
    // satisfies. This sum carried `e^{+i(G+q)·d}` — the conjugated smooth half — for months:
    // the oracle contraction is identically real over its ±d pairs, the folding test's
    // `q = b/2` makes the shifted set symmetric under negation, and every internal identity
    // conjugates both halves together, so nothing could see it until the splitting-independence
    // test compared the imaginary part across two values of α and found it moving in the
    // fourth digit.
    let volume = cell.measure();
    let inverse_four_alpha_squared = 1.0 / (4.0 * alpha * alpha);
    for (is_macroscopic, g) in shifted_reciprocal(cell, q, params.gmax) {
        let g2 = g.norm2();
        if g2 < 1.0e-20 {
            continue;
        }
        // The `G = 0` member — `|G + q| = |q|` — is the macroscopic field. See [`Macroscopic`].
        if is_macroscopic && macroscopic == Macroscopic::Exclude {
            continue;
        }
        let prefactor = 2.0 * TAU / volume * (-g2 * inverse_four_alpha_squared).exp() / g2;
        let gv = [g.x, g.y, g.z];
        for (slot, d) in out.iter_mut().zip(displacements) {
            let angle = g.dot(*d);
            let (sin, cos) = angle.sin_cos();
            // `e^{−iG·d}` contributes `cos − i sin`; its `d`-derivative multiplies by `−iG`.
            accumulate(&mut slot.value, [cos, -sin], prefactor);
            for (alpha_index, ga) in gv.iter().enumerate() {
                accumulate(
                    &mut slot.gradient[alpha_index],
                    [-sin, -cos],
                    prefactor * ga,
                );
                for (beta, gb) in gv.iter().enumerate() {
                    accumulate(
                        &mut slot.hessian[alpha_index][beta],
                        [-cos, sin],
                        prefactor * ga * gb,
                    );
                }
            }
        }
    }

    // The `G = 0` term and its neutralizing background exist only when the phase is one; away
    // from the reciprocal lattice the shifted sum above already covers everything.
    if unphased {
        let background = -PI / (alpha * alpha * volume);
        for slot in out.iter_mut() {
            accumulate(&mut slot.value, [1.0, 0.0], background);
        }
    }

    remove_gaussian_self_term(displacements, alpha, out);
}

/// The two-dimensional phased sum: Parry's slab formulation over the full shifted in-plane set.
///
/// Each `G + q` contributes `(π/(A|G+q|)) e^{−i(G+q)·d} K(|G+q|, z)` with `z = d·n̂` and `K` the
/// slab kernel of [`crate::pbc::ewald`]. Three things differ from the unphased sum, and each is
/// a place to go wrong:
///
/// * **No ±G folding.** `|G + q| ≠ |−G + q|`, so the sum runs the full shifted set and the
///   prefactor is `π/(A|G+q|)`, not `2π/(A|G|)` — at `q = 0` the full set with `π` reproduces
///   the half set with `2π`, which is what the splitting-independence test leans on: a wrong
///   factor here is compensated by `α` in a way only that test exposes.
/// * **`(G+q) ⊥ n̂`**, so the cross terms of the pair Hessian do not collapse:
///   `∂²/∂d_α∂d_β = e^{−i(G+q)·d}[−(G+q)_α(G+q)_β K − i((G+q)_α n̂_β + n̂_α(G+q)_β) K_z
///   + n̂_α n̂_β K_zz]`.
/// * **No `G = 0` case away from the reciprocal lattice.** The shifted set has no `G + q = 0`
///   member there — the "`G = 0` slot" is just the `g = |q|` member of the ordinary formula,
///   which is `(π/(A|q|)) K(|q|, z) = 2π/(A|q|) + sheet(z) + O(|q|)` as `q → 0` (from
///   `K(0, z) = 2` and `∂K/∂g|₀ = −2z erf(αz) − (2/(α√π))e^{−α²z²}`). The divergent piece is
///   `d`-independent, so only the *value* is delicate near `Γ`; the gradient stays finite but
///   `q̂`-dependent — the 2D LO–TO non-analyticity, which is physics and is left alone — and
///   the Hessian's small-`q` limit is smooth. The smeared-sheet term is restored only where `q`
///   *is* a reciprocal lattice vector, with phase one, exactly as the 3D background is.
fn phased_2d(
    cell: &Cell,
    displacements: &[Vec3],
    q: Vec3,
    params: &EwaldParams,
    unphased: bool,
    out: &mut [PhasedKernel],
) {
    let alpha = params.alpha;
    real_space_half(cell, displacements, q, params, unphased, out);

    let periodic = cell.periodic_indices();
    let normal = cell
        .vector(periodic[0])
        .cross(cell.vector(periodic[1]))
        .normalized();
    let nhat = [normal.x, normal.y, normal.z];
    let area = cell.measure();

    // The macroscopic flag is ignored here, and deliberately: in 2D the `q² · v(q)` product goes
    // to zero linearly, so the `G = 0` member carries no divergence to separate out — the slab's
    // `q → 0` limit is continuous with a kink rather than a splitting. See
    // [`crate::pbc::lo_to::non_analytic_term`], which refuses anything but 3D for the same reason.
    for (_, g) in shifted_reciprocal(cell, q, params.gmax) {
        let g2 = g.norm2();
        if g2 < 1.0e-20 {
            // Only possible when `q` is a reciprocal lattice vector; the sheet below owns it.
            continue;
        }
        let magnitude = g2.sqrt();
        // π/(A|G+q|): the full shifted set, so no factor of two for a folded ±G pair.
        let prefactor = PI / (area * magnitude);
        let gv = [g.x, g.y, g.z];
        for (slot, d) in out.iter_mut().zip(displacements) {
            let z = d.dot(normal);
            let (kernel, dkernel_dz, _, dkernel_dzz) = slab_kernel(magnitude, z, alpha);
            let angle = g.dot(*d);
            let (sin, cos) = angle.sin_cos();
            // `e^{−i(G+q)·d}`, minus sign as in the 3D sum above — the re-indexing identity
            // `Φ_q(d + T₀) = e^{−iq·T₀} Φ_q(d)` pins it here identically.
            let phase = [cos, -sin];
            let rotated = [-sin, -cos]; // −i·e^{−iθ}, the in-plane derivative's factor
            accumulate(&mut slot.value, phase, prefactor * kernel);
            #[allow(clippy::needless_range_loop)] // Cartesian tensor indices, as elsewhere
            for a in 0..3 {
                accumulate(&mut slot.gradient[a], rotated, prefactor * kernel * gv[a]);
                accumulate(
                    &mut slot.gradient[a],
                    phase,
                    prefactor * dkernel_dz * nhat[a],
                );
                for b in 0..3 {
                    accumulate(
                        &mut slot.hessian[a][b],
                        phase,
                        -prefactor * kernel * gv[a] * gv[b],
                    );
                    accumulate(
                        &mut slot.hessian[a][b],
                        rotated,
                        prefactor * dkernel_dz * (gv[a] * nhat[b] + nhat[a] * gv[b]),
                    );
                    accumulate(
                        &mut slot.hessian[a][b],
                        phase,
                        prefactor * dkernel_dzz * nhat[a] * nhat[b],
                    );
                }
            }
        }
    }

    // The Gaussian-smeared charged sheet, `−(2π/A)[z erf(αz) + e^{−α²z²}/(α√π)]` — the `G = 0`
    // term. It exists only where the phase is one, exactly like the 3D background: away from
    // the reciprocal lattice the shifted set has no `G + q = 0` member to replace.
    if unphased {
        let sqrt_pi = PI.sqrt();
        for (slot, d) in out.iter_mut().zip(displacements) {
            let z = d.dot(normal);
            let gaussian = (-(alpha * z).powi(2)).exp();
            let sheet = -TAU / area * (z * erf(alpha * z) + gaussian / (alpha * sqrt_pi));
            accumulate(&mut slot.value, [1.0, 0.0], sheet);
            let dsheet_dz = -TAU / area * erf(alpha * z);
            let dsheet_dzz = -TAU / area * 2.0 * alpha / sqrt_pi * gaussian;
            #[allow(clippy::needless_range_loop)] // Cartesian tensor indices, as elsewhere
            for a in 0..3 {
                accumulate(&mut slot.gradient[a], [1.0, 0.0], dsheet_dz * nhat[a]);
                for b in 0..3 {
                    accumulate(
                        &mut slot.hessian[a][b],
                        [1.0, 0.0],
                        dsheet_dzz * nhat[a] * nhat[b],
                    );
                }
            }
        }
    }

    remove_gaussian_self_term(displacements, alpha, out);
}

/// How far the direct chain sum is pushed beyond the requested image count when the Abel tail
/// is poorly conditioned.
///
/// The repeated Abel transformation's optimal-truncation error goes as `e^{−N/|w|}` with
/// `|w| = 1/(2|sin(q·a/2)|)` (see [`abel_tail`]), so the summed range is widened to keep
/// `N/|w|` at least this large: at 40 the truncation floor is `~e^{−40} ≈ 4·10⁻¹⁸` relative,
/// below everything else in this crate. The cost is linear in the extension and each term is a
/// dozen floating-point operations, so this is cheap everywhere except within `~1/40` of a
/// reciprocal lattice vector, where the sum itself is near its (physical) divergence.
const CHAIN_TAIL_CONDITIONING: f64 = 40.0;

/// Hard ceiling on the widened 1D image count, so a `q` pathologically close to — but not at —
/// a reciprocal lattice vector degrades in accuracy rather than in run time. The phased sum is
/// genuinely near-divergent there (`Φ_q ~ −(2/L) ln|1 − e^{iq·a}|`), so no truncation strategy
/// makes that region cheap and sharp at once.
const CHAIN_IMAGES_CEILING: usize = 100_000;

/// The one-dimensional phased sum: the chain path's direct summation, phased image by image.
///
/// No Ewald splitting, as in [`crate::pbc::ewald`]'s `direct_1d` — there is no `α` to get
/// wrong. What changes with `q` is the *tail*, and it changes character rather than merely
/// size:
///
/// * **At a reciprocal lattice vector** every phase is one. The per-displacement sum
///   `Σ_n 1/|d + na|` diverges logarithmically, and the value is defined by the same
///   neutralizing line-charge convention the chain energy uses: subtract
///   `2(H_N + ln(L/L₀))/L`, which is `d`-independent and therefore invisible to the
///   derivatives and to any neutral contraction. What remains converges with a `1/n³`
///   Euler–Maclaurin tail, corrected analytically below to `O(1/N⁴)`.
/// * **Away from the reciprocal lattice** the phase oscillates and the sum converges on its
///   own (Dirichlet); the line charge and the dipole tail are `q = 0` artefacts and are *not*
///   carried over — doing so would make `Φ_q` discontinuous as `q → 0` in a way no `q = 0`
///   test could catch, because `Σ_n e^{iqnL}/(nL)` is finite while `Σ_n 1/(nL)` is what the
///   line charge exists to cancel. But the truncated tail is now `O(1/N)`, not `O(1/N³)`:
///   partial sums of `e^{iqnL}` are bounded by `1/(2|sin(qL/2)|)`, so Dirichlet gives
///   convergence but not speed. [`abel_tail`] sums that tail by parts instead.
fn phased_1d(
    cell: &Cell,
    displacements: &[Vec3],
    q: Vec3,
    params: &EwaldParams,
    unphased: bool,
    out: &mut [PhasedKernel],
) {
    let axis_index = cell.periodic_indices()[0];
    let axis = cell.vector(axis_index);
    let length = axis.norm();
    let theta = q.dot(axis); // radians per cell; exactly 2πn (up to rounding) when unphased

    let requested = params.chain_images.max(1);
    let images = if unphased {
        requested
    } else {
        // Widen the direct sum where the Abel tail would be ill-conditioned; see
        // [`CHAIN_TAIL_CONDITIONING`].
        let w_magnitude = 0.5 / (0.5 * theta).sin().abs();
        let floor = (CHAIN_TAIL_CONDITIONING * w_magnitude).ceil() as usize;
        requested.max(floor).min(CHAIN_IMAGES_CEILING)
    };

    // The q = 0 constants, computed once: the line-charge convention and the Euler–Maclaurin
    // lattice tail Σ_{n>N} n⁻³ = 1/(2N²) − 1/(2N³) + 1/(4N⁴) + O(N⁻⁶) — all three terms, for
    // the same reason `direct_1d` keeps all three.
    let (line_charge, lattice_tail) = if unphased {
        let harmonic: f64 = (1..=images).map(|n| 1.0 / n as f64).sum();
        let extent = harmonic + (length / CHAIN_REFERENCE_LENGTH).ln();
        let n = images as f64;
        (
            -2.0 * extent / length,
            0.5 / (n * n) - 0.5 / (n * n * n) + 0.25 / (n * n * n * n),
        )
    } else {
        (0.0, 0.0)
    };
    let axis_hat = axis * (1.0 / length);

    for (slot, d) in out.iter_mut().zip(displacements) {
        // T = 0, phase one at every q.
        slot.add_bare([1.0, 0.0], *d);

        // ±n shells, both signs in the same iteration so the cancellation that converges the
        // unphased sum happens before rounding — the same discipline `direct_1d` keeps.
        for n in 1..=images {
            let shift = axis * n as f64;
            let (phase_plus, phase_minus) = if unphased {
                ([1.0, 0.0], [1.0, 0.0])
            } else {
                let angle = theta * n as f64;
                let (sin, cos) = angle.sin_cos();
                ([cos, sin], [cos, -sin])
            };
            slot.add_bare(phase_plus, *d + shift);
            slot.add_bare(phase_minus, *d - shift);
        }

        if unphased {
            // The line-charge convention constant — value only, being `d`-independent.
            accumulate(&mut slot.value, [1.0, 0.0], line_charge);

            // The analytic tail. Pairing ±n cancels the odd orders, so
            //   1/|d+na| + 1/|d−na| = 2/(nL) + C(d)/n³ + O(1/n⁵),
            // with `C(d) = (3(d·â)² − |d|²)/L³`. The `2/(nL)` monopole is what the line-charge
            // subtraction accounts for at every `n` including the tail; what is left beyond `N`
            // is `C(d)·Σ_{n>N} n⁻³` and its derivatives, which follow by differentiating `C`:
            // the expansion is uniform in `d`, so ∇ and ∇∇ pass through the `1/n³` coefficient.
            // The first neglected term is the `1/n⁵` order, `O(1/N⁴)` after summing. Measured
            // on a 6 Bohr chain: the residual against 4000 images is 5·10⁻¹² at 150 images and
            // 1·10⁻¹³ at the default 400 — a 50× drop where `(400/150)⁴ ≈ 51` — pinned by
            // `the_unphased_chain_tail_converges_as_the_fourth_power`.
            let along = d.dot(axis_hat);
            let l3 = length * length * length;
            let c = (3.0 * along * along - d.norm2()) / l3;
            accumulate(&mut slot.value, [1.0, 0.0], c * lattice_tail);
            let ahat = [axis_hat.x, axis_hat.y, axis_hat.z];
            let dv = [d.x, d.y, d.z];
            #[allow(clippy::needless_range_loop)] // Cartesian tensor indices, as elsewhere
            for a in 0..3 {
                let dc = (6.0 * along * ahat[a] - 2.0 * dv[a]) / l3;
                accumulate(&mut slot.gradient[a], [1.0, 0.0], dc * lattice_tail);
                for b in 0..3 {
                    let delta = if a == b { 1.0 } else { 0.0 };
                    let ddc = (6.0 * ahat[a] * ahat[b] - 2.0 * delta) / l3;
                    accumulate(&mut slot.hessian[a][b], [1.0, 0.0], ddc * lattice_tail);
                }
            }
        } else {
            // The Abel-transformed tails of the two half-lines, `n > N` and `n < −N`.
            let plus = abel_tail(*d, axis, theta, images + 1);
            let minus = abel_tail(*d, axis * -1.0, -theta, images + 1);
            for tail in [plus, minus] {
                slot.value[0] += tail[0][0];
                slot.value[1] += tail[0][1];
                #[allow(clippy::needless_range_loop)] // Cartesian tensor indices, as elsewhere
                for a in 0..3 {
                    slot.gradient[a][0] += tail[1 + a][0];
                    slot.gradient[a][1] += tail[1 + a][1];
                    for b in 0..3 {
                        slot.hessian[a][b][0] += tail[4 + 3 * a + b][0];
                        slot.hessian[a][b][1] += tail[4 + 3 * a + b][1];
                    }
                }
            }
        }
    }
}

/// Zero periodic directions: the single `T = 0` term, which is the bare kernel itself. The
/// phase of the only translation there is equals one for every `q`, which is also what
/// [`is_reciprocal_lattice_vector`] reports for an empty reciprocal basis.
fn phased_0d(displacements: &[Vec3], out: &mut [PhasedKernel]) {
    for (slot, d) in out.iter_mut().zip(displacements) {
        slot.add_bare([1.0, 0.0], *d);
    }
}

/// How many Abel orders [`abel_tail`] carries at most. Twenty is past the optimal truncation
/// point everywhere the conditioning floor allows (`N/|w| ≥ 40` gives term twenty a relative
/// size of `20!/40²¹ ≈ 5·10⁻¹⁶`), and the early-exit below stops sooner wherever the series
/// turns first.
const ABEL_ORDERS: usize = 20;

/// One component ordering shared by [`abel_tail`] and its caller: value, the three gradient
/// components, then the nine Hessian entries row-major.
type Bundle = [f64; 13];

/// The bare kernel bundle at one point: `1/r`, `∇(1/r)`, `∇∇(1/r)`.
fn bare_bundle(v: Vec3) -> Bundle {
    let mut out = [0.0; 13];
    let r = v.norm();
    if r < 1.0e-12 {
        return out;
    }
    let i1 = 1.0 / r;
    let i3 = i1 * i1 * i1;
    let i5 = i3 * i1 * i1;
    let dv = [v.x, v.y, v.z];
    out[0] = i1;
    for a in 0..3 {
        out[1 + a] = -dv[a] * i3;
        for b in 0..3 {
            let delta = if a == b { 1.0 } else { 0.0 };
            out[4 + 3 * a + b] = 3.0 * dv[a] * dv[b] * i5 - delta * i3;
        }
    }
    out
}

/// `Σ_{n ≥ first} z^n f(n)` for one half-line of the chain, by repeated Abel transformation,
/// with `z = e^{iθ}` and `f(n)` the kernel bundle at `d + n·step`.
///
/// # Why summation by parts, and not more images or a Bessel split
///
/// The naive truncation error is `O(1/N)`: Dirichlet's test converges the sum because the
/// partial sums of `z^n` are bounded — by `|w| = |1/(1−z)| = 1/(2|sin(θ/2)|)` — but that same
/// bound multiplies the first neglected `f(N+1) ~ 1/(NL)`, and it blows up as `q → 0`. Abel's
/// transformation trades the oscillating sum for one over differences of `f`:
///
/// ```text
///   Σ_{n>N} z^n f(n) = w z^{N+1} f(N+1) − z w Σ_{n>N} z^n Δf(n),      Δf(n) = f(n) − f(n+1)
/// ```
///
/// (verified by telescoping the partial sums; the geometric case `Δf = 0` fixes the first
/// term). Applying it to its own remainder `K` times over gives
///
/// ```text
///   Σ_{n>N} z^n f(n) = Σ_{j=0}^{K−1} (−zw)^j · w z^{N+1} · Δʲf(N+1)  +  (−zw)^K R_K
/// ```
///
/// where `Δʲf(N+1)` is the exact `j`-th forward difference — evaluated from `f` at
/// `N+1 … N+1+j`, not from an asymptotic formula — and `|R_K| ≲ K!/(K N^K L)`. The terms decay
/// like `(|w|/N)^{j+1} j!`: an asymptotic series in `1/N` with the conditioning parameter
/// `|w|`, best truncated at its smallest term. The loop below does exactly that — it stops
/// before the first term that grows — so the error is of the size of the first omitted term,
/// `~e^{−N/|w|}` at the optimum, which [`CHAIN_TAIL_CONDITIONING`] keeps at least `e^{−40}` by
/// widening `N`. The alternative — a Bessel-`K₀` Ewald split — needs a special function this
/// crate does not have, for accuracy this construction already exceeds.
fn abel_tail(d: Vec3, step: Vec3, theta: f64, first: usize) -> [[f64; 2]; 13] {
    // Exact forward differences from samples at `first .. first + ABEL_ORDERS`.
    let mut table: Vec<Bundle> = (0..=ABEL_ORDERS)
        .map(|j| bare_bundle(d + step * (first + j) as f64))
        .collect();
    let mut differences: Vec<Bundle> = Vec::with_capacity(ABEL_ORDERS + 1);
    differences.push(table[0]);
    for level in 1..=ABEL_ORDERS {
        for i in 0..=(ABEL_ORDERS - level) {
            let next = table[i + 1];
            for (slot, sub) in table[i].iter_mut().zip(next.iter()) {
                *slot -= sub;
            }
        }
        differences.push(table[0]);
    }

    let (sin, cos) = theta.sin_cos();
    let z = [cos, sin];
    // w = 1/(1 − z). |1 − z|² = 2(1 − cos θ), nonzero away from the reciprocal lattice, which
    // is the only place this is called.
    let denom2 = (1.0 - cos) * (1.0 - cos) + sin * sin;
    let w = [(1.0 - cos) / denom2, sin / denom2];
    let ratio = {
        let zw = complex_mul(z, w);
        [-zw[0], -zw[1]]
    };
    let (sin_first, cos_first) = (theta * first as f64).sin_cos();
    // coefficient_j = w · z^{first} · (−zw)^j
    let mut coefficient = complex_mul(w, [cos_first, sin_first]);

    let mut out = [[0.0; 2]; 13];
    let mut previous = f64::INFINITY;
    for difference in differences.iter() {
        let weight = (coefficient[0] * coefficient[0] + coefficient[1] * coefficient[1]).sqrt();
        let largest = difference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let magnitude = weight * largest;
        // Optimal truncation of an asymptotic series: stop *before* the first growing term.
        if magnitude > previous {
            break;
        }
        for (slot, value) in out.iter_mut().zip(difference.iter()) {
            slot[0] += coefficient[0] * value;
            slot[1] += coefficient[1] * value;
        }
        if magnitude < 1.0e-18 {
            break;
        }
        previous = magnitude;
        coefficient = complex_mul(coefficient, ratio);
    }
    out
}

/// The longest displacement, so the real-space image search covers every pair it is asked about.
fn longest(displacements: &[Vec3]) -> f64 {
    displacements
        .iter()
        .map(|d| d.norm())
        .fold(0.0_f64, f64::max)
}

/// Reciprocal vectors `G` with `|G + q| ≤ gmax`, the full set rather than a half-space. The
/// enumeration runs over the reciprocal basis of the periodic subspace, so in 2D it yields the
/// shifted in-plane set.
/// The shifted set `{G + q}` inside `gmax`, each flagged with whether it is the `G = 0` member.
///
/// The flag comes from the integer indices rather than from comparing the vector against `q`:
/// at small `q` the `G = 0` member is arbitrarily close to its neighbours in length, and a
/// floating-point test for "is this one `q`" would be a coin toss exactly where it matters.
fn shifted_reciprocal(cell: &Cell, q: Vec3, gmax: f64) -> Vec<(bool, Vec3)> {
    let basis = cell.reciprocal_basis();
    let mut limits = [0i32; 3];
    for (index, b) in &basis {
        let length = b.norm();
        // Widened by `|q|`, because the bound is on `|G + q|` and `q` can be as long as half a
        // reciprocal vector.
        limits[*index] = if length > 0.0 {
            ((gmax + q.norm()) / length).ceil() as i32
        } else {
            0
        };
    }
    let vector_of = |index: usize| -> Vec3 {
        basis
            .iter()
            .find(|(i, _)| *i == index)
            .map(|(_, b)| *b)
            .unwrap_or_else(Vec3::zero)
    };
    let (b0, b1, b2) = (vector_of(0), vector_of(1), vector_of(2));
    let gmax2 = gmax * gmax;
    let mut out = Vec::new();
    for n0 in -limits[0]..=limits[0] {
        for n1 in -limits[1]..=limits[1] {
            for n2 in -limits[2]..=limits[2] {
                let g = b0 * n0 as f64 + b1 * n1 as f64 + b2 * n2 as f64 + q;
                if g.norm2() <= gmax2 {
                    out.push((n0 == 0 && n1 == 0 && n2 == 0, g));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbc::ewald_reference::RefSite;

    fn cubic(edge: f64) -> Cell {
        Cell::cubic(edge).unwrap()
    }

    /// A slab periodic in a skewed `xy` plane, so nothing accidentally aligns with the axes.
    fn slab(vacuum: f64) -> Cell {
        Cell::new(
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(1.2, 5.6, 0.0),
            Vec3::new(0.0, 0.0, vacuum),
            [true, true, false],
        )
        .unwrap()
    }

    /// A chain along `x` with generous transverse padding, which the 1D sum must never read.
    fn chain(padding: f64) -> Cell {
        Cell::new(
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(0.0, padding, 0.0),
            Vec3::new(0.0, 0.0, padding),
            [true, false, false],
        )
        .unwrap()
    }

    fn probes() -> Vec<Vec3> {
        vec![
            Vec3::new(1.3, 0.0, 0.0),
            Vec3::new(0.7, -1.1, 2.4),
            Vec3::new(-2.0, 0.5, 0.9),
        ]
    }

    /// A generic interior wavevector for `cell`, built from its own reciprocal basis so it lies
    /// in the periodic subspace whatever the dimensionality.
    fn interior_q(cell: &Cell) -> Vec3 {
        let fractions = [0.31_f64, -0.23, 0.17];
        let mut q = Vec3::zero();
        for (index, b) in &cell.reciprocal_basis() {
            q += *b * fractions[*index];
        }
        q
    }

    /// A neutral four-charge fixture and a `q` chosen away from both `Γ` and the zone boundary,
    /// where the direct sum is sharpest. See [`crate::pbc::ewald_reference::direct_phased_sum`].
    fn oracle_fixture() -> (Cell, Vec<RefSite>, Vec3) {
        let cell = cubic(6.0);
        let sites = vec![
            RefSite {
                position: Vec3::new(0.4, 0.6, 0.9),
                charge: 0.8,
            },
            RefSite {
                position: Vec3::new(3.1, 1.2, 2.4),
                charge: -0.5,
            },
            RefSite {
                position: Vec3::new(1.7, 4.3, 3.8),
                charge: 0.35,
            },
            RefSite {
                position: Vec3::new(4.6, 3.4, 1.1),
                charge: -0.65,
            },
        ];
        let q = interior_q(&cell);
        (cell, sites, q)
    }

    /// The slab twin of [`oracle_fixture`]: neutral, spread through the cell in-plane and a few
    /// Bohr out of it, with `q` a generic interior in-plane wavevector.
    fn slab_oracle_fixture() -> (Cell, Vec<RefSite>, Vec3) {
        let cell = slab(40.0);
        let sites = vec![
            RefSite {
                position: Vec3::new(0.4, 0.6, 0.9),
                charge: 0.8,
            },
            RefSite {
                position: Vec3::new(3.1, 1.2, -1.4),
                charge: -0.5,
            },
            RefSite {
                position: Vec3::new(1.7, 4.3, 1.8),
                charge: 0.35,
            },
            RefSite {
                position: Vec3::new(4.6, 3.4, -0.7),
                charge: -0.65,
            },
        ];
        let q = interior_q(&cell);
        (cell, sites, q)
    }

    /// The chain twin: neutral, with transverse structure the axial sum has to carry exactly.
    /// The charges are 2.5× the other fixtures' — a chain reaches its neighbours through one
    /// axis instead of six faces, and the smaller contraction was failing the non-triviality
    /// floor the oracle comparisons insist on.
    fn chain_oracle_fixture() -> (Cell, Vec<RefSite>, Vec3) {
        let cell = chain(40.0);
        let sites = vec![
            RefSite {
                position: Vec3::new(0.4, 0.6, 0.9),
                charge: 2.0,
            },
            RefSite {
                position: Vec3::new(3.1, 1.2, -1.4),
                charge: -1.25,
            },
            RefSite {
                position: Vec3::new(1.7, -2.1, 1.8),
                charge: 0.875,
            },
            RefSite {
                position: Vec3::new(4.6, 1.6, -0.7),
                charge: -1.625,
            },
        ];
        let q = interior_q(&cell);
        (cell, sites, q)
    }

    /// Contract a set of kernels the way [`crate::pbc::dfpt`] does: every ordered site pair,
    /// weighted by the charges, with `∂/∂r_s = −∂/∂d` and `∂/∂r_t = +∂/∂d`.
    #[allow(clippy::type_complexity, clippy::needless_range_loop)] // parallel tensors, by index
    fn contract(
        sites: &[RefSite],
        kernels: &[PhasedKernel],
    ) -> ([f64; 2], Vec<[[f64; 2]; 3]>, Vec<Vec<[[[f64; 2]; 3]; 3]>>) {
        let n = sites.len();
        let mut value = [0.0; 2];
        let mut gradient = vec![[[0.0; 2]; 3]; n];
        let mut hessian = vec![vec![[[[0.0; 2]; 3]; 3]; n]; n];
        let mut index = 0;
        for (s, site_s) in sites.iter().enumerate() {
            for (t, site_t) in sites.iter().enumerate() {
                let k = kernels[index];
                index += 1;
                let w = site_s.charge * site_t.charge;
                for part in 0..2 {
                    value[part] += w * k.value[part];
                    for a in 0..3 {
                        gradient[s][a][part] -= w * k.gradient[a][part];
                        gradient[t][a][part] += w * k.gradient[a][part];
                        for b in 0..3 {
                            let h = w * k.hessian[a][b][part];
                            hessian[s][s][a][b][part] += h;
                            hessian[t][t][a][b][part] += h;
                            hessian[s][t][a][b][part] -= h;
                            hessian[t][s][a][b][part] -= h;
                        }
                    }
                }
            }
        }
        (value, gradient, hessian)
    }

    fn pair_displacements(sites: &[RefSite]) -> Vec<Vec3> {
        let mut displacements = Vec::new();
        for a in sites {
            for b in sites {
                displacements.push(b.position - a.position);
            }
        }
        displacements
    }

    /// The shared body of the oracle comparisons: contract the phased kernels over the fixture's
    /// charges and hold value, gradient **and** Hessian to the direct sum, after asserting the
    /// compared quantities are of a non-trivial scale so the test cannot pass by comparing
    /// noise with noise.
    #[allow(clippy::needless_range_loop)] // walking three parallel tensors
    fn assert_matches_direct(
        cell: &Cell,
        sites: &[RefSite],
        q: Vec3,
        params: &EwaldParams,
        shells: i32,
        tolerance: f64,
    ) {
        let kernels = ewald_phased(cell, &pair_displacements(sites), q, params).unwrap();
        let (value, gradient, hessian) = contract(sites, &kernels);
        let reference = crate::pbc::ewald_reference::direct_phased_sum(cell, sites, q, shells);

        let scale = reference.value[0].abs();
        assert!(
            scale > 0.05,
            "the fixture's value collapsed to {scale}, so nothing is being tested"
        );
        let largest = reference
            .hessian
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .map(|c| c[0].abs())
            .fold(0.0_f64, f64::max);
        assert!(
            largest > 0.01,
            "the fixture's Hessian collapsed to {largest}"
        );

        for part in 0..2 {
            let error = (value[part] - reference.value[part]).abs();
            assert!(
                error < tolerance,
                "the phased value's part {part} is {} against a direct sum of {} (error {error:.2e})",
                value[part],
                reference.value[part]
            );
        }
        for s in 0..sites.len() {
            for a in 0..3 {
                for part in 0..2 {
                    let (got, want) = (gradient[s][a][part], reference.gradient[s][a][part]);
                    assert!(
                        (got - want).abs() < tolerance,
                        "gradient[{s}][{a}] part {part}: {got} vs {want}"
                    );
                }
            }
        }
        for s in 0..sites.len() {
            for u in 0..sites.len() {
                for a in 0..3 {
                    for b in 0..3 {
                        for part in 0..2 {
                            let got = hessian[s][u][a][b][part];
                            let want = reference.hessian[s][u][a][b][part];
                            assert!(
                                (got - want).abs() < tolerance,
                                "hessian[{s}][{u}][{a}][{b}] part {part}: {got} vs {want}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// **The independent check.** Everything else in this module compares the sum against itself.
    ///
    /// The other tests here confirm that the sum reduces to the unphased one at a reciprocal
    /// lattice vector, that its derivatives are the finite differences of its own value, and that
    /// reversing `q` conjugates it. All three hold for a sum carrying a wrong prefactor, a
    /// mis-signed self term or a dropped `G`, because all three are statements about internal
    /// consistency. This one compares against a computation that shares none of the algebra: no
    /// reciprocal sum, no `erfc`, no self term, no background — just `1/r`, summed over whole
    /// cells so the phase can make it converge.
    #[test]
    fn the_phased_sum_matches_a_direct_lattice_sum() {
        let (cell, sites, q) = oracle_fixture();
        let params = EwaldParams::tuned(18.0, 1.0e-14);
        assert_matches_direct(&cell, &sites, q, &params, 32, 1.0e-6);
    }

    /// The slab's independent check, against the same cell-bundled direct sum — which is
    /// dimension-aware and ring-truncates for a slab. The oracle's own 2D convergence is
    /// asserted before it is leaned on, so the tolerance is a statement about the Parry sum and
    /// not about the oracle's truncation.
    #[test]
    fn the_phased_slab_sum_matches_a_direct_lattice_sum() {
        let (cell, sites, q) = slab_oracle_fixture();
        let params = EwaldParams::tuned(18.0, 1.0e-14);
        let coarse = crate::pbc::ewald_reference::direct_phased_sum(&cell, &sites, q, 32).value[0];
        let fine = crate::pbc::ewald_reference::direct_phased_sum(&cell, &sites, q, 48).value[0];
        assert!(
            (fine - coarse).abs() < 2.0e-6,
            "the 2D oracle moved {:.2e} between 32 and 48 rings; widen the shells",
            (fine - coarse).abs()
        );
        assert_matches_direct(&cell, &sites, q, &params, 48, 1.0e-5);
    }

    /// The chain's independent check. The 1D oracle bundles fall off as `1/n³` with bounded
    /// phase partial sums, so a thousand segments put its own truncation well below the
    /// tolerance — asserted, not assumed.
    #[test]
    fn the_phased_chain_sum_matches_a_direct_lattice_sum() {
        let (cell, sites, q) = chain_oracle_fixture();
        let params = EwaldParams::tuned(18.0, 1.0e-14);
        let coarse = crate::pbc::ewald_reference::direct_phased_sum(&cell, &sites, q, 500).value[0];
        let fine = crate::pbc::ewald_reference::direct_phased_sum(&cell, &sites, q, 1000).value[0];
        assert!(
            (fine - coarse).abs() < 5.0e-7,
            "the 1D oracle moved {:.2e} between 500 and 1000 segments",
            (fine - coarse).abs()
        );
        assert_matches_direct(&cell, &sites, q, &params, 1000, 1.0e-6);
    }

    /// The direct sum is a sharp instrument only away from `Γ` and the zone boundary, and the
    /// documentation says by how much. This keeps those numbers honest.
    #[test]
    fn the_direct_sum_converges_where_the_oracle_claims_it_does() {
        use crate::pbc::ewald_reference::{direct_phased_sum, extrapolate_shells};
        let (cell, sites, q) = oracle_fixture();

        // A generic interior q: the raw sum is already converged.
        let coarse = direct_phased_sum(&cell, &sites, q, 24).value[0];
        let fine = direct_phased_sum(&cell, &sites, q, 32).value[0];
        assert!(
            (fine - coarse).abs() < 1.0e-5,
            "the interior q should be settled by 24 shells; it moved {:.2e}",
            (fine - coarse).abs()
        );

        // The zone boundary: not settled, and extrapolation is what recovers it.
        let mut boundary = Vec3::zero();
        for (index, b) in &cell.reciprocal_basis() {
            boundary += *b * [0.5_f64, 0.0, 0.0][*index];
        }
        let raw = direct_phased_sum(&cell, &sites, boundary, 32).value[0];
        let extrapolated = extrapolate_shells(&cell, &sites, boundary, 16, 32)[0];

        let params = EwaldParams::tuned(18.0, 1.0e-14);
        let kernels = ewald_phased(&cell, &pair_displacements(&sites), boundary, &params).unwrap();
        let truth = contract(&sites, &kernels).0[0];

        assert!(
            (raw - truth).abs() > 1.0e-4,
            "the raw zone-boundary sum is documented as unconverged; it agreed to {:.2e}, so the \
             documentation is now wrong rather than the code",
            (raw - truth).abs()
        );
        assert!(
            (extrapolated - truth).abs() < 5.0e-4,
            "extrapolating the zone boundary gave {extrapolated} against {truth}"
        );
    }

    /// At **any** reciprocal lattice vector the phases are one, so the phased sum must reproduce
    /// the unphased one — value, gradient and Hessian, with a vanishing imaginary part — in
    /// every dimensionality that has a reciprocal lattice.
    ///
    /// Testing only `q = 0` would pass while the shifted reciprocal set silently dropped the
    /// `G + q = 0` term at every other lattice vector, and those are exactly the wavevectors a
    /// supercell comparison lands on. For the chain the load-bearing part is the 1D path
    /// recognizing `q·a = 2πn` and taking the line-charge branch, rather than handing `2πn` to
    /// an Abel tail whose `w = 1/(1 − e^{iθ})` is singular there.
    #[test]
    fn the_phase_disappears_at_every_reciprocal_lattice_vector() {
        for cell in [cubic(9.0), slab(40.0), chain(40.0)] {
            let params = EwaldParams::for_cell(&cell, 1.0e-12);
            let displacements = probes();
            let reference = ewald_phased(&cell, &displacements, Vec3::zero(), &params).unwrap();

            let b = cell.reciprocal_basis();
            for (index, vector) in &b {
                for multiple in [1.0_f64, -1.0, 2.0] {
                    let q = *vector * multiple;
                    assert!(
                        is_reciprocal_lattice_vector(&cell, q),
                        "b{index} × {multiple} should be a lattice vector"
                    );
                    let got = ewald_phased(&cell, &displacements, q, &params).unwrap();
                    for (a, r) in got.iter().zip(&reference) {
                        assert!(
                            (a.value[0] - r.value[0]).abs() < 1.0e-9,
                            "value {} vs {}",
                            a.value[0],
                            r.value[0]
                        );
                        assert!(a.value[1].abs() < 1.0e-12, "imaginary {}", a.value[1]);
                        for alpha in 0..3 {
                            assert!((a.gradient[alpha][0] - r.gradient[alpha][0]).abs() < 1.0e-9);
                            assert!(a.gradient[alpha][1].abs() < 1.0e-12);
                            for beta in 0..3 {
                                assert!(
                                    (a.hessian[alpha][beta][0] - r.hessian[alpha][beta][0]).abs()
                                        < 1.0e-8
                                );
                                assert!(a.hessian[alpha][beta][1].abs() < 1.0e-11);
                            }
                        }
                    }
                }
            }
        }
    }

    /// The gradient and Hessian must be the derivatives of the value, at a `q` that is *not* a
    /// lattice vector so the imaginary parts are live — in every dimensionality.
    ///
    /// Differencing the value is the only way to check a derivative, and doing it in the complex
    /// plane checks the phase factors too: a `+i` where a `−i` belongs leaves the magnitudes
    /// right and the derivative wrong. The displacement is differenced along all three Cartesian
    /// axes: in 2D that plays the `∂/∂z` slab-kernel half against the in-plane `i(G+q)` half,
    /// and in 1D the transverse directions against the axial one.
    #[test]
    fn the_derivatives_are_the_derivatives_of_the_value() {
        for cell in [cubic(9.0), slab(40.0), chain(40.0)] {
            let params = EwaldParams::for_cell(&cell, 1.0e-12);
            let mut q = Vec3::zero();
            for (index, b) in &cell.reciprocal_basis() {
                q += *b * [0.37_f64, 0.11, 0.0][*index];
            }
            let base = Vec3::new(0.9, -1.4, 2.1);
            let analytic = ewald_phased(&cell, &[base], q, &params).unwrap()[0];

            let step = 1.0e-5;
            for alpha in 0..3 {
                let shifted = |signed: f64| {
                    let mut d = base;
                    match alpha {
                        0 => d.x += signed * step,
                        1 => d.y += signed * step,
                        _ => d.z += signed * step,
                    }
                    ewald_phased(&cell, &[d], q, &params).unwrap()[0]
                };
                let plus = shifted(1.0);
                let minus = shifted(-1.0);
                for part in 0..2 {
                    let numeric = (plus.value[part] - minus.value[part]) / (2.0 * step);
                    let got = analytic.gradient[alpha][part];
                    assert!(
                        (got - numeric).abs() < 1.0e-6 * numeric.abs().max(1.0),
                        "gradient[{alpha}][{part}]: {got} vs {numeric}"
                    );
                    for beta in 0..3 {
                        let numeric =
                            (plus.gradient[beta][part] - minus.gradient[beta][part]) / (2.0 * step);
                        let got = analytic.hessian[alpha][beta][part];
                        assert!(
                            (got - numeric).abs() < 1.0e-5 * numeric.abs().max(1.0),
                            "hessian[{alpha}][{beta}][{part}]: {got} vs {numeric}"
                        );
                    }
                }
            }
        }
    }

    /// `Φ_{−q}(d) = Φ_q(d)*`, because the sum is real term by term and only the phase is complex
    /// — in every dimensionality.
    ///
    /// This is what makes the dynamical matrix satisfy `D(−q) = D(q)*` and therefore have real
    /// frequencies; a sign error in the phase would break it here first.
    #[test]
    fn reversing_the_wavevector_conjugates_the_sum() {
        for cell in [cubic(9.0), slab(40.0), chain(40.0)] {
            let params = EwaldParams::for_cell(&cell, 1.0e-12);
            let basis = cell.reciprocal_basis();
            let q = basis[basis.len() - 1].1 * 0.29;
            let displacements = probes();
            let forward = ewald_phased(&cell, &displacements, q, &params).unwrap();
            let reverse = ewald_phased(&cell, &displacements, q * -1.0, &params).unwrap();
            for (f, r) in forward.iter().zip(&reverse) {
                assert!((f.value[0] - r.value[0]).abs() < 1.0e-12);
                assert!((f.value[1] + r.value[1]).abs() < 1.0e-12);
                for alpha in 0..3 {
                    assert!((f.gradient[alpha][0] - r.gradient[alpha][0]).abs() < 1.0e-12);
                    assert!((f.gradient[alpha][1] + r.gradient[alpha][1]).abs() < 1.0e-12);
                }
            }
        }
    }

    /// Moving the real/reciprocal split must not move the phased sum — in 2D *or* 3D (1D has no
    /// split and no `α`).
    ///
    /// In 2D this pins the `π/(A|G+q|)` prefactor: a factor-of-two error from keeping the
    /// folded prefactor over the unfolded set changes how much weight the reciprocal half
    /// carries relative to the real half, and every *internal* identity — conjugation,
    /// lattice-vector reduction, derivative consistency — is blind to it because both halves
    /// scale together under it. Run at a generic `q` and at a reciprocal lattice vector, so the
    /// sheet branch is held to the same standard.
    ///
    /// It also checks the **imaginary** part, and that is not pedantry: this is the test that
    /// found the conjugated reciprocal phase the 3D sum shipped with. The oracle contraction is
    /// identically real over its ±d pairs, the folding comparison sits at `q = b/2` where the
    /// shifted set is symmetric under negation, and every internal identity conjugates both
    /// halves together — so an `e^{+i(G+q)·d}` where `e^{−i(G+q)·d}` belongs moved nothing
    /// anywhere until two values of `α` disagreed about `Im Φ_q` in the fourth digit.
    #[test]
    fn the_phased_sum_is_independent_of_the_splitting_parameter() {
        for cell in [cubic(9.0), slab(40.0)] {
            let displacements = probes();
            for q in [interior_q(&cell), cell.reciprocal_basis()[0].1] {
                let reference =
                    ewald_phased(&cell, &displacements, q, &EwaldParams::tuned(14.0, 1.0e-14))
                        .unwrap();
                let other =
                    ewald_phased(&cell, &displacements, q, &EwaldParams::tuned(20.0, 1.0e-14))
                        .unwrap();
                for (a, r) in other.iter().zip(&reference) {
                    for part in 0..2 {
                        assert!(
                            (a.value[part] - r.value[part]).abs() < 1.0e-9,
                            "value part {part} moved from {} to {} with the splitting",
                            r.value[part],
                            a.value[part]
                        );
                        for alpha in 0..3 {
                            assert!(
                                (a.gradient[alpha][part] - r.gradient[alpha][part]).abs() < 1.0e-9,
                                "gradient[{alpha}][{part}] moved with the splitting"
                            );
                            for beta in 0..3 {
                                assert!(
                                    (a.hessian[alpha][beta][part] - r.hessian[alpha][beta][part])
                                        .abs()
                                        < 1.0e-8,
                                    "hessian[{alpha}][{beta}][{part}] moved with the splitting"
                                );
                            }
                        }
                    }
                }
                let scale = reference
                    .iter()
                    .map(|k| k.value[0].abs().max(k.value[1].abs()))
                    .fold(0.0_f64, f64::max);
                assert!(scale > 0.05, "the values collapsed to {scale}");
            }
        }
    }

    /// `Φ_q(d + T₀) = e^{−iq·T₀} Φ_q(d)` for any lattice translation `T₀` — exact, from
    /// re-indexing the sum — and the same for the gradient and Hessian, which are derivatives
    /// in `d` and see `T₀` only through that envelope.
    ///
    /// This is the identity that fixes the *sign* of the reciprocal-space phase. Conjugation,
    /// derivative consistency and the (real) oracle contraction all survive an
    /// `e^{+i(G+q)·d}`; this does not, because the real-space half satisfies the identity with
    /// the minus sign and a conjugated smooth half satisfies it with the plus.
    #[test]
    fn translating_the_displacement_by_a_lattice_vector_rephases_the_sum() {
        for cell in [cubic(9.0), slab(40.0), chain(40.0)] {
            let params = EwaldParams::for_cell(&cell, 1.0e-12);
            let q = interior_q(&cell);
            let d = Vec3::new(0.9, -1.4, 2.1);
            let periodic = cell.periodic_indices();
            let mut t0 = cell.vector(periodic[0]);
            if periodic.len() > 1 {
                t0 += cell.vector(periodic[1]) * 2.0;
            }
            let angle = -q.dot(t0);
            let envelope = [angle.cos(), angle.sin()];

            let base = ewald_phased(&cell, &[d], q, &params).unwrap()[0];
            let moved = ewald_phased(&cell, &[d + t0], q, &params).unwrap()[0];

            let check = |got: [f64; 2], reference: [f64; 2], label: &str| {
                let want = complex_mul(envelope, reference);
                assert!(
                    (got[0] - want[0]).abs() < 1.0e-9 && (got[1] - want[1]).abs() < 1.0e-9,
                    "{label}: ({}, {}) vs e^(-iq*T0) * base = ({}, {})",
                    got[0],
                    got[1],
                    want[0],
                    want[1]
                );
            };
            check(moved.value, base.value, "value");
            assert!(
                base.value[1].abs() > 1.0e-3,
                "the base imaginary part collapsed to {}, so the envelope is untested",
                base.value[1]
            );
            for alpha in 0..3 {
                check(
                    moved.gradient[alpha],
                    base.gradient[alpha],
                    &format!("gradient[{alpha}]"),
                );
                for beta in 0..3 {
                    check(
                        moved.hessian[alpha][beta],
                        base.hessian[alpha][beta],
                        &format!("hessian[{alpha}][{beta}]"),
                    );
                }
            }
        }
    }

    /// As `q → 0` in-plane, the `g = |q|` member carries the divergence and the sheet:
    /// `Φ_q(d) − 2π/(A|q|) → Φ_0(d)` at rate `O(|q|)`, and the imaginary part approaches the
    /// finite, `q̂`-dependent `(q̂·d)·2π/A` — the 2D LO–TO non-analyticity, which is physics
    /// and must be preserved rather than smoothed.
    ///
    /// This is the conditioning statement in the module documentation, measured: the divergent
    /// piece is `d`-independent, so subtracting it once leaves a limit that is the unphased
    /// kernel itself.
    #[test]
    fn the_shifted_sheet_term_approaches_the_smeared_sheet() {
        let cell = slab(40.0);
        let d = Vec3::new(0.7, -1.1, 1.3);
        let params = EwaldParams::tuned(16.0, 1.0e-14);
        let area = cell.measure();
        let unphased = ewald_phased(&cell, &[d], Vec3::zero(), &params).unwrap()[0];

        let b0 = cell.reciprocal_basis()[0].1;
        let mut residuals = Vec::new();
        let mut imaginary_errors = Vec::new();
        for t in [1.0e-3_f64, 5.0e-4] {
            let q = b0 * t;
            let kernel = ewald_phased(&cell, &[d], q, &params).unwrap()[0];
            let divergent = TAU / (area * q.norm());
            residuals.push((kernel.value[0] - divergent - unphased.value[0]).abs());
            // −(q̂·d): the divergent member is (π/(A|q|)) K e^{−iq·d}, with the minus the
            // re-indexing identity Φ_q(d+T₀) = e^{−iq·T₀} Φ_q(d) forces on every member.
            let qhat = q * (1.0 / q.norm());
            imaginary_errors.push((kernel.value[1] + qhat.dot(d) * TAU / area).abs());
        }
        // O(|q|): halving |q| halves the residual. The window is generous because the residual
        // also carries the O(q) motion of every G ≠ 0 member, not only the sheet expansion.
        let ratio = residuals[0] / residuals[1];
        assert!(
            (1.6..2.4).contains(&ratio),
            "the sheet residual shrinks by {ratio:.3}x per halving of |q| \
             ({:.3e} then {:.3e}); 2 is the claimed O(|q|)",
            residuals[0],
            residuals[1]
        );
        assert!(
            residuals[1] < 1.0e-2,
            "the residual should be small in absolute terms too: {:.3e}",
            residuals[1]
        );
        assert!(
            imaginary_errors[1] < imaginary_errors[0] && imaginary_errors[1] < 1.0e-2,
            "the imaginary part should approach (q̂·d)·2π/A: errors {:.3e}, {:.3e}",
            imaginary_errors[0],
            imaginary_errors[1]
        );
    }

    /// The phased chain sum must not depend on how many images were summed directly, because
    /// the Abel tail accounts for the rest.
    ///
    /// The naive truncation error at `q ≠ 0` is `O(1/N)` — percent-level on the value at 150
    /// images — so agreement to 1e-12 across an eightfold range of `N` is direct evidence the
    /// transformed tail is right, of a kind no fixed tolerance on the value could give.
    #[test]
    fn the_phased_chain_sum_is_independent_of_the_image_count() {
        let cell = chain(40.0);
        let displacements = probes();
        let q = interior_q(&cell);
        let kernels_at = |images: usize| {
            let mut params = EwaldParams::tuned(18.0, 1.0e-14);
            params.chain_images = images;
            ewald_phased(&cell, &displacements, q, &params).unwrap()
        };
        let reference = kernels_at(1200);
        let scale = reference
            .iter()
            .map(|k| k.value[0].abs())
            .fold(0.0_f64, f64::max);
        assert!(scale > 0.05, "the chain values collapsed to {scale}");
        for images in [150, 400] {
            let got = kernels_at(images);
            for (a, r) in got.iter().zip(&reference) {
                for part in 0..2 {
                    assert!(
                        (a.value[part] - r.value[part]).abs() < 1.0e-12,
                        "{images} images: value part {part} is {} vs {}",
                        a.value[part],
                        r.value[part]
                    );
                    for alpha in 0..3 {
                        assert!(
                            (a.gradient[alpha][part] - r.gradient[alpha][part]).abs() < 1.0e-12
                        );
                        for beta in 0..3 {
                            assert!(
                                (a.hessian[alpha][beta][part] - r.hessian[alpha][beta][part]).abs()
                                    < 1.0e-12
                            );
                        }
                    }
                }
            }
        }
    }

    /// The `q = 0` chain tail really is the `1/N⁴` object its comment claims: the residual
    /// against a 4000-image reference must shrink by the fourth power of the image count, not
    /// merely shrink.
    ///
    /// Asserting the exponent is the point, exactly as in the unphased chain-energy tests: a
    /// tolerance alone could be met by a slowly-growing error that had not grown yet, while a
    /// ratio of residuals cannot. Measured when this was written: `5.2·10⁻¹²` at 150 images,
    /// `1.0·10⁻¹³` at 400, ratio 50 against the `(400/150)⁴ ≈ 51` of a fourth-power law.
    #[test]
    fn the_unphased_chain_tail_converges_as_the_fourth_power() {
        let cell = chain(40.0);
        let displacements = probes();
        let kernels_at = |images: usize| {
            let mut params = EwaldParams::tuned(18.0, 1.0e-14);
            params.chain_images = images;
            ewald_phased(&cell, &displacements, Vec3::zero(), &params).unwrap()
        };
        let reference = kernels_at(4000);
        let worst_at = |images: usize| {
            let got = kernels_at(images);
            let mut worst = 0.0_f64;
            for (a, r) in got.iter().zip(&reference) {
                worst = worst.max((a.value[0] - r.value[0]).abs());
                for alpha in 0..3 {
                    worst = worst.max((a.gradient[alpha][0] - r.gradient[alpha][0]).abs());
                    for beta in 0..3 {
                        worst = worst
                            .max((a.hessian[alpha][beta][0] - r.hessian[alpha][beta][0]).abs());
                    }
                }
            }
            worst
        };
        let coarse = worst_at(150);
        let fine = worst_at(400);
        assert!(
            coarse < 2.0e-11,
            "the 150-image residual is {coarse:.3e}; the tail correction has stopped working"
        );
        let ratio = coarse / fine;
        assert!(
            (25.0..110.0).contains(&ratio),
            "the residual shrinks by {ratio:.1}x from 150 to 400 images \
             ({coarse:.3e} to {fine:.3e}); (400/150)^4 = 51 is the claimed fourth power"
        );
    }

    /// The `q = 0` kernel contraction must reproduce [`crate::pbc::ewald`]'s energy — the
    /// Parry slab sum and the cell-grouped chain sum, neither of which shares a line of code
    /// with the phased assembly.
    ///
    /// The derivative tests cannot pin the *value*, and the neutral oracle contraction cannot
    /// see a `d`-independent constant, so this is the only place the chain's line-charge
    /// convention — `−2(H_N + ln(L/L₀))/L` per displacement — is held to agree with the one the
    /// SCF's own potentials use. A wrong reference length `L₀` would shift the charged case by
    /// `~Q² ln(L)/L` eV and leave every other test in this module green.
    ///
    /// Measured agreement when this was written: below `10⁻¹¹` eV for the neutral slab and
    /// chain, and `2.7·10⁻⁶` eV for the *charged* chain at 2000 images — the last is not
    /// rounding but the charged per-displacement `C(d)/n³` tail this kernel corrects and the
    /// cell-grouped path deliberately skips (a charged distribution's dipole is
    /// origin-dependent, so its grouped tail formula means nothing there).
    #[test]
    fn the_kernel_contraction_reproduces_the_ewald_energy() {
        use crate::constants::PM3_EV;
        use crate::pbc::ewald::{ewald, ChargeSite};

        let (slab_cell, slab_sites, _) = slab_oracle_fixture();
        let (chain_cell, chain_sites, _) = chain_oracle_fixture();
        let mut charged_sites = chain_sites.clone();
        charged_sites[0].charge += 1.0;

        for (label, cell, sites, tolerance) in [
            ("neutral slab", &slab_cell, &slab_sites, 1.0e-11),
            ("neutral chain", &chain_cell, &chain_sites, 1.0e-11),
            ("charged chain", &chain_cell, &charged_sites, 5.0e-6),
        ] {
            let charge_sites: Vec<ChargeSite> = sites
                .iter()
                .enumerate()
                .map(|(index, s)| ChargeSite::on_atom(s.position, s.charge, index))
                .collect();
            let mut params = EwaldParams::tuned(16.0, 1.0e-13);
            params.chain_images = 2000;
            let reference = ewald(cell, &charge_sites, &params).unwrap().energy_ev;

            let kernels =
                ewald_phased(cell, &pair_displacements(sites), Vec3::zero(), &params).unwrap();
            let mut energy = 0.0;
            for (i, site_i) in sites.iter().enumerate() {
                for (j, site_j) in sites.iter().enumerate() {
                    energy +=
                        0.5 * site_i.charge * site_j.charge * kernels[i * sites.len() + j].value[0];
                }
            }
            energy *= PM3_EV;

            assert!(
                reference.abs() > 0.3,
                "{label}: the reference energy collapsed to {reference}"
            );
            assert!(
                (energy - reference).abs() < tolerance,
                "{label}: kernel contraction {energy} vs ewald {reference} \
                 (difference {:.3e}, tolerance {tolerance:.0e})",
                (energy - reference).abs()
            );
        }
    }

    /// A charged 1D contraction is finite and image-count-independent at `q ≠ 0`, with no line
    /// charge anywhere in sight — where at `q = 0` the same contraction is defined only within
    /// the line-charge convention.
    ///
    /// This is the discontinuity the module documentation gates on the reciprocal lattice:
    /// `Q² Σ_n e^{iqnL}/(nL)` converges on its own, and carrying the `q = 0` neutralization
    /// over would shift `Φ_q` by a constant that no `q = 0` test would ever see.
    #[test]
    fn a_charged_chain_contraction_is_finite_at_nonzero_q() {
        let cell = chain(40.0);
        let sites = vec![
            RefSite {
                position: Vec3::new(0.4, 0.6, 0.9),
                charge: 0.7,
            },
            RefSite {
                position: Vec3::new(3.1, 1.2, -1.4),
                charge: -0.2,
            },
            RefSite {
                position: Vec3::new(4.6, -0.6, 0.7),
                charge: 0.4,
            },
        ];
        let total: f64 = sites.iter().map(|s| s.charge).sum();
        assert!(total.abs() > 0.5, "the fixture must actually be charged");
        let q = interior_q(&cell);
        let displacements = pair_displacements(&sites);

        let value_at = |images: usize| {
            let mut params = EwaldParams::tuned(18.0, 1.0e-14);
            params.chain_images = images;
            let kernels = ewald_phased(&cell, &displacements, q, &params).unwrap();
            contract(&sites, &kernels).0
        };
        let coarse = value_at(200);
        let fine = value_at(800);
        assert!(coarse[0].is_finite() && coarse[1].is_finite());
        assert!(
            coarse[0].abs() + coarse[1].abs() > 0.01,
            "the charged contraction collapsed to ({}, {})",
            coarse[0],
            coarse[1]
        );
        for part in 0..2 {
            assert!(
                (coarse[part] - fine[part]).abs() < 1.0e-10,
                "part {part}: 200 images give {}, 800 give {}",
                coarse[part],
                fine[part]
            );
        }
    }

    /// Zero periodic directions: the kernel is the bare `1/r` and its derivatives, exactly.
    #[test]
    fn an_isolated_cell_reduces_to_the_bare_kernel() {
        let cell = Cell::isolated();
        let displacements = vec![Vec3::new(1.3, -0.4, 2.2), Vec3::zero()];
        let params = EwaldParams::default();
        // Any q phases the single T = 0 term by one.
        let kernels =
            ewald_phased(&cell, &displacements, Vec3::new(0.3, 0.2, -0.1), &params).unwrap();

        let d = displacements[0];
        let r = d.norm();
        let dv = [d.x, d.y, d.z];
        assert!((kernels[0].value[0] - 1.0 / r).abs() < 1.0e-14);
        assert!(kernels[0].value[1].abs() < 1.0e-15);
        for a in 0..3 {
            assert!((kernels[0].gradient[a][0] + dv[a] / r.powi(3)).abs() < 1.0e-14);
            for b in 0..3 {
                let delta = if a == b { 1.0 } else { 0.0 };
                let want = 3.0 * dv[a] * dv[b] / r.powi(5) - delta / r.powi(3);
                assert!((kernels[0].hessian[a][b][0] - want).abs() < 1.0e-14);
            }
        }
        // The zero displacement is the caller's self-interaction and carries nothing here.
        assert!(kernels[1].value[0].abs() < 1.0e-15);
    }

    /// A wavevector with a component outside the periodic subspace is refused, with the reason:
    /// no translation would ever read it, so accepting it would silently answer for the
    /// projected `q` instead.
    #[test]
    fn a_wavevector_off_the_periodic_subspace_is_refused() {
        let params = EwaldParams::tuned(14.0, 1.0e-12);
        let slab_error = ewald_phased(&slab(40.0), &probes(), Vec3::new(0.0, 0.0, 0.2), &params)
            .expect_err("a slab has no out-of-plane wavevector");
        assert!(slab_error.to_string().contains("periodic subspace"));

        let chain_error = ewald_phased(&chain(40.0), &probes(), Vec3::new(0.1, 0.3, 0.0), &params)
            .expect_err("a chain has no transverse wavevector");
        assert!(chain_error.to_string().contains("periodic subspace"));
    }
}
