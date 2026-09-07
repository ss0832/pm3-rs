// SPDX-License-Identifier: GPL-3.0-or-later

//! Second derivatives of the Ewald sum with respect to atomic positions.
//!
//! The force constants of an ionic or polar crystal are dominated at long range by exactly this
//! term, so a periodic Hessian without it is not merely imprecise — it misses the interaction that
//! makes the optical branches behave the way they do.
//!
//! # At fixed charges
//!
//! Everything here holds the site charges fixed and moves only the positions. That is the
//! *skeleton* second derivative; the charges' own response to the displacement is the CPHF
//! problem and belongs to [`crate::pbc::hessian`]. Splitting them this way is what lets the two be
//! tested separately.
//!
//! # Reduced straight to atoms
//!
//! A multipole site is rigidly attached to its atom — the offsets `dd` and `qq` are atomic
//! parameters, not degrees of freedom — so `∂/∂R_atom = Σ_{sites of atom} ∂/∂r_site`, and the
//! `3n_site × 3n_site` object is contracted into a `3n_atom × 3n_atom` one as it is built. For a
//! system with ten sites per atom that is a hundredfold saving in both memory and the work of
//! writing it down.
//!
//! # Dimensionality
//!
//! 3D keeps its dedicated implementation below. 1D and 2D go through the phased lattice sum at
//! `q = 0` ([`crate::pbc::phased`]), which is the same object — `Σ_T Φ''(d + T)` — computed by
//! the machinery that already knows the wire and sheet kernels. Only an *isolated* cell (or an
//! empty site list) reports `None` now, matching the convention [`crate::pbc::ewald`] uses for
//! the virial: a missing derivative is reported as missing rather than as a zero matrix,
//! because a silently zero block would let a frequency analysis "succeed" with nonsense in it —
//! and an isolated system's Hessian belongs to the molecular path, not here.

use std::f64::consts::{PI, TAU};

use crate::cell::Cell;
use crate::error::Result;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::neighbor::NeighborList;
use crate::pbc::ewald::{ChargeSite, EwaldParams};
use crate::special::{erf, erfc};

const PM3_EV: f64 = crate::constants::PM3_EV;

/// `∂²E_Ewald/∂R_i∂R_j` at fixed charges, `3n_atoms × 3n_atoms` (eV/Bohr²).
///
/// `None` only for an isolated cell or an empty site list.
pub fn ewald_atom_hessian(
    cell: &Cell,
    sites: &[ChargeSite],
    params: &EwaldParams,
    n_atoms: usize,
) -> Result<Option<Matrix>> {
    if cell.n_periodic() == 0 || sites.is_empty() {
        return Ok(None);
    }
    if cell.n_periodic() != 3 {
        return reduced_dimensionality(cell, sites, params, n_atoms).map(Some);
    }
    let mut hessian = Matrix::zeros(3 * n_atoms, 3 * n_atoms);
    real_space(cell, sites, params, &mut hessian);
    reciprocal(cell, sites, params, n_atoms, &mut hessian);
    remove_excluded_smooth_part(sites, params, &mut hessian);
    // The self and background terms do not depend on where the sites are, so they contribute
    // nothing here.
    Ok(Some(hessian))
}

/// The 1D and 2D Hessian, by the phased lattice sum at `q = 0`.
///
/// Not a separate implementation: [`crate::pbc::dfpt`]'s `phased_lattice_sum` *is* the Ewald
/// Hessian at any `q` including zero, and it is validated against the dedicated 3D path above.
/// This assembles the same reduction here so a Γ-point Hessian in reduced dimensionality does
/// not need the whole dynamical-matrix machinery.
///
/// # Why no exclusions appear
///
/// The 3D path skips same-owner pairs at `T = 0` by hand and then walks back the `erf` half its
/// reciprocal sum kept anyway. Here both are **automatic**, because the exclusion cancels in
/// the atom reduction rather than in the sum: the phased assembly writes `2·plain` onto the two
/// diagonal blocks and `−(mixed + mixed*)` onto the two mixed ones, and for a same-owner pair
/// all four land on the *same* `(a, a)` block. At `T = 0` `plain` and `mixed` both carry the
/// unphased term — the phase of `T = 0` is one at every `q` — so it cancels identically, and so
/// does every image term, `+H` against `−H` on the same block. A pair of sites riding the same
/// atom contributes nothing to the `q = 0` atom Hessian in either implementation (moving the
/// atom moves both ends), so same-owner pairs and self-pairs are simply not enumerated.
fn reduced_dimensionality(
    cell: &Cell,
    sites: &[ChargeSite],
    params: &EwaldParams,
    n_atoms: usize,
) -> Result<Matrix> {
    let mut displacements = Vec::new();
    let mut weights = Vec::new();
    for (i, site_i) in sites.iter().enumerate() {
        for site_j in sites.iter().skip(i + 1) {
            if site_i.owner == site_j.owner {
                continue; // contributes exactly zero — see above
            }
            displacements.push(site_j.position - site_i.position);
            weights.push((site_i.owner, site_j.owner, site_i.charge * site_j.charge));
        }
    }
    let kernels = crate::pbc::phased::ewald_phased(cell, &displacements, Vec3::zero(), params)?;

    let mut hessian = Matrix::zeros(3 * n_atoms, 3 * n_atoms);
    for (kernel, (a, b, qq)) in kernels.iter().zip(&weights) {
        let coefficient = PM3_EV * qq;
        for alpha in 0..3 {
            for beta in 0..3 {
                // At q = 0 the kernel is real; its imaginary slot holds rounding noise only.
                let h = coefficient * kernel.hessian[alpha][beta][0];
                hessian[(3 * a + alpha, 3 * a + beta)] += h;
                hessian[(3 * b + alpha, 3 * b + beta)] += h;
                hessian[(3 * a + alpha, 3 * b + beta)] -= h;
                hessian[(3 * b + alpha, 3 * a + beta)] -= h;
            }
        }
    }
    Ok(hessian)
}

/// Scatter one site-pair `3×3` block into the atom Hessian.
///
/// `block` is `∂²E/∂r_a∂r_b` for the *pair*, i.e. the mixed block. The two diagonal blocks are its
/// negative, which is the pair's own statement of translational invariance.
fn scatter(hessian: &mut Matrix, owner_a: usize, owner_b: usize, block: &[[f64; 3]; 3]) {
    for (alpha, row) in block.iter().enumerate() {
        for (beta, value) in row.iter().enumerate() {
            let (ia, ib) = (3 * owner_a + alpha, 3 * owner_b + beta);
            hessian[(ia, ib)] += value;
            hessian[(ib, ia)] += value;
            hessian[(3 * owner_a + alpha, 3 * owner_a + beta)] -= value;
            hessian[(3 * owner_b + alpha, 3 * owner_b + beta)] -= value;
        }
    }
}

/// The `3×3` mixed block of a central pair potential `K q_i q_j f(r)`.
///
/// `∂²/∂r_a∂r_b [f(r)] = −[ (f'' − f'/r) r̂⊗r̂ + (f'/r) I ]`, the leading minus because moving `b`
/// away is moving `a` toward.
fn central_block(d: Vec3, r: f64, coefficient: f64, first: f64, second: f64) -> [[f64; 3]; 3] {
    let unit = d * (1.0 / r);
    let radial = second - first / r;
    let u = unit.to_array();
    let mut block = [[0.0; 3]; 3];
    for (alpha, row) in block.iter_mut().enumerate() {
        for (beta, slot) in row.iter_mut().enumerate() {
            let isotropic = if alpha == beta { first / r } else { 0.0 };
            *slot = -coefficient * (radial * u[alpha] * u[beta] + isotropic);
        }
    }
    block
}

/// `Σ q_i q_j erfc(αr)/r` over images inside the cutoff.
fn real_space(cell: &Cell, sites: &[ChargeSite], params: &EwaldParams, hessian: &mut Matrix) {
    let positions: Vec<Vec3> = sites.iter().map(|s| s.position).collect();
    let list = NeighborList::build_from_positions(&positions, Some(cell), params.real_cutoff);
    let alpha = params.alpha;
    let two_alpha_over_sqrt_pi = 2.0 * alpha / PI.sqrt();

    for pair in list.unique() {
        let r = pair.r;
        if r <= 0.0 || excluded(sites, pair.a, pair.b, pair.t) {
            continue;
        }
        let gaussian = (-(alpha * r).powi(2)).exp();
        let complement = erfc(alpha * r);
        // f  = erfc(αr)/r
        // f' = −erfc(αr)/r² − (2α/√π) e^{−α²r²}/r
        // f''=  2erfc(αr)/r³ + (2α/√π) e^{−α²r²} (2α² + 2/r²)
        let first = -(complement / (r * r)) - two_alpha_over_sqrt_pi * gaussian / r;
        let second = 2.0 * complement / (r * r * r)
            + two_alpha_over_sqrt_pi * gaussian * (2.0 * alpha * alpha + 2.0 / (r * r));
        let coefficient = PM3_EV * sites[pair.a].charge * sites[pair.b].charge;
        let block = central_block(pair.dvec, r, coefficient, first, second);
        scatter(hessian, sites[pair.a].owner, sites[pair.b].owner, &block);
    }
}

/// The `erf` half that the reciprocal sum keeps for pairs the real-space sum excluded.
///
/// The mirror of [`crate::pbc::ewald`]'s own correction, and just as easy to forget: without it the
/// force constants carry a spurious intra-atomic term that no symmetry check would catch, since it
/// is translationally invariant like everything else.
fn remove_excluded_smooth_part(sites: &[ChargeSite], params: &EwaldParams, hessian: &mut Matrix) {
    let alpha = params.alpha;
    let two_alpha_over_sqrt_pi = 2.0 * alpha / PI.sqrt();
    for a in 0..sites.len() {
        for b in (a + 1)..sites.len() {
            if sites[a].owner != sites[b].owner {
                continue;
            }
            let d = sites[b].position - sites[a].position;
            let r = d.norm();
            if r <= 0.0 {
                continue;
            }
            let gaussian = (-(alpha * r).powi(2)).exp();
            let smooth = erf(alpha * r);
            // g  = erf(αr)/r
            // g' = −erf(αr)/r² + (2α/√π) e^{−α²r²}/r
            // g''=  2erf(αr)/r³ − (2α/√π) e^{−α²r²} (2α² + 2/r²)
            //
            // `f + g = 1/r` and `f'' + g'' = 2/r³`, which is the check that fixes both signs.
            let first = -(smooth / (r * r)) + two_alpha_over_sqrt_pi * gaussian / r;
            let second = 2.0 * smooth / (r * r * r)
                - two_alpha_over_sqrt_pi * gaussian * (2.0 * alpha * alpha + 2.0 / (r * r));
            let coefficient = -PM3_EV * sites[a].charge * sites[b].charge;
            let block = central_block(d, r, coefficient, first, second);
            scatter(hessian, sites[a].owner, sites[b].owner, &block);
        }
    }
}

#[inline]
fn excluded(sites: &[ChargeSite], a: usize, b: usize, t: [i32; 3]) -> bool {
    t == [0, 0, 0] && sites[a].owner == sites[b].owner
}

/// The reciprocal sum's second derivative.
///
/// `E = K P Σ_G A(G) |S(G)|²` with `S(G) = Σ_i q_i e^{iG·r_i}`, so
///
/// ```text
/// ∂²|S|²/∂r_iα∂r_jβ = +2 q_i q_j G_α G_β cos(G·r_ij)      (i ≠ j)
/// ```
///
/// and the on-site block is minus the sum of that row — the reciprocal sum satisfies the acoustic
/// sum rule term by term, which is worth knowing when a phonon calculation comes out with a
/// non-zero acoustic branch: the cause is not here.
fn reciprocal(
    cell: &Cell,
    sites: &[ChargeSite],
    params: &EwaldParams,
    n_atoms: usize,
    hessian: &mut Matrix,
) {
    use rayon::prelude::*;

    let volume = cell.measure();
    // 2π/V, doubled because only half of each ±G pair is enumerated.
    let prefactor = 2.0 * TAU / volume;
    let inv_four_alpha2 = 1.0 / (4.0 * params.alpha * params.alpha);
    let gvectors = crate::pbc::ewald::reciprocal_vectors_for(cell, params.gmax);

    // **Per-atom structure factors, not a loop over site pairs.**
    //
    // This was `O(N_G · N_sites²)` and is the largest single phase of a Γ-point Hessian —
    // measured at 62% of a 48-atom run, against 38% for the coupled-perturbed response everyone
    // assumes is the expensive part and under 1% for the skeleton
    // (`examples/hessian_profile.rs`).
    //
    // The inner sum is a structure factor in disguise. What the site loop accumulates for an
    // atom pair `(A, B)` is `Σ_{a∈A, b∈B} q_a q_b cos(G·(r_a − r_b))`, and that is exactly
    // `Re[S_A S_B*]` for `S_A(G) = Σ_{a∈A} q_a e^{iG·r_a}`. For the same-atom pairs the site
    // loop visits — which exist because the real-space sum excludes them and the reciprocal sum
    // does not — the sum over `a < b` within `A` is `½(|S_A|² − Σ_{a∈A} q_a²)`.
    //
    // So `N_sites²` becomes `N_sites` to build the factors plus `N_atoms²` to combine them. PM3
    // puts four to five multipole sites on every heavy atom, so that is roughly a twentyfold cut
    // in the inner work before any parallelism.
    //
    // The site list is grouped by owner in increasing order, which is what makes the two forms
    // visit the same unordered pairs. `debug_assert` rather than a runtime check: it is a
    // property of how `pbc::gamma` builds the list, not of the input.
    debug_assert!(
        sites.windows(2).all(|w| w[0].owner <= w[1].owner),
        "the reciprocal Hessian's structure-factor form needs the sites grouped by owner"
    );

    // One partial Hessian per chunk of G, summed in chunk order. Fixed chunks reduced in index
    // order rather than `par_iter().reduce()`, so the last bits do not depend on which thread
    // finished first — the same reason the Ewald energy is written that way.
    let chunk = gvectors
        .len()
        .div_ceil(rayon::current_num_threads().max(1) * 4)
        .max(1);
    let partials: Vec<Matrix> = gvectors
        .par_chunks(chunk)
        .map(|gs| {
            let mut partial = Matrix::zeros(3 * n_atoms, 3 * n_atoms);
            let mut sre = vec![0.0f64; n_atoms];
            let mut sim = vec![0.0f64; n_atoms];
            let mut q2 = vec![0.0f64; n_atoms];
            for g in gs {
                let g2 = g.norm2();
                let amplitude = (-g2 * inv_four_alpha2).exp() / g2;
                if amplitude == 0.0 {
                    continue;
                }
                let scale = PM3_EV * prefactor * amplitude * 2.0;
                let components = g.to_array();

                sre.iter_mut().for_each(|v| *v = 0.0);
                sim.iter_mut().for_each(|v| *v = 0.0);
                q2.iter_mut().for_each(|v| *v = 0.0);
                for site in sites {
                    let (sin, cos) = g.dot(site.position).sin_cos();
                    sre[site.owner] += site.charge * cos;
                    sim[site.owner] += site.charge * sin;
                    q2[site.owner] += site.charge * site.charge;
                }

                let mut place = |a: usize, b: usize, weight: f64| {
                    let coefficient = scale * weight;
                    let mut block = [[0.0; 3]; 3];
                    for (alpha, row) in block.iter_mut().enumerate() {
                        for (beta, slot) in row.iter_mut().enumerate() {
                            *slot = coefficient * components[alpha] * components[beta];
                        }
                    }
                    scatter(&mut partial, a, b, &block);
                };

                for a in 0..n_atoms {
                    // The site pairs inside atom `a`, which the site loop reached as `a < b`
                    // with both owners equal.
                    place(a, a, 0.5 * (sre[a] * sre[a] + sim[a] * sim[a] - q2[a]));
                    for b in (a + 1)..n_atoms {
                        place(a, b, sre[a] * sre[b] + sim[a] * sim[b]);
                    }
                }
            }
            partial
        })
        .collect();

    for partial in partials {
        for i in 0..hessian.rows {
            for j in 0..hessian.cols {
                hessian[(i, j)] += partial[(i, j)];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbc::ewald::ewald;

    fn cubic_pair(edge: f64) -> (Cell, Vec<ChargeSite>) {
        let cell = Cell::cubic(edge).unwrap();
        let sites = vec![
            ChargeSite {
                position: Vec3::new(0.3, 0.2, 0.1),
                charge: 1.0,
                owner: 0,
            },
            ChargeSite {
                position: Vec3::new(2.1, 0.4, 0.7),
                charge: -0.6,
                owner: 1,
            },
            ChargeSite {
                position: Vec3::new(1.0, 2.6, 0.2),
                charge: -0.4,
                owner: 2,
            },
        ];
        (cell, sites)
    }

    /// Move an atom's sites the way the real code does — rigidly.
    fn displaced(sites: &[ChargeSite], atom: usize, axis: usize, delta: f64) -> Vec<ChargeSite> {
        sites
            .iter()
            .map(|s| {
                let mut moved = *s;
                if s.owner == atom {
                    match axis {
                        0 => moved.position.x += delta,
                        1 => moved.position.y += delta,
                        _ => moved.position.z += delta,
                    }
                }
                moved
            })
            .collect()
    }

    #[test]
    fn matches_finite_differences_of_the_analytic_gradient() {
        let (cell, sites) = cubic_pair(9.0);
        let params = EwaldParams::for_cell(&cell, 1.0e-12);
        let hessian = ewald_atom_hessian(&cell, &sites, &params, 3)
            .unwrap()
            .expect("3D reports a Hessian");

        let step = 1.0e-5;
        for atom in 0..3 {
            for axis in 0..3 {
                let plus = ewald(&cell, &displaced(&sites, atom, axis, step), &params).unwrap();
                let minus = ewald(&cell, &displaced(&sites, atom, axis, -step), &params).unwrap();
                // Reduce each site gradient onto its atom, then difference.
                let reduce = |field: &crate::pbc::ewald::EwaldOutput| -> Vec<f64> {
                    let mut out = vec![0.0; 9];
                    for (site, gradient) in sites.iter().zip(&field.site_gradient) {
                        let g = gradient.to_array();
                        for beta in 0..3 {
                            out[3 * site.owner + beta] += g[beta];
                        }
                    }
                    out
                };
                let (gp, gm) = (reduce(&plus), reduce(&minus));
                for (index, (p, m)) in gp.iter().zip(&gm).enumerate() {
                    let numerical = (p - m) / (2.0 * step);
                    let exact = hessian[(3 * atom + axis, index)];
                    assert!(
                        (numerical - exact).abs() < 1.0e-6,
                        "({}, {index}): analytic {exact} vs finite difference {numerical}",
                        3 * atom + axis
                    );
                }
            }
        }
    }

    /// Symmetric, as any second derivative of a scalar must be.
    #[test]
    fn the_hessian_is_symmetric() {
        let (cell, sites) = cubic_pair(9.0);
        let params = EwaldParams::for_cell(&cell, 1.0e-12);
        let hessian = ewald_atom_hessian(&cell, &sites, &params, 3)
            .unwrap()
            .unwrap();
        for i in 0..9 {
            for j in 0..9 {
                let difference = (hessian[(i, j)] - hessian[(j, i)]).abs();
                assert!(
                    difference < 1.0e-12,
                    "({i},{j}) asymmetric by {difference:.3e}"
                );
            }
        }
    }

    /// The acoustic sum rule: translating everything together changes nothing, so every `3×3`
    /// row-block must sum to zero. This holds term by term in the construction, so a failure here
    /// means a scatter went to the wrong place rather than that a physical term is missing.
    #[test]
    fn the_acoustic_sum_rule_holds() {
        let (cell, sites) = cubic_pair(9.0);
        let params = EwaldParams::for_cell(&cell, 1.0e-12);
        let hessian = ewald_atom_hessian(&cell, &sites, &params, 3)
            .unwrap()
            .unwrap();
        for row in 0..9 {
            for beta in 0..3 {
                let total: f64 = (0..3).map(|atom| hessian[(row, 3 * atom + beta)]).sum();
                assert!(
                    total.abs() < 1.0e-9,
                    "row {row}, axis {beta}: sums to {total:.3e}"
                );
            }
        }
    }

    /// The splitting parameter is a computational choice, so no derivative may depend on it.
    #[test]
    fn the_hessian_is_independent_of_the_splitting() {
        let (cell, sites) = cubic_pair(9.0);
        let reference = ewald_atom_hessian(&cell, &sites, &EwaldParams::tuned(4.5, 1.0e-14), 3)
            .unwrap()
            .unwrap();
        let other = ewald_atom_hessian(&cell, &sites, &EwaldParams::tuned(4.0, 1.0e-14), 3)
            .unwrap()
            .unwrap();
        for i in 0..9 {
            for j in 0..9 {
                let difference = (reference[(i, j)] - other[(i, j)]).abs();
                assert!(
                    difference < 1.0e-8,
                    "({i},{j}) moved by {difference:.3e} with the splitting"
                );
            }
        }
    }

    /// A slab periodic in `xy`, with the sites spread out of the plane so the wire of the
    /// machinery being tested — the slab kernel's `z` derivatives — is actually loaded.
    fn slab_pair() -> (Cell, Vec<ChargeSite>) {
        let cell = Cell::new(
            Vec3::new(9.0, 0.0, 0.0),
            Vec3::new(1.1, 8.4, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, true, false],
        )
        .unwrap();
        let sites = vec![
            ChargeSite {
                position: Vec3::new(0.3, 0.2, 1.3),
                charge: 1.0,
                owner: 0,
            },
            ChargeSite {
                position: Vec3::new(2.1, 0.4, -0.9),
                charge: -0.6,
                owner: 1,
            },
            ChargeSite {
                position: Vec3::new(1.0, 2.6, 0.4),
                charge: -0.4,
                owner: 2,
            },
        ];
        (cell, sites)
    }

    /// A chain along `x` with transverse structure the axial sum has to carry exactly.
    fn chain_pair() -> (Cell, Vec<ChargeSite>) {
        let cell = Cell::new(
            Vec3::new(7.0, 0.0, 0.0),
            Vec3::new(0.0, 45.0, 0.0),
            Vec3::new(0.0, 0.0, 45.0),
            [true, false, false],
        )
        .unwrap();
        let sites = vec![
            ChargeSite {
                position: Vec3::new(0.3, 0.2, 1.3),
                charge: 1.0,
                owner: 0,
            },
            ChargeSite {
                position: Vec3::new(2.1, 0.4, -0.9),
                charge: -0.6,
                owner: 1,
            },
            ChargeSite {
                position: Vec3::new(1.0, 2.6, 0.4),
                charge: -0.4,
                owner: 2,
            },
        ];
        (cell, sites)
    }

    /// The reduced-dimensionality Hessian against finite differences of the *unphased* Ewald
    /// gradients — the same cross-implementation check the 3D path gets, and the sharper one
    /// here: the Hessian comes from the phased sum at `q = 0` while the gradients come from
    /// [`ewald`]'s Parry slab and direct chain paths, which share no code with it.
    #[test]
    fn reduced_dimensionality_matches_finite_differences_of_the_gradient() {
        for (cell, sites) in [slab_pair(), chain_pair()] {
            let params = EwaldParams::for_cell(&cell, 1.0e-12);
            let hessian = ewald_atom_hessian(&cell, &sites, &params, 3)
                .unwrap()
                .expect("every periodic cell reports a Hessian now");

            // The compared entries must be of a non-trivial scale, or agreement means nothing.
            let mut largest = 0.0_f64;
            for i in 0..9 {
                for j in 0..9 {
                    largest = largest.max(hessian[(i, j)].abs());
                }
            }
            assert!(
                largest > 1.0e-3,
                "the {}D Hessian collapsed to {largest:.3e}",
                cell.n_periodic()
            );

            let step = 1.0e-5;
            for atom in 0..3 {
                for axis in 0..3 {
                    let plus = ewald(&cell, &displaced(&sites, atom, axis, step), &params).unwrap();
                    let minus =
                        ewald(&cell, &displaced(&sites, atom, axis, -step), &params).unwrap();
                    let reduce = |field: &crate::pbc::ewald::EwaldOutput| -> Vec<f64> {
                        let mut out = vec![0.0; 9];
                        for (site, gradient) in sites.iter().zip(&field.site_gradient) {
                            let g = gradient.to_array();
                            for beta in 0..3 {
                                out[3 * site.owner + beta] += g[beta];
                            }
                        }
                        out
                    };
                    let (gp, gm) = (reduce(&plus), reduce(&minus));
                    for (index, (p, m)) in gp.iter().zip(&gm).enumerate() {
                        let numerical = (p - m) / (2.0 * step);
                        let exact = hessian[(3 * atom + axis, index)];
                        assert!(
                            (numerical - exact).abs() < 1.0e-6,
                            "{}D ({}, {index}): analytic {exact} vs finite difference {numerical}",
                            cell.n_periodic(),
                            3 * atom + axis
                        );
                    }
                }
            }
        }
    }

    /// Symmetry and the acoustic sum rule in reduced dimensionality. The sum rule holds by
    /// construction — the four scatter targets of each pair cancel along any row — so a failure
    /// here means a scatter went to the wrong place, exactly as in 3D.
    #[test]
    fn the_reduced_dimensionality_hessian_is_symmetric_and_acoustic() {
        for (cell, sites) in [slab_pair(), chain_pair()] {
            let params = EwaldParams::for_cell(&cell, 1.0e-12);
            let hessian = ewald_atom_hessian(&cell, &sites, &params, 3)
                .unwrap()
                .unwrap();
            for i in 0..9 {
                for j in 0..9 {
                    let difference = (hessian[(i, j)] - hessian[(j, i)]).abs();
                    assert!(
                        difference < 1.0e-10,
                        "{}D ({i},{j}) asymmetric by {difference:.3e}",
                        cell.n_periodic()
                    );
                }
            }
            for row in 0..9 {
                for beta in 0..3 {
                    let total: f64 = (0..3).map(|atom| hessian[(row, 3 * atom + beta)]).sum();
                    assert!(
                        total.abs() < 1.0e-9,
                        "{}D row {row}, axis {beta}: sums to {total:.3e}",
                        cell.n_periodic()
                    );
                }
            }
        }
    }

    /// The slab Hessian must not move with the splitting parameter — the check that a wrong
    /// `π/(A|G|)` prefactor after unfolding the ±G sum cannot survive, because `α` decides how
    /// much of the answer flows through the mis-weighted half. (1D has no splitting.)
    #[test]
    fn the_slab_hessian_is_independent_of_the_splitting() {
        let (cell, sites) = slab_pair();
        let reference = ewald_atom_hessian(&cell, &sites, &EwaldParams::tuned(14.0, 1.0e-14), 3)
            .unwrap()
            .unwrap();
        let other = ewald_atom_hessian(&cell, &sites, &EwaldParams::tuned(20.0, 1.0e-14), 3)
            .unwrap()
            .unwrap();
        for i in 0..9 {
            for j in 0..9 {
                let difference = (reference[(i, j)] - other[(i, j)]).abs();
                assert!(
                    difference < 1.0e-8,
                    "({i},{j}) moved by {difference:.3e} with the splitting"
                );
            }
        }
    }

    /// Only an isolated cell reports no Hessian now; the reduced dimensionalities answer.
    #[test]
    fn only_an_isolated_cell_reports_no_hessian() {
        let (_, sites) = cubic_pair(9.0);
        let params = EwaldParams::default();
        assert!(
            ewald_atom_hessian(&Cell::isolated(), &sites, &params, 3)
                .unwrap()
                .is_none(),
            "an isolated system's Hessian belongs to the molecular path"
        );
        for (cell, sites) in [slab_pair(), chain_pair()] {
            let params = EwaldParams::for_cell(&cell, 1.0e-12);
            assert!(ewald_atom_hessian(&cell, &sites, &params, 3)
                .unwrap()
                .is_some());
        }
    }
}
