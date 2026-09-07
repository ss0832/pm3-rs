// SPDX-License-Identifier: GPL-3.0-or-later

//! The rigid-body subspace of a mass-weighted Hessian, and the projector that removes it.
//!
//! # Why this is a projection and not a threshold
//!
//! Three translations and (up to) three rotations cost no energy, so a Hessian evaluated at a
//! stationary point has six directions along which its eigenvalue should be exactly zero. It
//! never is: the second derivative is assembled from a converged-but-not-exact density and, for
//! the numerical path, from finite differences, so those six come out at whatever the accumulated
//! error happens to be. Reporting them is honest but not useful, and the temptation is to decide
//! afterwards that anything below some number of wavenumbers "was" a translation.
//!
//! That rule is wrong in both directions. A floppy torsion or a soft mode near a phase transition
//! is a real vibration below any threshold worth using, and a badly converged rigid mode can land
//! above one. Worse, the threshold has to be chosen, and it was chosen three separate times in
//! this repository's own tests — 50, 100 and 300 cm⁻¹ for the same quantity.
//!
//! The subspace is known in closed form from the geometry and the masses. Removing it before
//! diagonalizing costs one `3N × 3N` product and makes the answer exact: the modes that come back
//! along those directions are zero because nothing was left there, not because they were small
//! enough to round off. What the Hessian *would* have said about them is still worth knowing —
//! it measures how well converged the calculation was — so it is reported separately as a Rayleigh
//! quotient rather than thrown away.
//!
//! # The rank is discovered, not assumed
//!
//! Six is wrong for a linear molecule (five), for a single atom (three) and for a periodic cell,
//! where the rotations are not symmetries at all (three). Rather than branching on geometry, the
//! six candidate vectors are orthonormalized and the ones that are already spanned fall out with
//! zero norm. A linear molecule's third rotation generator is a linear combination of the other
//! two and simply does not survive Gram–Schmidt.
//!
//! Sparkles and MOPAC's `+`/`−` point charges have a tabulated mass of exactly zero. They carry
//! `√m = 0` into every vector here, so they contribute nothing to the subspace — which is right:
//! a massless site cannot carry momentum, and its rows of the mass-weighted Hessian are zeroed
//! for the same reason.

use crate::linalg::Matrix;
use crate::math::Vec3;

/// Directions that cost no energy: translations always, rotations only for a molecule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RigidMotions {
    /// Three translations and three rotations — an isolated molecule.
    TranslationsAndRotations,
    /// Three translations only. A crystal is not invariant under rotating its contents inside a
    /// fixed lattice, so the rotational generators are *not* null vectors of a periodic Hessian
    /// and projecting them out would delete real restoring force.
    TranslationsOnly,
}

/// An orthonormal basis for the rigid-body subspace in **mass-weighted** coordinates.
///
/// `positions` are Cartesian (any consistent length unit — only differences enter) and `masses`
/// is one entry per atom. The returned vectors have length `3 * masses.len()`; there are three,
/// five or six of them depending on the geometry, and the count is the answer to "how many modes
/// should be zero".
pub fn rigid_body_basis(
    positions: &[Vec3],
    masses: &[f64],
    motions: RigidMotions,
) -> Vec<Vec<f64>> {
    let nat = masses.len();
    let ndof = 3 * nat;

    // Rotations are generated about the centre of mass. Translations do not care where the
    // origin is, but rotations do: about any other point a rotation is a rotation plus a
    // translation, which is still in the span and so still projected out correctly — the centre
    // is used because it makes the two families orthogonal before Gram–Schmidt rather than after.
    let centre = {
        let mut total = 0.0;
        let mut moment = [0.0; 3];
        for (index, position) in positions.iter().enumerate().take(nat) {
            let m = masses[index];
            total += m;
            for (slot, value) in moment.iter_mut().zip(position.to_array()) {
                *slot += m * value;
            }
        }
        moment.map(|v| if total > 0.0 { v / total } else { 0.0 })
    };

    let mut candidates: Vec<Vec<f64>> = Vec::with_capacity(6);
    for axis in 0..3 {
        let mut vector = vec![0.0; ndof];
        for atom in 0..nat {
            vector[3 * atom + axis] = masses[atom].sqrt();
        }
        candidates.push(vector);
    }
    if motions == RigidMotions::TranslationsAndRotations {
        for axis in 0..3 {
            let mut vector = vec![0.0; ndof];
            for atom in 0..nat {
                let r = positions[atom].to_array();
                let d = [r[0] - centre[0], r[1] - centre[1], r[2] - centre[2]];
                // The rotation generator about `axis`: `e_axis × d`.
                let cross = match axis {
                    0 => [0.0, -d[2], d[1]],
                    1 => [d[2], 0.0, -d[0]],
                    _ => [-d[1], d[0], 0.0],
                };
                let root = masses[atom].sqrt();
                for (component, value) in cross.iter().enumerate() {
                    vector[3 * atom + component] = root * value;
                }
            }
            candidates.push(vector);
        }
    }

    orthonormalize(candidates)
}

/// Modified Gram–Schmidt, dropping directions already spanned.
///
/// This is where the rank is discovered. The `1e-6` is not a threshold on a physical quantity —
/// it separates "this vector is a linear combination of the ones before it" from "this vector is
/// new", and after one reorthogonalization pass a dependent vector's norm is at round-off while
/// an independent one's is order one. There is no continuum in between for the six vectors this
/// is ever handed.
fn orthonormalize(candidates: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for mut candidate in candidates {
        // Twice: one pass leaves a dependent vector's residual at √ε rather than ε, which for a
        // near-linear molecule is the difference between discovering five directions and six.
        for _ in 0..2 {
            for kept in &basis {
                let overlap: f64 = candidate.iter().zip(kept).map(|(a, b)| a * b).sum();
                for (value, base) in candidate.iter_mut().zip(kept) {
                    *value -= overlap * base;
                }
            }
        }
        let norm: f64 = candidate.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1.0e-6 {
            for value in candidate.iter_mut() {
                *value /= norm;
            }
            basis.push(candidate);
        }
    }
    basis
}

/// Remove the rigid-body content from each column of `modes`, in place.
///
/// Used for infrared intensities, where the modes themselves are wanted with their translational
/// and rotational content gone but the frequencies are not being recomputed.
pub fn project_columns(basis: &[Vec<f64>], modes: &Matrix) -> Matrix {
    let ndof = modes.rows;
    let mut out = modes.clone();
    for mode in 0..modes.cols {
        for kept in basis {
            let overlap: f64 = (0..ndof).map(|row| kept[row] * out[(row, mode)]).sum();
            for (row, base) in kept.iter().enumerate() {
                out[(row, mode)] -= overlap * base;
            }
        }
    }
    out
}

/// `A ← (1 − P) A (1 − P)`, with `P` the projector onto `basis`.
///
/// Applied from both sides, which is the whole point: a one-sided correction makes the matrix
/// non-symmetric, and a symmetric eigensolver reading one triangle then silently re-symmetrizes
/// it into something that no longer has the null space that was just imposed.
pub fn project_out_symmetric(basis: &[Vec<f64>], matrix: &mut Matrix) {
    if basis.is_empty() {
        return;
    }
    let n = matrix.rows;
    // `(1−P) A (1−P) = A − PA − AP + PAP`, accumulated through `w_k = A b_k` so the cost is
    // `k` matrix-vector products rather than two matrix-matrix ones.
    let mut w: Vec<Vec<f64>> = Vec::with_capacity(basis.len());
    for b in basis {
        let mut column = vec![0.0; n];
        for (i, slot) in column.iter_mut().enumerate() {
            *slot = (0..n).map(|j| matrix[(i, j)] * b[j]).sum();
        }
        w.push(column);
    }
    // `b_kᵀ A b_l`, needed for the `PAP` term.
    let mut inner = vec![vec![0.0; basis.len()]; basis.len()];
    for (k, wk) in w.iter().enumerate() {
        for (l, bl) in basis.iter().enumerate() {
            inner[k][l] = (0..n).map(|i| bl[i] * wk[i]).sum();
        }
    }
    for i in 0..n {
        for j in 0..n {
            let mut value = matrix[(i, j)];
            for (k, bk) in basis.iter().enumerate() {
                value -= bk[i] * w[k][j];
                value -= w[k][i] * bk[j];
                for (l, bl) in basis.iter().enumerate() {
                    value += bk[i] * inner[k][l] * bl[j];
                }
            }
            matrix[(i, j)] = value;
        }
    }
}

/// `A ← (1 − P) A (1 − P)` for a complex Hermitian `A` and a **real** basis.
///
/// The basis being real is what makes this the same formula: the projector commutes with taking
/// real and imaginary parts, so no conjugation appears anywhere. Used at `q = 0`, where the
/// dynamical matrix is real up to round-off but is carried as complex because the same code path
/// serves every other wavevector.
pub fn project_out_hermitian(basis: &[Vec<f64>], matrix: &mut crate::cmatrix::CMatrix) {
    if basis.is_empty() {
        return;
    }
    let n = matrix.rows;
    let zero = faer::c64::new(0.0, 0.0);
    let mut w: Vec<Vec<faer::c64>> = Vec::with_capacity(basis.len());
    for b in basis {
        let mut column = vec![zero; n];
        for (i, slot) in column.iter_mut().enumerate() {
            let mut sum = zero;
            for j in 0..n {
                sum += matrix[(i, j)] * b[j];
            }
            *slot = sum;
        }
        w.push(column);
    }
    let mut inner = vec![vec![zero; basis.len()]; basis.len()];
    for (k, wk) in w.iter().enumerate() {
        for (l, bl) in basis.iter().enumerate() {
            let mut sum = zero;
            for i in 0..n {
                sum += wk[i] * bl[i];
            }
            inner[k][l] = sum;
        }
    }
    for i in 0..n {
        for j in 0..n {
            let mut value = matrix[(i, j)];
            for (k, bk) in basis.iter().enumerate() {
                value -= w[k][j] * bk[i];
                value -= w[k][i] * bk[j];
                for (l, bl) in basis.iter().enumerate() {
                    value += inner[k][l] * (bk[i] * bl[j]);
                }
            }
            matrix[(i, j)] = value;
        }
    }
}

/// The eigenvalue each rigid direction carried **before** it was projected out.
///
/// A Rayleigh quotient `b_kᵀ A b_k`, not a second diagonalization: it is exactly the number the
/// null modes would have reported, and it is the honest measure of how well converged the Hessian
/// is. After projection the modes themselves are zero by construction and say nothing.
pub fn rayleigh_quotients(basis: &[Vec<f64>], matrix: &Matrix) -> Vec<f64> {
    let n = matrix.rows;
    basis
        .iter()
        .map(|b| {
            (0..n)
                .map(|i| {
                    let row: f64 = (0..n).map(|j| matrix[(i, j)] * b[j]).sum();
                    b[i] * row
                })
                .sum()
        })
        .collect()
}

/// Which eigenvectors span the rigid subspace: the `basis.len()` columns of `modes` with the
/// largest weight in it.
///
/// A **count**, decided by geometry, applied to a ranking — not a magnitude cutoff. The
/// alternative (call anything below some wavenumber a translation) is what this module exists to
/// avoid, and it misclassifies a genuine soft mode in exactly the systems where soft modes are
/// the interesting part.
pub fn rigid_mode_indices(basis: &[Vec<f64>], modes: &Matrix) -> Vec<usize> {
    if basis.is_empty() {
        return Vec::new();
    }
    let ndof = modes.rows;
    let mut weights: Vec<(usize, f64)> = (0..modes.cols)
        .map(|mode| {
            let weight: f64 = basis
                .iter()
                .map(|b| {
                    let overlap: f64 = (0..ndof).map(|row| b[row] * modes[(row, mode)]).sum();
                    overlap * overlap
                })
                .sum();
            (mode, weight)
        })
        .collect();
    weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut chosen: Vec<usize> = weights.iter().take(basis.len()).map(|(m, _)| *m).collect();
    chosen.sort_unstable();
    chosen
}

/// [`rigid_mode_indices`] for a complex eigenvector matrix and a real basis.
pub fn rigid_mode_indices_complex(
    basis: &[Vec<f64>],
    modes: &crate::cmatrix::CMatrix,
) -> Vec<usize> {
    if basis.is_empty() {
        return Vec::new();
    }
    let ndof = modes.rows;
    let mut weights: Vec<(usize, f64)> = (0..modes.cols)
        .map(|mode| {
            let weight: f64 = basis
                .iter()
                .map(|b| {
                    // `|<b|v>|²` with a real `b`, so the real and imaginary overlaps add in
                    // quadrature rather than one of them being dropped.
                    let (mut re, mut im) = (0.0, 0.0);
                    for row in 0..ndof {
                        re += b[row] * modes[(row, mode)].re;
                        im += b[row] * modes[(row, mode)].im;
                    }
                    re * re + im * im
                })
                .sum();
            (mode, weight)
        })
        .collect();
    weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut chosen: Vec<usize> = weights.iter().take(basis.len()).map(|(m, _)| *m).collect();
    chosen.sort_unstable();
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    fn water() -> (Vec<Vec3>, Vec<f64>) {
        (
            vec![
                Vec3::new(0.0, 0.0, 0.2215),
                Vec3::new(0.0, 1.4309, -0.8863),
                Vec3::new(0.0, -1.4309, -0.8863),
            ],
            vec![15.999, 1.008, 1.008],
        )
    }

    /// Six for a bent triatomic, five for a linear one, three for a lone atom — and the code is
    /// told none of that. It is the property the whole approach rests on: if the rank were
    /// assumed, a linear molecule would lose a real vibration to a rotation that does not exist.
    #[test]
    fn the_rank_is_discovered_from_the_geometry() {
        let (positions, masses) = water();
        let bent = rigid_body_basis(&positions, &masses, RigidMotions::TranslationsAndRotations);
        assert_eq!(bent.len(), 6, "a bent triatomic has three of each");

        let linear = rigid_body_basis(
            &[
                Vec3::new(0.0, 0.0, -2.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 2.0),
            ],
            &masses,
            RigidMotions::TranslationsAndRotations,
        );
        assert_eq!(linear.len(), 5, "rotation about the axis moves nothing");

        let atom = rigid_body_basis(
            &[Vec3::new(0.0, 0.0, 0.0)],
            &[15.999],
            RigidMotions::TranslationsAndRotations,
        );
        assert_eq!(atom.len(), 3, "a single atom has no rotations either");

        let crystal = rigid_body_basis(&positions, &masses, RigidMotions::TranslationsOnly);
        assert_eq!(crystal.len(), 3, "a lattice is not rotationally invariant");
    }

    /// Exactly zero is reachable, and it is reachable by assignment rather than by arithmetic.
    ///
    /// This is the guarantee the frequency tests rest on when they assert `== 0.0` instead of
    /// `< 1e-6`, and it is worth pinning because the reasoning is not obvious from the call site.
    /// The projection alone leaves the null directions at round-off — some *negative*, which a
    /// signed square root turns into a spurious imaginary mode at the head of the spectrum. What
    /// makes the reported value exact is that [`rigid_mode_indices`] names those modes by their
    /// overlap with the subspace, which is a count applied to a ranking with no magnitude in it,
    /// and the caller then writes `0.0` into them.
    ///
    /// So a tolerance would be asserting less than is true, and would let a regression that
    /// stopped doing the assignment pass unnoticed.
    #[test]
    fn the_projection_leaves_roundoff_and_the_index_pass_is_what_makes_it_exact() {
        let (positions, masses) = water();
        let basis = rigid_body_basis(&positions, &masses, RigidMotions::TranslationsAndRotations);
        let ndof = 9;
        // A positive-definite matrix plus rigid-body content, so the internal modes are clearly
        // non-zero and the null space is unambiguous.
        let mut a = Matrix::zeros(ndof, ndof);
        for i in 0..ndof {
            a[(i, i)] = 3.0 + i as f64;
            for j in 0..i {
                let v = 0.35 / (1.0 + (i - j) as f64);
                a[(i, j)] = v;
                a[(j, i)] = v;
            }
        }
        project_out_symmetric(&basis, &mut a);
        let (values, vectors) = crate::linalg::symmetric_eigen(&a).unwrap();

        let indices = rigid_mode_indices(&basis, &vectors);
        assert_eq!(
            indices.len(),
            basis.len(),
            "one index per removed direction"
        );

        // The projection alone does *not* give exact zeros -- if it did, none of this would be
        // needed. They are round-off, and at least one is typically negative.
        let residual = indices
            .iter()
            .map(|&i| values[i].abs())
            .fold(0.0_f64, f64::max);
        assert!(
            residual < 1.0e-10,
            "the projected directions should be at round-off, largest is {residual:e}"
        );

        // And the internal modes are nowhere near it, so the ranking cannot pick the wrong ones.
        let smallest_internal = (0..ndof)
            .filter(|i| !indices.contains(i))
            .map(|i| values[i].abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            smallest_internal > 1.0e-3,
            "an internal mode at {smallest_internal:e} is too close to the null space to rank"
        );

        // The assignment is what the callers do, and it is exact: `signed_wavenumber(0.0)` is
        // `521.47 * (0.0).sqrt()`, which IEEE-754 gives as exactly `0.0` on every platform.
        let mut eigs = values.clone();
        for &index in &indices {
            eigs[index] = 0.0;
        }
        let zeros = eigs
            .iter()
            .filter(|v| crate::hessian::signed_wavenumber(**v) == 0.0)
            .count();
        assert_eq!(zeros, basis.len(), "the assignment must survive the sqrt");
    }

    /// A massless site contributes nothing, and does not produce a NaN doing it.
    #[test]
    fn a_massless_site_carries_no_rigid_motion() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        // MOPAC's `+` point charge has a tabulated mass of exactly zero.
        let basis = rigid_body_basis(
            &positions,
            &[15.999, 1.008, 0.0],
            RigidMotions::TranslationsAndRotations,
        );
        assert!(basis.iter().flatten().all(|v| v.is_finite()));
        for b in &basis {
            for slot in b.iter().skip(6) {
                assert_eq!(*slot, 0.0, "the massless site must sit at zero");
            }
        }
    }

    /// The basis is orthonormal, which every later step assumes.
    #[test]
    fn the_basis_is_orthonormal() {
        let (positions, masses) = water();
        let basis = rigid_body_basis(&positions, &masses, RigidMotions::TranslationsAndRotations);
        for (i, a) in basis.iter().enumerate() {
            for (j, b) in basis.iter().enumerate() {
                let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-12, "<{i}|{j}> = {dot}");
            }
        }
    }

    /// The projection annihilates the subspace **exactly**, and does it symmetrically.
    ///
    /// Symmetry is the part worth testing: a one-sided correction leaves a matrix a symmetric
    /// eigensolver will silently re-symmetrize, undoing half of what was just imposed.
    #[test]
    fn the_projection_is_exact_and_symmetric() {
        let (positions, masses) = water();
        let basis = rigid_body_basis(&positions, &masses, RigidMotions::TranslationsAndRotations);
        let ndof = 9;
        // An arbitrary symmetric matrix with no particular null space.
        let mut a = Matrix::zeros(ndof, ndof);
        for i in 0..ndof {
            for j in 0..ndof {
                a[(i, j)] = ((i * 7 + j * 3) % 11) as f64 - 5.0;
            }
        }
        for i in 0..ndof {
            for j in 0..i {
                let v = 0.5 * (a[(i, j)] + a[(j, i)]);
                a[(i, j)] = v;
                a[(j, i)] = v;
            }
        }
        let before = rayleigh_quotients(&basis, &a);
        assert!(
            before.iter().any(|v| v.abs() > 1e-6),
            "the test matrix must actually have rigid-body content to remove"
        );

        project_out_symmetric(&basis, &mut a);

        for i in 0..ndof {
            for j in 0..ndof {
                assert!(
                    (a[(i, j)] - a[(j, i)]).abs() < 1e-12,
                    "the projection broke symmetry at ({i}, {j})"
                );
            }
        }
        for (k, value) in rayleigh_quotients(&basis, &a).iter().enumerate() {
            assert!(value.abs() < 1e-10, "direction {k} survived at {value}");
        }
        // And `A b_k` itself, not just the quotient: the subspace is a true null space.
        for b in &basis {
            for i in 0..ndof {
                let row: f64 = (0..ndof).map(|j| a[(i, j)] * b[j]).sum();
                assert!(row.abs() < 1e-10, "A b is not zero: {row}");
            }
        }
    }
}
