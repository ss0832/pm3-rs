// SPDX-License-Identifier: GPL-3.0-or-later

//! Complex dense matrices and the Hermitian eigensolver the k-point SCF needs.
//!
//! A deliberately small mirror of [`crate::linalg`]: row-major storage, the same indexing, and
//! one solver. Everything here exists because a Bloch-transformed Fock matrix
//! `F(k) = Σ_T e^{ik·T} F(T)` is complex Hermitian rather than real symmetric.
//!
//! # Why there is no generalized eigenproblem
//!
//! NDDO assumes an orthogonal AO basis, so `S(T) = δ_T0 δ_μν` and `S(k) = I` at every `k`. The
//! Bloch overlap matrix is the identity, and the k-point SCF solves a plain Hermitian
//! eigenproblem — the same structural simplification that removes the Pulay term from the
//! gradient.
//!
//! # Gauge
//!
//! Eigenvectors of a Hermitian matrix are defined only up to a phase. Nothing that reaches an
//! observable depends on it — the density `Σ_n f_n c_n c_n†` is gauge invariant — but comparisons
//! between eigenvectors are meaningless without fixing one, so [`hermitian_eigen`] returns each
//! column with its largest-magnitude entry made real and positive.

use faer::c64;

use crate::error::{Pm3Error, Result};

/// Row-major dense complex matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct CMatrix {
    pub rows: usize,
    pub cols: usize,
    data: Vec<c64>,
}

impl CMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![c64::new(0.0, 0.0); rows * cols],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = c64::new(1.0, 0.0);
        }
        m
    }

    /// Promote a real matrix, for the `k = Γ` case and for tests that compare the two paths.
    pub fn from_real(a: &crate::linalg::Matrix) -> Self {
        let mut out = Self::zeros(a.rows, a.cols);
        for (slot, value) in out.data.iter_mut().zip(a.as_slice()) {
            *slot = c64::new(*value, 0.0);
        }
        out
    }

    #[inline]
    pub fn as_slice(&self) -> &[c64] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [c64] {
        &mut self.data
    }

    /// `A†`.
    pub fn adjoint(&self) -> CMatrix {
        let mut out = CMatrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[(j, i)] = self[(i, j)].conj();
            }
        }
        out
    }

    /// Largest `|A_ij − A_ji*|`, which is zero exactly when the matrix is Hermitian.
    pub fn hermitian_defect(&self) -> f64 {
        let mut worst = 0.0_f64;
        for i in 0..self.rows {
            for j in 0..self.cols {
                worst = worst.max((self[(i, j)] - self[(j, i)].conj()).norm());
            }
        }
        worst
    }

    /// `Re Tr[A† B]`, the real inner product the SCF convergence tests use.
    pub fn real_dot(&self, other: &CMatrix) -> f64 {
        debug_assert_eq!(self.data.len(), other.data.len());
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a.re * b.re + a.im * b.im)
            .sum()
    }

    /// Root-mean-square `|A − B|` between two equally shaped matrices.
    pub fn rms_difference(&self, other: &CMatrix) -> f64 {
        debug_assert_eq!(self.data.len(), other.data.len());
        if self.data.is_empty() {
            return 0.0;
        }
        let total: f64 = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| {
                let d = a - b;
                d.re * d.re + d.im * d.im
            })
            .sum();
        (total / self.data.len() as f64).sqrt()
    }

    /// `self · other`.
    pub fn matmul(&self, other: &CMatrix) -> CMatrix {
        assert_eq!(self.cols, other.rows, "complex matmul dimension mismatch");
        let a = faer::MatRef::from_row_major_slice(self.as_slice(), self.rows, self.cols);
        let b = faer::MatRef::from_row_major_slice(other.as_slice(), other.rows, other.cols);
        let product = a * b;
        let mut out = CMatrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                out[(i, j)] = product[(i, j)];
            }
        }
        out
    }

    /// `C_occ diag(weights) C_occ†` over the leading `count` columns — one k-point's contribution
    /// to the density matrix.
    ///
    /// The result is Hermitian by construction, which is what guarantees a real occupation and a
    /// real energy however the eigenvector phases came out.
    pub fn occupied_density(&self, weights: &[f64]) -> CMatrix {
        let n = self.rows;
        let mut out = CMatrix::zeros(n, n);
        for (column, weight) in weights.iter().enumerate() {
            if *weight == 0.0 {
                continue;
            }
            for i in 0..n {
                let ci = self[(i, column)];
                for j in 0..n {
                    out[(i, j)] += ci * self[(j, column)].conj() * *weight;
                }
            }
        }
        out
    }
}

impl std::ops::Index<(usize, usize)> for CMatrix {
    type Output = c64;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &c64 {
        &self.data[i * self.cols + j]
    }
}

impl std::ops::IndexMut<(usize, usize)> for CMatrix {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut c64 {
        &mut self.data[i * self.cols + j]
    }
}

/// Hermitian eigendecomposition (faer, pure Rust — no LAPACK/BLAS).
///
/// Returns `(eigenvalues, eigenvectors)` with the eigenvalues **ascending** and the eigenvectors
/// as **columns**, so `A = V diag(λ) V†`. Each column is phase-fixed as described in the module
/// note.
pub fn hermitian_eigen(a: &CMatrix) -> Result<(Vec<f64>, CMatrix)> {
    let n = a.rows;
    if a.cols != n {
        return Err(Pm3Error::LinearAlgebra(
            "hermitian_eigen requires a square matrix".to_string(),
        ));
    }
    if n == 0 {
        return Ok((Vec::new(), CMatrix::zeros(0, 0)));
    }
    let fa = faer::MatRef::from_row_major_slice(a.as_slice(), n, n);
    let eigen = fa.self_adjoint_eigen(faer::Side::Lower).map_err(|e| {
        Pm3Error::LinearAlgebra(format!("faer Hermitian eigendecomposition failed: {e:?}"))
    })?;
    let s = eigen.S();
    let u = eigen.U();

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        s[i].re
            .partial_cmp(&s[j].re)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let values: Vec<f64> = order.iter().map(|&k| s[k].re).collect();

    let mut vectors = CMatrix::zeros(n, n);
    for (new_column, &old_column) in order.iter().enumerate() {
        // Fix the gauge on the largest entry, which is the numerically safest choice: a small
        // entry's phase is dominated by rounding.
        let mut pivot = c64::new(1.0, 0.0);
        let mut largest = 0.0_f64;
        for i in 0..n {
            let value = u[(i, old_column)];
            if value.norm() > largest {
                largest = value.norm();
                pivot = value;
            }
        }
        let phase = if largest > 0.0 {
            c64::new(pivot.re / largest, -pivot.im / largest)
        } else {
            c64::new(1.0, 0.0)
        };
        for i in 0..n {
            vectors[(i, new_column)] = u[(i, old_column)] * phase;
        }
    }
    Ok((values, vectors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{symmetric_eigen, Matrix};

    fn hermitian(n: usize, seed: u64) -> CMatrix {
        // A deterministic pseudo-random Hermitian matrix; the generator only has to be varied,
        // not statistically good.
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        let mut a = CMatrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = c64::new(next(), 0.0);
            for j in 0..i {
                let value = c64::new(next(), next());
                a[(i, j)] = value;
                a[(j, i)] = value.conj();
            }
        }
        a
    }

    #[test]
    fn eigenvalues_are_ascending_and_reconstruct_the_matrix() {
        let a = hermitian(7, 12345);
        let (values, vectors) = hermitian_eigen(&a).unwrap();
        assert!(values.windows(2).all(|w| w[0] <= w[1]));

        // A = V diag(λ) V†.
        let mut scaled = vectors.clone();
        for j in 0..7 {
            for i in 0..7 {
                scaled[(i, j)] *= c64::new(values[j], 0.0);
            }
        }
        let reconstructed = scaled.matmul(&vectors.adjoint());
        for i in 0..7 {
            for j in 0..7 {
                let difference = (reconstructed[(i, j)] - a[(i, j)]).norm();
                assert!(difference < 1.0e-12, "({i},{j}) off by {difference:.3e}");
            }
        }
    }

    #[test]
    fn eigenvectors_are_orthonormal() {
        let (_, vectors) = hermitian_eigen(&hermitian(9, 777)).unwrap();
        let gram = vectors.adjoint().matmul(&vectors);
        for i in 0..9 {
            for j in 0..9 {
                let expected = if i == j {
                    c64::new(1.0, 0.0)
                } else {
                    c64::new(0.0, 0.0)
                };
                assert!((gram[(i, j)] - expected).norm() < 1.0e-12);
            }
        }
    }

    /// A real symmetric matrix promoted to complex must give the real solver's eigenvalues. This
    /// is what lets the k-point path be checked against the Γ-point one at `k = 0`.
    #[test]
    fn a_real_matrix_agrees_with_the_real_solver() {
        let n = 6;
        let mut real = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let value = ((i * 7 + j * 3) as f64).sin();
                real[(i, j)] = value;
                real[(j, i)] = value;
            }
        }
        let (real_values, _) = symmetric_eigen(&real).unwrap();
        let (complex_values, _) = hermitian_eigen(&CMatrix::from_real(&real)).unwrap();
        for (a, b) in real_values.iter().zip(&complex_values) {
            assert!((a - b).abs() < 1.0e-12, "{a} vs {b}");
        }
    }

    /// The gauge convention has to be deterministic, or nothing that compares eigenvectors across
    /// k-points or SCF iterations means anything.
    #[test]
    fn the_phase_convention_is_applied() {
        let (_, vectors) = hermitian_eigen(&hermitian(5, 99)).unwrap();
        for column in 0..5 {
            let mut largest = 0.0_f64;
            let mut pivot = c64::new(0.0, 0.0);
            for i in 0..5 {
                if vectors[(i, column)].norm() > largest {
                    largest = vectors[(i, column)].norm();
                    pivot = vectors[(i, column)];
                }
            }
            assert!(pivot.im.abs() < 1.0e-14, "phase not fixed: {pivot:?}");
            assert!(pivot.re > 0.0, "pivot not made positive: {pivot:?}");
        }
    }

    /// The density built from occupied columns must be Hermitian and must have the right trace,
    /// whatever the eigenvector phases were.
    #[test]
    fn the_occupied_density_is_hermitian_with_the_expected_trace() {
        let (_, vectors) = hermitian_eigen(&hermitian(8, 4242)).unwrap();
        let weights = [1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
        let density = vectors.occupied_density(&weights);
        assert!(density.hermitian_defect() < 1.0e-13);
        let trace: f64 = (0..8).map(|i| density[(i, i)].re).sum();
        assert!((trace - 2.5).abs() < 1.0e-12, "trace {trace}");
    }
}
