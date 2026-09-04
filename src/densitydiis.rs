// SPDX-License-Identifier: GPL-3.0-or-later

//! Pulay DIIS on a density, shared by the k-point and divide-and-conquer SCFs.
//!
//! # Why not the molecular accelerator
//!
//! [`crate::scf`] extrapolates the **Fock matrix** against the `[F, P]` commutator, which is the
//! better-conditioned choice when there is one Fock matrix to extrapolate. Neither of the callers
//! here has that: the k-point SCF has a different Fock matrix at every `k` while the quantity
//! actually iterated is one real direct-space density, and the divide-and-conquer SCF assembles
//! its density from subsystem eigenvectors rather than obtaining it from a global diagonalization.
//! For both, the density is the natural iterate, so this is Pulay's original formulation:
//! extrapolate the iterate against its own residual `P_out − P_in`.
//!
//! # The Gram matrix is kept, not rebuilt
//!
//! `⟨r_i, r_j⟩` costs `O(depth · n)` to extend by one row and `O(depth² · n)` to rebuild. Rebuilding
//! it — and rebuilding it again for each shorter suffix the solve retries — turned out to dominate
//! a divide-and-conquer iteration outright: at a thousand basis functions and depth eight, more
//! time than the Fock build and every subsystem diagonalization combined. Extending it costs
//! nothing in comparison, and the suffixes are then sub-blocks of a matrix that already exists.

/// Largest history the accelerator will keep.
const MAX_DEPTH: usize = 8;

/// Largest `Σ|c_i|` accepted from the solve.
///
/// The extrapolation is an interpolation only in spirit — the coefficients are unconstrained in
/// sign — and once the residuals go nearly linearly dependent the solve answers with large
/// cancelling weights that amplify whatever noise is left. Refusing those and retrying on a
/// shorter history is what keeps the tail of an SCF monotone.
const MAX_WEIGHT: f64 = 20.0;

pub(crate) struct DensityDiis {
    depth: usize,
    density: Vec<Vec<f64>>,
    residual: Vec<Vec<f64>>,
    /// `⟨r_i, r_j⟩`, maintained incrementally.
    gram: Vec<Vec<f64>>,
}

impl DensityDiis {
    /// A history sized to fit `memory_mb`, holding two vectors of `len` doubles per slot.
    ///
    /// `0` means no budget. The depth never drops below two, which is the minimum for any
    /// extrapolation at all; trimming it costs a few extra iterations rather than an allocation
    /// failure.
    pub fn new(len: usize, memory_mb: usize) -> Self {
        let depth = if memory_mb == 0 || len == 0 {
            MAX_DEPTH
        } else {
            let per_slot = 2 * len * std::mem::size_of::<f64>();
            (memory_mb * 1024 * 1024 / per_slot.max(1)).clamp(2, MAX_DEPTH)
        };
        Self {
            depth,
            density: Vec::new(),
            residual: Vec::new(),
            gram: Vec::new(),
        }
    }

    pub fn push(&mut self, density: Vec<f64>, residual: Vec<f64>) {
        let row: Vec<f64> = self
            .residual
            .iter()
            .map(|other| dot(other, &residual))
            .chain(std::iter::once(dot(&residual, &residual)))
            .collect();
        for (index, existing) in self.gram.iter_mut().enumerate() {
            existing.push(row[index]);
        }
        self.gram.push(row);
        self.density.push(density);
        self.residual.push(residual);

        while self.density.len() > self.depth {
            self.density.remove(0);
            self.residual.remove(0);
            self.gram.remove(0);
            for row in &mut self.gram {
                row.remove(0);
            }
        }
    }

    /// `Σ c_i P_i` over the longest suffix of the history that solves cleanly, or `None` when
    /// nothing does and the caller should fall back to plain damping.
    pub fn extrapolate(&self) -> Option<Vec<f64>> {
        let n = self.density.len();
        for first in 0..n.saturating_sub(1) {
            let block: Vec<Vec<f64>> = self.gram[first..]
                .iter()
                .map(|row| row[first..].to_vec())
                .collect();
            let Some(coefficients) = crate::scf::diis_coeffs_from_gram(&block) else {
                continue;
            };
            let kept = n - first;
            let weight: f64 = coefficients.iter().take(kept).map(|c| c.abs()).sum();
            if !weight.is_finite() || weight > MAX_WEIGHT {
                continue;
            }
            let mut out = vec![0.0; self.density[0].len()];
            for (c, slot) in coefficients.iter().zip(&self.density[first..]) {
                for (target, value) in out.iter_mut().zip(slot) {
                    *target += c * value;
                }
            }
            return Some(out);
        }
        None
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The incrementally maintained Gram matrix must equal the one a rebuild would produce.
    /// Everything the accelerator does rests on it, and an incremental update is exactly the kind
    /// of thing that drifts silently.
    #[test]
    fn the_gram_matrix_matches_a_rebuild() {
        let mut diis = DensityDiis::new(4, 0);
        let vectors: Vec<Vec<f64>> = (0..6)
            .map(|i| {
                (0..4)
                    .map(|j| ((i * 7 + j * 3) as f64).sin())
                    .collect::<Vec<f64>>()
            })
            .collect();
        for v in &vectors {
            diis.push(v.clone(), v.iter().map(|x| x * 0.5).collect());
        }
        for (i, row) in diis.gram.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                let expected = dot(&diis.residual[i], &diis.residual[j]);
                assert!(
                    (value - expected).abs() < 1.0e-12,
                    "({i},{j}): kept {value} vs rebuilt {expected}"
                );
            }
        }
    }

    /// The history is trimmed from the front, and the Gram matrix must be trimmed with it.
    #[test]
    fn trimming_keeps_the_gram_matrix_consistent() {
        let mut diis = DensityDiis::new(3, 0);
        for i in 0..20 {
            let v: Vec<f64> = (0..3).map(|j| ((i + j) as f64).cos()).collect();
            diis.push(v.clone(), v);
        }
        assert_eq!(diis.density.len(), MAX_DEPTH);
        assert_eq!(diis.gram.len(), MAX_DEPTH);
        for row in &diis.gram {
            assert_eq!(row.len(), MAX_DEPTH);
        }
        for (i, row) in diis.gram.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                let expected = dot(&diis.residual[i], &diis.residual[j]);
                assert!((value - expected).abs() < 1.0e-12, "({i},{j}) drifted");
            }
        }
    }

    /// A memory budget must shorten the history rather than fail.
    #[test]
    fn the_memory_budget_trims_the_depth() {
        // 1 MiB against slots of 2 × 100_000 doubles (1.6 MiB each) leaves room for none, so the
        // floor of two applies.
        let diis = DensityDiis::new(100_000, 1);
        assert_eq!(diis.depth, 2);
        // A generous budget keeps the full depth.
        assert_eq!(DensityDiis::new(100, 512).depth, MAX_DEPTH);
        assert_eq!(DensityDiis::new(100_000, 0).depth, MAX_DEPTH);
    }

    /// With one entry there is nothing to extrapolate between.
    #[test]
    fn a_single_entry_yields_nothing() {
        let mut diis = DensityDiis::new(3, 0);
        diis.push(vec![1.0, 2.0, 3.0], vec![0.1, 0.2, 0.3]);
        assert!(diis.extrapolate().is_none());
    }

    /// Two iterates whose residuals bracket zero must extrapolate to the point between them where
    /// the residual vanishes — the whole purpose of the thing.
    #[test]
    fn it_interpolates_to_the_zero_residual_point() {
        let mut diis = DensityDiis::new(1, 0);
        // Residuals +1 and −1 at iterates 0 and 4: the zero crossing is at 2.
        diis.push(vec![0.0], vec![1.0]);
        diis.push(vec![4.0], vec![-1.0]);
        let extrapolated = diis.extrapolate().expect("two entries can be combined");
        assert!(
            (extrapolated[0] - 2.0).abs() < 1.0e-10,
            "extrapolated to {} rather than 2",
            extrapolated[0]
        );
    }
}
