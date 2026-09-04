// SPDX-License-Identifier: GPL-3.0-or-later

//! Which density-matrix entries a partitioning can ever populate.
//!
//! # Why this exists
//!
//! A divide-and-conquer density is assembled block by block: subsystem `α` contributes to
//! `P_μν` only when it holds **both** `μ` and `ν`. Every other entry is exactly zero — not
//! small, not truncated, structurally absent. That is the approximation divide-and-conquer
//! makes, and [`Partition::dropped_pairs`] is what reports its extent.
//!
//! Storing the result in a dense `nao × nao` matrix therefore keeps `O(N²)` numbers of which
//! `O(N)` can be nonzero, and every operation the SCF performs on the density — damping,
//! the RMS change, the DIIS extrapolation — pays for the zeros. At 960 atoms that is 3.7
//! million entries per spin of which about 300 thousand are live, and the DIIS history alone
//! moved 177 MB per iteration to carry them.
//!
//! This module records the live set once, as a CSR sparsity pattern, and stores everything the
//! SCF iterates on — both densities, both Fock matrices and the workspaces between them — on it
//! rather than dense. Measured on a chain of water molecules, for the eight matrices the loop
//! holds across an iteration:
//!
//! | waters | nao | dense | on the pattern |
//! |---:|---:|---:|---:|
//! | 8 | 48 | 0.14 MiB | 0.141 MiB |
//! | 16 | 96 | 0.56 | 0.466 |
//! | 32 | 192 | 2.25 | 1.122 |
//! | 64 | 384 | 9.00 | 2.434 |
//! | 128 | 768 | 36.00 | 5.059 |
//!
//! The dense column quadruples with every doubling — a log-log slope of exactly 2. The sparse
//! one grows by 2.08 over the last doubling, a slope of **1.06**. Below about thirty atoms the
//! pattern is nearly dense and there is nothing to save; what grows is the ratio.
//!
//! # Why this changes no number
//!
//! The pattern is the union of the subsystem blocks, which is exactly the set the assembly
//! writes to. Gathering, operating, and scattering back therefore reproduces the dense path
//! entry for entry: what is skipped is arithmetic on zeros that stay zero. The equivalence is
//! pinned by [`tests::the_pattern_covers_everything_the_assembly_writes`], which checks the
//! containment directly rather than trusting the argument.
//!
//! # Why the pattern is not the Fock matrix's
//!
//! The Fock build reads the density at pairs its integral cutoffs reach, which is not the same
//! set. Where the two disagree the density really is zero, and reading a zero is correct — the
//! partitioning, not the pattern, is what decided that. Widening the pattern to match the Fock
//! build's reach would store zeros that the assembly never writes.

use crate::dc::partition::Partition;
use crate::linalg::Matrix;

/// The nonzero structure of a divide-and-conquer density, in compressed sparse row form.
pub(crate) struct DensityPattern {
    nao: usize,
    /// `row_start[mu]..row_start[mu + 1]` indexes [`DensityPattern::cols`] for row `mu`.
    row_start: Vec<usize>,
    /// Column indices, ascending within each row.
    cols: Vec<u32>,
}

impl DensityPattern {
    /// The union of every subsystem's orbital block.
    pub fn from_partition(partitioning: &Partition, nao: usize) -> Self {
        // Collect per row first. A row's column list is short — the orbitals of the subsystems
        // that contain this one — so sorting and deduplicating per row is cheaper than one
        // global sort, and gives the ascending order the gather relies on for locality.
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); nao];
        for subsystem in &partitioning.subsystems {
            for &mu in &subsystem.orbitals {
                let row = &mut rows[mu];
                for &nu in &subsystem.orbitals {
                    row.push(nu as u32);
                }
            }
        }

        let mut row_start = Vec::with_capacity(nao + 1);
        let mut cols = Vec::new();
        row_start.push(0);
        for row in &mut rows {
            row.sort_unstable();
            row.dedup();
            cols.extend_from_slice(row);
            row_start.push(cols.len());
        }

        Self {
            nao,
            row_start,
            cols,
        }
    }

    /// How many entries the density can hold.
    pub fn nnz(&self) -> usize {
        self.cols.len()
    }

    /// Read one spin channel into `out`, which must be [`DensityPattern::nnz`] long.
    pub fn gather(&self, dense: &Matrix, out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.nnz());
        for mu in 0..self.nao {
            let span = self.row_start[mu]..self.row_start[mu + 1];
            for index in span {
                out[index] = dense[(mu, self.cols[index] as usize)];
            }
        }
    }

    fn scatter(&self, values: &[f64], dense: &mut Matrix) {
        for mu in 0..self.nao {
            let span = self.row_start[mu]..self.row_start[mu + 1];
            for index in span {
                dense[(mu, self.cols[index] as usize)] = values[index];
            }
        }
    }

    /// `out = a + b`, on the pattern.
    pub fn add_sparse(&self, a: &SparseMatrix, b: &SparseMatrix, out: &mut SparseMatrix) {
        for (slot, (x, y)) in out
            .values_mut()
            .iter_mut()
            .zip(a.values().iter().zip(b.values()))
        {
            *slot = x + y;
        }
    }

    /// `current ← damping · current + (1 − damping) · fresh`, on the pattern.
    pub fn damp_sparse(&self, current: &mut SparseMatrix, fresh: &SparseMatrix, damping: f64) {
        for (slot, value) in current.values_mut().iter_mut().zip(fresh.values()) {
            *slot = damping * *slot + (1.0 - damping) * value;
        }
    }

    /// `Σ a_μν b_μν` over the live entries — the full Frobenius product whenever one of them is
    /// zero everywhere else, which a density on this pattern is.
    pub fn dot_sparse(&self, a: &SparseMatrix, b: &SparseMatrix) -> f64 {
        a.values().iter().zip(b.values()).map(|(x, y)| x * y).sum()
    }

    /// The same root-mean-square difference the dense form reported, with the
    /// `nao²` denominator kept so the convergence threshold keeps its meaning.
    pub fn rms_difference_sparse(&self, a: &SparseMatrix, b: &SparseMatrix) -> f64 {
        let sum: f64 = a
            .values()
            .iter()
            .zip(b.values())
            .map(|(x, y)| (x - y) * (x - y))
            .sum();
        (sum / (self.nao * self.nao).max(1) as f64).sqrt()
    }

    /// Read both spin channels into one vector, α first — the shape the extrapolator works in.
    pub fn gather_pair_sparse(&self, alpha: &SparseMatrix, beta: &SparseMatrix, out: &mut [f64]) {
        let (first, second) = out.split_at_mut(self.nnz());
        first.copy_from_slice(alpha.values());
        second.copy_from_slice(beta.values());
    }

    /// Write both spin channels back.
    pub fn scatter_pair_sparse(
        &self,
        values: &[f64],
        alpha: &mut SparseMatrix,
        beta: &mut SparseMatrix,
    ) {
        let (first, second) = values.split_at(self.nnz());
        alpha.values_mut().copy_from_slice(first);
        beta.values_mut().copy_from_slice(second);
    }

    /// Where `(mu, nu)` sits in a [`SparseMatrix`]'s value array, or `None` if it is outside.
    ///
    /// Binary search within the row. The column lists are short — the orbitals of the subsystems
    /// containing this one — and ascending, which is what
    /// [`DensityPattern::from_partition`] sorts them for.
    #[inline]
    pub fn index_of(&self, mu: usize, nu: usize) -> Option<usize> {
        let span = self.row_start[mu]..self.row_start[mu + 1];
        let row = &self.cols[span.clone()];
        row.binary_search(&(nu as u32)).ok().map(|k| span.start + k)
    }

    /// An all-zero matrix on this pattern.
    pub fn zeros(&self) -> SparseMatrix {
        SparseMatrix {
            values: vec![0.0; self.nnz()],
        }
    }

    /// A dense matrix read onto this pattern. What falls outside is dropped, which is lossless
    /// for anything only ever read back through the pattern.
    pub fn read_dense(&self, dense: &Matrix) -> SparseMatrix {
        let mut out = self.zeros();
        self.gather(dense, &mut out.values);
        out
    }

    /// The dense matrix a sparse one stands for, with structural zeros written out.
    ///
    /// Allocates `nao²`. It exists for the boundary where a dense matrix has to be handed on —
    /// the public `DcResult::density` — and nowhere inside the iteration.
    pub fn to_dense(&self, sparse: &SparseMatrix) -> Matrix {
        let mut out = Matrix::zeros(self.nao, self.nao);
        self.scatter(&sparse.values, &mut out);
        out
    }
}

/// A matrix stored only where a [`DensityPattern`] says it can be nonzero.
///
/// Everything the divide-and-conquer SCF iterates on — both densities, both Fock matrices and
/// the workspaces between them — is read back only through the pattern that produced it, so
/// storing the rest is storing zeros. At 960 atoms the eight dense `nao × nao` matrices the loop
/// used to hold came to about a gigabyte, of which a few megabytes could ever be nonzero.
///
/// Reading outside the pattern returns **zero**, and that is a statement about the model rather
/// than a convenience: the divide-and-conquer density *is* zero where no subsystem holds both
/// orbitals, and the Fock matrix is only ever consulted where the subsystem gather looks, which
/// is inside the pattern by construction. See the module note.
#[derive(Clone)]
pub(crate) struct SparseMatrix {
    values: Vec<f64>,
}

impl SparseMatrix {
    /// The live entries, in the pattern's own order.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// `A_{μν}`, or zero outside the pattern.
    #[inline]
    pub fn get(&self, pattern: &DensityPattern, mu: usize, nu: usize) -> f64 {
        pattern.index_of(mu, nu).map_or(0.0, |k| self.values[k])
    }

    /// Add `value` at `(mu, nu)`, ignoring positions outside the pattern.
    ///
    /// Ignoring rather than refusing is deliberate: a Fock build walks every pair its integral
    /// cutoffs reach, and the ones no subsystem shares contribute to entries the solve never
    /// reads. Dropping them is what the pattern is for.
    #[inline]
    pub fn add(&mut self, pattern: &DensityPattern, mu: usize, nu: usize, value: f64) {
        if let Some(k) = pattern.index_of(mu, nu) {
            self.values[k] += value;
        }
    }

    pub fn fill(&mut self, value: f64) {
        self.values.fill(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::Basis;
    use crate::dc::partition::{partition, DcOptions};
    use crate::params::Pm3Parameters;
    use crate::system::Molecule;

    fn chain(n: usize) -> Molecule {
        let mut lines = format!("{}\nchain\n", 3 * n);
        for i in 0..n {
            let x = 3.2 * i as f64;
            lines.push_str(&format!("O {x:.4} 0.0 0.0\n"));
            lines.push_str(&format!("H {:.4} 0.7572 0.5865\n", x + 0.2));
            lines.push_str(&format!("H {:.4} -0.7572 0.5865\n", x + 0.2));
        }
        Molecule::from_xyz_str(&lines, 0.0).unwrap()
    }

    fn setup(n: usize) -> (Basis, Partition) {
        let molecule = chain(n);
        let params = Pm3Parameters::standard().unwrap();
        let basis = Basis::build(&molecule, &params).unwrap();
        let options = DcOptions {
            core_radius: 4.0,
            buffer_radius: 9.0,
            ..DcOptions::default()
        };
        let partitioning = partition(&molecule, &basis, &options).unwrap();
        (basis, partitioning)
    }

    /// The pattern must contain every entry a subsystem can write, or the SCF would silently
    /// discard density. This checks the containment directly instead of trusting the
    /// construction, because it is the whole safety argument for gathering at all.
    #[test]
    fn the_pattern_covers_everything_the_assembly_writes() {
        let (basis, partitioning) = setup(6);
        let pattern = DensityPattern::from_partition(&partitioning, basis.nao);

        // Mark the pattern, then walk the assembly's own loop and demand every destination is
        // marked.
        let mut covered = vec![false; basis.nao * basis.nao];
        for mu in 0..basis.nao {
            for index in pattern.row_start[mu]..pattern.row_start[mu + 1] {
                covered[mu * basis.nao + pattern.cols[index] as usize] = true;
            }
        }
        for subsystem in &partitioning.subsystems {
            for &mu in &subsystem.orbitals {
                for &nu in &subsystem.orbitals {
                    assert!(
                        covered[mu * basis.nao + nu],
                        "subsystem writes ({mu}, {nu}) but the pattern does not hold it"
                    );
                }
            }
        }
    }

    /// Gather then scatter must be the identity on anything the pattern holds.
    #[test]
    fn a_round_trip_leaves_the_density_alone() {
        let (basis, partitioning) = setup(5);
        let pattern = DensityPattern::from_partition(&partitioning, basis.nao);

        let mut alpha = Matrix::zeros(basis.nao, basis.nao);
        let mut beta = Matrix::zeros(basis.nao, basis.nao);
        for subsystem in &partitioning.subsystems {
            for &mu in &subsystem.orbitals {
                for &nu in &subsystem.orbitals {
                    alpha[(mu, nu)] = 0.01 * (mu as f64) - 0.003 * (nu as f64);
                    beta[(mu, nu)] = 0.002 * (mu as f64 + nu as f64);
                }
            }
        }
        let (reference_alpha, reference_beta) = (alpha.clone(), beta.clone());

        let mut sparse_alpha = pattern.read_dense(&alpha);
        let mut sparse_beta = pattern.read_dense(&beta);
        let mut buffer = vec![0.0; 2 * pattern.nnz()];
        pattern.gather_pair_sparse(&sparse_alpha, &sparse_beta, &mut buffer);
        sparse_alpha = pattern.zeros();
        sparse_beta = pattern.zeros();
        pattern.scatter_pair_sparse(&buffer, &mut sparse_alpha, &mut sparse_beta);
        alpha = pattern.to_dense(&sparse_alpha);
        beta = pattern.to_dense(&sparse_beta);

        for mu in 0..basis.nao {
            for nu in 0..basis.nao {
                assert_eq!(alpha[(mu, nu)], reference_alpha[(mu, nu)], "α ({mu}, {nu})");
                assert_eq!(beta[(mu, nu)], reference_beta[(mu, nu)], "β ({mu}, {nu})");
            }
        }
    }

    /// A sparse read outside the pattern is a structural zero, and inside it is the value.
    ///
    /// This is the property the whole substitution rests on: the divide-and-conquer density *is*
    /// zero where no subsystem holds both orbitals, so a matrix that stores nothing there and a
    /// dense one that stores zeros there are the same matrix. If [`SparseMatrix::get`] ever
    /// returned something else outside the pattern — or missed a value inside it — every energy
    /// in this module would move.
    #[test]
    fn a_sparse_read_is_the_dense_value_everywhere() {
        // Long enough that the buffer does not reach end to end; a short chain gives a pattern
        // that is dense, where the test would hold for the wrong reason.
        let (basis, partitioning) = setup(16);
        let pattern = DensityPattern::from_partition(&partitioning, basis.nao);

        let mut dense = Matrix::zeros(basis.nao, basis.nao);
        for subsystem in &partitioning.subsystems {
            for &mu in &subsystem.orbitals {
                for &nu in &subsystem.orbitals {
                    dense[(mu, nu)] = 0.07 * (mu as f64) - 0.011 * (nu as f64) + 1.0;
                }
            }
        }
        let sparse = pattern.read_dense(&dense);
        let mut inside = 0usize;
        for mu in 0..basis.nao {
            for nu in 0..basis.nao {
                assert_eq!(
                    sparse.get(&pattern, mu, nu),
                    dense[(mu, nu)],
                    "({mu}, {nu})"
                );
                if pattern.index_of(mu, nu).is_some() {
                    inside += 1;
                }
            }
        }
        assert_eq!(inside, pattern.nnz(), "the pattern disagrees with itself");
        assert!(
            inside < basis.nao * basis.nao,
            "the pattern is dense, so this proves nothing"
        );
    }

    /// The point of the pattern: its size grows with `N`, not with `N²`. Doubling the chain
    /// must not much more than double the stored entries, while the dense count quadruples.
    #[test]
    fn the_stored_entries_grow_linearly_with_the_chain() {
        let (small_basis, small) = setup(8);
        let (large_basis, large) = setup(16);
        let small_nnz = DensityPattern::from_partition(&small, small_basis.nao).nnz() as f64;
        let large_nnz = DensityPattern::from_partition(&large, large_basis.nao).nnz() as f64;

        let growth = large_nnz / small_nnz;
        assert!(
            growth < 2.6,
            "sparse entries grew {growth:.2}x on doubling; linear would be 2.0"
        );

        let dense_growth =
            (large_basis.nao * large_basis.nao) as f64 / (small_basis.nao * small_basis.nao) as f64;
        assert!(
            dense_growth > 3.5,
            "the dense comparison should have quadrupled, got {dense_growth:.2}x"
        );
    }
}
