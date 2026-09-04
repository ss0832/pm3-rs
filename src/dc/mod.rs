// SPDX-License-Identifier: GPL-3.0-or-later

//! Divide-and-conquer SCF (Yang–Lee).
//!
//! # The idea in one line
//!
//! Diagonalizing an `N × N` Fock matrix costs `O(N³)`. Diagonalizing `N/m` matrices of size `m`
//! costs `O(N m²)`, which is linear in `N` — so cut the system into pieces, solve each in the
//! presence of a buffer, and put the density back together.
//!
//! # What makes it work, and what makes it approximate
//!
//! The density matrix of a system with a gap is **near-sighted**: `P_μν` decays exponentially with
//! the distance between the orbitals. A subsystem large enough to contain that decay length
//! reproduces its own core's density to within the truncation, and the pieces reassemble exactly
//! — see [`partition`] for the weights. The only approximation is that `P_μν` is set to zero for
//! orbital pairs no single subsystem contains.
//!
//! That is also why the buffer radius is the one knob that matters: increasing it must converge
//! the result onto the full diagonalization, monotonically and from one side. If it does not, the
//! system is not near-sighted — it is metallic, or the gap has closed — and divide-and-conquer is
//! the wrong method rather than a poorly converged one.
//!
//! # One chemical potential, not one per subsystem
//!
//! Subsystems are diagonalized independently but they are not independent systems: electrons flow
//! between them. Filling each to its own electron count would freeze whatever charge distribution
//! the partitioning happened to imply. Instead every subsystem is occupied from a **single global
//! Fermi level**, bisected so the assembled density holds the right number of electrons. That is
//! what lets charge redistribute, and it is why a charged system needs no special handling here:
//! the constraint is the electron count, and a charged system simply has a different one.
//!
//! Because the occupation must be a smooth function of that potential for the bisection to work,
//! the local eigenvalues are occupied with a small Fermi–Dirac broadening rather than a step. See
//! [`DcOptions::smearing_ev`].

pub mod derivatives;
pub mod partition;
pub(crate) mod pattern;
pub mod scf;

pub use derivatives::{dc_gradient, dc_periodic_gradient, DcGradient, DcPeriodicGradient};
pub use partition::{partition, DcOptions, Partition, Subsystem};
pub use scf::{run_dc, run_dc_gamma, DcPeriodicResult, DcResult};

/// How many density-matrix entries a partitioning can populate, out of `nao x nao`.
///
/// The ratio is what the partitioned SCF stores instead of a dense matrix, and it is the
/// quantity to watch when asking whether a system is large enough for the method to pay: it
/// grows linearly with the atom count while `nao x nao` grows quadratically.
pub fn pattern_nnz(partitioning: &Partition, nao: usize) -> usize {
    pattern::DensityPattern::from_partition(partitioning, nao).nnz()
}
