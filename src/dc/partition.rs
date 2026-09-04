// SPDX-License-Identifier: GPL-3.0-or-later

//! Spatial partitioning for divide-and-conquer: cores, buffers, and the weights that put the
//! pieces back together.
//!
//! # The core is a partition; the buffer is not
//!
//! Every atom belongs to exactly **one** core. That is what makes the reassembly exact in the
//! bookkeeping sense — no density element is counted twice or dropped by accident. The buffer is
//! the opposite: it is whatever lies within `buffer_radius` of the core, it overlaps freely with
//! other subsystems' cores and buffers, and it exists only so that the orbitals of the core see a
//! realistic environment before being truncated.
//!
//! # The Yang–Lee weights
//!
//! A density element `P_μν` is assembled from the subsystems that can see both orbitals:
//!
//! ```text
//! D^α_μν = 1     both μ and ν in core α
//!        = 1/2   one in core α, the other in α's buffer
//!        = 0     otherwise
//! ```
//!
//! and these sum to exactly 1 over `α` whenever the pair is covered at all. Take `μ ∈ core α` and
//! `ν ∈ core β` with `α ≠ β`: if each core reaches into the other's buffer, both contribute a
//! half. If neither does, the element is set to **zero** — and that, rather than any weighting
//! subtlety, is the entire divide-and-conquer approximation. Its error is the density between
//! atoms further apart than the buffer radius, which decays exponentially with a gap.
//!
//! [`Partition::weight_sum`] reports the actual sum for every pair, so a test can check that the
//! only pairs not summing to one are the ones deliberately dropped.

use crate::basis::Basis;
use crate::error::{Pm3Error, Result};
use crate::math::Vec3;
use crate::system::Molecule;

/// How the system is cut up.
#[derive(Clone, Copy, Debug)]
pub struct DcOptions {
    /// Target radius of a core region (Bohr). Cores are grown to roughly this size.
    pub core_radius: f64,
    /// How far beyond the core the buffer reaches (Bohr).
    pub buffer_radius: f64,
    /// Never make a core smaller than this; below a handful of atoms the buffer does all the work
    /// and the partitioning is pure overhead.
    pub min_core_atoms: usize,
    /// Tolerance on the electron count when the global chemical potential is bisected.
    pub electron_tol: f64,
    /// Fermi–Dirac broadening (eV) used to occupy the local eigenvalues.
    ///
    /// Not optional the way it is for a k-mesh: subsystems are diagonalized independently, so a
    /// hard cutoff would let an orbital jump between "occupied here, empty there" from one
    /// iteration to the next and the electron count would oscillate. A small broadening makes the
    /// count a smooth function of `μ`, which is what the bisection needs.
    pub smearing_ev: f64,
    /// Distance (Bohr) beyond which two atoms interact through a point-charge model instead of
    /// a resident NDDO pair table, or `None` to keep every pair exact.
    ///
    /// # What this buys and what it costs
    ///
    /// `None` — the default — builds the two-electron table for all `N(N−1)/2` pairs and keeps
    /// it resident for the whole SCF. That is `O(N²)` in both time and memory and it is what
    /// dominates a large divide-and-conquer run: the diagonalization the method exists to
    /// remove is not the bottleneck at reachable sizes.
    ///
    /// `Some(r)` runs the same Coulomb split the periodic path uses, at zero dimensions: a
    /// neighbour-list pair table inside `r`, and outside it the point-charge model that PM3's
    /// own multipoles define. The pair tables then grow linearly with the system rather than
    /// quadratically, and only the point-charge sum stays quadratic — with about ten operations
    /// per site pair rather than hundreds per atom pair.
    ///
    /// The cost is the documented Klopman–Ohno switch (see [`crate::pbc::screen`]): beyond `r`
    /// the screened form is handed back to its point limit. Measured on water, that is **3 µeV
    /// per atom at `r = 22`**, flat from 12 to 288 atoms — an intensive error rather than one
    /// that accumulates. It is roughly ten times the divide-and-conquer truncation error per
    /// atom at the default buffer, so it is opt-in rather than automatic. Widening `r` shrinks
    /// it as `r⁻³` while the near-field pair count grows as `r³`.
    ///
    /// Refused for molecules containing `Cb`: the capped-bond energy correction is not part of
    /// this path.
    pub long_range_cutoff: Option<f64>,
}

impl Default for DcOptions {
    fn default() -> Self {
        Self {
            core_radius: 6.0,
            buffer_radius: 9.0,
            min_core_atoms: 1,
            electron_tol: 1.0e-9,
            smearing_ev: 0.1,
            long_range_cutoff: None,
        }
    }
}

/// One subsystem: a core, and the buffer that surrounds it.
#[derive(Clone, Debug)]
pub struct Subsystem {
    /// Atoms whose density this subsystem is responsible for.
    pub core: Vec<usize>,
    /// Every atom in the local problem, core first, then buffer. These index the local matrices.
    pub atoms: Vec<usize>,
    /// AO indices of `atoms`, in the same order — the rows and columns extracted from the global
    /// matrices.
    pub orbitals: Vec<usize>,
    /// `true` for the entries of [`Subsystem::orbitals`] that belong to a core atom.
    pub in_core: Vec<bool>,
}

impl Subsystem {
    pub fn n_orbitals(&self) -> usize {
        self.orbitals.len()
    }
}

/// A complete partitioning.
#[derive(Clone, Debug)]
pub struct Partition {
    pub subsystems: Vec<Subsystem>,
    /// Which core each atom belongs to.
    pub core_of: Vec<usize>,
    pub nao: usize,
}

impl Partition {
    /// `Σ_α D^α_μν` for every orbital pair — 1 where the pair is covered, 0 where it is dropped.
    ///
    /// Exists to be asserted on. The reassembly is only meaningful if these are exactly 1 or 0.
    pub fn weight_sum(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.nao * self.nao];
        for subsystem in &self.subsystems {
            for (local_mu, &mu) in subsystem.orbitals.iter().enumerate() {
                for (local_nu, &nu) in subsystem.orbitals.iter().enumerate() {
                    out[mu * self.nao + nu] +=
                        weight(subsystem.in_core[local_mu], subsystem.in_core[local_nu]);
                }
            }
        }
        out
    }

    /// How many orbital pairs the partitioning drops, and the largest subsystem it produced —
    /// the two numbers that say whether a partitioning is worth using.
    pub fn dropped_pairs(&self) -> usize {
        self.weight_sum().iter().filter(|w| **w < 0.5).count()
    }

    pub fn largest_subsystem(&self) -> usize {
        self.subsystems
            .iter()
            .map(|s| s.n_orbitals())
            .max()
            .unwrap_or(0)
    }
}

/// The Yang–Lee weight for one orbital pair in one subsystem.
#[inline]
pub fn weight(mu_in_core: bool, nu_in_core: bool) -> f64 {
    match (mu_in_core, nu_in_core) {
        (true, true) => 1.0,
        (true, false) | (false, true) => 0.5,
        (false, false) => 0.0,
    }
}

/// Partition a molecule into cores and buffers.
///
/// Cores are grown greedily: take the atom furthest from the centroid that is not yet assigned,
/// and claim everything within `core_radius` of it that is still free. Greedy growth from the
/// outside in produces compact cores and leaves no straggler that would need a core of its own —
/// growing from the inside out tends to leave a shell of ones and twos at the surface.
pub fn partition(molecule: &Molecule, basis: &Basis, options: &DcOptions) -> Result<Partition> {
    let nat = molecule.atoms.len();
    if nat == 0 {
        return Err(Pm3Error::InvalidInput(
            "cannot partition an empty molecule".to_string(),
        ));
    }
    if options.core_radius <= 0.0 || options.buffer_radius < 0.0 {
        return Err(Pm3Error::InvalidInput(
            "divide-and-conquer needs a positive core radius and a non-negative buffer".to_string(),
        ));
    }
    let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let centroid = positions.iter().fold(Vec3::zero(), |sum, p| sum + *p) * (1.0 / nat as f64);

    let mut core_of = vec![usize::MAX; nat];
    let mut cores: Vec<Vec<usize>> = Vec::new();
    let mut remaining = nat;
    while remaining > 0 {
        // The unassigned atom furthest from the centre seeds the next core.
        let seed = (0..nat)
            .filter(|a| core_of[*a] == usize::MAX)
            .max_by(|a, b| {
                let da = (positions[*a] - centroid).norm2();
                let db = (positions[*b] - centroid).norm2();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("remaining > 0");
        let index = cores.len();
        let mut core = Vec::new();
        for atom in 0..nat {
            if core_of[atom] != usize::MAX {
                continue;
            }
            if (positions[atom] - positions[seed]).norm() <= options.core_radius {
                core_of[atom] = index;
                core.push(atom);
                remaining -= 1;
            }
        }
        // The seed is always within its own radius, so `core` is never empty and the loop always
        // makes progress.
        debug_assert!(!core.is_empty());

        // Absorb an undersized core into the nearest existing one rather than leaving a subsystem
        // too small for its buffer to mean anything.
        if core.len() < options.min_core_atoms && index > 0 {
            let nearest = (0..index)
                .min_by(|a, b| {
                    let da = core_distance(&positions, &cores[*a], &core);
                    let db = core_distance(&positions, &cores[*b], &core);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("index > 0");
            for atom in &core {
                core_of[*atom] = nearest;
            }
            cores[nearest].extend(core);
            continue;
        }
        cores.push(core);
    }

    let reach = options.core_radius + options.buffer_radius;
    let mut subsystems = Vec::with_capacity(cores.len());
    for core in &cores {
        let mut atoms = core.clone();
        for atom in 0..nat {
            if core_of[atom] == core_of[core[0]] {
                continue;
            }
            let close = core
                .iter()
                .any(|c| (positions[atom] - positions[*c]).norm() <= reach);
            if close {
                atoms.push(atom);
            }
        }
        let mut orbitals = Vec::new();
        let mut in_core = Vec::new();
        for atom in &atoms {
            let is_core = core_of[*atom] == core_of[core[0]];
            let off = basis.atom_offset[*atom];
            for k in 0..basis.atom_norb[*atom] {
                orbitals.push(off + k);
                in_core.push(is_core);
            }
        }
        subsystems.push(Subsystem {
            core: core.clone(),
            atoms,
            orbitals,
            in_core,
        });
    }

    Ok(Partition {
        subsystems,
        core_of,
        nao: basis.nao,
    })
}

/// Closest approach between two groups of atoms.
fn core_distance(positions: &[Vec3], a: &[usize], b: &[usize]) -> f64 {
    let mut best = f64::INFINITY;
    for i in a {
        for j in b {
            best = best.min((positions[*i] - positions[*j]).norm());
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Pm3Parameters;

    /// A chain of `n` water molecules, spaced far enough apart to make the partitioning obvious.
    fn water_chain(n: usize) -> Molecule {
        let mut lines = format!("{}\nchain\n", 3 * n);
        for i in 0..n {
            let x = 6.0 * i as f64;
            lines.push_str(&format!("O {:.4} 0.0 0.0\n", x));
            lines.push_str(&format!("H {:.4} 0.0 0.0\n", x + 0.9584));
            lines.push_str(&format!("H {:.4} 0.9278 0.0\n", x - 0.24));
        }
        Molecule::from_xyz_str(&lines, 0.0).unwrap()
    }

    fn parts(n: usize, options: &DcOptions) -> (Partition, Basis) {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = water_chain(n);
        let basis = Basis::build(&molecule, &params).unwrap();
        (partition(&molecule, &basis, options).unwrap(), basis)
    }

    /// Every atom in exactly one core: the property the reassembly rests on.
    #[test]
    fn every_atom_belongs_to_exactly_one_core() {
        let (partitioning, _) = parts(8, &DcOptions::default());
        let mut seen = vec![0usize; 24];
        for subsystem in &partitioning.subsystems {
            for atom in &subsystem.core {
                seen[*atom] += 1;
            }
        }
        assert!(
            seen.iter().all(|count| *count == 1),
            "core membership is not a partition: {seen:?}"
        );
        for (atom, count) in seen.iter().enumerate() {
            assert_eq!(*count, 1, "atom {atom} is in {count} cores");
            assert!(partitioning.core_of[atom] < partitioning.subsystems.len());
        }
    }

    /// The Yang–Lee weights sum to exactly one for every pair the partitioning keeps, and to zero
    /// for the ones it drops. Anything in between would mean a density element scaled by an
    /// arbitrary factor — a failure mode that shows up as a plausible but wrong energy.
    #[test]
    fn the_weights_sum_to_one_or_zero() {
        let (partitioning, _) = parts(6, &DcOptions::default());
        for (index, sum) in partitioning.weight_sum().iter().enumerate() {
            let ok = (sum - 1.0).abs() < 1.0e-12 || sum.abs() < 1.0e-12;
            assert!(
                ok,
                "pair {index} has weight {sum}, which is neither one nor zero"
            );
        }
    }

    /// A buffer large enough to reach everything drops nothing — the partitioning then describes
    /// the same density matrix the full calculation does.
    #[test]
    fn a_reaching_buffer_drops_nothing() {
        let options = DcOptions {
            buffer_radius: 1000.0,
            ..DcOptions::default()
        };
        let (partitioning, _) = parts(6, &options);
        assert_eq!(partitioning.dropped_pairs(), 0);
    }

    /// A small buffer drops the distant pairs, which is the point of the method.
    #[test]
    fn a_small_buffer_drops_distant_pairs() {
        let options = DcOptions {
            core_radius: 3.0,
            buffer_radius: 3.0,
            ..DcOptions::default()
        };
        let (partitioning, basis) = parts(8, &options);
        assert!(
            partitioning.dropped_pairs() > 0,
            "a 3 Bohr buffer on a 42 Bohr chain should drop something"
        );
        assert!(
            partitioning.largest_subsystem() < basis.nao,
            "no subsystem should be the whole system"
        );
    }

    /// Subsystems grow with the buffer and never shrink — the knob has to be monotone or "increase
    /// the buffer until it converges" is not a usable procedure.
    #[test]
    fn subsystems_grow_monotonically_with_the_buffer() {
        let mut previous = 0;
        for buffer in [2.0_f64, 4.0, 8.0, 16.0] {
            let options = DcOptions {
                core_radius: 3.0,
                buffer_radius: buffer,
                ..DcOptions::default()
            };
            let (partitioning, _) = parts(8, &options);
            let size = partitioning.largest_subsystem();
            assert!(
                size >= previous,
                "buffer {buffer} gave a smaller subsystem ({size}) than the one before ({previous})"
            );
            previous = size;
        }
    }

    #[test]
    fn an_empty_molecule_is_refused() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = Molecule::default();
        let basis = Basis::build(&molecule, &params).unwrap();
        assert!(partition(&molecule, &basis, &DcOptions::default()).is_err());
    }
}
