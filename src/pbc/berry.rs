// SPDX-License-Identifier: GPL-3.0-or-later

//! **Berry-phase electronic polarization** — King-Smith and Vanderbilt's modern theory.
//!
//! # Why polarization is not `Σ q r`
//!
//! In a periodic crystal the dipole per cell is not a function of the charge density. Moving the
//! cell boundary moves a charge from one side to the other and changes `Σ q r` by a lattice
//! vector times that charge, so the "dipole of the unit cell" depends on where the cell was
//! drawn. No amount of care with the sum fixes this — the quantity is genuinely not defined.
//!
//! What *is* defined is the **change** in polarization along an adiabatic path, and the object
//! whose changes those are is a Berry phase of the occupied Bloch states:
//!
//! ```text
//! P_el,α = −(e/Ω) (1/2π) a_α (1/N_⊥) Σ_{k_⊥} Im ln Π_j det S(k_j, k_{j+1})
//! ```
//!
//! The product runs along a **string** of k points spanning the Brillouin zone in direction `α`,
//! and `S` is the overlap of the occupied manifolds at neighbouring points. The result is defined
//! only **modulo the quantum** `e a_α / Ω`: a different branch of the logarithm assigns the
//! electrons to a different unit cell, which is an equally valid choice. That ambiguity is the
//! physics rather than a defect, and [`BerryPolarization::quantum`] reports it so that
//! [`BerryPolarization::difference`] can reduce a difference onto the branch nearest zero.
//!
//! # The overlap in an NDDO basis
//!
//! `S_mn(k, k+b) = ⟨u_mk | u_n,k+b⟩` is over the cell-periodic parts. This crate builds
//! `H(k) = Σ_T e^{ik·T} H(0, T)` — the phase carried on the lattice translation alone — so the
//! coefficients are in the cell gauge and
//!
//! ```text
//! S_mn(k, k+b) = Σ_μ c*_{μm}(k) e^{−i b·τ_μ} c_{μn}(k+b)
//! ```
//!
//! with `τ_μ` the position of the atom carrying orbital `μ`. NDDO assumes an orthonormal AO basis
//! and puts each orbital at its atom, which is the same approximation
//! [`crate::dipole::dipole_matrix`] makes when it writes `R_a` on the diagonal of atom `a`'s
//! block. Using anything else here would make the Berry phase and the dipole disagree about where
//! an orbital sits.
//!
//! What it drops is the intra-atomic `s`–`p` hybridization moment `dd_a`, which the dipole
//! operator does carry. That is a real difference between the two routes and it is measured
//! rather than argued away — see `tests/pbc_berry.rs`.
//!
//! # Closing the string, and why it needs no extra factor
//!
//! In the cell gauge `H(k) = Σ_T e^{ik·T} H(0, T)` is **exactly periodic** in `k`: `G·T` is a
//! multiple of `2π` for every lattice translation, so `H(k + G) = H(k)` term by term and the
//! coefficients at `k_0 + G` are the coefficients at `k_0`. The last link is therefore an
//! ordinary link that happens to reuse the `k_0` vectors, with the same `e^{−ib·τ_μ}` as every
//! other, and the `e^{−iG·τ_μ}` that closes the loop appears on its own: `J` links each carrying
//! `b = G/J` multiply to exactly that.
//!
//! Worth stating because the atomic gauge -- `H(k) = Σ_T e^{ik·(T + τ_ν − τ_μ)} H(0, T)` -- is
//! not periodic in `k`, and there the closing link *does* need an explicit `e^{−iG·τ_μ}` on top.
//! Applying that correction here instead put the Born charge at `+21.8 e` against a CPHF value of
//! `−0.33`, which is what the cross-check in `tests/pbc_berry.rs` is for.
//!
//! Either way the product is manifestly gauge invariant in the other sense: whatever phase the
//! diagonalizer put on an eigenvector at an interior point appears once as `c*` and once as `c`
//! and cancels.

// A note on `clippy::needless_range_loop`, allowed below.
//
// The loop variables here are Cartesian directions (`alpha`, `beta`, `axis`), atom indices, or
// the rows and columns of a matrix being eliminated. The index *is* the meaning: `for alpha in
// 0..3` says which direction, where `for (alpha, row) in out.iter_mut().enumerate()` says it
// less clearly and no more safely. Several of these loops also index two different tensors by
// the same direction, which no single iterator expresses.
#![allow(clippy::needless_range_loop)]

use crate::basis::Basis;
use crate::cmatrix::CMatrix;
use crate::error::{Pm3Error, Result};
use crate::math::Vec3;
use crate::params::Pm3Parameters;
use crate::pbc::gamma::PeriodicOptions;
use crate::pbc::kpoints::KPoint;
use crate::pbc::kscf::{KpointOptions, KpointResult};
use crate::system::Molecule;
use crate::Pm3Options;
use faer::c64;

/// The polarization of a periodic cell, in `e·Bohr / Bohr³` (atomic units).
#[derive(Clone, Debug)]
pub struct BerryPolarization {
    /// Electronic contribution, from the Berry phase of the occupied manifold.
    pub electronic: Vec3,
    /// Ionic contribution, `(1/Ω) Σ_A Z_A τ_A` over the core charges.
    pub ionic: Vec3,
    /// Their sum. Defined **modulo** [`Self::quantum`].
    pub total: Vec3,
    /// The Berry phase along each lattice direction, in units of `2π`. The raw, branch-dependent
    /// number, reported because everything else here is derived from it.
    pub phase: [f64; 3],
    /// The polarization quantum along each lattice vector, `e a_α / Ω`. Two polarizations are the
    /// same physical state if they differ by an integer combination of these.
    pub quantum: [Vec3; 3],
    /// How many k points each string used.
    pub string_length: usize,
}

impl BerryPolarization {
    /// `other − self`, reduced onto the branch nearest zero along each lattice direction.
    ///
    /// The only physically meaningful thing to do with two polarizations. Subtracting the `total`
    /// fields is wrong whenever the two landed on different branches, which for a finite
    /// displacement is common and gives an answer off by exactly one quantum — a number that
    /// looks like a catastrophic error rather than like a bookkeeping choice.
    pub fn difference(&self, other: &Self) -> Vec3 {
        let mut delta = other.total - self.total;
        // A lattice reduction of the difference: the quanta are the lattice vectors scaled by
        // `e/Ω`, so rounding the projection onto each and subtracting lands on the nearest branch.
        for q in &self.quantum {
            let n2 = q.norm2();
            if n2 < 1.0e-30 {
                continue;
            }
            let n = (delta.dot(*q) / n2).round();
            delta -= *q * n;
        }
        delta
    }
}

/// Determinant of a complex matrix by Gaussian elimination with partial pivoting.
///
/// Only ever applied to an occupied-by-occupied overlap block, which is small — the cost here is
/// nothing beside the diagonalization that produced its inputs.
///
/// Returns zero for a singular matrix rather than failing: a vanishing overlap between adjacent
/// points on a string is a real condition (the string is too coarse to follow the manifold), and
/// the caller detects it from the product rather than from an error deep in a loop.
fn determinant(mut a: Vec<Vec<c64>>) -> c64 {
    let n = a.len();
    let mut det = c64::new(1.0, 0.0);
    for col in 0..n {
        let mut pivot = col;
        let mut best = a[col][col].norm();
        for row in (col + 1)..n {
            let size = a[row][col].norm();
            if size > best {
                best = size;
                pivot = row;
            }
        }
        if best == 0.0 {
            return c64::new(0.0, 0.0);
        }
        if pivot != col {
            a.swap(pivot, col);
            det = -det;
        }
        det *= a[col][col];
        let diagonal = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / diagonal;
            if factor == c64::new(0.0, 0.0) {
                continue;
            }
            for k in col..n {
                let value = a[col][k] * factor;
                a[row][k] -= value;
            }
        }
    }
    det
}

/// `S_mn = Σ_μ c*_{μm}(left) e^{−i b·τ_μ} c_{μn}(right)`, over the lowest `n_occ` bands.
///
/// `phase_per_ao` is `e^{−i b·τ_μ}` precomputed per orbital, since it is the same for every pair
/// of bands and depends only on the step.
fn overlap_block(
    left: &CMatrix,
    right: &CMatrix,
    phase_per_ao: &[c64],
    n_occ: usize,
) -> Vec<Vec<c64>> {
    let nao = phase_per_ao.len();
    let mut s = vec![vec![c64::new(0.0, 0.0); n_occ]; n_occ];
    for (m, row) in s.iter_mut().enumerate() {
        for (n, entry) in row.iter_mut().enumerate() {
            let mut total = c64::new(0.0, 0.0);
            for mu in 0..nao {
                total += left[(mu, m)].conj() * phase_per_ao[mu] * right[(mu, n)];
            }
            *entry = total;
        }
    }
    s
}

/// Compute the Berry-phase polarization of a periodic cell.
///
/// `strings` is the number of k points along each Brillouin-zone string. It is the convergence
/// parameter, and the answer must become independent of it — `tests/pbc_berry.rs` checks that it
/// does rather than assuming a value is enough.
///
/// The transverse sampling comes from `kopts`. A string is a one-dimensional integration for each
/// transverse point, so the total work is `strings × (transverse points)` diagonalizations in a
/// potential that is converged once.
///
/// # What it refuses
///
/// A cell that is not fully periodic. The quantum is `e a_α / Ω` and `Ω` has to be a volume; a
/// slab or a chain has a polarization along its periodic directions only, which this does not
/// separate out.
pub fn berry_polarization(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    kopts: &KpointOptions,
    strings: usize,
) -> Result<BerryPolarization> {
    crate::pbc::refuse_field(molecule, options)?;
    let cell = molecule
        .cell
        .ok_or_else(|| Pm3Error::InvalidInput("a Berry phase needs a periodic cell".to_string()))?;
    if cell.n_periodic() != 3 {
        return Err(Pm3Error::InvalidInput(
            "the Berry-phase polarization is defined here for a three-dimensional cell: the \
             quantum is `e a/Ω` and Ω has to be a volume. A slab or a chain has a polarization \
             along its periodic directions only, which this does not separate out."
                .to_string(),
        ));
    }
    if strings < 3 {
        return Err(Pm3Error::InvalidInput(
            "a Berry-phase string needs at least 3 k points: the discretized phase is a product \
             of nearest-neighbour overlaps, and two points cannot resolve a winding"
                .to_string(),
        ));
    }

    let scf: KpointResult =
        crate::pbc::kscf::run_kpoints(molecule, params, options, periodic, kopts)?;
    let setup = crate::pbc::gamma::build_setup(molecule, params, periodic)?;
    let basis = Basis::build(molecule, params)?;
    let potential = crate::pbc::kscf::converged_potential(molecule, params, &setup, &scf)?;

    // The position of the atom each orbital sits on. This is the NDDO approximation the dipole
    // operator already makes, and using anything else would make the two disagree.
    let mut tau = vec![Vec3::zero(); basis.nao];
    for (a, atom) in molecule.atoms.iter().enumerate() {
        let start = basis.atom_offset[a];
        for slot in tau[start..(start + basis.atom_norb[a])].iter_mut() {
            *slot = atom.position;
        }
    }

    // How many bands are occupied, and how many electrons each holds. A restricted calculation
    // carries two per band and has one manifold; an unrestricted one has two manifolds of one.
    let n_elec: f64 = molecule
        .atoms
        .iter()
        .map(|atom| params.element(atom.z).map(|e| e.core_charge).unwrap_or(0.0))
        .sum::<f64>()
        - molecule.charge;
    let manifolds: Vec<(bool, usize, f64)> = if scf.unrestricted {
        let total = n_elec.round() as usize;
        let unpaired = options.multiplicity.saturating_sub(1);
        let n_beta = (total - unpaired) / 2;
        let n_alpha = total - n_beta;
        vec![(false, n_alpha, 1.0), (true, n_beta, 1.0)]
    } else {
        vec![(false, (n_elec / 2.0).round() as usize, 2.0)]
    };
    for (_, n_occ, _) in &manifolds {
        if *n_occ == 0 || *n_occ > basis.nao {
            return Err(Pm3Error::InvalidInput(format!(
                "the occupied manifold has {n_occ} bands against {} orbitals; a Berry phase needs \
                 a filled manifold to follow",
                basis.nao
            )));
        }
    }

    // `reciprocal_basis` returns `(axis, vector)` pairs for the periodic directions only. Every
    // direction is periodic here -- the refusal above saw to that -- so this fills all three.
    let mut reciprocal = [Vec3::zero(); 3];
    for (axis, vector) in cell.reciprocal_basis() {
        reciprocal[axis] = vector;
    }

    // The transverse sampling. `Explicit` points are a band path rather than a mesh and carry no
    // divisions to read, so they get one transverse point per direction: a string is still a
    // correct one-dimensional integration, just an unaveraged one.
    let divisions = match kopts.spec {
        crate::pbc::kpoints::KpointSpec::Mesh { divisions, .. } => divisions,
        _ => [1, 1, 1],
    };
    let mut phase = [0.0_f64; 3];

    for axis in 0..3 {
        let (t1, t2) = ((axis + 1) % 3, (axis + 2) % 3);
        let mut total_phase = 0.0_f64;
        let mut transverse_count = 0usize;

        for i1 in 0..divisions[t1].max(1) {
            for i2 in 0..divisions[t2].max(1) {
                let mut base = [0.0_f64; 3];
                base[t1] = i1 as f64 / divisions[t1].max(1) as f64;
                base[t2] = i2 as f64 / divisions[t2].max(1) as f64;

                let mut string_phase = 0.0_f64;
                for (beta_spin, n_occ, weight) in &manifolds {
                    // The occupied coefficients at each point of the string.
                    let mut coefficients: Vec<CMatrix> = Vec::with_capacity(strings);
                    for j in 0..strings {
                        let mut frac = base;
                        frac[axis] = j as f64 / strings as f64;
                        let k = KPoint { frac, weight: 1.0 };
                        let (_, vectors) =
                            crate::pbc::kscf::bands_at(&setup, &potential, &k, *beta_spin)?;
                        coefficients.push(vectors);
                    }

                    // `b`, the step between adjacent points, in Cartesian reciprocal space.
                    let step = reciprocal[axis] / strings as f64;
                    let phase_per_ao: Vec<c64> = tau
                        .iter()
                        .map(|position| {
                            let angle = -step.dot(*position);
                            c64::new(angle.cos(), angle.sin())
                        })
                        .collect();
                    let mut accumulated = c64::new(1.0, 0.0);
                    for j in 0..strings {
                        // The last link closes onto `k_0 + G`, whose coefficients *are* the `k_0`
                        // coefficients in this gauge -- see the module note. So it is an ordinary
                        // link with the same step factor, and the `e^{−iG·τ}` that closes the loop
                        // is the product of the `J` steps rather than an extra term on one of them.
                        let right = if j + 1 == strings { 0 } else { j + 1 };
                        let block = overlap_block(
                            &coefficients[j],
                            &coefficients[right],
                            &phase_per_ao,
                            *n_occ,
                        );
                        let d = determinant(block);
                        if d.norm() == 0.0 {
                            return Err(Pm3Error::InvalidInput(format!(
                                "the overlap between adjacent points on the string along axis \
                                 {axis} is singular, so the occupied manifold cannot be followed \
                                 from one to the next. Increase `strings` above {strings}."
                            )));
                        }
                        // Accumulated as a product and normalized each step: `strings` complex
                        // multiplications otherwise overflow or underflow the modulus long before
                        // the argument, which is the only part that matters.
                        accumulated = (accumulated * d) / (accumulated * d).norm();
                    }
                    string_phase += weight * accumulated.arg() / (2.0 * std::f64::consts::PI);
                }
                total_phase += string_phase;
                transverse_count += 1;
            }
        }
        phase[axis] = total_phase / transverse_count as f64;
    }

    let volume = cell.measure();
    // `P_el = (e/Ω) Σ_α φ_α a_α`, with `φ` in units of `2π`.
    //
    // The sign follows from the `e^{−ib·τ_μ}` in the overlap and is fixed by a limit rather than
    // by a convention quoted from elsewhere. Take one orbital in a large box: `c = 1`, every link
    // contributes `e^{−ib·τ}`, and the `J` of them multiply to `e^{−iG·τ}`, so `φ = −τ_α/a_α` in
    // units of `2π`. An electron there carries `P = −(occupancy) τ_α / Ω`, which is
    // `+a_α φ_α · occupancy / Ω` — a plus, because the minus of the electron's charge and the
    // minus already inside `φ` cancel. Writing the textbook `−(e/Ω) φ a` on top of this overlap
    // convention double-counts that sign and puts the Born charge of fluorine at `+14.4 e`
    // instead of `−0.33`.
    let mut electronic = Vec3::zero();
    for axis in 0..3 {
        electronic += cell.vector(axis) * (phase[axis] / volume);
    }

    let mut ionic = Vec3::zero();
    for atom in &molecule.atoms {
        let z = params.element(atom.z)?.core_charge;
        ionic += atom.position * (z / volume);
    }

    let quantum = [
        cell.vector(0) / volume,
        cell.vector(1) / volume,
        cell.vector(2) / volume,
    ];

    Ok(BerryPolarization {
        electronic,
        ionic,
        total: electronic + ionic,
        phase,
        quantum,
        string_length: strings,
    })
}
