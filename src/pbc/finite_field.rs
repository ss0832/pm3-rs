// SPDX-License-Identifier: GPL-3.0-or-later

//! A finite electric field **along** a periodic direction, by the Berry-phase electric enthalpy.
//!
//! # Why `𝓔·R` cannot be used here
//!
//! Under a lattice, `𝓔·R` shifts by `𝓔·T` on translation by `T`, so it is lattice-periodic exactly
//! when `𝓔·T = 0` for every lattice vector. A field orthogonal to all of them — normal to a slab,
//! transverse to a chain — is an ordinary calculation and goes through [`Pm3Options::field`].
//! Along a periodic direction the potential is unbounded, the spectrum has no lower bound, and no
//! care in the assembly repairs that: the ground state of `H − 𝓔·R` on a lattice does not exist.
//!
//! # What replaces it
//!
//! The **electric enthalpy** of Nunes and Gonze, minimized in place of the energy:
//!
//! ```text
//! F[ψ, 𝓔] = E[ψ] − Ω 𝓔·P[ψ]
//! ```
//!
//! with `P` the Berry-phase polarization of [`crate::pbc::berry`] rather than `⟨r⟩`. Because `P`
//! is built from overlaps between **neighbouring** k-points, its derivative couples them: the
//! field term at `k` reads the coefficients at `k ± b`. The k-points can no longer be solved one
//! at a time, which is the structural reason this is not a small change to the SCF.
//!
//! # The coupling, derived here rather than quoted
//!
//! Sign and factor conventions for a Berry phase differ between sources, and a wrong factor does
//! not fail — it returns a plausible polarizability. So this is derived from *this crate's own*
//! polarization convention, fixed and documented in [`crate::pbc::berry`], which is not the one a
//! textbook expression assumes:
//!
//! ```text
//! P_el = (f/Ω) Σ_α a_α φ_α,   φ_α = (1/2π)(1/N⊥) Σ_{k⊥} Im ln Π_j det S_j
//! S_j  = C_j† Δ C_{j+1},      Δ = diag(e^{−i b·τ_μ}),   b = G_α/J,   C_J ≡ C_0
//! ```
//!
//! `f` is the occupancy (2, restricted). `C_J ≡ C_0` with no extra phase is the cell gauge: see
//! [`crate::pbc::berry`], where applying the closure factor a textbook derivation calls for put a
//! Born charge at `+21.8 e` against a true `−0.33`.
//!
//! Differentiate, treating `C` and `C*` as independent. From `S_j` alone,
//! `∂ ln det S_j/∂C*(k_j) = Δ C_{j+1} S_j⁻¹`; the conjugate half picks up `S_{j−1}`, giving
//!
//! ```text
//! ∂(Im ln Z)/∂C*(k_j) = (1/2i) [ Δ C_{j+1} S_j⁻¹ − Δ† C_{j−1} (S_{j−1}⁻¹)† ]  ≡ (1/2i)(W₊ − W₋)
//! ```
//!
//! The energy's own gradient is `w_k f H C(k_j)` with `w_k = 1/(J N⊥)`. Dividing the enthalpy's
//! gradient through by that same `w_k f` turns it into an operator — the `N⊥` cancels against the
//! transverse average already in `φ`, and `1/i = −i` flips the sign:
//!
//! ```text
//! ΔH C(k_j) = i λ_α (W₊ − W₋),     λ_α = (𝓔·a_α) J / 4π
//! ```
//!
//! `ΔH` itself is recovered by projecting onto the occupied manifold,
//! `M = i λ (W₊ − W₋) C_j†`, and made Hermitian as **`M + M†`** — not `½(M + M†)`.
//!
//! The half would be right for a general matrix and is wrong here, because `M` is one-sided:
//! `M|v⟩ = G C†|v⟩ = 0` for any virtual `v`, since `C†` annihilates everything outside the
//! occupied span. So `M` carries the whole virtual-occupied block and none of the
//! occupied-virtual one, `M†` is its mirror, and adding them fills two disjoint blocks once each.
//! Averaging instead halves the occupied-virtual block — which is the block the linear response
//! is made of, so the polarizability comes out at exactly half. That was measured before it was
//! explained: ratio `0.5001` against the CPHF value with the half in place, `1.0003` without.
//!
//! The occupied-occupied block is doubled by this and it does not matter: it mixes occupied
//! orbitals among themselves, leaving the density and hence the polarization untouched.
//!
//! # What says the factor is right
//!
//! Not the derivation. `tests/pbc_finite_field.rs` takes `α = Ω ∂P/∂𝓔` by finite differences of
//! this and compares it against the **CPHF** polarizability from
//! [`crate::pbc::dielectric::polarizability`] — two formalisms sharing only the SCF. A factor of
//! two, a missing `J`, or a sign shows up there and nowhere else.
//!
//! This is a semiempirical model, so neither number is a prediction of experiment; what is being
//! checked is that the crate computes its own model's polarizability consistently by two routes.

// A note on `clippy::needless_range_loop`, allowed below.
//
// The loop variables here are Cartesian directions (`alpha`, `beta`, `axis`), atom indices, or
// the rows and columns of a matrix being eliminated. The index *is* the meaning: `for alpha in
// 0..3` says which direction, where `for (alpha, row) in out.iter_mut().enumerate()` says it
// less clearly and no more safely. Several of these loops also index two different tensors by
// the same direction, which no single iterator expresses.
#![allow(clippy::needless_range_loop)]

use crate::cmatrix::{hermitian_eigen, CMatrix};
use crate::error::{Pm3Error, Result};
use crate::math::Vec3;
use crate::params::Pm3Parameters;
use crate::pbc::gamma::PeriodicOptions;
use crate::pbc::kpoints::{KPoint, KpointSpec};
use crate::pbc::kscf::{KpointOptions, KpointResult};
use crate::system::Molecule;
use crate::Pm3Options;
use faer::c64;

/// A converged finite-field calculation.
#[derive(Clone, Debug)]
pub struct FiniteFieldResult {
    /// The self-consistent state in the field.
    pub scf: KpointResult,
    /// The applied field, eV per `e·Bohr`.
    pub field: Vec3,
    /// Berry phases in turns, one per lattice direction, averaged over transverse strings.
    pub phase: [f64; 3],
    /// Electronic polarization, `e/Bohr²`.
    pub electronic_polarization: Vec3,
    /// Ionic polarization, same units.
    pub ionic_polarization: Vec3,
    /// `electronic + ionic`, modulo the quantum `e a_α/Ω`.
    pub polarization: Vec3,
    /// `E − Ω 𝓔·P` (eV): the quantity actually minimized.
    pub enthalpy_ev: f64,
    /// Outer (field-operator) iterations taken.
    pub iterations: usize,
    pub converged: bool,
    /// Per axis, whether the mesh had the three k-points a Berry phase needs.
    ///
    /// An unresolved axis contributes **zero** to [`Self::phase`] and to
    /// [`Self::electronic_polarization`], which is not the same as its contribution being zero.
    /// A `[6, 1, 1]` mesh resolves `x` alone, which is enough to measure `α_xx` — the `y` and `z`
    /// components cancel in a `±𝓔` difference — and is not enough to read the polarization
    /// vector itself.
    pub resolved: [bool; 3],
}

/// Outer-loop settings for the field operator.
#[derive(Clone, Copy, Debug)]
pub struct FiniteFieldOptions {
    /// Threshold on the largest change in `ΔH` between outer iterations, eV.
    pub tol: f64,
    pub max_iter: usize,
    /// Linear mixing on `ΔH`.
    ///
    /// The field operator is a strongly non-local function of the coefficients — it reads two
    /// neighbouring k-points through a matrix inverse — and a full step oscillates on anything
    /// but the smallest fields. Unlike the response solver in [`crate::pbc::dfpt`], which
    /// extrapolates because its problem is linear, this one is not linear in `C` and mixing is
    /// the appropriate control.
    pub mixing: f64,
}

impl Default for FiniteFieldOptions {
    fn default() -> Self {
        Self {
            tol: 1.0e-8,
            max_iter: 60,
            mixing: 0.5,
        }
    }
}

/// Inverse of a small complex matrix by Gauss-Jordan with partial pivoting.
///
/// Applied only to an occupied-by-occupied overlap block. A singular one means the string is too
/// coarse to follow the manifold from one point to the next, which is reported rather than
/// papered over.
fn invert(a: &[Vec<c64>]) -> Result<Vec<Vec<c64>>> {
    let n = a.len();
    let mut m: Vec<Vec<c64>> = a.to_vec();
    let mut inv: Vec<Vec<c64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        c64::new(1.0, 0.0)
                    } else {
                        c64::new(0.0, 0.0)
                    }
                })
                .collect()
        })
        .collect();

    for col in 0..n {
        let mut pivot = col;
        let mut best = m[col][col].norm();
        for row in (col + 1)..n {
            if m[row][col].norm() > best {
                best = m[row][col].norm();
                pivot = row;
            }
        }
        if best == 0.0 {
            return Err(Pm3Error::InvalidInput(
                "the overlap between adjacent k-points on the string is singular, so the occupied \
                 manifold cannot be followed from one to the next. Use a denser mesh along the \
                 field direction, or a smaller field."
                    .to_string(),
            ));
        }
        m.swap(pivot, col);
        inv.swap(pivot, col);

        let diagonal = m[col][col];
        for k in 0..n {
            m[col][k] /= diagonal;
            inv[col][k] /= diagonal;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = m[row][col];
            if factor == c64::new(0.0, 0.0) {
                continue;
            }
            for k in 0..n {
                let a = m[col][k] * factor;
                m[row][k] -= a;
                let b = inv[col][k] * factor;
                inv[row][k] -= b;
            }
        }
    }
    Ok(inv)
}

/// `S_mn = Σ_μ c*_{μm}(left) e^{−i b·τ_μ} c_{μn}(right)` over the lowest `n_occ` bands.
fn overlap(left: &CMatrix, right: &CMatrix, phase: &[c64], n_occ: usize) -> Vec<Vec<c64>> {
    let nao = phase.len();
    let mut s = vec![vec![c64::new(0.0, 0.0); n_occ]; n_occ];
    for (m, row) in s.iter_mut().enumerate() {
        for (n, entry) in row.iter_mut().enumerate() {
            let mut total = c64::new(0.0, 0.0);
            for mu in 0..nao {
                total += left[(mu, m)].conj() * phase[mu] * right[(mu, n)];
            }
            *entry = total;
        }
    }
    s
}

/// `det` of a small complex matrix, for the phase.
fn determinant(mut a: Vec<Vec<c64>>) -> c64 {
    let n = a.len();
    let mut det = c64::new(1.0, 0.0);
    for col in 0..n {
        let mut pivot = col;
        let mut best = a[col][col].norm();
        for row in (col + 1)..n {
            if a[row][col].norm() > best {
                best = a[row][col].norm();
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
            for k in col..n {
                let value = a[col][k] * factor;
                a[row][k] -= value;
            }
        }
    }
    det
}

/// Where a grid point sits in the explicit k-point list this module builds.
fn index_of(divisions: [usize; 3], i: [usize; 3]) -> usize {
    (i[0] * divisions[1] + i[1]) * divisions[2] + i[2]
}

/// The explicit, unreduced k-point list.
///
/// Explicit rather than a mesh because the field operator is **odd** under `k → −k`: the
/// time-reversal folding that is exact for the ground state would average it against its own
/// negative. Explicit also pins the ordering the terms are indexed by.
fn grid(divisions: [usize; 3]) -> Vec<KPoint> {
    let total = divisions[0] * divisions[1] * divisions[2];
    let weight = 1.0 / total as f64;
    let mut points = Vec::with_capacity(total);
    for i0 in 0..divisions[0] {
        for i1 in 0..divisions[1] {
            for i2 in 0..divisions[2] {
                points.push(KPoint {
                    frac: [
                        i0 as f64 / divisions[0] as f64,
                        i1 as f64 / divisions[1] as f64,
                        i2 as f64 / divisions[2] as f64,
                    ],
                    weight,
                });
            }
        }
    }
    points
}

/// Solve the periodic SCF in a finite field with a component along a periodic direction.
///
/// `divisions` is the k-mesh; the component along each field direction is that direction's string
/// length, and is the convergence parameter the answer must become independent of.
///
/// # What it refuses
///
/// A cell that is not fully periodic (the polarization quantum needs a volume — and a field
/// orthogonal to every lattice vector needs none of this machinery, so use [`Pm3Options::field`]).
/// Fermi smearing, since a Berry phase needs a gapped, integer-filled manifold. An open shell,
/// which would need each spin manifold's phase separately. And a molecular field set at the same
/// time, which would be the same perturbation applied twice.
pub fn run_finite_field(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    divisions: [usize; 3],
    field: Vec3,
    ff: &FiniteFieldOptions,
) -> Result<FiniteFieldResult> {
    let cell = molecule.cell.ok_or_else(|| {
        Pm3Error::InvalidInput("a finite-field calculation needs a cell".to_string())
    })?;
    if cell.n_periodic() != 3 {
        return Err(Pm3Error::InvalidInput(
            "the Berry-phase finite field is implemented for a three-dimensional cell, matching \
             `pbc::berry`. A field orthogonal to every lattice vector needs none of this \
             machinery: set `Pm3Options::field`."
                .to_string(),
        ));
    }
    if options.field.is_some() {
        return Err(Pm3Error::InvalidInput(
            "`Pm3Options::field` and the Berry-phase finite field are two treatments of the same \
             perturbation; pass the field to `run_finite_field` only. The `𝓔·R` form is for a \
             field orthogonal to every lattice vector, this one for a field along a periodic one."
                .to_string(),
        ));
    }
    if divisions.contains(&0) {
        return Err(Pm3Error::InvalidInput(
            "a k-mesh division of zero is not a mesh".to_string(),
        ));
    }

    let mut reciprocal = [Vec3::zero(); 3];
    for (axis, vector) in cell.reciprocal_basis() {
        reciprocal[axis] = vector;
    }

    // Which axes the field couples to, and which the mesh can resolve a phase along. These are
    // **not** the same set, and conflating them was a bug: an axis the field does not touch still
    // carries polarization, so its phase has to be computed even though it contributes no `ΔH`.
    // Computing only the coupled axes made a zero field — or one orthogonal to every lattice
    // vector — report an electronic polarization of exactly zero, which is not its value.
    //
    // The threshold is relative to `|𝓔| |a|` rather than an exact zero: in a non-orthogonal cell
    // a field the caller placed perpendicular to a lattice vector lands at `1e-17` rather than
    // `0`, and an exact test would then demand three k-points along an axis that contributes
    // nothing measurable.
    let mut active: Vec<usize> = Vec::new();
    for axis in 0..3 {
        let a = cell.vector(axis);
        let scale = field.norm() * a.norm();
        if scale > 0.0 && field.dot(a).abs() > 1.0e-12 * scale {
            active.push(axis);
        }
    }
    for axis in &active {
        if divisions[*axis] < 3 {
            return Err(Pm3Error::InvalidInput(format!(
                "the field has a component along lattice vector {axis}, whose string has \
                 {} k-points. A discretized Berry phase is a product of nearest-neighbour \
                 overlaps and two points cannot resolve a winding; use at least 3.",
                divisions[*axis]
            )));
        }
    }
    // An axis needs three points for a phase whether or not the field reaches it. One that has
    // fewer is reported as unresolved rather than as zero.
    let resolved = [divisions[0] >= 3, divisions[1] >= 3, divisions[2] >= 3];

    let kpoints = grid(divisions);
    let kopt = KpointOptions {
        spec: KpointSpec::Explicit(kpoints.clone()),
        ..KpointOptions::default()
    };
    if options.multiplicity > 1 {
        return Err(Pm3Error::InvalidInput(
            "the Berry-phase finite field is restricted-only, matching `pbc::berry`: an \
             open-shell cell would need the phase of each spin manifold separately"
                .to_string(),
        ));
    }

    let basis = crate::basis::Basis::build(molecule, params)?;
    let nao = basis.nao;
    let setup = crate::pbc::gamma::build_setup(molecule, params, periodic)?;

    // Orbital positions, the same NDDO placement the dipole operator and the Berry phase use.
    let mut tau = vec![Vec3::zero(); nao];
    for (a, atom) in molecule.atoms.iter().enumerate() {
        let start = basis.atom_offset[a];
        for slot in tau[start..(start + basis.atom_norb[a])].iter_mut() {
            *slot = atom.position;
        }
    }

    let n_elec: f64 = molecule
        .atoms
        .iter()
        .map(|atom| params.element(atom.z).map(|e| e.core_charge).unwrap_or(0.0))
        .sum::<f64>()
        - molecule.charge;
    let n_occ = (n_elec / 2.0).round() as usize;
    if n_occ == 0 || n_occ >= nao {
        return Err(Pm3Error::InvalidInput(format!(
            "the occupied manifold has {n_occ} bands against {nao} orbitals; a Berry phase needs \
             a filled manifold to follow"
        )));
    }
    let occupancy = 2.0;

    let mut k_terms: Vec<CMatrix> = (0..kpoints.len())
        .map(|_| CMatrix::zeros(nao, nao))
        .collect();
    let mut scf =
        crate::pbc::kscf::run_kpoints_with_terms(molecule, params, options, periodic, &kopt, None)?;
    let mut phase = [0.0_f64; 3];
    let mut converged = false;
    let mut iterations = 0usize;
    // Kept outside the loop so a failure can report how far off it was rather than only that it
    // gave up -- the difference between "needs more iterations" and "is not converging".
    let mut largest_change = f64::INFINITY;

    for iteration in 0..ff.max_iter {
        iterations = iteration + 1;

        scf = crate::pbc::kscf::run_kpoints_with_terms(
            molecule,
            params,
            options,
            periodic,
            &kopt,
            Some(&k_terms),
        )?;

        // The coefficients from the same Hamiltonian the SCF diagonalized: its own Fock plus the
        // field term that was held fixed through it.
        let potential = crate::pbc::kscf::converged_potential(molecule, params, &setup, &scf)?;
        let mut coefficients: Vec<CMatrix> = Vec::with_capacity(kpoints.len());
        for (index, k) in kpoints.iter().enumerate() {
            let mut fk = crate::pbc::kscf::bloch_fock(
                &setup,
                &potential.f_onsite_alpha,
                &potential.p_alpha.images,
                k,
            );
            for row in 0..nao {
                for col in 0..nao {
                    fk[(row, col)] += k_terms[index][(row, col)];
                }
            }
            let (_, vectors) = hermitian_eigen(&fk)?;
            coefficients.push(vectors);
        }

        // Build the new field operator, and the phase, from those coefficients.
        let mut next: Vec<CMatrix> = (0..kpoints.len())
            .map(|_| CMatrix::zeros(nao, nao))
            .collect();
        let mut new_phase = [0.0_f64; 3];

        // Every axis the mesh resolves, not only the ones the field couples to: the phase is a
        // property of the state and all three components of the polarization are reported.
        // `lambda` is zero on the axes the field misses, so those contribute no operator.
        for axis in (0..3).filter(|axis| resolved[*axis]) {
            let j_count = divisions[axis];
            let step = reciprocal[axis] / j_count as f64;
            let delta: Vec<c64> = tau
                .iter()
                .map(|position| {
                    let angle = -step.dot(*position);
                    c64::new(angle.cos(), angle.sin())
                })
                .collect();
            let lambda =
                field.dot(cell.vector(axis)) * j_count as f64 / (4.0 * std::f64::consts::PI);

            let (t1, t2) = ((axis + 1) % 3, (axis + 2) % 3);
            let mut axis_phase = 0.0_f64;
            let mut strings = 0usize;

            for a1 in 0..divisions[t1] {
                for a2 in 0..divisions[t2] {
                    // The k-point indices along this string, in order.
                    let mut line = Vec::with_capacity(j_count);
                    for j in 0..j_count {
                        let mut i = [0usize; 3];
                        i[axis] = j;
                        i[t1] = a1;
                        i[t2] = a2;
                        line.push(index_of(divisions, i));
                    }

                    // `S_j` and its inverse for every link, including the one that closes the
                    // loop onto `k_0` -- an ordinary link in this gauge, see `pbc::berry`.
                    let mut s_inv = Vec::with_capacity(j_count);
                    let mut running = c64::new(1.0, 0.0);
                    for j in 0..j_count {
                        let right = line[(j + 1) % j_count];
                        let s =
                            overlap(&coefficients[line[j]], &coefficients[right], &delta, n_occ);
                        let d = determinant(s.clone());
                        if d.norm() == 0.0 {
                            return Err(Pm3Error::InvalidInput(format!(
                                "the overlap along axis {axis} is singular; the string is too \
                                 coarse to follow the occupied manifold"
                            )));
                        }
                        // Normalized each step: the modulus of a product of `J` determinants
                        // over- or underflows long before its argument, which is all that matters.
                        running = (running * d) / (running * d).norm();
                        s_inv.push(invert(&s)?);
                    }
                    axis_phase += running.arg() / (2.0 * std::f64::consts::PI);
                    strings += 1;

                    // `ΔH C(k_j) = i λ (W₊ − W₋)`, then `M = i λ (W₊ − W₋) C_j†`, Hermitized.
                    for j in 0..j_count {
                        let here = line[j];
                        let ahead = line[(j + 1) % j_count];
                        let behind = line[(j + j_count - 1) % j_count];
                        let s_here = &s_inv[j];
                        let s_back = &s_inv[(j + j_count - 1) % j_count];

                        // W₊ = Δ C_{j+1} S_j⁻¹ ; W₋ = Δ† C_{j−1} (S_{j−1}⁻¹)†
                        let mut w = vec![vec![c64::new(0.0, 0.0); n_occ]; nao];
                        for mu in 0..nao {
                            for m in 0..n_occ {
                                let mut plus = c64::new(0.0, 0.0);
                                let mut minus = c64::new(0.0, 0.0);
                                for n in 0..n_occ {
                                    plus += coefficients[ahead][(mu, n)] * s_here[n][m];
                                    // `(S⁻¹)†` is the conjugate transpose, so index [m][n]
                                    // conjugated rather than [n][m].
                                    minus += coefficients[behind][(mu, n)] * s_back[m][n].conj();
                                }
                                w[mu][m] = delta[mu] * plus - delta[mu].conj() * minus;
                            }
                        }

                        // M = i λ W C_j†, then ΔH += M + M† -- **not** ½(M + M†).
                        //
                        // `M` is one-sided: `M|v⟩ = G C†|v⟩ = 0` for a virtual `v`, because `C†`
                        // annihilates anything outside the occupied span. So its occupied-virtual
                        // block is zero while its virtual-occupied block is the whole coupling,
                        // and `M†` is the mirror image. Adding them fills the two disjoint blocks
                        // once each; the conventional ½ would halve the occupied-virtual block,
                        // which is the one the linear response is made of.
                        //
                        // Measured, not argued: with the ½ in place the polarizability came out
                        // at exactly half the CPHF value (ratio 0.5001 at a 32 Bohr cell on an
                        // 8×1×1 mesh, with `dd` removed from both routes so nothing else
                        // differed). Without it, 1.0003.
                        //
                        // The occupied-occupied block is doubled by this, and that is harmless:
                        // it mixes occupied orbitals among themselves and leaves the density,
                        // and therefore the polarization, unchanged.
                        let i_lambda = c64::new(0.0, lambda);
                        for mu in 0..nao {
                            for nu in 0..nao {
                                let mut value = c64::new(0.0, 0.0);
                                for m in 0..n_occ {
                                    value += w[mu][m] * coefficients[here][(nu, m)].conj();
                                }
                                let entry = i_lambda * value;
                                next[here][(mu, nu)] += entry;
                                next[here][(nu, mu)] += entry.conj();
                            }
                        }
                    }
                }
            }
            new_phase[axis] = axis_phase / strings as f64;
        }

        // Convergence on the operator itself, which is what the outer loop is solving for.
        let mut largest = 0.0_f64;
        for (fresh, old) in next.iter().zip(&k_terms) {
            for row in 0..nao {
                for col in 0..nao {
                    largest = largest.max((fresh[(row, col)] - old[(row, col)]).norm());
                }
            }
        }
        phase = new_phase;
        largest_change = largest;
        if largest < ff.tol {
            converged = true;
            break;
        }
        for (target, fresh) in k_terms.iter_mut().zip(&next) {
            for row in 0..nao {
                for col in 0..nao {
                    let mixed =
                        (*target)[(row, col)] * (1.0 - ff.mixing) + fresh[(row, col)] * ff.mixing;
                    (*target)[(row, col)] = mixed;
                }
            }
        }
    }

    if !converged {
        return Err(Pm3Error::ScfNotConverged {
            iterations,
            error: largest_change,
        });
    }

    let volume = cell.measure();
    let mut electronic = Vec3::zero();
    for axis in 0..3 {
        // The same sign `pbc::berry` derives, with the occupancy carried explicitly.
        electronic += cell.vector(axis) * (occupancy * phase[axis] / volume);
    }
    let mut ionic = Vec3::zero();
    for atom in &molecule.atoms {
        ionic += atom.position * (params.element(atom.z)?.core_charge / volume);
    }
    let polarization = electronic + ionic;

    Ok(FiniteFieldResult {
        enthalpy_ev: scf.total_ev - volume * field.dot(polarization),
        scf,
        field,
        phase,
        electronic_polarization: electronic,
        ionic_polarization: ionic,
        polarization,
        iterations,
        converged,
        resolved,
    })
}
