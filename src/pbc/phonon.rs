// SPDX-License-Identifier: GPL-3.0-or-later

//! Real-space force constants `Φ(T)` from a supercell, and the dynamical matrix they interpolate.
//!
//! # The other route to `D(q)`
//!
//! [`crate::pbc::dfpt`] computes the response at one wavevector directly, from a primitive cell,
//! for any `q`. This module takes the opposite road: displace atoms in a **supercell**, read
//! `Φ(0κ, Tκ')` off its Γ-point Hessian, and Fourier transform.
//!
//! ```text
//! D(q) = Σ_T Φ(0, T) e^{iq·T}
//! ```
//!
//! Both are worth having, and for opposite reasons:
//!
//! | | reach in `q` | cost | long range |
//! |---|---|---|---|
//! | DFPT | any `q`, one at a time | a primitive cell per `q` | carried by the phased Ewald sum |
//! | supercell | only the `q` the supercell can represent | one Hessian, then every `q` free | **truncated away** |
//!
//! So a dispersion over hundreds of `q` costs one supercell Hessian here and hundreds of response
//! solves there — and the supercell's `q` are restricted to the commensurate set, with everything
//! between them an interpolation rather than a calculation.
//!
//! # What the truncation costs, and what to do about it
//!
//! `Φ(T)` is read off a finite supercell, so it is zero beyond that supercell by construction.
//! In a polar crystal the true `Φ(T)` has a `1/T³` dipole tail, and cutting it off removes the
//! macroscopic field — which is exactly the LO–TO splitting. That is why
//! [`crate::pbc::lo_to::non_analytic_term`] exists and why adding it to a matrix from *this*
//! module is right, while adding it to a finite-`q` DFPT matrix would count the same physics
//! twice.
//!
//! # Determinism
//!
//! The blocks are held in a **sorted vector**, not a hash map, and every sum over them runs in
//! that order. A Fourier sum whose accumulation order varies between runs gives answers that vary
//! in their last digits, and in a near-degenerate mode that is not a last-digit effect: the same
//! calculation repeated can move a frequency by hundreds of wavenumbers. Sorting once at
//! construction costs nothing and removes the possibility.

use crate::cmatrix::CMatrix;
use crate::error::{Pm3Error, Result};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::params::Pm3Parameters;
use crate::pbc::gamma::PeriodicOptions;
use crate::scf::Pm3Options;
use crate::system::{Atom, Molecule};

/// Real-space force constants `Φ(0κα, Tκ'β)` in eV/Bohr², one block per translation.
#[derive(Clone, Debug)]
pub struct ForceConstants {
    /// `(T, Φ(0, T))` sorted by `T`, so every sum over them is order-stable. Row `3a + i` is
    /// atom `a` of the home cell, column `3b + j` atom `b` of cell `T`.
    blocks: Vec<([i32; 3], Matrix)>,
    /// Atomic masses (amu) of the **primitive** cell, in its atom order.
    masses: Vec<f64>,
    /// The primitive cell the translations are measured in.
    cell: crate::cell::Cell,
    /// How many primitive cells the supercell held, per axis.
    supercell: [usize; 3],
    /// Atoms in the primitive cell.
    nat: usize,
}

/// The primitive cells of an `n₁ × n₂ × n₃` supercell, in a fixed order.
///
/// The order is the one [`build_supercell`] lays atoms out in, and both read it from here so they
/// cannot disagree about which block belongs to which translation.
fn supercell_cells(supercell: [usize; 3]) -> Vec<[i32; 3]> {
    let mut out = Vec::with_capacity(supercell[0] * supercell[1] * supercell[2]);
    for n0 in 0..supercell[0] {
        for n1 in 0..supercell[1] {
            for n2 in 0..supercell[2] {
                out.push([n0 as i32, n1 as i32, n2 as i32]);
            }
        }
    }
    out
}

/// Replicate a periodic cell into an `n₁ × n₂ × n₃` supercell.
///
/// Atoms come out in `cell · nat + primitive` order, which is what
/// [`ForceConstants::from_supercell`] reads them back in. A non-periodic axis must be replicated
/// once: there is no translation along it to replicate *by*, and asking for more would silently
/// stack copies on top of each other.
pub fn build_supercell(primitive: &Molecule, supercell: [usize; 3]) -> Result<Molecule> {
    let cell = primitive.cell.ok_or_else(|| {
        Pm3Error::InvalidInput("a supercell needs a periodic cell to replicate".to_string())
    })?;
    for (axis, count) in supercell.iter().enumerate() {
        if *count == 0 {
            return Err(Pm3Error::InvalidInput(format!(
                "supercell division along axis {axis} is zero; use 1 for no replication"
            )));
        }
        if !cell.pbc[axis] && *count != 1 {
            return Err(Pm3Error::InvalidInput(format!(
                "axis {axis} is not periodic, so there is no lattice vector to replicate along: \
                 a division of {count} there would stack {count} copies on the same coordinates. \
                 Use 1."
            )));
        }
    }

    let mut atoms = Vec::with_capacity(primitive.atoms.len() * supercell.iter().product::<usize>());
    for offset in supercell_cells(supercell) {
        let shift = cell.vector(0) * offset[0] as f64
            + cell.vector(1) * offset[1] as f64
            + cell.vector(2) * offset[2] as f64;
        for atom in &primitive.atoms {
            atoms.push(Atom {
                z: atom.z,
                position: atom.position + shift,
            });
        }
    }

    let rows = [
        (cell.vector(0) * supercell[0] as f64).to_array(),
        (cell.vector(1) * supercell[1] as f64).to_array(),
        (cell.vector(2) * supercell[2] as f64).to_array(),
    ];
    let big_cell = crate::cell::Cell::from_rows(rows, cell.pbc)?;

    let copies = supercell.iter().product::<usize>() as f64;
    let mut out = Molecule {
        atoms,
        // Charge and multiplicity are per cell, so a supercell holds that many times as much.
        // A multiplicity of `2S+1` scales as `S` does: `2·(copies·S)+1`.
        charge: primitive.charge * copies,
        multiplicity: ((primitive.multiplicity as f64 - 1.0) * copies) as usize + 1,
        cell: Some(big_cell),
    };
    out.cell = Some(big_cell);
    Ok(out)
}

impl ForceConstants {
    /// Read `Φ(0, T)` off an `n₁ × n₂ × n₃` supercell's Γ-point Hessian.
    ///
    /// Only the home cell's rows are needed: translational invariance makes every other cell's a
    /// copy, so `Φ(Lκ, L'κ') = Φ(0κ, (L'−L)κ')` and taking `L = 0` loses nothing.
    pub fn from_supercell(
        primitive: &Molecule,
        params: &Pm3Parameters,
        options: &Pm3Options,
        periodic: &PeriodicOptions,
        supercell: [usize; 3],
    ) -> Result<Self> {
        let cell = primitive.cell.ok_or_else(|| {
            Pm3Error::InvalidInput("force constants need a periodic cell".to_string())
        })?;
        let big = build_supercell(primitive, supercell)?;
        let hessian = crate::pbc::hessian::periodic_hessian(&big, params, options, periodic)?;

        let nat = primitive.atoms.len();
        let cells = supercell_cells(supercell);
        let mut blocks = Vec::with_capacity(cells.len());
        for (index, offset) in cells.iter().enumerate() {
            let mut block = Matrix::zeros(3 * nat, 3 * nat);
            for a in 0..nat {
                for b in 0..nat {
                    // Home cell is index 0 by construction of `supercell_cells`.
                    let row = 3 * a;
                    let column = 3 * (index * nat + b);
                    for i in 0..3 {
                        for j in 0..3 {
                            block[(3 * a + i, 3 * b + j)] = hessian[(row + i, column + j)];
                        }
                    }
                }
            }
            blocks.push((*offset, block));
        }
        blocks.sort_by_key(|(t, _)| *t);

        let masses = primitive
            .atoms
            .iter()
            .map(|atom| params.element(atom.z).map(|e| e.mass))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            blocks,
            masses,
            cell,
            supercell,
            nat,
        })
    }

    /// `D(q) = Σ_T Φ(0, T) e^{iq·T}`, in eV/Bohr², **not** mass weighted.
    ///
    /// The phase uses fractional coordinates — `q·T = 2π(f₁n₁ + f₂n₂ + f₃n₃)` — so the Cartesian
    /// reciprocal basis never enters and there is one fewer place for a convention to slip.
    pub fn dynamical_matrix(&self, q_frac: [f64; 3]) -> crate::pbc::dfpt::DynamicalMatrix {
        let ndof = 3 * self.nat;
        let mut matrix = CMatrix::zeros(ndof, ndof);
        // In sorted order. See the module note on determinism.
        for (t, block) in &self.blocks {
            let angle = std::f64::consts::TAU
                * (q_frac[0] * t[0] as f64 + q_frac[1] * t[1] as f64 + q_frac[2] * t[2] as f64);
            let (sin, cos) = angle.sin_cos();
            for i in 0..ndof {
                for j in 0..ndof {
                    let value = block[(i, j)];
                    matrix[(i, j)] += faer::c64::new(value * cos, value * sin);
                }
            }
        }
        // A truncated `Φ(T)` is not exactly symmetric under `T → −T`, so the transform is not
        // exactly Hermitian. The defect is reported the way the DFPT path reports its own.
        let hermitian_defect = matrix.hermitian_defect();
        let mut hermitized = CMatrix::zeros(ndof, ndof);
        for i in 0..ndof {
            for j in 0..ndof {
                let (a, b) = (matrix[(i, j)], matrix[(j, i)]);
                hermitized[(i, j)] = faer::c64::new(0.5 * (a.re + b.re), 0.5 * (a.im - b.im));
            }
        }
        crate::pbc::dfpt::DynamicalMatrix {
            q_frac,
            matrix: hermitized,
            masses: self.masses.clone(),
            hermitian_defect,
        }
    }

    /// Frequencies at `q_frac` in cm⁻¹, ascending, imaginary modes as negatives.
    pub fn frequencies(&self, q_frac: [f64; 3]) -> Result<Vec<f64>> {
        crate::pbc::dfpt::frequencies_of(&self.dynamical_matrix(q_frac))
    }

    /// Frequencies along a path of fractional wavevectors — a phonon band structure.
    ///
    /// Free after the one Hessian: each point is a Fourier sum and a `3N × 3N` diagonalization,
    /// which is what makes this route worth having next to DFPT.
    pub fn band_structure(&self, path: &[[f64; 3]]) -> Result<Vec<Vec<f64>>> {
        path.iter().map(|q| self.frequencies(*q)).collect()
    }

    /// The wavevectors this supercell can represent exactly, rather than interpolate.
    ///
    /// `q = (m₁/n₁, m₂/n₂, m₃/n₃)`. At these the Fourier sum is a finite exact transform of the
    /// supercell's own Γ-point Hessian; everywhere else it is an interpolation, and how good an
    /// interpolation depends on how far `Φ(T)` had decayed by the supercell boundary.
    pub fn commensurate_q(&self) -> Vec<[f64; 3]> {
        let mut out = Vec::new();
        for m0 in 0..self.supercell[0] {
            for m1 in 0..self.supercell[1] {
                for m2 in 0..self.supercell[2] {
                    out.push([
                        m0 as f64 / self.supercell[0] as f64,
                        m1 as f64 / self.supercell[1] as f64,
                        m2 as f64 / self.supercell[2] as f64,
                    ]);
                }
            }
        }
        out
    }

    /// `max |Σ_b Σ_T Φ_{aα,bβ}(T)|` — the acoustic sum rule's residual.
    ///
    /// Translating the crystal costs no energy, so this is zero for exact force constants. On a
    /// truncated set it is a measure of what the truncation threw away, and it is the number to
    /// look at before trusting an acoustic branch near `Γ`, where the frequencies are square
    /// roots of nearly cancelling quantities.
    pub fn acoustic_sum_rule_residual(&self) -> f64 {
        let ndof = 3 * self.nat;
        let mut worst = 0.0_f64;
        for row in 0..ndof {
            for beta in 0..3 {
                let mut total = 0.0;
                for (_, block) in &self.blocks {
                    for b in 0..self.nat {
                        total += block[(row, 3 * b + beta)];
                    }
                }
                worst = worst.max(total.abs());
            }
        }
        worst
    }

    /// Impose the acoustic sum rule by subtracting the violation from the home-cell self term.
    ///
    /// Check [`Self::acoustic_sum_rule_residual`] first. If it is not already small the force
    /// constants are wrong rather than rounded, and flattening it hides that.
    /// The correction is **symmetrized before it is applied**. `Φ` is symmetric, and the raw
    /// violation `C_a[α][β] = Σ_T Σ_b Φ_T[3a+α, 3b+β]` need not be symmetric in `αβ`, so
    /// subtracting it as it stands leaves each on-site `3×3` block asymmetric. That would be
    /// undone anyway — the dynamical matrix is Hermitized on the way to the frequencies, and
    /// `symmetric_eigen` reads one triangle — so an asymmetric correction is half discarded by
    /// whichever step comes next, silently. Subtracting `(C + Cᵀ)/2` keeps the block symmetric
    /// and leaves behind only the antisymmetric part of a quantity that was rounding-sized to
    /// begin with.
    pub fn enforce_acoustic_sum_rule(&mut self) {
        let home = self
            .blocks
            .iter()
            .position(|(t, _)| *t == [0, 0, 0])
            .expect("the home cell is always present");
        for atom in 0..self.nat {
            let mut violation = [[0.0f64; 3]; 3];
            for (alpha, row_of) in violation.iter_mut().enumerate() {
                let row = 3 * atom + alpha;
                for (beta, slot) in row_of.iter_mut().enumerate() {
                    let mut total = 0.0;
                    for (_, block) in &self.blocks {
                        for b in 0..self.nat {
                            total += block[(row, 3 * b + beta)];
                        }
                    }
                    *slot = total;
                }
            }
            // All of it onto the diagonal self-term of the row's own atom, which is where a
            // translation-invariant set would have put it — symmetrized, per the note above.
            for (alpha, row_of) in violation.iter().enumerate() {
                for (beta, value) in row_of.iter().enumerate() {
                    let share = 0.5 * (value + violation[beta][alpha]);
                    self.blocks[home].1[(3 * atom + alpha, 3 * atom + beta)] -= share;
                }
            }
        }
    }

    /// The primitive cell these were measured in.
    pub fn cell(&self) -> crate::cell::Cell {
        self.cell
    }

    /// Atomic masses (amu) in the order the matrices index them.
    pub fn masses(&self) -> &[f64] {
        &self.masses
    }

    /// How many primitive cells the supercell held, per axis.
    pub fn supercell(&self) -> [usize; 3] {
        self.supercell
    }
}

/// A straight-line path through fractional reciprocal space, `points` per segment.
///
/// The corners are included; each segment contributes `points` samples and the final corner is
/// appended once, so a two-corner path of `n` points has `n + 1` entries.
pub fn q_path(corners: &[[f64; 3]], points: usize) -> Vec<[f64; 3]> {
    if corners.len() < 2 || points == 0 {
        return corners.to_vec();
    }
    let mut out = Vec::with_capacity((corners.len() - 1) * points + 1);
    for pair in corners.windows(2) {
        let (from, to) = (pair[0], pair[1]);
        for step in 0..points {
            let fraction = step as f64 / points as f64;
            out.push([
                from[0] + (to[0] - from[0]) * fraction,
                from[1] + (to[1] - from[1]) * fraction,
                from[2] + (to[2] - from[2]) * fraction,
            ]);
        }
    }
    out.push(*corners.last().expect("checked non-empty"));
    out
}

/// The Cartesian wavevector a fractional one names, for callers pairing this with
/// [`crate::pbc::lo_to`], whose direction is Cartesian.
pub fn cartesian_q(cell: &crate::cell::Cell, q_frac: [f64; 3]) -> Vec3 {
    let mut q = Vec3::zero();
    for (index, b) in cell.reciprocal_basis() {
        q += b * q_frac[index];
    }
    q
}
