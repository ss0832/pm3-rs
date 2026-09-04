// SPDX-License-Identifier: GPL-3.0-or-later

//! Periodic cell: lattice vectors, dimensionality, fractional/Cartesian maps, the reciprocal
//! lattice of the **periodic subspace**, lattice-translation enumeration, and strain.
//!
//! # Dimensionality
//!
//! `pbc[i]` marks lattice vector `i` as periodic, so 1D (chains, nanotubes), 2D (slabs,
//! monolayers) and 3D (crystals) are all first-class. Everything that is summed over images —
//! the neighbour list, the screened NDDO short-range terms, the classical D3/H4/X corrections —
//! only ever sees [`Cell::translations_within`], which enumerates the periodic subspace. The
//! only genuinely dimension-dependent piece of physics is the reciprocal-space part of the
//! Ewald sum, and it consumes [`Cell::measure`] and [`Cell::reciprocal_basis`], both of which
//! are defined per dimensionality here:
//!
//! | `n_periodic` | [`Cell::measure`]      | [`Cell::reciprocal_basis`]                 |
//! |--------------|------------------------|--------------------------------------------|
//! | 3            | volume `|det h|`       | `b_i` with `b_i·a_j = 2π δ_ij`              |
//! | 2            | area `|a_i × a_j|`     | in-plane `b_i` with `b_i·a_j = 2π δ_ij`     |
//! | 1            | length `|a_i|`         | `b = 2π a/|a|²`                             |
//! | 0            | `1`                    | none (the molecular path)                   |
//!
//! Non-periodic lattice vectors are never summed over and never enter `measure`, so a slab may
//! carry whatever vacuum vector the caller likes (or none at all) without changing any result.
//!
//! Lengths are **Bohr**, matching the rest of the crate.

use crate::error::{Pm3Error, Result};
use crate::math::{Mat3, Vec3};

/// Lattice vectors below this length are treated as absent.
const NULL_VECTOR: f64 = 1.0e-10;

/// A periodic cell. Lattice vectors are the **columns** of `h`, so `r_cart = h · r_frac`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell {
    /// Lattice vectors `a1, a2, a3` as columns (Bohr), exactly as supplied by the caller.
    pub h: Mat3,
    /// Which lattice vectors are periodic.
    pub pbc: [bool; 3],
}

impl Cell {
    /// Build a cell from three lattice vectors (Bohr) and per-direction periodicity flags.
    ///
    /// Every **periodic** vector must be non-degenerate, and the periodic vectors must be
    /// linearly independent; non-periodic vectors are unconstrained (they may be zero).
    pub fn new(a1: Vec3, a2: Vec3, a3: Vec3, pbc: [bool; 3]) -> Result<Self> {
        let cell = Self {
            h: Mat3::from_columns(a1, a2, a3),
            pbc,
        };
        cell.validate()?;
        Ok(cell)
    }

    /// Build from lattice vectors given as **rows** — the ASE / extended-XYZ convention.
    pub fn from_rows(rows: [[f64; 3]; 3], pbc: [bool; 3]) -> Result<Self> {
        Self::new(
            Vec3::new(rows[0][0], rows[0][1], rows[0][2]),
            Vec3::new(rows[1][0], rows[1][1], rows[1][2]),
            Vec3::new(rows[2][0], rows[2][1], rows[2][2]),
            pbc,
        )
    }

    /// A fully periodic orthorhombic cell with the given edge lengths (Bohr).
    pub fn orthorhombic(a: f64, b: f64, c: f64) -> Result<Self> {
        Self::new(
            Vec3::new(a, 0.0, 0.0),
            Vec3::new(0.0, b, 0.0),
            Vec3::new(0.0, 0.0, c),
            [true; 3],
        )
    }

    /// A fully periodic cubic cell of edge `a` (Bohr).
    pub fn cubic(a: f64) -> Result<Self> {
        Self::orthorhombic(a, a, a)
    }

    /// No periodic direction at all: an isolated system, treated as the zero-dimensional member
    /// of the same family.
    ///
    /// The lattice vectors are the unit cube and are never read — with no periodic direction
    /// there are no images, no reciprocal space, and no volume to normalize by. What the cell
    /// carries here is the *absence* of periodicity, which is what lets one set of machinery
    /// serve a molecule and a crystal: see [`crate::pbc::ewald::direct_0d`].
    pub fn isolated() -> Self {
        Self::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            [false; 3],
        )
        .expect("the unit cube is a valid lattice")
    }

    /// Lattice vector `i` (Bohr).
    #[inline]
    pub fn vector(&self, i: usize) -> Vec3 {
        self.h.col[i]
    }

    /// Lattice vectors as rows — the ASE / extended-XYZ layout.
    pub fn to_rows(&self) -> [[f64; 3]; 3] {
        [
            self.h.col[0].to_array(),
            self.h.col[1].to_array(),
            self.h.col[2].to_array(),
        ]
    }

    /// Number of periodic directions (0–3).
    #[inline]
    pub fn n_periodic(&self) -> usize {
        self.pbc.iter().filter(|p| **p).count()
    }

    /// Indices of the periodic lattice vectors, ascending.
    pub fn periodic_indices(&self) -> Vec<usize> {
        (0..3).filter(|&i| self.pbc[i]).collect()
    }

    /// Reject cells whose periodic vectors are degenerate or linearly dependent.
    pub fn validate(&self) -> Result<()> {
        let periodic = self.periodic_indices();
        for &i in &periodic {
            if self.h.col[i].norm() <= NULL_VECTOR {
                return Err(Pm3Error::InvalidInput(format!(
                    "lattice vector {i} is flagged periodic but is zero-length"
                )));
            }
        }
        let degenerate = match periodic.as_slice() {
            [] | [_] => false,
            [i, j] => {
                let cross = self.h.col[*i].cross(self.h.col[*j]).norm();
                cross <= NULL_VECTOR * self.h.col[*i].norm() * self.h.col[*j].norm()
            }
            _ => {
                let v = self.h.col[0].dot(self.h.col[1].cross(self.h.col[2])).abs();
                let scale = self.h.col[0].norm() * self.h.col[1].norm() * self.h.col[2].norm();
                v <= NULL_VECTOR * scale.max(1.0)
            }
        };
        if degenerate {
            return Err(Pm3Error::InvalidInput(
                "the periodic lattice vectors are linearly dependent (degenerate cell)".to_string(),
            ));
        }
        Ok(())
    }

    /// Measure of the periodic subspace: volume (3D, Bohr³), area (2D, Bohr²), length
    /// (1D, Bohr), or `1` for a non-periodic cell.
    ///
    /// This is the normalizing factor of every reciprocal-space Ewald term, which is why it has
    /// to be dimension-aware: a slab's Ewald sum divides by the in-plane **area**, never by a
    /// volume that would depend on how much vacuum the caller happened to include.
    pub fn measure(&self) -> f64 {
        let periodic = self.periodic_indices();
        match periodic.as_slice() {
            [] => 1.0,
            [i] => self.h.col[*i].norm(),
            [i, j] => self.h.col[*i].cross(self.h.col[*j]).norm(),
            _ => self.h.col[0].dot(self.h.col[1].cross(self.h.col[2])).abs(),
        }
    }

    /// Cell volume `|det h|` (Bohr³). Meaningful only when all three vectors are supplied;
    /// for reduced dimensionality prefer [`Cell::measure`].
    pub fn volume(&self) -> f64 {
        self.h.col[0].dot(self.h.col[1].cross(self.h.col[2])).abs()
    }

    /// Reciprocal basis of the **periodic subspace**: one vector per periodic direction,
    /// paired with that direction's index, satisfying `b_i · a_j = 2π δ_ij`.
    ///
    /// For 2D the returned vectors lie in the periodic plane; for 1D the single vector is
    /// parallel to the chain axis. In both cases they are independent of the non-periodic
    /// lattice vectors — including a slab's vacuum thickness.
    pub fn reciprocal_basis(&self) -> Vec<(usize, Vec3)> {
        let periodic = self.periodic_indices();
        let two_pi = std::f64::consts::TAU;
        match periodic.as_slice() {
            [] => Vec::new(),
            [i] => {
                let a = self.h.col[*i];
                vec![(*i, a * (two_pi / a.norm2()))]
            }
            [i, j] => {
                // In-plane reciprocal vectors: b_i = 2π (a_j × n)/A, b_j = 2π (n × a_i)/A,
                // with n the unit normal and A the cell area. b_i·a_i = 2π, b_i·a_j = 0.
                let (ai, aj) = (self.h.col[*i], self.h.col[*j]);
                let cross = ai.cross(aj);
                let area = cross.norm();
                let n = cross / area;
                vec![
                    (*i, aj.cross(n) * (two_pi / area)),
                    (*j, n.cross(ai) * (two_pi / area)),
                ]
            }
            _ => {
                let (a1, a2, a3) = (self.h.col[0], self.h.col[1], self.h.col[2]);
                let v = a1.dot(a2.cross(a3));
                vec![
                    (0, a2.cross(a3) * (two_pi / v)),
                    (1, a3.cross(a1) * (two_pi / v)),
                    (2, a1.cross(a2) * (two_pi / v)),
                ]
            }
        }
    }

    /// Perpendicular width of the periodic sublattice along each periodic direction (Bohr):
    /// the spacing between adjacent lattice planes (3D), lines (2D), or points (1D).
    ///
    /// This is what bounds the translation search: a translation with index `n_i` is at least
    /// `|n_i| · width_i` away, so `n_i` never has to exceed `ceil(cutoff / width_i)`.
    pub fn periodic_widths(&self) -> Vec<(usize, f64)> {
        let periodic = self.periodic_indices();
        let measure = self.measure();
        match periodic.as_slice() {
            [] => Vec::new(),
            [i] => vec![(*i, self.h.col[*i].norm())],
            [i, j] => vec![
                (*i, measure / self.h.col[*j].norm()),
                (*j, measure / self.h.col[*i].norm()),
            ],
            _ => {
                let (a1, a2, a3) = (self.h.col[0], self.h.col[1], self.h.col[2]);
                vec![
                    (0, measure / a2.cross(a3).norm()),
                    (1, measure / a3.cross(a1).norm()),
                    (2, measure / a1.cross(a2).norm()),
                ]
            }
        }
    }

    /// Every lattice translation `T = Σ n_i a_i` (periodic directions only) with `|T| ≤ cutoff`,
    /// as `(n, T)` pairs. `T = 0` is included and always comes first.
    ///
    /// The enumeration box comes from [`Cell::periodic_widths`], so it is tight for skewed
    /// cells too; candidates outside the sphere are then filtered by length.
    pub fn translations_within(&self, cutoff: f64) -> Vec<([i32; 3], Vec3)> {
        let mut out = vec![([0, 0, 0], Vec3::zero())];
        if cutoff <= 0.0 || self.n_periodic() == 0 {
            return out;
        }
        let mut limits = [0i32; 3];
        for (i, width) in self.periodic_widths() {
            limits[i] = if width > NULL_VECTOR {
                (cutoff / width).ceil() as i32
            } else {
                0
            };
        }
        let cutoff2 = cutoff * cutoff;
        for n0 in -limits[0]..=limits[0] {
            for n1 in -limits[1]..=limits[1] {
                for n2 in -limits[2]..=limits[2] {
                    if n0 == 0 && n1 == 0 && n2 == 0 {
                        continue;
                    }
                    let t = self.h.col[0] * n0 as f64
                        + self.h.col[1] * n1 as f64
                        + self.h.col[2] * n2 as f64;
                    if t.norm2() <= cutoff2 {
                        out.push(([n0, n1, n2], t));
                    }
                }
            }
        }
        out
    }

    /// Fractional coordinates of a Cartesian position (`h⁻¹ r`). Errors on a singular `h`,
    /// which can only happen when a non-periodic lattice vector was left zero.
    pub fn to_fractional(&self, r: Vec3) -> Result<Vec3> {
        let inv = invert(&self.h).ok_or_else(|| {
            Pm3Error::InvalidInput(
                "fractional coordinates need three non-degenerate lattice vectors".to_string(),
            )
        })?;
        Ok(inv.mul_vec(r))
    }

    /// Cartesian position of a fractional coordinate (`h f`).
    #[inline]
    pub fn to_cartesian(&self, f: Vec3) -> Vec3 {
        self.h.mul_vec(f)
    }

    /// Wrap a Cartesian position into the cell along the **periodic** directions only.
    pub fn wrap(&self, r: Vec3) -> Result<Vec3> {
        if self.n_periodic() == 0 {
            return Ok(r);
        }
        let mut f = self.to_fractional(r)?;
        for i in 0..3 {
            if self.pbc[i] {
                let v = f.get(i);
                set_component(&mut f, i, v - v.floor());
            }
        }
        Ok(self.to_cartesian(f))
    }

    /// The shortest periodic image of a displacement, found by direct search over
    /// [`Cell::translations_within`] rather than by the naive fractional rounding, which is
    /// wrong for skewed cells.
    pub fn minimum_image(&self, d: Vec3) -> Vec3 {
        if self.n_periodic() == 0 {
            return d;
        }
        // Any image is within `|d| + max periodic vector length` of the origin, so a search
        // radius of twice the longest periodic vector plus `|d|` certainly contains the best.
        let longest = self
            .periodic_indices()
            .iter()
            .map(|&i| self.h.col[i].norm())
            .fold(0.0_f64, f64::max);
        let mut best = d;
        let mut best2 = d.norm2();
        for (_, t) in self.translations_within(d.norm() + 2.0 * longest) {
            let candidate = d + t;
            let n2 = candidate.norm2();
            if n2 < best2 {
                best2 = n2;
                best = candidate;
            }
        }
        best
    }

    /// The cell under an infinitesimal or finite strain: `a_i ← (1 + ε) a_i`.
    ///
    /// `eps` is the strain tensor with **columns** `ε·e_x, ε·e_y, ε·e_z`, i.e. the same column
    /// convention as `h`. This is the map whose derivative the analytic stress reports.
    pub fn strained(&self, eps: &Mat3) -> Self {
        let deform = |v: Vec3| v + eps.mul_vec(v);
        Self {
            h: Mat3::from_columns(
                deform(self.h.col[0]),
                deform(self.h.col[1]),
                deform(self.h.col[2]),
            ),
            pbc: self.pbc,
        }
    }
}

#[inline]
fn set_component(v: &mut Vec3, i: usize, value: f64) {
    match i {
        0 => v.x = value,
        1 => v.y = value,
        _ => v.z = value,
    }
}

/// Inverse of a column-major 3×3 matrix, or `None` when it is singular.
fn invert(m: &Mat3) -> Option<Mat3> {
    let (a, b, c) = (m.col[0], m.col[1], m.col[2]);
    let det = a.dot(b.cross(c));
    let scale = a.norm() * b.norm() * c.norm();
    if det.abs() <= NULL_VECTOR * scale.max(1.0) {
        return None;
    }
    // Rows of the inverse are the reciprocal (cofactor) vectors divided by the determinant;
    // expressed column-major, the inverse's columns are those rows transposed.
    let r0 = b.cross(c) / det;
    let r1 = c.cross(a) / det;
    let r2 = a.cross(b) / det;
    Some(Mat3::from_columns(
        Vec3::new(r0.x, r1.x, r2.x),
        Vec3::new(r0.y, r1.y, r2.y),
        Vec3::new(r0.z, r1.z, r2.z),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn measure_is_dimension_aware() {
        // A slab: 3 × 4 in-plane, with 100 Bohr of vacuum along c. The measure must be the
        // in-plane area (12), never the 1200 Bohr³ that includes the arbitrary vacuum.
        let slab = Cell::new(
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(0.0, 0.0, 100.0),
            [true, true, false],
        )
        .unwrap();
        assert!(approx(slab.measure(), 12.0, 1e-12));
        assert!(approx(slab.volume(), 1200.0, 1e-12));

        let chain = Cell::new(
            Vec3::new(2.5, 0.0, 0.0),
            Vec3::new(0.0, 50.0, 0.0),
            Vec3::new(0.0, 0.0, 50.0),
            [true, false, false],
        )
        .unwrap();
        assert!(approx(chain.measure(), 2.5, 1e-12));

        let crystal = Cell::cubic(5.0).unwrap();
        assert!(approx(crystal.measure(), 125.0, 1e-12));
    }

    /// The vacuum thickness of a slab (or the padding of a chain) must not touch the
    /// reciprocal basis — that independence is what makes the reduced-dimension Ewald sums
    /// vacuum-free.
    #[test]
    fn reciprocal_basis_ignores_non_periodic_padding() {
        let thin = Cell::new(
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(1.0, 4.0, 0.0),
            Vec3::new(0.0, 0.0, 20.0),
            [true, true, false],
        )
        .unwrap();
        let thick = Cell::new(
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(1.0, 4.0, 0.0),
            Vec3::new(0.0, 0.0, 500.0),
            [true, true, false],
        )
        .unwrap();
        let (rt, rk) = (thin.reciprocal_basis(), thick.reciprocal_basis());
        assert_eq!(rt.len(), 2);
        for ((it, bt), (ik, bk)) in rt.iter().zip(&rk) {
            assert_eq!(it, ik);
            assert!((*bt - *bk).norm() < 1e-12);
        }
        assert!(approx(thin.measure(), thick.measure(), 1e-12));
    }

    #[test]
    fn reciprocal_basis_is_biorthogonal() {
        let cells = [
            Cell::new(
                Vec3::new(4.0, 0.3, -0.2),
                Vec3::new(-0.4, 5.0, 0.6),
                Vec3::new(0.1, -0.7, 6.0),
                [true, true, true],
            )
            .unwrap(),
            Cell::new(
                Vec3::new(3.0, 0.0, 0.0),
                Vec3::new(1.5, 2.6, 0.0),
                Vec3::new(0.0, 0.0, 40.0),
                [true, true, false],
            )
            .unwrap(),
            Cell::new(
                Vec3::new(2.4, 0.0, 0.0),
                Vec3::new(0.0, 30.0, 0.0),
                Vec3::new(0.0, 0.0, 30.0),
                [true, false, false],
            )
            .unwrap(),
        ];
        for cell in cells {
            for (i, b) in cell.reciprocal_basis() {
                for j in cell.periodic_indices() {
                    let expected = if i == j { std::f64::consts::TAU } else { 0.0 };
                    assert!(
                        approx(b.dot(cell.vector(j)), expected, 1e-10),
                        "b_{i}·a_{j} = {}",
                        b.dot(cell.vector(j))
                    );
                }
            }
        }
    }

    #[test]
    fn translations_cover_exactly_the_sphere() {
        // A deliberately skewed 2D cell: a naive ±1 box would miss images inside the cutoff.
        let cell = Cell::new(
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(2.7, 1.3, 0.0),
            Vec3::new(0.0, 0.0, 25.0),
            [true, true, false],
        )
        .unwrap();
        let cutoff = 12.0;
        let found = cell.translations_within(cutoff);
        assert_eq!(found[0].0, [0, 0, 0]);
        for (n, t) in &found {
            assert!(t.norm() <= cutoff + 1e-12);
            assert_eq!(n[2], 0, "a non-periodic direction was translated");
        }
        // Brute force over a box far larger than the enumeration bound.
        let mut expected = 0usize;
        for n0 in -40..=40 {
            for n1 in -40..=40 {
                let t = cell.vector(0) * n0 as f64 + cell.vector(1) * n1 as f64;
                if t.norm() <= cutoff + 1e-12 {
                    expected += 1;
                }
            }
        }
        assert_eq!(
            found.len(),
            expected,
            "translation enumeration is not tight"
        );
    }

    #[test]
    fn fractional_roundtrip_and_wrap() {
        let cell = Cell::new(
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(1.0, 5.0, 0.0),
            Vec3::new(0.2, -0.3, 6.0),
            [true, true, false],
        )
        .unwrap();
        let r = Vec3::new(9.3, -7.1, 2.4);
        let f = cell.to_fractional(r).unwrap();
        assert!((cell.to_cartesian(f) - r).norm() < 1e-12);

        let wrapped = cell.wrap(r).unwrap();
        let wf = cell.to_fractional(wrapped).unwrap();
        assert!((0.0..1.0).contains(&wf.x) && (0.0..1.0).contains(&wf.y));
        // The non-periodic direction is left exactly alone.
        assert!(approx(wf.z, f.z, 1e-12));
    }

    #[test]
    fn minimum_image_beats_fractional_rounding_on_a_skewed_cell() {
        // Highly skewed: rounding fractional coordinates picks the wrong image here.
        let cell = Cell::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.95, 0.2, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            [true, true, true],
        )
        .unwrap();
        let d = Vec3::new(0.6, 0.15, 0.4);
        let mi = cell.minimum_image(d);
        let mut brute = d;
        for n0 in -6..=6 {
            for n1 in -6..=6 {
                for n2 in -6..=6 {
                    let t = cell.vector(0) * n0 as f64
                        + cell.vector(1) * n1 as f64
                        + cell.vector(2) * n2 as f64;
                    if (d + t).norm2() < brute.norm2() {
                        brute = d + t;
                    }
                }
            }
        }
        assert!(
            approx(mi.norm(), brute.norm(), 1e-12),
            "{mi:?} vs {brute:?}"
        );
    }

    #[test]
    fn strain_scales_the_measure_by_the_jacobian() {
        let cell = Cell::cubic(4.0).unwrap();
        let e = 1.0e-3;
        // Isotropic strain: volume scales as (1+e)³.
        let eps = Mat3::from_columns(
            Vec3::new(e, 0.0, 0.0),
            Vec3::new(0.0, e, 0.0),
            Vec3::new(0.0, 0.0, e),
        );
        let strained = cell.strained(&eps);
        assert!(approx(
            strained.measure(),
            cell.measure() * (1.0 + e).powi(3),
            1e-9
        ));
    }

    #[test]
    fn degenerate_periodic_cells_are_rejected() {
        // Zero-length periodic vector.
        assert!(Cell::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(0.0, 0.0, 4.0),
            [true, true, true],
        )
        .is_err());
        // Two parallel periodic vectors.
        assert!(Cell::new(
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 4.0),
            [true, true, true],
        )
        .is_err());
        // The same degeneracy is fine when those directions are not periodic.
        assert!(Cell::new(
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 4.0),
            [true, false, true],
        )
        .is_ok());
    }
}
