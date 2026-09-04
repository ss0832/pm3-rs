// SPDX-License-Identifier: GPL-3.0-or-later

//! Cutoff neighbour lists over lattice images, shared by every distance-based term:
//! the screened NDDO short-range corrections, the resonance and exchange blocks, the
//! core-core repulsion, the classical D3/H4/X corrections, and the real-space part of
//! the Ewald sum. A molecule is the `cell = None` special case and takes the same path
//! with a single `T = 0` translation.
//!
//! # Counting
//!
//! Getting the lattice sum's *counting* right is the single most error-prone part of a
//! periodic implementation — a term counted twice, or a self-image dropped, produces a
//! plausible-looking energy that is silently wrong. This module fixes the convention in
//! one place and exposes it through two views:
//!
//! * [`NeighborList::pairs_of`] — **every** ordered `(a, b, T)` within the cutoff,
//!   excluding only the self-interaction `a = b, T = 0`. Matrix assembly needs this:
//!   the `H(0,T)` block for `(a,b,T)` and the one for `(b,a,−T)` are different matrix
//!   elements and both have to be written.
//!
//! * [`NeighborList::unique`] — each distinct *interaction* exactly once, for sums of
//!   the form `E_cell = ½ Σ_{a,b,T}' f(|r_b + T − r_a|)`. The half-set rule is:
//!   take `T = 0` with `a < b`, and take all ordered `(a, b)` — self-images `a = b`
//!   included — for one translation out of each `±T` pair. Summing `f` over
//!   [`NeighborList::unique`] with weight 1 equals the primed double sum with weight ½,
//!   because relabelling `a ↔ b` maps the `+T` half onto the `−T` half.
//!
//! The `Γ`-point-versus-supercell folding tests are what actually prove this: a term
//! counted 1.5 or 2 times shows up immediately as an energy per cell that does not match
//! the supercell energy divided by the number of cells.

use crate::cell::Cell;
use crate::math::Vec3;
use crate::system::Molecule;

/// One ordered atom pair together with the lattice image the partner sits in.
#[derive(Clone, Copy, Debug)]
pub struct PairImage {
    /// Atom index in the reference cell.
    pub a: usize,
    /// Partner atom index (also a reference-cell index; `t` says which image).
    pub b: usize,
    /// Lattice translation of the partner's image, in units of the lattice vectors.
    pub t: [i32; 3],
    /// `r_b + T − r_a` in Bohr — the displacement every integral kernel is written in.
    pub dvec: Vec3,
    /// `|dvec|` in Bohr.
    pub r: f64,
}

impl PairImage {
    /// Whether this pair is the representative of its interaction under the half-set rule
    /// documented on [`NeighborList`]: `T = 0` keeps `a < b`; a non-zero `T` keeps the
    /// lexicographically positive member of the `±T` pair, for all ordered `(a, b)`.
    #[inline]
    pub fn is_unique_representative(&self) -> bool {
        match first_nonzero(&self.t) {
            None => self.a < self.b,
            Some(sign) => sign > 0,
        }
    }
}

/// Sign of the first non-zero component of a translation index, or `None` for `T = 0`.
#[inline]
fn first_nonzero(t: &[i32; 3]) -> Option<i32> {
    t.iter().copied().find(|v| *v != 0).map(i32::signum)
}

/// Neighbour list for one geometry and one cutoff.
#[derive(Clone, Debug)]
pub struct NeighborList {
    /// The cutoff the list was built with (Bohr).
    pub cutoff: f64,
    /// All ordered pairs, sorted by `a` so `pairs_of` is a slice.
    pairs: Vec<PairImage>,
    /// `offsets[a]..offsets[a + 1]` is atom `a`'s slice of `pairs`.
    offsets: Vec<usize>,
}

impl NeighborList {
    /// Build the list for `molecule` at `cutoff` Bohr, honouring `molecule.cell`
    /// (a molecule with no cell contributes the single `T = 0` translation).
    pub fn build(molecule: &Molecule, cutoff: f64) -> Self {
        let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
        Self::build_from_positions(&positions, molecule.cell.as_ref(), cutoff)
    }

    /// Build from raw positions (Bohr) and an optional cell.
    pub fn build_from_positions(positions: &[Vec3], cell: Option<&Cell>, cutoff: f64) -> Self {
        let n = positions.len();
        let translations = match cell {
            // A partner image can only reach atom `a` if the translation is within the
            // cutoff plus the spread of the cell's own atoms, so that is the search bound.
            Some(c) => c.translations_within(cutoff + bounding_span(positions)),
            None => vec![([0, 0, 0], Vec3::zero())],
        };

        let mut grid = Grid::new(positions, cutoff);
        let mut pairs: Vec<PairImage> = Vec::new();
        let cutoff2 = cutoff * cutoff;
        for (t, shift) in &translations {
            let zero_translation = *t == [0, 0, 0];
            for (b, &pos_b) in positions.iter().enumerate() {
                let image = pos_b + *shift;
                grid.for_each_near(image, |a| {
                    if zero_translation && a == b {
                        return; // the self-interaction is not a pair
                    }
                    let dvec = image - positions[a];
                    let r2 = dvec.norm2();
                    if r2 <= cutoff2 {
                        pairs.push(PairImage {
                            a,
                            b,
                            t: *t,
                            dvec,
                            r: r2.sqrt(),
                        });
                    }
                });
            }
        }

        pairs.sort_unstable_by_key(|p| p.a);
        let mut offsets = vec![0usize; n + 1];
        for pair in &pairs {
            offsets[pair.a + 1] += 1;
        }
        for a in 0..n {
            offsets[a + 1] += offsets[a];
        }
        Self {
            cutoff,
            pairs,
            offsets,
        }
    }

    /// Every ordered pair, in `a`-major order.
    #[inline]
    pub fn all(&self) -> &[PairImage] {
        &self.pairs
    }

    /// Atom `a`'s neighbours (all images within the cutoff).
    #[inline]
    pub fn pairs_of(&self, a: usize) -> &[PairImage] {
        &self.pairs[self.offsets[a]..self.offsets[a + 1]]
    }

    /// Each distinct interaction exactly once — the view a pairwise energy sums over.
    pub fn unique(&self) -> impl Iterator<Item = &PairImage> {
        self.pairs.iter().filter(|p| p.is_unique_representative())
    }

    /// Number of ordered pairs.
    #[inline]
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

/// Diagonal of the axis-aligned bounding box of `positions` (Bohr) — an upper bound on the
/// distance between any two atoms of the reference cell, and hence on how far a translation
/// has to reach before it can no longer bring an image inside the cutoff.
fn bounding_span(positions: &[Vec3]) -> f64 {
    if positions.is_empty() {
        return 0.0;
    }
    let mut lo = positions[0];
    let mut hi = positions[0];
    for p in positions {
        lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    }
    (hi - lo).norm()
}

/// Uniform Cartesian bucket grid over the reference-cell atoms, with a bucket edge of one
/// cutoff, so a query only has to scan the 27 buckets around the query point.
///
/// This is what keeps the neighbour build linear in the atom count: without it every
/// translation would cost an `O(N²)` scan, and the divide-and-conquer path could never
/// reach linear scaling no matter how cheap its diagonalization became.
struct Grid {
    origin: Vec3,
    inv_edge: f64,
    dims: [i64; 3],
    /// Atom indices bucketed by cell, flattened; `starts[i]..starts[i + 1]` is bucket `i`.
    items: Vec<usize>,
    starts: Vec<usize>,
}

impl Grid {
    fn new(positions: &[Vec3], cutoff: f64) -> Self {
        let edge = cutoff.max(1.0e-6);
        if positions.is_empty() {
            return Self {
                origin: Vec3::zero(),
                inv_edge: 1.0 / edge,
                dims: [1, 1, 1],
                items: Vec::new(),
                starts: vec![0, 0],
            };
        }
        let mut lo = positions[0];
        let mut hi = positions[0];
        for p in positions {
            lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
            hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
        }
        let extent = hi - lo;
        let dims = [
            ((extent.x / edge).floor() as i64 + 1).max(1),
            ((extent.y / edge).floor() as i64 + 1).max(1),
            ((extent.z / edge).floor() as i64 + 1).max(1),
        ];
        let bucket_count = (dims[0] * dims[1] * dims[2]) as usize;
        let mut counts = vec![0usize; bucket_count + 1];
        let inv_edge = 1.0 / edge;
        let index_of = |p: Vec3| -> usize {
            let i = (((p.x - lo.x) * inv_edge).floor() as i64).clamp(0, dims[0] - 1);
            let j = (((p.y - lo.y) * inv_edge).floor() as i64).clamp(0, dims[1] - 1);
            let k = (((p.z - lo.z) * inv_edge).floor() as i64).clamp(0, dims[2] - 1);
            ((i * dims[1] + j) * dims[2] + k) as usize
        };
        for p in positions {
            counts[index_of(*p) + 1] += 1;
        }
        for i in 0..bucket_count {
            counts[i + 1] += counts[i];
        }
        let starts = counts.clone();
        let mut cursor = counts;
        let mut items = vec![0usize; positions.len()];
        for (index, p) in positions.iter().enumerate() {
            let bucket = index_of(*p);
            items[cursor[bucket]] = index;
            cursor[bucket] += 1;
        }
        Self {
            origin: lo,
            inv_edge,
            dims,
            items,
            starts,
        }
    }

    /// Call `f` for every reference-cell atom in the 27 buckets around `point`. Atoms
    /// outside the cutoff may be visited; the caller filters on the actual distance.
    fn for_each_near(&mut self, point: Vec3, mut f: impl FnMut(usize)) {
        let base = [
            ((point.x - self.origin.x) * self.inv_edge).floor() as i64,
            ((point.y - self.origin.y) * self.inv_edge).floor() as i64,
            ((point.z - self.origin.z) * self.inv_edge).floor() as i64,
        ];
        for di in -1..=1 {
            let i = base[0] + di;
            if i < 0 || i >= self.dims[0] {
                continue;
            }
            for dj in -1..=1 {
                let j = base[1] + dj;
                if j < 0 || j >= self.dims[1] {
                    continue;
                }
                for dk in -1..=1 {
                    let k = base[2] + dk;
                    if k < 0 || k >= self.dims[2] {
                        continue;
                    }
                    let bucket = ((i * self.dims[1] + j) * self.dims[2] + k) as usize;
                    for &atom in &self.items[self.starts[bucket]..self.starts[bucket + 1]] {
                        f(atom);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::Atom;

    fn molecule(positions: &[[f64; 3]], cell: Option<Cell>) -> Molecule {
        let mut m = Molecule::new(
            positions
                .iter()
                .map(|p| Atom {
                    z: 1,
                    position: Vec3::new(p[0], p[1], p[2]),
                })
                .collect(),
        );
        m.cell = cell;
        m
    }

    /// The grid-accelerated build must find exactly the pairs a brute-force scan finds.
    #[test]
    fn grid_build_matches_brute_force() {
        let cell = Cell::new(
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(1.5, 5.5, 0.0),
            Vec3::new(0.0, 0.0, 7.0),
            [true, true, true],
        )
        .unwrap();
        let positions = [
            [0.1, 0.2, 0.3],
            [2.7, 1.1, 3.4],
            [4.9, 4.3, 1.2],
            [1.3, 3.8, 6.1],
            [5.5, 0.4, 5.0],
        ];
        let m = molecule(&positions, Some(cell));
        let cutoff = 11.0;
        let list = NeighborList::build(&m, cutoff);

        // Brute force over a translation box far larger than the build's own bound.
        let mut expected = 0usize;
        for a in 0..positions.len() {
            for b in 0..positions.len() {
                for n0 in -6..=6 {
                    for n1 in -6..=6 {
                        for n2 in -6..=6 {
                            if n0 == 0 && n1 == 0 && n2 == 0 && a == b {
                                continue;
                            }
                            let t = cell.vector(0) * n0 as f64
                                + cell.vector(1) * n1 as f64
                                + cell.vector(2) * n2 as f64;
                            let d = m.atoms[b].position + t - m.atoms[a].position;
                            if d.norm() <= cutoff {
                                expected += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(list.len(), expected, "grid build missed or invented pairs");
    }

    /// The half-set must cover every interaction exactly once: summing over `unique`
    /// with weight 1 has to equal half the sum over every ordered pair.
    #[test]
    fn unique_view_is_exactly_half_the_ordered_sum() {
        let cell = Cell::new(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.8, 4.6, 0.0),
            Vec3::new(0.0, 0.6, 5.3),
            [true, true, true],
        )
        .unwrap();
        let m = molecule(
            &[[0.0, 0.0, 0.0], [2.1, 1.4, 0.7], [3.9, 3.3, 2.8]],
            Some(cell),
        );
        let list = NeighborList::build(&m, 13.0);
        // Any smooth decaying pair function will do; use a screened Coulomb form so the
        // test would notice a sign or ordering mistake as well as a counting one.
        let f = |r: f64| (-0.3 * r).exp() / r;
        let ordered: f64 = list.all().iter().map(|p| f(p.r)).sum();
        let unique: f64 = list.unique().map(|p| f(p.r)).sum();
        assert!(
            (unique - 0.5 * ordered).abs() < 1.0e-10 * ordered.abs().max(1.0),
            "unique {unique} vs half of ordered {}",
            0.5 * ordered
        );
        assert!(unique > 0.0, "the test function should not vanish");
    }

    /// Self-images (`a = b`, `T ≠ 0`) are real interactions and must be present exactly
    /// once in the unique view — dropping them is the classic periodic-energy bug.
    #[test]
    fn self_images_are_counted_once() {
        let cell = Cell::cubic(4.0).unwrap();
        let m = molecule(&[[0.0, 0.0, 0.0]], Some(cell));
        let list = NeighborList::build(&m, 4.5);
        // Cubic cell, cutoff 4.5: the 6 face and 12 edge neighbours are within 4.5 Bohr
        // (4.0 and 5.66 -> only the 6 faces), so 6 ordered pairs, 3 unique.
        assert_eq!(list.len(), 6);
        assert_eq!(list.unique().count(), 3);
        for p in list.unique() {
            assert_eq!(p.a, 0);
            assert_eq!(p.b, 0);
            assert!((p.r - 4.0).abs() < 1e-12);
        }
    }

    /// A molecule takes the same path with a single translation and must reproduce the
    /// plain `i < j` pair loop the non-periodic code already uses.
    #[test]
    fn molecular_case_is_the_plain_pair_loop() {
        let m = molecule(
            &[
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [0.0, 1.9, 0.0],
                [3.0, 3.0, 3.0],
            ],
            None,
        );
        let list = NeighborList::build(&m, 100.0);
        assert_eq!(list.len(), 4 * 3, "every ordered pair, no self-interaction");
        assert_eq!(list.unique().count(), 4 * 3 / 2);
        for p in list.unique() {
            assert!(p.a < p.b);
            assert_eq!(p.t, [0, 0, 0]);
        }
    }

    /// Reduced dimensionality must only translate along the periodic directions.
    #[test]
    fn reduced_dimensionality_translates_only_periodic_directions() {
        let slab = Cell::new(
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(0.0, 0.0, 60.0),
            [true, true, false],
        )
        .unwrap();
        let m = molecule(&[[0.0, 0.0, 0.0], [2.0, 2.0, 1.0]], Some(slab));
        let list = NeighborList::build(&m, 9.0);
        assert!(!list.is_empty());
        for p in list.all() {
            assert_eq!(p.t[2], 0, "translated along a non-periodic direction");
        }
        // The vacuum thickness must not change the neighbour set at all.
        let thicker = Cell::new(
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(0.0, 0.0, 600.0),
            [true, true, false],
        )
        .unwrap();
        let m2 = molecule(&[[0.0, 0.0, 0.0], [2.0, 2.0, 1.0]], Some(thicker));
        assert_eq!(NeighborList::build(&m2, 9.0).len(), list.len());
    }

    /// `pairs_of` must partition the full list.
    #[test]
    fn per_atom_slices_partition_the_list() {
        let m = molecule(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            Some(Cell::cubic(9.0).unwrap()),
        );
        let list = NeighborList::build(&m, 9.5);
        let total: usize = (0..3).map(|a| list.pairs_of(a).len()).sum();
        assert_eq!(total, list.len());
        for a in 0..3 {
            for p in list.pairs_of(a) {
                assert_eq!(p.a, a);
            }
        }
    }
}
