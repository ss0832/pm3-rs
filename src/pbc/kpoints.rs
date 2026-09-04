// SPDX-License-Identifier: GPL-3.0-or-later

//! Monkhorst–Pack meshes, time-reversal reduction, and band paths.
//!
//! A k-point is stored in **fractional** coordinates of the reciprocal basis, because that is the
//! form every consumer wants: the Bloch phase for a translation `T = Σ n_i a_i` is
//! `exp(2πi Σ k_i n_i)`, which needs the integer image indices and the fractional k and never the
//! Cartesian ones. [`KPoint::phase`] is the only place that arithmetic appears.
//!
//! # Non-periodic directions
//!
//! A slab has no band dispersion perpendicular to it, so a division other than 1 there is not a
//! coarse mesh — it is a request to sample a dimension that does not exist. [`KpointSpec::mesh`]
//! rejects it rather than silently sampling one point, because silently correcting it would hide
//! a mistake in the caller's setup.
//!
//! # Time reversal
//!
//! With a real Hamiltonian, `H(−k) = H(k)*`, so `−k` has the same eigenvalues as `k` and a density
//! contribution that is the complex conjugate. Summing a `±k` pair therefore gives twice the real
//! part of either one, and only one member of each pair needs to be diagonalized. That is a
//! factor-of-two saving on everything downstream, and it is exact rather than approximate — see
//! [`reduce`].
//!
//! Note that the density matrix is still built correctly for the *dropped* member: the k-point SCF
//! forms `P(T) = Σ_k w_k e^{−ik·T} P(k)` and takes the real part, which for a `±k` pair with
//! doubled weight is exactly the same thing.

use std::collections::HashMap;

use crate::cell::Cell;
use crate::error::{Pm3Error, Result};
use crate::math::Vec3;

/// One sampling point of the Brillouin zone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KPoint {
    /// Fractional coordinates along the reciprocal basis, in `[-0.5, 0.5)`.
    pub frac: [f64; 3],
    /// Integration weight. The weights of a mesh sum to 1.
    pub weight: f64,
}

impl KPoint {
    pub const GAMMA: KPoint = KPoint {
        frac: [0.0, 0.0, 0.0],
        weight: 1.0,
    };

    /// `exp(i k·T)` for the lattice translation with integer image indices `t`.
    ///
    /// In fractional coordinates this is `exp(2πi Σ k_i t_i)` — no reciprocal basis needed, which
    /// is why k-points are stored fractionally.
    #[inline]
    pub fn phase(&self, t: [i32; 3]) -> faer::c64 {
        frac_phase(self.frac, t)
    }

    /// True when `k` and `−k` are the same point of the reciprocal lattice, so time reversal
    /// relates the point to itself and its weight must not be doubled.
    pub fn is_time_reversal_invariant(&self) -> bool {
        self.frac
            .iter()
            .all(|f| (2.0 * f - (2.0 * f).round()).abs() < TOLERANCE)
    }

    /// Cartesian `k` (inverse Bohr), for band plots and diagnostics.
    pub fn cartesian(&self, cell: &Cell) -> Vec3 {
        let mut out = Vec3::zero();
        for (index, b) in cell.reciprocal_basis() {
            out += b * self.frac[index];
        }
        out
    }
}

/// Numerical slack for identifying k-points on the fractional grid. The coordinates are exact
/// rationals `(2n − N + 1)/2N`, so anything this close is the same point.
const TOLERANCE: f64 = 1.0e-9;

/// `exp(2πi Σ k_i t_i)` for a **fractional** wavevector and integer image indices.
///
/// The only place this arithmetic lives, so everything phased by a wavevector agrees to the last
/// bit.
///
/// It matters most at a reciprocal lattice vector. The phase there is not *exactly* one —
/// `cos(TAU * n)` is a floating-point cosine like any other, and comes out a few ulp short — but
/// the **angle** is `TAU * n` to full precision, where the Cartesian route reaches it through a
/// reciprocal basis and a dot product and arrives with an error that grows with `|T|`. Measured
/// through the identity that depends on it, `D(q + G) = D(q)`: `1e-9` this way, `1e-6` the
/// other.
#[inline]
pub fn frac_phase(frac: [f64; 3], t: [i32; 3]) -> faer::c64 {
    let turns = frac[0] * t[0] as f64 + frac[1] * t[1] as f64 + frac[2] * t[2] as f64;
    let angle = std::f64::consts::TAU * turns;
    faer::c64::new(angle.cos(), angle.sin())
}

/// `k + q`, folded back into `[-0.5, 0.5)`, keeping `k`'s weight.
///
/// Folding costs nothing here, and that is worth saying because it is not true of every method.
/// [`crate::pbc::kscf::bloch_fock`] assembles `F(k) = Σ_T e^{ik·T} F(T)` in the periodic-gauge AO
/// basis, with no `e^{iG·r}` factors anywhere, so `F(k + G) ≡ F(k)` element for element and the
/// eigenvectors are literally the same. A plane-wave code has to track the umklapp; this does
/// not.
pub fn shift(k: &KPoint, q_frac: [f64; 3]) -> KPoint {
    KPoint {
        frac: [
            fold(k.frac[0] + q_frac[0]),
            fold(k.frac[1] + q_frac[1]),
            fold(k.frac[2] + q_frac[2]),
        ],
        weight: k.weight,
    }
}

/// Exact lookup from a folded fractional coordinate to its position in a mesh.
///
/// Used only as an optimization: when `k + q` happens to be a mesh point its eigenpairs are
/// already in hand, and when it does not they are obtained by diagonalizing at that point. The
/// answer does not depend on which case it is — see [`shift`].
pub struct MeshIndex {
    slots: HashMap<[i64; 3], usize>,
}

impl MeshIndex {
    pub fn build(points: &[KPoint]) -> Self {
        Self {
            slots: points
                .iter()
                .enumerate()
                .map(|(index, point)| (key(point.frac), index))
                .collect(),
        }
    }

    pub fn find(&self, frac: [f64; 3]) -> Option<usize> {
        self.slots
            .get(&key([fold(frac[0]), fold(frac[1]), fold(frac[2])]))
            .copied()
    }
}

/// Whether `q` maps this mesh onto itself, so every `k + q` is again one of its points.
///
/// The right definition is membership in the mesh's **difference set** `{k_i − k_j}`, not
/// "`q` is a rational fraction of a reciprocal vector": a shifted Monkhorst–Pack mesh contains
/// plenty of rational `q` that move its points off it.
///
/// Reported as a diagnostic rather than enforced. A commensurate `q` makes the answer an exact
/// supercell result; an incommensurate one makes it a Fourier interpolation of the same thing,
/// which is a statement about what was asked for, not about whether it was computed correctly.
pub fn is_commensurate(points: &[KPoint], q_frac: [f64; 3]) -> bool {
    let index = MeshIndex::build(points);
    points
        .iter()
        .all(|point| index.find(shift(point, q_frac).frac).is_some())
}

/// How the Brillouin zone is to be sampled.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum KpointSpec {
    /// The Γ point alone — the `[1, 1, 1]` mesh, named because it is the common case.
    #[default]
    Gamma,
    /// A Monkhorst–Pack mesh with an optional shift, in units of half a mesh spacing.
    Mesh {
        divisions: [usize; 3],
        shift: [f64; 3],
    },
    /// Explicit points with explicit weights, for band paths and for tests.
    Explicit(Vec<KPoint>),
}

impl KpointSpec {
    /// An unshifted Monkhorst–Pack mesh.
    pub fn mesh(divisions: [usize; 3]) -> Self {
        Self::Mesh {
            divisions,
            shift: [0.0; 3],
        }
    }

    /// Expand into the irreducible list of points with weights summing to 1.
    pub fn generate(&self, cell: &Cell) -> Result<Vec<KPoint>> {
        match self {
            Self::Gamma => Ok(vec![KPoint::GAMMA]),
            Self::Mesh { divisions, shift } => {
                let full = monkhorst_pack(cell, *divisions, *shift)?;
                Ok(reduce(&full))
            }
            Self::Explicit(points) => {
                if points.is_empty() {
                    return Err(Pm3Error::InvalidInput(
                        "an explicit k-point list must not be empty".to_string(),
                    ));
                }
                for point in points {
                    check_non_periodic(cell, point.frac)?;
                }
                Ok(points.clone())
            }
        }
    }
}

/// A Γ-centred Monkhorst–Pack mesh, before symmetry reduction.
///
/// The points are `k_i = (n_i + s_i)/N_i` folded into `[-0.5, 0.5)`, with `n_i` running over
/// `0..N_i`. Non-periodic directions are pinned to a single point at zero.
///
/// **Γ-centred, not zone-centred.** The original Monkhorst–Pack prescription centres the mesh on
/// the zone, which for an even division puts no point at `Γ` at all. Both are legitimate
/// quadratures, but only the Γ-centred mesh is the exact set of `k` allowed by an `N₁×N₂×N₃`
/// supercell's periodicity, so only it satisfies `E(mesh) == E(supercell at Γ)/N` — the identity
/// the k-point path is validated against. `shift` is there for callers who want the other
/// convention, at half a mesh spacing per unit.
/// The largest mesh this will construct.
///
/// Sixteen million points is already far past any calculation that finishes; the bound exists so
/// that an absurd request is a message rather than an allocation failure or, worse, a wrapped
/// product that yields a small mesh with wrong weights.
const MAX_MESH_POINTS: usize = 1 << 24;

pub fn monkhorst_pack(cell: &Cell, divisions: [usize; 3], shift: [f64; 3]) -> Result<Vec<KPoint>> {
    let mut effective = [1usize; 3];
    for axis in 0..3 {
        if divisions[axis] == 0 {
            return Err(Pm3Error::InvalidInput(format!(
                "k-point division along axis {axis} is zero; use 1 for no sampling"
            )));
        }
        if !cell.pbc[axis] && divisions[axis] != 1 {
            return Err(Pm3Error::InvalidInput(format!(
                "axis {axis} is not periodic, so it has no band dispersion to sample: \
                 a division of {} is meaningless there and 1 is the only valid value",
                divisions[axis]
            )));
        }
        if !cell.pbc[axis] && shift[axis] != 0.0 {
            return Err(Pm3Error::InvalidInput(format!(
                "axis {axis} is not periodic, so a k-point shift of {} has no meaning there",
                shift[axis]
            )));
        }
        effective[axis] = divisions[axis];
    }

    // The product, checked. Zero was already refused above and nothing bounded the other end, so
    // `divisions` large enough to overflow `usize` wrapped to a small number — producing a mesh
    // with the wrong point count and the wrong weights, silently, in release. A mesh of a million
    // per axis is a mistake rather than a request in any case; saying so is cheaper than
    // allocating for it.
    let total = effective[0]
        .checked_mul(effective[1])
        .and_then(|n| n.checked_mul(effective[2]))
        .filter(|n| *n <= MAX_MESH_POINTS)
        .ok_or_else(|| {
            Pm3Error::InvalidInput(format!(
                "a {}x{}x{} mesh is {} k-points, past the {MAX_MESH_POINTS} this will build; \
                 a mesh that large is a typo rather than a sampling choice",
                effective[0],
                effective[1],
                effective[2],
                // Saturating, because the honest product may not be representable.
                (effective[0] as u128) * (effective[1] as u128) * (effective[2] as u128)
            ))
        })?;
    let weight = 1.0 / total as f64;
    let mut points = Vec::with_capacity(total);
    for n0 in 0..effective[0] {
        for n1 in 0..effective[1] {
            for n2 in 0..effective[2] {
                let raw = [
                    (n0 as f64 + shift[0]) / effective[0] as f64,
                    (n1 as f64 + shift[1]) / effective[1] as f64,
                    (n2 as f64 + shift[2]) / effective[2] as f64,
                ];
                points.push(KPoint {
                    frac: [fold(raw[0]), fold(raw[1]), fold(raw[2])],
                    weight,
                });
            }
        }
    }
    Ok(points)
}

/// Fold a fractional coordinate into `[-0.5, 0.5)`.
fn fold(value: f64) -> f64 {
    let mut folded = value - value.round();
    // `round` ties away from zero, so exactly +0.5 lands on −0.5; make that canonical.
    if folded >= 0.5 - TOLERANCE {
        folded -= 1.0;
    }
    if folded.abs() < TOLERANCE {
        folded = 0.0;
    }
    folded
}

/// Merge each `±k` pair into its representative, doubling the weight.
///
/// Exact, not approximate: with a real Hamiltonian `H(−k) = H(k)*`, so the two points have
/// identical eigenvalues and conjugate densities, and the pair's contribution to any real
/// observable is twice the real part of one member's. Points that are their own negative modulo a
/// reciprocal lattice vector — `Γ` and the zone-boundary points — pair with themselves and keep
/// their weight.
pub fn reduce(points: &[KPoint]) -> Vec<KPoint> {
    let mut index: HashMap<[i64; 3], usize> = HashMap::new();
    let mut out: Vec<KPoint> = Vec::new();
    for point in points {
        if point.is_time_reversal_invariant() {
            out.push(*point);
            continue;
        }
        let negated = key([-point.frac[0], -point.frac[1], -point.frac[2]]);
        if let Some(slot) = index.get(&negated) {
            out[*slot].weight += point.weight;
            continue;
        }
        index.insert(key(point.frac), out.len());
        out.push(*point);
    }
    out
}

/// Quantize a folded fractional coordinate for exact hash lookup.
fn key(frac: [f64; 3]) -> [i64; 3] {
    let quantize = |v: f64| (fold(v) / TOLERANCE).round() as i64;
    [quantize(frac[0]), quantize(frac[1]), quantize(frac[2])]
}

fn check_non_periodic(cell: &Cell, frac: [f64; 3]) -> Result<()> {
    for (axis, component) in frac.iter().enumerate() {
        if !cell.pbc[axis] && component.abs() > TOLERANCE {
            return Err(Pm3Error::InvalidInput(format!(
                "k-point component {component} along non-periodic axis {axis} must be zero"
            )));
        }
    }
    Ok(())
}

/// A band path: straight segments through the given fractional corner points, sampled at roughly
/// uniform Cartesian spacing, with equal weights.
///
/// Weights on a band path are meaningless as an integration measure — a path is not a mesh — so
/// they are set uniform and should not be used to average anything.
pub fn band_path(
    cell: &Cell,
    corners: &[[f64; 3]],
    points_per_segment: usize,
) -> Result<Vec<KPoint>> {
    if corners.len() < 2 {
        return Err(Pm3Error::InvalidInput(
            "a band path needs at least two corner points".to_string(),
        ));
    }
    if points_per_segment == 0 {
        return Err(Pm3Error::InvalidInput(
            "a band path needs at least one point per segment".to_string(),
        ));
    }
    for corner in corners {
        check_non_periodic(cell, *corner)?;
    }
    let mut path = Vec::new();
    for (segment, window) in corners.windows(2).enumerate() {
        let (from, to) = (window[0], window[1]);
        // Every segment contributes its start; only the last contributes its end, so corners are
        // not duplicated where segments meet.
        let last = segment + 2 == corners.len();
        let steps = points_per_segment;
        for step in 0..=steps {
            if step == steps && !last {
                break;
            }
            let fraction = step as f64 / steps as f64;
            path.push(KPoint {
                frac: [
                    from[0] + (to[0] - from[0]) * fraction,
                    from[1] + (to[1] - from[1]) * fraction,
                    from[2] + (to[2] - from[2]) * fraction,
                ],
                weight: 0.0,
            });
        }
    }
    let uniform = 1.0 / path.len() as f64;
    for point in &mut path {
        point.weight = uniform;
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cubic() -> Cell {
        Cell::cubic(10.0).unwrap()
    }

    fn slab() -> Cell {
        Cell::new(
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, true, false],
        )
        .unwrap()
    }

    #[test]
    fn a_one_by_one_mesh_is_the_gamma_point() {
        let points = KpointSpec::mesh([1, 1, 1]).generate(&cubic()).unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].frac, [0.0, 0.0, 0.0]);
        assert!((points[0].weight - 1.0).abs() < 1.0e-15);
    }

    #[test]
    fn mesh_weights_sum_to_one_before_and_after_reduction() {
        for divisions in [[2, 2, 2], [3, 3, 3], [4, 2, 1], [5, 1, 1]] {
            let full = monkhorst_pack(&cubic(), divisions, [0.0; 3]).unwrap();
            assert_eq!(full.len(), divisions.iter().product::<usize>());
            let total: f64 = full.iter().map(|k| k.weight).sum();
            assert!((total - 1.0).abs() < 1.0e-12, "{divisions:?}: {total}");

            let reduced = reduce(&full);
            let total: f64 = reduced.iter().map(|k| k.weight).sum();
            assert!((total - 1.0).abs() < 1.0e-12, "{divisions:?}: {total}");
            assert!(reduced.len() <= full.len());
        }
    }

    /// Time reversal must halve the count, up to the self-paired points it cannot touch.
    ///
    /// For an odd `N³` mesh only Γ is self-paired, so the reduced count is `(N³ + 1)/2`. For an
    /// even mesh the zone-boundary points are self-paired too.
    #[test]
    fn time_reversal_halves_an_odd_mesh_exactly() {
        for n in [3usize, 5, 7] {
            let full = monkhorst_pack(&cubic(), [n, n, n], [0.0; 3]).unwrap();
            let reduced = reduce(&full);
            assert_eq!(
                reduced.len(),
                (n * n * n).div_ceil(2),
                "{n}³ mesh reduced to {}",
                reduced.len()
            );
        }
    }

    #[test]
    fn self_paired_points_keep_their_weight() {
        let full = monkhorst_pack(&cubic(), [2, 2, 2], [0.0; 3]).unwrap();
        let reduced = reduce(&full);
        // Every point of a Γ-centred 2×2×2 mesh is 0 or −½ in each component, so all eight are
        // their own negatives and nothing is merged.
        assert_eq!(reduced.len(), 8);
        for point in &reduced {
            assert!(point.is_time_reversal_invariant());
            assert!((point.weight - 0.125).abs() < 1.0e-12);
        }
    }

    /// The Bloch phase is what every consumer actually uses, so pin its conventions down.
    #[test]
    fn the_bloch_phase_is_consistent() {
        let gamma = KPoint::GAMMA;
        for t in [[0, 0, 0], [1, 0, 0], [-2, 3, 1]] {
            let phase = gamma.phase(t);
            assert!((phase.re - 1.0).abs() < 1.0e-15 && phase.im.abs() < 1.0e-15);
        }

        // At the zone boundary the phase alternates in sign with the image index.
        let boundary = KPoint {
            frac: [0.5, 0.0, 0.0],
            weight: 1.0,
        };
        assert!((boundary.phase([1, 0, 0]).re + 1.0).abs() < 1.0e-14);
        assert!((boundary.phase([2, 0, 0]).re - 1.0).abs() < 1.0e-14);

        // `k` and `−k` give conjugate phases, which is the identity time-reversal reduction rests
        // on.
        let k = KPoint {
            frac: [0.25, -0.125, 0.375],
            weight: 1.0,
        };
        let minus = KPoint {
            frac: [-0.25, 0.125, -0.375],
            weight: 1.0,
        };
        for t in [[1, 0, 0], [0, 1, 1], [2, -1, 3]] {
            let difference = k.phase(t) - minus.phase(t).conj();
            assert!(difference.norm() < 1.0e-14);
        }
    }

    #[test]
    fn a_slab_refuses_sampling_perpendicular_to_itself() {
        let error = KpointSpec::mesh([4, 4, 4])
            .generate(&slab())
            .expect_err("a slab has no dispersion along its normal");
        assert!(error.to_string().contains("not periodic"));

        // In-plane sampling of the same slab is fine.
        let points = KpointSpec::mesh([4, 4, 1]).generate(&slab()).unwrap();
        assert!(points.iter().all(|k| k.frac[2] == 0.0));
    }

    #[test]
    fn zero_divisions_are_refused() {
        let error = KpointSpec::mesh([2, 0, 2])
            .generate(&cubic())
            .expect_err("a zero division is not a valid mesh");
        assert!(error.to_string().contains("zero"));
    }

    #[test]
    fn an_explicit_point_off_a_non_periodic_axis_is_refused() {
        let spec = KpointSpec::Explicit(vec![KPoint {
            frac: [0.0, 0.0, 0.25],
            weight: 1.0,
        }]);
        let error = spec
            .generate(&slab())
            .expect_err("k must vanish along a non-periodic axis");
        assert!(error.to_string().contains("non-periodic"));
    }

    #[test]
    fn a_band_path_visits_its_corners_once_each() {
        let corners = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]];
        let path = band_path(&cubic(), &corners, 4).unwrap();
        // Two segments of four steps, sharing the middle corner: 4 + 4 + 1 points.
        assert_eq!(path.len(), 9);
        assert_eq!(path[0].frac, corners[0]);
        assert_eq!(path[4].frac, corners[1]);
        assert_eq!(path[8].frac, corners[2]);
        let total: f64 = path.iter().map(|k| k.weight).sum();
        assert!((total - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn a_mesh_point_maps_to_a_cartesian_vector_of_the_right_length() {
        let cell = cubic();
        let k = KPoint {
            frac: [0.5, 0.0, 0.0],
            weight: 1.0,
        };
        // |b₁| = 2π/a for a cubic cell, so the zone boundary sits at π/a.
        let expected = std::f64::consts::PI / 10.0;
        assert!((k.cartesian(&cell).norm() - expected).abs() < 1.0e-12);
    }
    /// `k + q` folded back into the zone, and the mesh lookup that finds it.
    #[test]
    fn shifting_a_k_point_folds_it_back_into_the_zone() {
        let cell = Cell::cubic(8.0).unwrap();
        let mesh = monkhorst_pack(&cell, [4, 4, 4], [0.0; 3]).unwrap();
        let index = MeshIndex::build(&mesh);

        for point in &mesh {
            for q in [[0.25, 0.0, 0.0], [0.5, -0.25, 0.75], [0.0; 3]] {
                let moved = shift(point, q);
                for component in moved.frac {
                    assert!(
                        (-0.5..0.5).contains(&component),
                        "{component} is outside [-0.5, 0.5)"
                    );
                }
                assert_eq!(moved.weight, point.weight, "shifting must not reweight");
                // A quarter-mesh shift of a four-fold mesh lands on the mesh again.
                assert!(
                    index.find(moved.frac).is_some(),
                    "{q:?} should map the mesh onto itself"
                );
            }
        }
    }

    /// Commensurability is membership in the mesh's **difference set**, not "a rational fraction
    /// of a reciprocal vector".
    ///
    /// The distinction is not academic: a shifted Monkhorst-Pack mesh contains plenty of rational
    /// `q` that move its points off it, and reading commensurability off the fraction alone would
    /// call those commensurate.
    #[test]
    fn commensurability_is_membership_in_the_difference_set() {
        let cell = Cell::cubic(8.0).unwrap();
        let mesh = monkhorst_pack(&cell, [4, 4, 4], [0.0; 3]).unwrap();
        assert!(is_commensurate(&mesh, [0.25, 0.0, 0.0]));
        assert!(is_commensurate(&mesh, [0.5, 0.25, -0.25]));
        assert!(!is_commensurate(&mesh, [0.1, 0.0, 0.0]));
        assert!(!is_commensurate(&mesh, [1.0 / 3.0, 0.0, 0.0]));

        // A half-shifted mesh: 1/4 still maps it onto itself (the shift is common to every
        // point), but 1/8 does not, and neither reads off the fraction alone.
        let shifted = monkhorst_pack(&cell, [4, 1, 1], [0.5, 0.0, 0.0]).unwrap();
        assert!(is_commensurate(&shifted, [0.25, 0.0, 0.0]));
        assert!(!is_commensurate(&shifted, [0.125, 0.0, 0.0]));
    }

    /// At a reciprocal lattice vector the phase is one to within a few ulp — and, more to the
    /// point, it stays that way as the image gets far away.
    ///
    /// Two earlier versions of this test were wrong, and each was wrong in a way worth keeping
    /// in the record.
    ///
    /// The first asserted the phase was *exactly* one. It is not: `cos(TAU * n)` is an ordinary
    /// floating-point cosine. The second allowed a fixed few ulp, which fails at four turns —
    /// because `TAU` is itself only good to one ulp, so `TAU * n` is off by `n` ulp and the
    /// residue grows **with the number of turns**, not with the image index.
    ///
    /// That is the honest statement and it is what this pins. Fractional coordinates do not make
    /// the phase exact; they make the angle exact up to the representation of `2π`, where a
    /// Cartesian `q·T` reaches the same angle through a reciprocal basis and a dot product and
    /// arrives with more error than that. The measured consequence is in
    /// `dfpt::tests::the_matrix_is_periodic_in_the_wavevector`: `1e-9` this way, `1e-6` the other.
    #[test]
    fn a_reciprocal_lattice_vector_phases_by_one_to_the_precision_of_two_pi() {
        for t in [[1, 0, 0], [3, -2, 7], [0, 0, -11], [40, -37, 53]] {
            for n in [1.0_f64, -1.0, 4.0] {
                let phase = frac_phase([n, 0.0, 0.0], t);
                // The angle is `TAU * n * t[0]`, so that many turns, and one ulp of `TAU` each.
                let turns = (n * t[0] as f64).abs().max(1.0);
                let bound = 4.0 * f64::EPSILON * turns;
                assert!(
                    (phase.re - 1.0).abs() < bound,
                    "t = {t:?}, n = {n}: real part {} is not one within {bound:.2e}",
                    phase.re
                );
                assert!(
                    phase.im.abs() < bound,
                    "t = {t:?}, n = {n}: imaginary part {} exceeds {bound:.2e}",
                    phase.im
                );
            }
        }
    }

    /// The phase is a homomorphism: `e^{ik(T+T')} = e^{ikT} e^{ikT'}`, and `e^{i(-k)T}` is the
    /// conjugate. Both are properties of the formula rather than of any particular use of it.
    #[test]
    fn the_phase_composes_and_conjugates() {
        let k = [0.31, -0.23, 0.17];
        for (a, b) in [([1, 0, 0], [0, 2, -1]), ([-3, 1, 4], [2, -2, 5])] {
            let sum = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
            let product = frac_phase(k, a) * frac_phase(k, b);
            let direct = frac_phase(k, sum);
            assert!((product.re - direct.re).abs() < 1.0e-14);
            assert!((product.im - direct.im).abs() < 1.0e-14);

            let reversed = frac_phase([-k[0], -k[1], -k[2]], a);
            let conjugate = frac_phase(k, a).conj();
            assert!((reversed.re - conjugate.re).abs() < 1.0e-15);
            assert!((reversed.im - conjugate.im).abs() < 1.0e-15);
        }
    }
}
