// SPDX-License-Identifier: GPL-3.0-or-later

//! Lattice sums for the classical D3/H4/X corrections.
//!
//! The corrections are functions of the geometry alone and are finite-ranged — `1/R⁶` for
//! dispersion, exponential for the hydrogen- and halogen-bond terms — so no Ewald machinery is
//! involved. What periodicity costs them is entirely a matter of *counting*: every distinct
//! tuple in the crystal has to contribute exactly once per unit cell, and the tuples that matter
//! most in a molecular crystal are precisely the ones that straddle a cell boundary. In ice, most
//! hydrogen bonds do.
//!
//! This module builds the image-expanded cluster those sums run over.
//! [`crate::corrections::correction_energy_cluster_g`] does the counting.
//!
//! # Why the cluster is built in the generic scalar type
//!
//! Image positions are `parent + T`, formed *from the caller's own position values*. When those
//! values carry derivative information — [`crate::dual::Dual`] for a gradient,
//! [`crate::dual2::Dual2`] for a Hessian — an image inherits its parent's derivative seed
//! automatically, because a lattice translation is a constant shift. Building the cluster in
//! `f64` and then re-seeding would lose that, and every image's contribution to the force on its
//! parent atom with it.

use crate::dual::Scalar;
use crate::params::Pm3Element;
use crate::system::Molecule;

/// Default cluster radius (Bohr) for the dispersion term.
///
/// `C6/R⁶` with a typical `C6 ≈ 30` atomic units is about `1e-6` eV per pair at 30 Bohr, and the
/// pair count at that range cannot lift the total above the tolerances used here.
pub const DEFAULT_D3_CUTOFF: f64 = 30.0;

/// Default cluster radius (Bohr) for the hydrogen- and halogen-bond terms.
///
/// Both are cut off internally well inside this — H4 at its donor–acceptor range, X at 8 Å — so
/// this only has to be comfortably beyond those, and staying tight keeps the `O(N²)` donor–
/// acceptor scan small.
pub const DEFAULT_SHORT_CUTOFF: f64 = 16.0;

/// Radius (Bohr) for the D3 coordination-number sum.
///
/// Unlike the others this is a **model parameter, not a convergence parameter**, and it cannot
/// be made one. D3's counting function `1/(1 + exp(-16(rco/r - 1)))` tends to `1/(1 + e^16)`
/// rather than to zero, so summing it over an ever-larger sphere adds a contribution growing as
/// the enclosed atom count — the periodic coordination number simply does not converge in the
/// cutoff. Every periodic D3 implementation therefore fixes a radius and states it. By 15 Bohr
/// the genuine part of the sum has decayed to about `1e-5` per pair, and the saturated
/// remainder amounts to a few times `1e-5` in the coordination number itself.
pub const DEFAULT_CN_CUTOFF: f64 = 15.0;

/// Cluster radii per correction term.
#[derive(Clone, Copy, Debug)]
pub struct CorrectionCutoffs {
    /// Radius for the dispersion sum (Bohr).
    pub dispersion: f64,
    /// Radius for the coordination-number sum (Bohr). See [`DEFAULT_CN_CUTOFF`] — this one
    /// defines the model rather than approximating it.
    pub coordination: f64,
    /// Radius for the hydrogen- and halogen-bond sums (Bohr).
    pub short_range: f64,
}

impl Default for CorrectionCutoffs {
    fn default() -> Self {
        Self {
            dispersion: DEFAULT_D3_CUTOFF,
            coordination: DEFAULT_CN_CUTOFF,
            short_range: DEFAULT_SHORT_CUTOFF,
        }
    }
}

/// The reference cell's atoms followed by every image within a radius.
pub struct Cluster {
    pub numbers: Vec<u8>,
    pub positions: Vec<[f64; 3]>,
    /// How many leading entries belong to the reference cell.
    pub n_cell: usize,
    /// For each entry, the reference-cell atom it is an image of (itself, for the first `n_cell`).
    pub parent: Vec<usize>,
    /// Which lattice translation each entry sits in, `[0, 0, 0]` for the reference cell.
    ///
    /// Discarded until phonons at finite `q` needed it: a phased displacement pattern gives the
    /// image in cell `T` a different displacement from its parent, and `T` is the only thing that
    /// says which.
    pub translation: Vec<[i32; 3]>,
    /// The radius the cluster was built to, in Bohr. What [`Cluster::coordination_is_exact`]
    /// compares against.
    pub cutoff: f64,
}

impl Cluster {
    /// Whether entry `index`'s coordination number comes out right on this cluster.
    ///
    /// The cluster holds everything within `cutoff` of a reference-cell atom. An entry's
    /// coordination number counts neighbours within `cn_cutoff` of *it*, so it is exact exactly
    /// when that whole ball is inside the cluster — which the triangle inequality makes a
    /// distance test against the nearest cell atom.
    pub fn coordination_is_exact(&self, index: usize, cn_cutoff: f64) -> bool {
        if index < self.n_cell {
            return true;
        }
        let here = self.positions[index];
        let mut nearest = f64::INFINITY;
        for cell in &self.positions[..self.n_cell] {
            let d = ((here[0] - cell[0]).powi(2)
                + (here[1] - cell[1]).powi(2)
                + (here[2] - cell[2]).powi(2))
            .sqrt();
            nearest = nearest.min(d);
        }
        nearest + cn_cutoff <= self.cutoff
    }
}

/// Build the image-expanded cluster for `molecule` out to `cutoff` Bohr.
///
/// A molecule with no cell yields itself, so callers do not need a periodic branch.
pub fn build_cluster(molecule: &Molecule, cutoff: f64) -> Cluster {
    let n_cell = molecule.atoms.len();
    let mut numbers: Vec<u8> = molecule.atoms.iter().map(|a| a.z).collect();
    let mut positions: Vec<[f64; 3]> = molecule
        .atoms
        .iter()
        .map(|a| [a.position.x, a.position.y, a.position.z])
        .collect();
    let mut parent: Vec<usize> = (0..n_cell).collect();
    let mut translation: Vec<[i32; 3]> = vec![[0; 3]; n_cell];

    if let Some(cell) = molecule.cell {
        // An image can only reach a cell atom if the translation is within the cutoff plus the
        // spread of the cell's own contents.
        let span = bounding_span(&positions);
        for (index, shift) in cell.translations_within(cutoff + span) {
            if index == [0, 0, 0] {
                continue;
            }
            for (atom, position) in molecule.atoms.iter().enumerate() {
                let shifted = position.position + shift;
                numbers.push(molecule.atoms[atom].z);
                positions.push([shifted.x, shifted.y, shifted.z]);
                parent.push(atom);
                translation.push(index);
            }
        }
    }
    Cluster {
        numbers,
        positions,
        n_cell,
        parent,
        translation,
        cutoff,
    }
}

/// The cluster's positions with each entry's displacement scaled by its own weight.
///
/// [`cluster_positions_g`] is the `weights = 1` case: every image inherits its parent's seed
/// whole, which is what a `q = 0` displacement means — moving an atom moves all of its copies by
/// the same amount. A phonon at finite `q` moves the copy in cell `T` by `e^{iq·T}` times as
/// much, and the weight is how that reaches the images.
///
/// `base` is the unperturbed cluster in the seeded scalar, so that `pos[p] − base[p]` is exactly
/// the seeded displacement and nothing else.
pub fn cluster_positions_weighted<S: Scalar>(
    cluster: &Cluster,
    pos: &[[S; 3]],
    weights: &[f64],
) -> Vec<[S; 3]> {
    let mut out = Vec::with_capacity(cluster.positions.len());
    for (index, parent) in cluster.parent.iter().enumerate() {
        let weight = weights[index];
        let base = &cluster.positions[*parent];
        let here = &cluster.positions[index];
        let mut entry = [S::cst(0.0); 3];
        for axis in 0..3 {
            // `x_here + w · (seeded parent − x_parent)`: the constant part is this entry's own
            // position, and only the displacement is scaled.
            let displacement = pos[*parent][axis] - S::cst(base[axis]);
            entry[axis] = S::cst(here[axis]) + displacement * S::cst(weight);
        }
        out.push(entry);
    }
    out
}

/// The cluster's positions in a generic scalar type, built from `pos` so derivative seeds carry
/// through to the images.
///
/// `pos` holds the reference cell's positions in whatever scalar the caller is differentiating
/// in; each image is `pos[parent] + T` with `T` a plain constant.
pub fn cluster_positions_g<S: Scalar>(cluster: &Cluster, pos: &[[S; 3]]) -> Vec<[S; 3]> {
    let mut out = Vec::with_capacity(cluster.positions.len());
    for (index, parent) in cluster.parent.iter().enumerate() {
        if index < cluster.n_cell {
            out.push(pos[*parent]);
        } else {
            // The translation is the difference between the image and its parent in f64; adding
            // it as a constant leaves the parent's derivatives untouched, which is exactly the
            // behaviour a rigid lattice translation should have.
            let base = &cluster.positions[*parent];
            let image = &cluster.positions[index];
            out.push([
                pos[*parent][0] + (image[0] - base[0]),
                pos[*parent][1] + (image[1] - base[1]),
                pos[*parent][2] + (image[2] - base[2]),
            ]);
        }
    }
    out
}

/// Whether every atom carries a PM3 parameter block the corrections understand.
pub fn supported(molecule: &Molecule, element: impl Fn(u8) -> Option<Pm3Element>) -> bool {
    molecule.atoms.iter().all(|a| element(a.z).is_some())
}

fn bounding_span(positions: &[[f64; 3]]) -> f64 {
    if positions.is_empty() {
        return 0.0;
    }
    let mut lo = positions[0];
    let mut hi = positions[0];
    for p in positions {
        for axis in 0..3 {
            lo[axis] = lo[axis].min(p[axis]);
            hi[axis] = hi[axis].max(p[axis]);
        }
    }
    ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt()
}

/// Cluster radius wide enough for `variant`.
pub fn cutoff_for(variant: crate::corrections::Variant, cutoffs: &CorrectionCutoffs) -> f64 {
    match variant {
        crate::corrections::Variant::Pm3 => 0.0,
        crate::corrections::Variant::Pm3D3 => cutoffs.dispersion,
        _ => cutoffs.dispersion.max(cutoffs.short_range),
    }
}

/// Per-cell correction energy (eV) for a periodic structure.
pub fn periodic_correction_energy(
    molecule: &Molecule,
    variant: crate::corrections::Variant,
    cutoffs: &CorrectionCutoffs,
) -> f64 {
    if variant == crate::corrections::Variant::Pm3 {
        return 0.0;
    }
    let cluster = build_cluster(molecule, cutoff_for(variant, cutoffs));
    crate::corrections::correction_energy_cluster_g::<f64>(
        &cluster.numbers,
        &cluster.positions,
        cluster.n_cell,
        Some(&cluster.parent),
        None,
        Some(cutoffs.dispersion),
        Some(cutoffs.coordination),
        variant,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;
    use crate::corrections::{correction_energy, Variant};
    use crate::math::Vec3;
    use crate::system::Atom;

    const VARIANTS: [Variant; 3] = [Variant::Pm3D3, Variant::Pm3D3H4, Variant::Pm3D3H4X];

    fn water_molecule() -> Molecule {
        Molecule::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n",
            0.0,
        )
        .unwrap()
    }

    /// A molecule alone in a large cell must give the molecular correction energy: the images are
    /// too far to contribute, so anything left is a counting error rather than physics.
    #[test]
    fn a_molecule_in_a_large_cell_matches_the_molecular_correction() {
        for variant in VARIANTS {
            let molecular = correction_energy(&water_molecule(), variant);
            let mut periodic = water_molecule();
            periodic.cell = Some(Cell::cubic(90.0).unwrap());
            let per_cell =
                periodic_correction_energy(&periodic, variant, &CorrectionCutoffs::default());
            assert!(
                (per_cell - molecular).abs() < 1.0e-9,
                "{variant:?}: per-cell {per_cell} vs molecular {molecular}"
            );
        }
    }

    /// The counting test: the energy per cell computed with a lattice sum must equal the energy
    /// of an `n × n × n` supercell divided by `n³`.
    ///
    /// This is what catches a term counted one and a half times, or a hydrogen bond crossing the
    /// cell edge that is credited to both cells, or one that is credited to neither. It works
    /// because the supercell calculation knows nothing about the counting rules — it just has
    /// more atoms.
    #[test]
    fn per_cell_energy_matches_the_supercell_folded_back() {
        // A dense enough water arrangement that images genuinely interact.
        let cell = Cell::cubic(11.0).unwrap();
        let base = vec![
            Atom {
                z: 8,
                position: Vec3::new(0.4, 0.3, 0.2),
            },
            Atom {
                z: 1,
                position: Vec3::new(2.2, 0.3, 0.2),
            },
            Atom {
                z: 1,
                position: Vec3::new(-0.05, 2.05, 0.2),
            },
            Atom {
                z: 8,
                position: Vec3::new(5.6, 5.2, 4.9),
            },
            Atom {
                z: 1,
                position: Vec3::new(7.4, 5.2, 4.9),
            },
            Atom {
                z: 1,
                position: Vec3::new(5.15, 6.95, 4.9),
            },
        ];
        let mut unit = Molecule::new(base.clone());
        unit.cell = Some(cell);

        for variant in VARIANTS {
            let per_cell =
                periodic_correction_energy(&unit, variant, &CorrectionCutoffs::default());

            // Build a 2×2×2 supercell of the same structure.
            let n = 2;
            let mut atoms = Vec::new();
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        let shift = cell.vector(0) * i as f64
                            + cell.vector(1) * j as f64
                            + cell.vector(2) * k as f64;
                        for atom in &base {
                            atoms.push(Atom {
                                z: atom.z,
                                position: atom.position + shift,
                            });
                        }
                    }
                }
            }
            let mut supercell = Molecule::new(atoms);
            supercell.cell = Some(Cell::cubic(11.0 * n as f64).unwrap());
            let folded =
                periodic_correction_energy(&supercell, variant, &CorrectionCutoffs::default())
                    / (n * n * n) as f64;

            assert!(
                (per_cell - folded).abs() < 1.0e-9 * per_cell.abs().max(1.0),
                "{variant:?}: per cell {per_cell} vs supercell/{}: {folded}",
                n * n * n
            );
            assert!(
                per_cell.abs() > 1.0e-4,
                "{variant:?}: the fixture produces no correction to speak of ({per_cell})"
            );
        }
    }

    /// The dispersion lattice sum has to converge in the cluster radius as `R⁻³`, and the
    /// residual at the default radius has to be negligible.
    ///
    /// `R⁻³` is not a tolerance chosen to pass: a shell at radius `R` holds `O(R²)` pairs each
    /// contributing `C₆/R⁶`, so the truncated tail is `∫R²·R⁻⁶ dR ∝ R⁻³`. Measuring the exponent
    /// says the sum is behaving like dispersion; a shallower law would mean something
    /// longer-ranged had crept in. The absolute residual is then quoted for what it is —
    /// about 1e-5 eV, or 2e-4 kcal/mol, at the 30 Bohr default.
    #[test]
    fn the_dispersion_lattice_sum_converges_as_r_cubed() {
        let mut unit = water_molecule();
        unit.cell = Some(Cell::cubic(12.0).unwrap());
        let at = |radius: f64| {
            periodic_correction_energy(
                &unit,
                Variant::Pm3D3,
                &CorrectionCutoffs {
                    dispersion: radius,
                    ..CorrectionCutoffs::default()
                },
            )
        };
        // Estimate the limit by extrapolating the R⁻³ tail from two large radii, then measure
        // how the residual falls.
        let converged = at(150.0);
        let residual = |radius: f64| (at(radius) - converged).abs();
        let (near, far) = (residual(30.0), residual(60.0));
        let exponent = (near / far).log2();
        assert!(
            (exponent - 3.0).abs() < 0.4,
            "the dispersion tail falls as R^-{exponent:.2}, not R^-3 (residuals {near:.3e}, {far:.3e})"
        );
        assert!(
            near < 5.0e-5,
            "the default 30 Bohr cutoff leaves {near:.3e} eV on the table"
        );
    }

    /// The short-range terms are cut off internally, so widening their cluster past those limits
    /// must change nothing at all.
    #[test]
    fn the_short_range_terms_are_insensitive_to_the_cluster_radius() {
        let mut unit = water_molecule();
        unit.cell = Some(Cell::cubic(12.0).unwrap());
        for variant in [Variant::Pm3D3H4, Variant::Pm3D3H4X] {
            // Hold the dispersion radius fixed so only the short-range one moves.
            let at = |radius: f64| {
                periodic_correction_energy(
                    &unit,
                    variant,
                    &CorrectionCutoffs {
                        short_range: radius,
                        ..CorrectionCutoffs::default()
                    },
                )
            };
            let difference = (at(DEFAULT_SHORT_CUTOFF) - at(2.5 * DEFAULT_SHORT_CUTOFF)).abs();
            assert!(
                difference < 1.0e-12,
                "{variant:?}: the short-range cluster radius changed the energy by {difference:.3e}"
            );
        }
    }

    /// Hydrogen bonds that cross a cell boundary must actually be counted. A cell holding one
    /// water, small enough that its images hydrogen-bond to it, has to produce an H4 term — if
    /// the triple search never left the reference cell it would be exactly zero.
    #[test]
    fn hydrogen_bonds_across_the_cell_boundary_contribute() {
        let mut unit = water_molecule();
        // Roughly the O–O distance in ice, so each water hydrogen-bonds to its own images.
        unit.cell = Some(Cell::cubic(5.3).unwrap());
        let with_images =
            periodic_correction_energy(&unit, Variant::Pm3D3H4, &CorrectionCutoffs::default());
        let molecular = correction_energy(&water_molecule(), Variant::Pm3D3H4);
        assert!(
            (with_images - molecular).abs() > 1.0e-3,
            "the periodic H4 term is indistinguishable from the isolated molecule \
             ({with_images} vs {molecular}); bonds across the boundary are being missed"
        );
    }

    /// A structure with no cell must still work, and give the molecular answer.
    #[test]
    fn a_molecule_without_a_cell_is_its_own_cluster() {
        let molecule = water_molecule();
        let cluster = build_cluster(&molecule, 30.0);
        assert_eq!(cluster.n_cell, 3);
        assert_eq!(cluster.numbers.len(), 3);
        for variant in VARIANTS {
            let via_cluster =
                periodic_correction_energy(&molecule, variant, &CorrectionCutoffs::default());
            let direct = correction_energy(&molecule, variant);
            assert!((via_cluster - direct).abs() < 1.0e-12, "{variant:?}");
        }
    }
}
