// SPDX-License-Identifier: GPL-3.0-or-later

//! Γ-point analytic forces and stress.
//!
//! # No Pulay term, and why that survives periodicity
//!
//! NDDO works in an orthonormal AO basis, so the energy is stationary with respect to the
//! density and the nuclear gradient is the derivative of the energy expression at the converged
//! density — no basis-set-superposition term, no overlap-constraint term. Making the system
//! periodic does not change that: `S(T) = δ_T0 δ_μν` still, because zero differential overlap is
//! a statement about the AOs, not about the boundary conditions. Every derivative below is
//! therefore a Hellmann–Feynman one.
//!
//! # The stress comes free with the gradient
//!
//! [`crate::dual::Dual`] carries derivatives with respect to the **pair displacement** `d`, and
//! the periodic pair displacement is `d + T`. Under a strain `ε` both scale together, so
//!
//! ```text
//! σ_αβ = (1/V) Σ_pairs (∂E/∂d_α) · (d + T)_β
//! ```
//!
//! The same per-pair derivative that gives the force gives the virial, with one extra outer
//! product and no new automatic differentiation. Only the Ewald sum needs its strain derivative
//! derived separately, and [`crate::pbc::ewald`] already returns it.
//!
//! # What is differentiated where
//!
//! | term | how |
//! |---|---|
//! | screened two-center integrals, resonance `β·S`, electron–core | forward-mode AD on `d + T` |
//! | the switch `f(r)` itself | the same AD — it multiplies the correction, so `f'` is part of the force |
//! | core–core short range | forward-mode AD on `r` |
//! | Ewald | analytic, from [`crate::pbc::ewald::EwaldOutput`] |
//! | D3/H4/X | forward-mode AD over the image cluster |
//!
//! The switch derivative is the easy one to lose. Evaluating `f` in `f64` and multiplying an
//! AD-derived correction by it gives a force that is wrong only inside the switching shell —
//! small enough to pass for noise, large enough to fail a finite-difference check.

use crate::basis::Basis;
use crate::corrections::periodic::{build_cluster, cluster_positions_g, CorrectionCutoffs};
use crate::corrections::{correction_energy_cluster_g, Variant};
use crate::dual::{Dual, Scalar};
use crate::error::{Pm3Error, Result};
use crate::integrals::{pack, pair_two_electron_g};
use crate::linalg::Matrix;
use crate::math::{Mat3, Vec3};
use crate::neighbor::NeighborList;
use crate::params::{Pm3Element, Pm3Parameters};
use crate::pbc::ewald::{ewald, ChargeSite, EwaldParams};
use crate::pbc::gamma::{run_gamma, PeriodicOptions, PeriodicResult};
use crate::pbc::multipole::AtomSites;
use crate::pbc::screen::point_pair_g;
use crate::scf::Pm3Options;
use crate::system::Molecule;

/// Forces and stress for a periodic structure, at the Γ point.
#[derive(Clone, Debug)]
pub struct PeriodicGradient {
    /// The converged SCF this was evaluated at.
    pub scf: PeriodicResult,
    /// Total energy per cell (eV).
    pub energy_ev: f64,
    /// `∂E/∂R` per atom (eV/Bohr).
    pub gradient: Vec<Vec3>,
    /// `−∂E/∂R` per atom (eV/Bohr).
    pub forces: Vec<Vec3>,
    /// `∂E/∂ε` (eV), the strain derivative, projected onto the periodic directions: any
    /// component touching a non-periodic axis is exactly zero, because that cell degree of
    /// freedom does not exist. `None` only for an isolated cell, which has no strain at all —
    /// deliberately not a zero matrix, which a caller would read as "no stress".
    pub virial: Option<Mat3>,
    /// `σ = ∂E/∂ε` divided by [`crate::cell::Cell::measure`] — a pressure in 3D (eV/Bohr³), a
    /// surface tension in 2D (eV/Bohr²), an axial tension in 1D (eV/Bohr). `None` for the same
    /// reason as `virial`.
    pub stress: Option<Mat3>,
    /// Largest gradient component magnitude (eV/Bohr).
    pub max_gradient: f64,
}

/// Where the gradient reads the density from.
///
/// Two terms — the resonance and the exchange — contract an **inter-atomic** density block, and
/// that block is `P_ij(T)` for the particular image being differentiated. At the Γ point every
/// image gets the same `P(Γ)`; with a k-mesh each gets its own. Everything else contracts an
/// on-site block, which is `P(0)` either way.
///
/// This is the same distinction the Fock build makes, and it has to be made here too or the
/// k-point forces would silently be Γ-point forces evaluated at a k-point density.
pub trait PairDensity {
    /// `P(0)` on one atom, in that atom's own orbital indices.
    fn onsite(&self, atom: usize, mu: usize, nu: usize) -> f64;
    /// The `norb_i × norb_j` block linking atom `i` in the reference cell to atom `j` in cell `t`.
    fn inter(&self, i: usize, j: usize, t: [i32; 3], norb_i: usize, norb_j: usize) -> InterBlock;
}

/// An inter-atomic density block, total and spin-resolved.
pub struct InterBlock {
    /// `P_ij(T)`, row-major with `cols` columns.
    pub total: Vec<f64>,
    /// `P^α − P^β` for the same block, when the calculation is unrestricted.
    pub spin: Option<Vec<f64>>,
    pub cols: usize,
}

impl InterBlock {
    #[inline]
    fn get(&self, mu: usize, la: usize) -> f64 {
        self.total[mu * self.cols + la]
    }

    /// `(P^α, P^β)` for one element.
    #[inline]
    fn spin_pair(&self, mu: usize, la: usize) -> (f64, f64) {
        let total = self.get(mu, la);
        match &self.spin {
            Some(spin) => {
                let difference = spin[mu * self.cols + la];
                (0.5 * (total + difference), 0.5 * (total - difference))
            }
            None => (0.5 * total, 0.5 * total),
        }
    }
}

/// The Γ-point density: one matrix, used for every image.
pub struct GammaDensity<'a> {
    pub density: &'a Matrix,
    pub spin: Option<&'a Matrix>,
    pub offsets: &'a [usize],
}

impl PairDensity for GammaDensity<'_> {
    fn onsite(&self, atom: usize, mu: usize, nu: usize) -> f64 {
        let off = self.offsets[atom];
        self.density[(off + mu, off + nu)]
    }

    fn inter(&self, i: usize, j: usize, _t: [i32; 3], norb_i: usize, norb_j: usize) -> InterBlock {
        let (oi, oj) = (self.offsets[i], self.offsets[j]);
        let mut total = vec![0.0; norb_i * norb_j];
        let mut spin = self.spin.map(|_| vec![0.0; norb_i * norb_j]);
        for mu in 0..norb_i {
            for la in 0..norb_j {
                total[mu * norb_j + la] = self.density[(oi + mu, oj + la)];
                if let (Some(target), Some(source)) = (spin.as_mut(), self.spin) {
                    target[mu * norb_j + la] = source[(oi + mu, oj + la)];
                }
            }
        }
        InterBlock {
            total,
            spin,
            cols: norb_j,
        }
    }
}

/// Converge the Γ-point SCF and evaluate the analytic forces and stress at that density.
pub fn periodic_gradient(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
) -> Result<PeriodicGradient> {
    let scf = run_gamma(molecule, params, options, periodic)?;
    let density = GammaDensity {
        density: &scf.density,
        spin: scf.spin_density.as_ref(),
        offsets: &Basis::build(molecule, params)?.atom_offset,
    };
    let (gradient, virial, stress) =
        forces_and_stress(molecule, params, options, periodic, &density, &scf.density)?;
    let forces: Vec<Vec3> = gradient.iter().map(|g| *g * -1.0).collect();
    let max_gradient = gradient
        .iter()
        .flat_map(|g| g.to_array())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    Ok(PeriodicGradient {
        energy_ev: scf.total_ev,
        scf,
        gradient,
        forces,
        virial,
        stress,
        max_gradient,
    })
}

/// The forces and the strain derivative at a fixed density.
///
/// `density` supplies the inter-atomic blocks the resonance and exchange need; `onsite_density`
/// is `P(0)` as a matrix, which is what the multipole charges and the Ewald sum are built from.
/// The two are separate arguments because only the first differs between the Γ-point and k-point
/// paths.
#[allow(clippy::type_complexity)]
pub fn forces_and_stress<D: PairDensity>(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    density: &D,
    onsite_density: &Matrix,
) -> Result<(Vec<Vec3>, Option<Mat3>, Option<Mat3>)> {
    let cell = molecule.cell.expect("a periodic gradient requires a cell");
    let nat = molecule.atoms.len();
    let basis = Basis::build(molecule, params)?;

    let mut gradient = vec![Vec3::zero(); nat];
    let mut virial = Mat3::zero();

    // --- short-range two-center terms, differentiated on the image displacement -------------
    let mut atom_sites = Vec::with_capacity(nat);
    for atom in &molecule.atoms {
        let elem = params.element(atom.z)?;
        atom_sites.push(AtomSites::build(elem).ok_or(Pm3Error::UnsupportedDOrbitals(atom.z))?);
    }
    let cutoff = periodic.short_range_cutoff.max(periodic.switch.off);
    let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let list = NeighborList::build_from_positions(&positions, Some(&cell), cutoff);

    for pair in list.unique() {
        let (a, b) = (pair.a, pair.b);
        let ea = params.element(molecule.atoms[a].z)?;
        let eb = params.element(molecule.atoms[b].z)?;
        let heavy_first = ea.n_orb >= eb.n_orb;
        let (first_index, second_index) = if heavy_first { (a, b) } else { (b, a) };
        let (first, second) = if heavy_first { (ea, eb) } else { (eb, ea) };
        let dvec = if heavy_first {
            pair.dvec
        } else {
            pair.dvec * -1.0
        };
        // The image index has to follow the orientation swap, or a k-point run would look up the
        // density of the wrong image whenever the lighter atom comes first.
        let image = if heavy_first {
            pair.t
        } else {
            [-pair.t[0], -pair.t[1], -pair.t[2]]
        };
        let seeded = [
            Dual::var(dvec.x, 0),
            Dual::var(dvec.y, 1),
            Dual::var(dvec.z, 2),
        ];

        let te = pair_two_electron_g::<Dual>(first, second, seeded);
        let point = point_pair_g::<Dual>(
            &atom_sites[first_index],
            first.core_charge,
            &atom_sites[second_index],
            second.core_charge,
            seeded,
        );
        let r_dual = dual_norm(&seeded);
        let switch = periodic.switch.at_g(r_dual);

        let mut derivative = [0.0f64; 3];
        let (na, nb) = (first.n_orb, second.n_orb);
        let inter = density.inter(first_index, second_index, image, na, nb);

        // Electron–core, both directions, screened. On-site blocks: `P(0)` either way.
        for mu in 0..na {
            for nu in 0..na {
                let coefficient = density.onsite(first_index, mu, nu);
                let term = (te.e1b[mu][nu] - point.e1b[mu * na + nu]) * switch;
                for (axis, slot) in derivative.iter_mut().enumerate() {
                    *slot += coefficient * term.d[axis];
                }
            }
        }
        for la in 0..nb {
            for si in 0..nb {
                let coefficient = density.onsite(second_index, la, si);
                let term = (te.e2a[la][si] - point.e2a[la * nb + si]) * switch;
                for (axis, slot) in derivative.iter_mut().enumerate() {
                    *slot += coefficient * term.d[axis];
                }
            }
        }

        // Resonance β·S, inside its own cutoff. Inter-atomic: this is one of the two terms that
        // needs the density *of this image*.
        let off_first = basis.atom_offset[first_index];
        let off_second = basis.atom_offset[second_index];
        if pair.r <= periodic.short_range_cutoff {
            let overlap = crate::overlap::diatom_overlap_dual(first, Vec3::zero(), second, dvec)?;
            // The index runs over three things at once — the overlap block, the AO table, and the
            // density block — so it stays an index.
            #[allow(clippy::needless_range_loop)]
            for mu in 0..na.min(4) {
                let bi = resonance_beta(first, basis.aos[off_first + mu].orb);
                for la in 0..nb.min(4) {
                    let bj = resonance_beta(second, basis.aos[off_second + la].orb);
                    // Both `(μ,λ)` and `(λ,μ)` are occupied by the same physical term, hence
                    // the density element appearing twice.
                    let coefficient = inter.get(mu, la) * (bi + bj);
                    for (axis, slot) in derivative.iter_mut().enumerate() {
                        *slot += coefficient * overlap[mu][la].d[axis];
                    }
                }
            }
        }

        // Two-electron: Coulomb from the screened table, exchange from the full one — the same
        // asymmetry the Fock build uses, for the same reason.
        let npack_j = second.n_orb * (second.n_orb + 1) / 2;
        let inside_exchange = pair.r <= periodic.short_range_cutoff;
        for mu in 0..na {
            for nu in 0..na {
                for la in 0..nb {
                    for si in 0..nb {
                        let index = pack(mu, nu) * npack_j + pack(la, si);
                        let coulomb = (te.w[index] - point.w[index]) * switch;
                        let weight = density.onsite(first_index, mu, nu)
                            * density.onsite(second_index, la, si);
                        for (axis, slot) in derivative.iter_mut().enumerate() {
                            *slot += weight * coulomb.d[axis];
                        }
                        if inside_exchange {
                            // Exchange uses the same integral element but the *full* table, and
                            // it is a sum over spin channels — which for a restricted density is
                            // the familiar `−½ P P`, since each channel carries half of it.
                            let (alpha_ml, beta_ml) = inter.spin_pair(mu, la);
                            let (alpha_ns, beta_ns) = inter.spin_pair(nu, si);
                            let exchange_weight = -(alpha_ml * alpha_ns + beta_ml * beta_ns);
                            for (axis, slot) in derivative.iter_mut().enumerate() {
                                *slot += exchange_weight * te.w[index].d[axis];
                            }
                        }
                    }
                }
            }
        }

        // Core-core: the PM3 pair energy with its Klopman-Ohno monopole switched to the point
        // form the lattice sum already carries.
        let full = crate::repulsion::pair_core_energy_scalar::<Dual>(
            params,
            first,
            second,
            molecule.atoms[first_index].z,
            molecule.atoms[second_index].z,
            r_dual,
        );
        let rho = first.po[9] + second.po[9];
        let bare = (r_dual * r_dual + rho * rho).sqrt().recip()
            * (crate::constants::PM3_EV * first.core_charge * second.core_charge);
        let point_monopole =
            r_dual.recip() * (crate::constants::PM3_EV * first.core_charge * second.core_charge);
        let core = full - bare + (bare - point_monopole) * switch;
        for (axis, slot) in derivative.iter_mut().enumerate() {
            *slot += core.d[axis];
        }

        // `derivative` is ∂E/∂d with `d` pointing from `first` to `second`.
        let contribution = Vec3::new(derivative[0], derivative[1], derivative[2]);
        gradient[second_index] += contribution;
        gradient[first_index] -= contribution;
        accumulate_outer(&mut virial, contribution, dvec);
    }

    // --- the lattice sum -------------------------------------------------------------------
    let (sites, offsets, ewald_params) = charge_sites(
        molecule,
        params,
        &basis,
        &atom_sites,
        onsite_density,
        periodic,
    )?;
    let field = ewald(&cell, &sites, &ewald_params)?;
    for (site, site_gradient) in sites.iter().zip(&field.site_gradient) {
        gradient[site.owner] += *site_gradient;
    }

    // The multipole offsets do **not** scale with the cell. `dd` and `qq` are atomic parameters,
    // fixed by the element's Slater exponents, so under a strain a site moves with its nucleus
    // and not with its own displacement from it. [`crate::pbc::ewald`] computes the strain
    // derivative on the assumption that every position it was handed scales, which is right for
    // bare point charges and wrong here by exactly
    //
    //     Σ_sites (∂E/∂r_site)_α · offset_β
    //
    // — the part of each site's motion that the strain does not actually produce. Left in, it
    // put the 3D virial 10% out and nothing but a finite difference would have shown it.
    let ewald_virial = field.virial.map(|full| {
        let mut corrected = full;
        for (offset, site_gradient) in offsets.iter().zip(&field.site_gradient) {
            accumulate_outer(&mut corrected, *site_gradient * -1.0, *offset);
        }
        corrected
    });

    // --- classical corrections --------------------------------------------------------------
    if options.variant != Variant::Pm3 {
        let (correction_gradient, correction_virial) =
            correction_derivatives(molecule, options.variant, &periodic.correction_cutoffs);
        for (slot, value) in gradient.iter_mut().zip(&correction_gradient) {
            *slot += *value;
        }
        add_in_place(&mut virial, &correction_virial);
    }

    let virial = match ewald_virial {
        Some(from_ewald) => {
            add_in_place(&mut virial, &from_ewald);
            Some(virial)
        }
        None => None,
    };
    // Keep only the strains the cell actually has. `∂E/∂ε` is a well-defined derivative in all
    // nine components — deforming a chain transversally really does move its atoms apart — but a
    // chain has no transverse cell vector to relax and a slab has no thickness to squeeze. Every
    // component touching a non-periodic axis is therefore zeroed, which is what stops a
    // variable-cell run from optimizing a degree of freedom that does not exist, and what makes
    // the Voigt tensor handed to ASE mean what ASE expects.
    let virial = virial.map(|v| project_onto_periodic(&v, &cell));
    let measure = cell.measure();
    let stress = virial.map(|v| scale(&v, 1.0 / measure));
    Ok((gradient, virial, stress))
}

/// The multipole charge sites for the converged density, cores included, plus the Ewald
/// parameters actually in force.
fn charge_sites(
    molecule: &Molecule,
    params: &Pm3Parameters,
    basis: &Basis,
    atom_sites: &[AtomSites],
    density: &Matrix,
    periodic: &PeriodicOptions,
) -> Result<(Vec<ChargeSite>, Vec<Vec3>, EwaldParams)> {
    let cell = molecule.cell.expect("periodic");
    let mut sites = Vec::new();
    let mut offsets = Vec::new();
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let n = basis.atom_norb[ia];
        let off = basis.atom_offset[ia];
        let mut block = vec![0.0; n * n];
        for mu in 0..n {
            for nu in 0..n {
                block[mu * n + nu] = density[(off + mu, off + nu)];
            }
        }
        // Cores and electrons together: the force is the derivative of the whole electrostatic
        // energy at fixed charges, and splitting it would only mean adding the halves back up.
        let charges = atom_sites[ia].charges(&block, elem.core_charge);
        for (offset, charge) in atom_sites[ia].offsets.iter().zip(&charges) {
            sites.push(ChargeSite {
                position: atom.position + *offset,
                charge: *charge,
                owner: ia,
            });
            offsets.push(*offset);
        }
    }
    let ewald_params = periodic
        .ewald
        .unwrap_or_else(|| EwaldParams::for_cell(&cell, crate::pbc::ewald::DEFAULT_ACCURACY));
    Ok((sites, offsets, ewald_params))
}

/// Forces and virial from the classical corrections, by forward-mode AD over the image cluster.
///
/// Seeding one reference-cell atom at a time and letting the cluster inherit the seed is what
/// makes an image contribute to the force on its parent — the physically required behaviour, and
/// the reason [`cluster_positions_g`] builds images from the caller's own values rather than
/// from `f64` copies.
fn correction_derivatives(
    molecule: &Molecule,
    variant: Variant,
    cutoffs: &CorrectionCutoffs,
) -> (Vec<Vec3>, Mat3) {
    let nat = molecule.atoms.len();
    let mut gradient = vec![Vec3::zero(); nat];
    let mut virial = Mat3::zero();
    let cluster = build_cluster(
        molecule,
        crate::corrections::periodic::cutoff_for(variant, cutoffs),
    );

    // The atom index addresses three different things — the seed, the cluster parent map and
    // the gradient slot — so iterating it directly is clearer than an iterator rewrite.
    #[allow(clippy::needless_range_loop)]
    for atom in 0..nat {
        let seeded: Vec<[Dual; 3]> = molecule
            .atoms
            .iter()
            .enumerate()
            .map(|(index, a)| {
                if index == atom {
                    [
                        Dual::var(a.position.x, 0),
                        Dual::var(a.position.y, 1),
                        Dual::var(a.position.z, 2),
                    ]
                } else {
                    [
                        Dual::constant(a.position.x),
                        Dual::constant(a.position.y),
                        Dual::constant(a.position.z),
                    ]
                }
            })
            .collect();
        let cluster_pos = cluster_positions_g(&cluster, &seeded);
        let energy = correction_energy_cluster_g::<Dual>(
            &cluster.numbers,
            &cluster_pos,
            cluster.n_cell,
            Some(&cluster.parent),
            None,
            Some(cutoffs.dispersion),
            Some(cutoffs.coordination),
            variant,
        );
        gradient[atom] += Vec3::new(energy.d[0], energy.d[1], energy.d[2]);
    }

    // The virial follows from the same derivatives: under a homogeneous strain every position —
    // reference-cell atoms and images alike — scales with the cell, so `∂E/∂ε_αβ = Σ_i
    // (∂E/∂r_i)_α r_iβ` summed over the cluster. Seeding the cluster this way collects the image
    // terms onto their parents, which is exactly the sum needed.
    let strain_seed: Vec<[Dual; 3]> = molecule
        .atoms
        .iter()
        .map(|a| {
            [
                Dual::constant(a.position.x),
                Dual::constant(a.position.y),
                Dual::constant(a.position.z),
            ]
        })
        .collect();
    let base = cluster_positions_g(&cluster, &strain_seed);
    for alpha in 0..3 {
        for beta in 0..3 {
            // Strain the whole cluster: r -> r + ε r, seeded in one direction at a time.
            let strained: Vec<[Dual; 3]> = base
                .iter()
                .map(|p| {
                    let mut out = *p;
                    let shift = Dual::var(0.0, 0) * p[beta].v;
                    out[alpha] = out[alpha] + shift;
                    out
                })
                .collect();
            let energy = correction_energy_cluster_g::<Dual>(
                &cluster.numbers,
                &strained,
                cluster.n_cell,
                Some(&cluster.parent),
                None,
                Some(cutoffs.dispersion),
                Some(cutoffs.coordination),
                variant,
            );
            set_component(&mut virial, alpha, beta, energy.d[0]);
        }
    }
    (gradient, virial)
}

pub(crate) fn resonance_beta(elem: &Pm3Element, orb: u8) -> f64 {
    match orb {
        0 => elem.beta_s,
        1..=3 => elem.beta_p,
        _ => elem.beta_d,
    }
}

#[inline]
fn dual_norm(d: &[Dual; 3]) -> Dual {
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

#[inline]
fn accumulate_outer(m: &mut Mat3, a: Vec3, b: Vec3) {
    m.col[0] += a * b.x;
    m.col[1] += a * b.y;
    m.col[2] += a * b.z;
}

#[inline]
fn add_in_place(target: &mut Mat3, other: &Mat3) {
    for axis in 0..3 {
        target.col[axis] += other.col[axis];
    }
}

#[inline]
/// Zero every component of a strain derivative that touches a non-periodic direction.
fn project_onto_periodic(m: &Mat3, cell: &crate::cell::Cell) -> Mat3 {
    let mut out = Mat3::zero();
    for &alpha in &cell.periodic_indices() {
        for &beta in &cell.periodic_indices() {
            let value = match alpha {
                0 => m.col[beta].x,
                1 => m.col[beta].y,
                _ => m.col[beta].z,
            };
            match alpha {
                0 => out.col[beta].x = value,
                1 => out.col[beta].y = value,
                _ => out.col[beta].z = value,
            }
        }
    }
    out
}

fn scale(m: &Mat3, factor: f64) -> Mat3 {
    Mat3::from_columns(m.col[0] * factor, m.col[1] * factor, m.col[2] * factor)
}

#[inline]
fn set_component(m: &mut Mat3, alpha: usize, beta: usize, value: f64) {
    let column = &mut m.col[beta];
    match alpha {
        0 => column.x = value,
        1 => column.y = value,
        _ => column.z = value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;
    use crate::pbc::gamma::run_gamma;

    const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";

    fn cell_with(xyz: &str, edge: f64) -> Molecule {
        let mut molecule = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        molecule.cell = Some(Cell::cubic(edge).unwrap());
        molecule
    }

    /// A dense fixture where images genuinely interact, so the derivative tests are exercising
    /// the lattice sums rather than an isolated molecule.
    fn dense_cell() -> Molecule {
        let mut molecule = Molecule::from_xyz_str(
            "6\ntwo waters\nO 0.20 0.15 0.10\nH 1.16 0.15 0.10\nH -0.04 1.08 0.10\n\
             O 2.90 2.75 2.60\nH 3.86 2.75 2.60\nH 2.66 3.68 2.60\n",
            0.0,
        )
        .unwrap();
        molecule.cell = Some(Cell::cubic(11.0).unwrap());
        molecule
    }

    fn energy_of(molecule: &Molecule, options: &Pm3Options) -> f64 {
        let params = Pm3Parameters::standard().unwrap();
        run_gamma(molecule, &params, options, &PeriodicOptions::default())
            .unwrap()
            .total_ev
    }

    /// Analytic forces against a central difference of the self-consistent energy.
    ///
    /// Differencing the *converged* energy rather than a fixed-density one is what makes this a
    /// real check: it would catch a missing orbital-relaxation term as readily as a mis-signed
    /// integral derivative, and confirms the Hellmann–Feynman claim rather than assuming it.
    #[test]
    fn forces_match_finite_differences() {
        let params = Pm3Parameters::standard().unwrap();
        let base = dense_cell();
        let options = Pm3Options::default();
        let analytic = periodic_gradient(&base, &params, &options, &PeriodicOptions::default())
            .unwrap()
            .gradient;

        let step = 2.0e-4;
        #[allow(clippy::needless_range_loop)]
        for atom in 0..base.atoms.len() {
            for axis in 0..3 {
                let mut plus = base.clone();
                let mut minus = base.clone();
                shift(&mut plus.atoms[atom].position, axis, step);
                shift(&mut minus.atoms[atom].position, axis, -step);
                let numeric =
                    (energy_of(&plus, &options) - energy_of(&minus, &options)) / (2.0 * step);
                let got = analytic[atom].get(axis);
                assert!(
                    (got - numeric).abs() < 2.0e-4 * numeric.abs().max(1.0),
                    "atom {atom} axis {axis}: analytic {got} vs numeric {numeric}"
                );
            }
        }
    }

    /// The same for a corrected variant, which adds the classical lattice sums to the derivative.
    #[test]
    fn forces_match_finite_differences_with_corrections() {
        let params = Pm3Parameters::standard().unwrap();
        let base = dense_cell();
        let options = Pm3Options {
            variant: Variant::Pm3D3H4,
            ..Pm3Options::default()
        };
        let analytic = periodic_gradient(&base, &params, &options, &PeriodicOptions::default())
            .unwrap()
            .gradient;
        let step = 2.0e-4;
        for atom in [0usize, 3] {
            for axis in 0..3 {
                let mut plus = base.clone();
                let mut minus = base.clone();
                shift(&mut plus.atoms[atom].position, axis, step);
                shift(&mut minus.atoms[atom].position, axis, -step);
                let numeric =
                    (energy_of(&plus, &options) - energy_of(&minus, &options)) / (2.0 * step);
                let got = analytic[atom].get(axis);
                assert!(
                    (got - numeric).abs() < 2.0e-4 * numeric.abs().max(1.0),
                    "atom {atom} axis {axis}: analytic {got} vs numeric {numeric}"
                );
            }
        }
    }

    /// A periodic system cannot accelerate itself: the forces must sum to zero.
    #[test]
    fn forces_sum_to_zero() {
        let params = Pm3Parameters::standard().unwrap();
        let result = periodic_gradient(
            &dense_cell(),
            &params,
            &Pm3Options::default(),
            &PeriodicOptions::default(),
        )
        .unwrap();
        let total = result.gradient.iter().fold(Vec3::zero(), |acc, g| acc + *g);
        assert!(
            total.norm() < 1.0e-6,
            "net periodic force is {total:?}, not zero"
        );
    }

    /// The stress against a central difference of the energy under strain — the only way to
    /// catch a sign or factor slip in a virial, since nothing else constrains it.
    #[test]
    fn stress_matches_strained_finite_differences() {
        let params = Pm3Parameters::standard().unwrap();
        let base = dense_cell();
        let options = Pm3Options::default();
        let analytic = periodic_gradient(&base, &params, &options, &PeriodicOptions::default())
            .unwrap()
            .virial
            .expect("3D reports a virial");

        let step = 5.0e-5;
        for alpha in 0..3 {
            for beta in 0..3 {
                let strained = |sign: f64| -> f64 {
                    let mut strain = Mat3::zero();
                    set_component(&mut strain, alpha, beta, sign * step);
                    let mut molecule = base.clone();
                    molecule.cell = Some(base.cell.unwrap().strained(&strain));
                    for atom in &mut molecule.atoms {
                        atom.position += strain.mul_vec(atom.position);
                    }
                    energy_of(&molecule, &options)
                };
                let numeric = (strained(1.0) - strained(-1.0)) / (2.0 * step);
                let got = component(&analytic, alpha, beta);
                assert!(
                    (got - numeric).abs() < 1.0e-3 * numeric.abs().max(1.0),
                    "virial[{alpha}][{beta}]: analytic {got} vs numeric {numeric}"
                );
            }
        }
    }

    /// A molecule in a large cell approaches the molecular forces as `1/L³`.
    ///
    /// Not to a fixed tolerance: water is polar, so what remains is the force its own dipole
    /// images exert on it — about 7e-5 eV/Bohr at 60 Bohr, consistent with the 1.27e-4 eV image
    /// *energy* measured at the same cell. Asserting the exponent identifies that residual as
    /// the dipole term rather than merely bounding it; a leftover monopole error would fall off
    /// as `1/L²` in the force, and a real bug would not fall off at all.
    #[test]
    fn a_molecule_in_a_large_cell_approaches_the_molecular_forces() {
        let params = Pm3Parameters::standard().unwrap();
        let options = Pm3Options::default();
        let molecular = crate::gradient::closed_form_gradient(
            &Molecule::from_xyz_str(WATER, 0.0).unwrap(),
            &params,
            &options,
        )
        .unwrap();
        let deviation = |edge: f64| -> f64 {
            let periodic = periodic_gradient(
                &cell_with(WATER, edge),
                &params,
                &options,
                &PeriodicOptions::default(),
            )
            .unwrap();
            periodic
                .gradient
                .iter()
                .zip(&molecular.gradient)
                .map(|(a, b)| (*a - *b).norm())
                .fold(0.0_f64, f64::max)
        };
        let (near, far) = (deviation(40.0), deviation(80.0));
        let exponent = (near / far).log2();
        assert!(
            (exponent - 3.0).abs() < 0.4,
            "the residual force falls as 1/L^{exponent:.2}, not 1/L³ ({near:.3e}, {far:.3e})"
        );
        assert!(far < 5.0e-5, "residual force {far:.3e} eV/Bohr at 80 Bohr");
    }

    /// A chain's axial stress against finite differences of the SCF energy under strain.
    ///
    /// This is the end-to-end statement, not the lattice sum's own: it goes through the SCF, the
    /// screened short-range tables, the switch, the core–core term and the 1D sum together, and
    /// it is what says the pieces share one convention. Only the axial component is compared,
    /// because only the axial one exists.
    #[test]
    fn a_chain_reports_an_axial_stress_that_matches_finite_differences() {
        let params = Pm3Parameters::standard().unwrap();
        let options = Pm3Options::default();
        let periodic = PeriodicOptions::default();

        // Trans-polyacetylene: a real 1D system with a real axial tension.
        let axis = 4.7;
        let mut chain = Molecule::from_xyz_str(
            "4\nchain\nC 0.0 0.0 0.0\nH 0.0 1.09 0.0\nC 1.24 -0.4 0.0\nH 1.24 -1.49 0.0\n",
            0.0,
        )
        .unwrap();
        chain.cell = Some(
            Cell::new(
                Vec3::new(axis, 0.0, 0.0),
                Vec3::new(0.0, 40.0, 0.0),
                Vec3::new(0.0, 0.0, 40.0),
                [true, false, false],
            )
            .unwrap(),
        );

        let analytic = periodic_gradient(&chain, &params, &options, &periodic).unwrap();
        let virial = analytic
            .virial
            .expect("a chain now reports an axial strain derivative");

        let step = 2.0e-4;
        let energy_at = |signed: f64| {
            let mut strain = Mat3::zero();
            strain.col[0].x = signed * step;
            let mut strained = chain.clone();
            strained.cell = Some(chain.cell.unwrap().strained(&strain));
            for atom in &mut strained.atoms {
                atom.position += strain.mul_vec(atom.position);
            }
            run_gamma(&strained, &params, &options, &periodic)
                .unwrap()
                .total_ev
        };
        let numeric = (energy_at(1.0) - energy_at(-1.0)) / (2.0 * step);
        let got = virial.col[0].x;
        assert!(
            (got - numeric).abs() < 2.0e-4 * numeric.abs().max(1.0),
            "chain axial virial: analytic {got} vs finite difference {numeric}"
        );

        // Everything that is not the periodic direction is zero, because a chain has no
        // transverse cell vector to relax.
        for (label, value) in [
            ("yy", virial.col[1].y),
            ("zz", virial.col[2].z),
            ("xy", virial.col[1].x),
            ("xz", virial.col[2].x),
        ] {
            assert_eq!(
                value, 0.0,
                "chain virial {label} should be absent, got {value}"
            );
        }
    }

    /// A **slab's** in-plane stress against finite differences of the SCF energy under strain.
    ///
    /// The end-to-end statement, as for the chain: through the SCF, the screened short-range
    /// tables, the switch, the core–core term and the Parry sum together. Only the two in-plane
    /// components exist — a slab's `z` extent is padding, not a degree of freedom — and the rest
    /// must be exactly zero rather than small, because a variable-cell optimizer reads a small
    /// number as a direction to move in.
    #[test]
    fn a_slab_reports_an_in_plane_stress_that_matches_finite_differences() {
        let params = Pm3Parameters::standard().unwrap();
        let options = Pm3Options::default();
        let periodic = PeriodicOptions::default();

        let mut slab = Molecule::from_xyz_str(WATER, 0.0).unwrap();
        slab.cell = Some(
            Cell::new(
                Vec3::new(11.0, 0.0, 0.0),
                Vec3::new(0.0, 11.5, 0.0),
                Vec3::new(0.0, 0.0, 60.0),
                [true, true, false],
            )
            .unwrap(),
        );

        let analytic = periodic_gradient(&slab, &params, &options, &periodic).unwrap();
        let virial = analytic
            .virial
            .expect("a slab now reports an in-plane strain derivative");

        let step = 2.0e-4;
        for (alpha, beta) in [(0usize, 0usize), (1, 1), (0, 1)] {
            let energy_at = |signed: f64| {
                let mut strain = Mat3::zero();
                set_component(&mut strain, alpha, beta, signed * step);
                let mut strained = slab.clone();
                strained.cell = Some(slab.cell.unwrap().strained(&strain));
                for atom in &mut strained.atoms {
                    atom.position += strain.mul_vec(atom.position);
                }
                run_gamma(&strained, &params, &options, &periodic)
                    .unwrap()
                    .total_ev
            };
            let numeric = (energy_at(1.0) - energy_at(-1.0)) / (2.0 * step);
            let got = component(&virial, alpha, beta);
            assert!(
                (got - numeric).abs() < 2.0e-4 * numeric.abs().max(1.0),
                "slab virial[{alpha}][{beta}]: analytic {got} vs finite difference {numeric}"
            );
        }

        // The out-of-plane block is not a degree of freedom and must be absent, not merely small.
        for (label, value) in [
            ("zz", virial.col[2].z),
            ("xz", virial.col[2].x),
            ("zy", virial.col[1].z),
        ] {
            assert_eq!(
                value, 0.0,
                "slab virial {label} should be absent, got {value}"
            );
        }

        // The forces are still available and still balance.
        let total = analytic
            .gradient
            .iter()
            .fold(Vec3::zero(), |acc, g| acc + *g);
        assert!(total.norm() < 1.0e-6);
    }

    /// Zero dimensions is the one case with no strain at all, and it still has to say so rather
    /// than hand back zeros a variable-cell optimizer would read as "already converged".
    #[test]
    fn an_isolated_cell_reports_no_stress() {
        let params = Pm3Parameters::standard().unwrap();
        let mut molecule = Molecule::from_xyz_str(WATER, 0.0).unwrap();
        molecule.cell = Some(Cell::isolated());
        let result = periodic_gradient(
            &molecule,
            &params,
            &Pm3Options::default(),
            &PeriodicOptions::default(),
        )
        .unwrap();
        assert!(result.virial.is_none());
        assert!(result.stress.is_none());
    }

    fn shift(v: &mut Vec3, axis: usize, delta: f64) {
        match axis {
            0 => v.x += delta,
            1 => v.y += delta,
            _ => v.z += delta,
        }
    }

    fn component(m: &Mat3, alpha: usize, beta: usize) -> f64 {
        m.col[beta].get(alpha)
    }
}
