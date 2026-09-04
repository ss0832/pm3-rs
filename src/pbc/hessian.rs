// SPDX-License-Identifier: GPL-3.0-or-later

//! Γ-point analytic Hessian and phonons for periodic systems.
//!
//! # What this is, in phonon language
//!
//! The matrix built here is the dynamical matrix at `q = 0`:
//!
//! ```text
//! D(0)_{κα,κ'β} = Σ_T Φ(0κα, Tκ'β)
//! ```
//!
//! — the second derivative of the energy per cell with respect to displacing *every copy* of atom
//! `κ` together. That is what a Γ-point calculation can produce and all it can produce: a
//! displacement pattern with a wavevector needs the density's response at that wavevector, which
//! one k-point cannot represent, for the same reason spelled out in [`crate::pbc::gamma`].
//!
//! # Two halves
//!
//! ```text
//! ∂²E/∂R∂R  =  skeleton (density held fixed)  +  response (the density moves)
//! ```
//!
//! The split is not a numerical convenience — it is what makes the two testable apart. The
//! skeleton is pure differentiation of the integrals and can be checked term by term; the response
//! is a linear solve whose right-hand side is the skeleton derivative of the Fock matrix.
//!
//! | contribution | how |
//! |---|---|
//! | two-center integrals, resonance, electron–core, core–core | second-order AD ([`Dual2`]) on the image displacement `d + T` |
//! | Ewald, at fixed charges | analytic, [`crate::pbc::ewald_hessian`] |
//! | D3/H4/X | second-order AD over the image cluster |
//! | density response | CPHF, with the periodic two-electron kernel |
//!
//! # Why a self-image contributes nothing
//!
//! A pair `(a, a, T)` has displacement `d = T·h`, which does not depend on where atom `a` is —
//! both endpoints move together. Its derivatives are therefore identically zero, and the scatter
//! below produces that automatically: `+H` on `(a,a)` twice and `−H` on `(a,a)` twice.
//!
//! # The acoustic sum rule
//!
//! Every term here is a function of interatomic displacements alone, so each one is individually
//! translation-invariant and the acoustic sum rule holds by construction rather than by
//! enforcement. [`enforce_acoustic_sum_rule`] exists to clean up rounding, not to rescue a missing
//! term — if it changes the result by more than roundoff, something is wrong.

use rayon::prelude::*;

use crate::basis::Basis;
use crate::corrections::periodic::{build_cluster, cluster_positions_g, CorrectionCutoffs};
use crate::corrections::{correction_energy_cluster_g, Variant};
use crate::dual::Scalar;
use crate::dual2::Dual2;
use crate::error::{Pm3Error, Result};
use crate::integrals::{pack, pair_two_electron_g};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::neighbor::NeighborList;
use crate::params::Pm3Parameters;
use crate::pbc::ewald_hessian::ewald_atom_hessian;
use crate::pbc::gamma::{run_gamma, PeriodicOptions, PeriodicResult};
use crate::pbc::multipole::AtomSites;
use crate::pbc::screen::point_pair_g;
use crate::scf::Pm3Options;
use crate::system::Molecule;

/// A Γ-point phonon calculation.
#[derive(Clone, Debug)]
pub struct PeriodicPhonons {
    /// The converged SCF the Hessian was evaluated at.
    pub scf: PeriodicResult,
    /// `∂²E/∂R∂R` per cell (eV/Bohr²) — the dynamical matrix at `q = 0`, before mass weighting.
    pub hessian: Matrix,
    /// Γ-point phonon frequencies (cm⁻¹), ascending. Negative values denote imaginary modes.
    pub frequencies_cm: Vec<f64>,
    /// Eigenvalues of the mass-weighted matrix, in the same order.
    pub eigenvalues: Vec<f64>,
    /// The three smallest `|ω|`, which are the acoustic modes and should be zero. Reported so a
    /// caller can see how well that came out instead of having to trust it.
    pub acoustic_residual_cm: f64,
}

/// The Γ-point Hessian: `∂²E/∂R_iα ∂R_jβ` per cell, in eV/Bohr².
pub fn periodic_hessian(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
) -> Result<Matrix> {
    let scf = run_gamma(molecule, params, options, periodic)?;
    hessian_at(molecule, params, options, periodic, &scf)
}

/// Γ-point phonon frequencies, from the analytic Hessian.
pub fn periodic_phonons(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
) -> Result<PeriodicPhonons> {
    let scf = run_gamma(molecule, params, options, periodic)?;
    let mut hessian = hessian_at(molecule, params, options, periodic, &scf)?;
    enforce_acoustic_sum_rule(&mut hessian);
    let (frequencies_cm, eigenvalues) = frequencies(molecule, params, &hessian)?;
    let mut acoustic: Vec<f64> = frequencies_cm.iter().map(|f| f.abs()).collect();
    acoustic.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let acoustic_residual_cm = acoustic.get(2).copied().unwrap_or(0.0);
    Ok(PeriodicPhonons {
        scf,
        hessian,
        frequencies_cm,
        eigenvalues,
        acoustic_residual_cm,
    })
}

fn hessian_at(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    scf: &PeriodicResult,
) -> Result<Matrix> {
    if scf.unrestricted {
        return Err(Pm3Error::InvalidInput(
            "the periodic analytic Hessian is restricted-only so far; an open-shell cell needs \
             the unrestricted CPHF response, which is not wired up yet"
                .to_string(),
        ));
    }
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let basis = Basis::build(molecule, params)?;

    let mut hessian = skeleton(molecule, params, options, periodic, scf, &basis)?;

    // The lattice sum's own second derivative, at fixed charges.
    let cell = molecule.cell.expect("run_gamma requires a cell");
    let (sites, ewald_params) = charge_sites(molecule, params, &basis, &scf.density, periodic)?;
    if let Some(from_ewald) = ewald_atom_hessian(&cell, &sites, &ewald_params, nat)? {
        for i in 0..ndof {
            for j in 0..ndof {
                hessian[(i, j)] += from_ewald[(i, j)];
            }
        }
    } else {
        // 1D and 2D answer through the phased sum at q = 0 now; only an isolated cell reports
        // no Ewald Hessian, and it has a better home than this path.
        return Err(Pm3Error::InvalidInput(
            "the cell has no periodic direction; use the molecular Hessian for an isolated \
             system"
                .to_string(),
        ));
    }

    // The classical corrections, lattice summed.
    if options.variant != Variant::Pm3 {
        let from_corrections =
            correction_hessian(molecule, options.variant, &periodic.correction_cutoffs);
        for i in 0..ndof {
            for j in 0..ndof {
                hessian[(i, j)] += from_corrections[(i, j)];
            }
        }
    }

    // The density's response.
    response(molecule, params, periodic, scf, &basis, &mut hessian)?;

    // Average away the asymmetry that different summation orders leave behind.
    for i in 0..ndof {
        for j in (i + 1)..ndof {
            let mean = 0.5 * (hessian[(i, j)] + hessian[(j, i)]);
            hessian[(i, j)] = mean;
            hessian[(j, i)] = mean;
        }
    }
    Ok(hessian)
}

/// One pair's contribution: the two atom indices and the `3×3` block of its displacement Hessian.
pub(crate) type PairBlock = (usize, usize, [[f64; 3]; 3]);

/// Scatter a pair's `3×3` displacement Hessian onto the four atom blocks it belongs to.
///
/// For `E(d)` with `d = R_b − R_a`, `∂²E/∂R_a∂R_a = ∂²E/∂R_b∂R_b = +H` and the mixed blocks are
/// `−H`. A self-image, where `a == b`, cancels to zero — which is right, since its displacement
/// does not depend on the atom's position at all.
fn scatter_pair(hessian: &mut Matrix, a: usize, b: usize, block: &[[f64; 3]; 3]) {
    for (alpha, row) in block.iter().enumerate() {
        for (beta, value) in row.iter().enumerate() {
            hessian[(3 * a + alpha, 3 * a + beta)] += value;
            hessian[(3 * b + alpha, 3 * b + beta)] += value;
            hessian[(3 * a + alpha, 3 * b + beta)] -= value;
            hessian[(3 * b + alpha, 3 * a + beta)] -= value;
        }
    }
}

/// The fixed-density second derivative of everything that depends on an image displacement.
pub(crate) fn skeleton(
    molecule: &Molecule,
    params: &Pm3Parameters,
    _options: &Pm3Options,
    periodic: &PeriodicOptions,
    scf: &PeriodicResult,
    basis: &Basis,
) -> Result<Matrix> {
    let cell = molecule.cell.expect("run_gamma requires a cell");
    let nat = molecule.atoms.len();
    let mut hessian = Matrix::zeros(3 * nat, 3 * nat);

    let mut atom_sites = Vec::with_capacity(nat);
    for atom in &molecule.atoms {
        let elem = params.element(atom.z)?;
        atom_sites.push(AtomSites::build(elem).ok_or(Pm3Error::UnsupportedDOrbitals(atom.z))?);
    }
    let cutoff = periodic.short_range_cutoff.max(periodic.switch.off);
    let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let list = NeighborList::build_from_positions(&positions, Some(&cell), cutoff);

    let pairs: Vec<_> = list.unique().collect();
    let blocks: Vec<Result<PairBlock>> = pairs
        .par_iter()
        .map(|pair| {
            pair_block(
                molecule,
                params,
                periodic,
                &scf.density,
                &[(&scf.density, 0.5)],
                basis,
                &atom_sites,
                pair.a,
                pair.b,
                pair.dvec,
                pair.r,
            )
        })
        .collect();
    for entry in blocks {
        let (a, b, block) = entry?;
        scatter_pair(&mut hessian, a, b, &block);
    }
    Ok(hessian)
}

/// One image pair's `3×3` second derivative with respect to its displacement.
///
/// The same terms the gradient differentiates once, differentiated twice. Keeping them in one
/// place is deliberate: the switch, the point-charge subtraction and the exchange cutoff all have
/// to agree between the energy, the gradient and the Hessian, and three copies of that agreement
/// is two too many to keep straight.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pair_block(
    molecule: &Molecule,
    params: &Pm3Parameters,
    periodic: &PeriodicOptions,
    // The density, not the whole result: this is a fixed-density second derivative, and taking
    // the matrix rather than the PeriodicResult is what lets the k-point path reuse it.
    density: &Matrix,
    // What **exchange** reads, and by how much: `[(P, ½)]` for a closed shell and
    // `[(P^α, 1), (P^β, 1)]` for an open one. The two agree exactly when `P^α = P^β = P/2`, which
    // is what keeps the restricted path bit-for-bit unchanged. Coulomb always reads `density`.
    exchange: &[(&Matrix, f64)],
    basis: &Basis,
    atom_sites: &[AtomSites],
    a: usize,
    b: usize,
    dvec_in: Vec3,
    r: f64,
) -> Result<PairBlock> {
    use crate::pbc::gradient::resonance_beta;

    let ea = params.element(molecule.atoms[a].z)?;
    let eb = params.element(molecule.atoms[b].z)?;
    let heavy_first = ea.n_orb >= eb.n_orb;
    let (first_index, second_index) = if heavy_first { (a, b) } else { (b, a) };
    let (first, second) = if heavy_first { (ea, eb) } else { (eb, ea) };
    let dvec = if heavy_first { dvec_in } else { dvec_in * -1.0 };

    let seeded = [
        Dual2::var(dvec.x, 0),
        Dual2::var(dvec.y, 1),
        Dual2::var(dvec.z, 2),
    ];
    let te = pair_two_electron_g::<Dual2>(first, second, seeded);
    let point = point_pair_g::<Dual2>(
        &atom_sites[first_index],
        first.core_charge,
        &atom_sites[second_index],
        second.core_charge,
        seeded,
    );
    let r_dual = (seeded[0] * seeded[0] + seeded[1] * seeded[1] + seeded[2] * seeded[2]).sqrt();
    let switch = periodic.switch.at_g(r_dual);

    let (na, nb) = (first.n_orb, second.n_orb);
    let off_first = basis.atom_offset[first_index];
    let off_second = basis.atom_offset[second_index];

    // Accumulate the whole pair energy as one Dual2, then read its Hessian off at the end. That
    // is cheaper than carrying a 3×3 accumulator through every term and, more to the point, it
    // cannot get the bookkeeping wrong.
    let mut energy = Dual2::constant(0.0);

    for mu in 0..na {
        for nu in 0..na {
            let coefficient = density[(off_first + mu, off_first + nu)];
            energy = energy + (te.e1b[mu][nu] - point.e1b[mu * na + nu]) * switch * coefficient;
        }
    }
    for la in 0..nb {
        for si in 0..nb {
            let coefficient = density[(off_second + la, off_second + si)];
            energy = energy + (te.e2a[la][si] - point.e2a[la * nb + si]) * switch * coefficient;
        }
    }

    let inside_short_range = r <= periodic.short_range_cutoff;
    if inside_short_range {
        let overlap = crate::overlap::diatom_overlap_dual2(first, Vec3::zero(), second, dvec)?;
        #[allow(clippy::needless_range_loop)]
        for mu in 0..na.min(4) {
            let bi = resonance_beta(first, basis.aos[off_first + mu].orb);
            for la in 0..nb.min(4) {
                let bj = resonance_beta(second, basis.aos[off_second + la].orb);
                let coefficient = density[(off_first + mu, off_second + la)] * (bi + bj);
                energy = energy + overlap[mu][la] * coefficient;
            }
        }
    }

    let npack_j = nb * (nb + 1) / 2;
    for mu in 0..na {
        for nu in 0..na {
            for la in 0..nb {
                for si in 0..nb {
                    let index = pack(mu, nu) * npack_j + pack(la, si);
                    let weight = density[(off_first + mu, off_first + nu)]
                        * density[(off_second + la, off_second + si)];
                    energy = energy + (te.w[index] - point.w[index]) * switch * weight;
                    if inside_short_range {
                        let mut weight = 0.0;
                        for (spin, scale) in exchange {
                            weight -= scale
                                * spin[(off_first + mu, off_second + la)]
                                * spin[(off_first + nu, off_second + si)];
                        }
                        energy = energy + te.w[index] * weight;
                    }
                }
            }
        }
    }

    let full = crate::repulsion::pair_core_energy_scalar::<Dual2>(
        params,
        first,
        second,
        molecule.atoms[first_index].z,
        molecule.atoms[second_index].z,
        r_dual,
    );
    let rho = first.po[9] + second.po[9];
    let charge_product = crate::constants::PM3_EV * first.core_charge * second.core_charge;
    let bare = (r_dual * r_dual + rho * rho).sqrt().recip() * charge_product;
    let point_monopole = r_dual.recip() * charge_product;
    energy = energy + full - bare + (bare - point_monopole) * switch;

    let mut block = [[0.0; 3]; 3];
    for (alpha, row) in block.iter_mut().enumerate() {
        for (beta, slot) in row.iter_mut().enumerate() {
            *slot = energy.h[alpha][beta];
        }
    }
    Ok((first_index, second_index, block))
}

/// The classical corrections' second derivative, lattice summed.
///
/// Structurally the molecular version with the cluster standing in for the molecule: seeding a
/// *parent* atom's coordinate propagates the seed to every image of it, which is exactly right,
/// because moving an atom moves all of its copies.
fn correction_hessian(
    molecule: &Molecule,
    variant: Variant,
    cutoffs: &CorrectionCutoffs,
) -> Matrix {
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let mut hessian = Matrix::zeros(ndof, ndof);
    if variant == Variant::Pm3 {
        return hessian;
    }
    let cluster = build_cluster(molecule, cutoffs.dispersion.max(cutoffs.short_range));
    let base: Vec<[Dual2; 3]> = cluster
        .positions
        .iter()
        .map(|p| {
            [
                Dual2::constant(p[0]),
                Dual2::constant(p[1]),
                Dual2::constant(p[2]),
            ]
        })
        .collect();
    let energy_of = |seeded: &[[Dual2; 3]]| -> Dual2 {
        let positions = cluster_positions_g::<Dual2>(&cluster, seeded);
        correction_energy_cluster_g::<Dual2>(
            &cluster.numbers,
            &positions,
            cluster.n_cell,
            Some(&cluster.parent),
            None,
            Some(cutoffs.dispersion),
            Some(cutoffs.coordination),
            variant,
        )
    };

    // Diagonal blocks: all three coordinates of one atom at once.
    let diagonal: Vec<[[f64; 3]; 3]> = (0..nat)
        .into_par_iter()
        .map(|a| {
            let mut seeded = base.clone();
            seeded[a] = [
                Dual2::var(cluster.positions[a][0], 0),
                Dual2::var(cluster.positions[a][1], 1),
                Dual2::var(cluster.positions[a][2], 2),
            ];
            energy_of(&seeded).h
        })
        .collect();
    for (a, block) in diagonal.iter().enumerate() {
        for (k, row) in block.iter().enumerate() {
            for (l, value) in row.iter().enumerate() {
                hessian[(3 * a + k, 3 * a + l)] = *value;
            }
        }
    }

    // Off-diagonal blocks: one coordinate of each atom, read off the mixed derivative.
    let pairs: Vec<(usize, usize)> = (0..nat)
        .flat_map(|a| ((a + 1)..nat).map(move |b| (a, b)))
        .collect();
    let blocks: Vec<[[f64; 3]; 3]> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let mut block = [[0.0; 3]; 3];
            for (k, row) in block.iter_mut().enumerate() {
                for (l, slot) in row.iter_mut().enumerate() {
                    let mut seeded = base.clone();
                    seeded[a][k] = Dual2::var(cluster.positions[a][k], 0);
                    seeded[b][l] = Dual2::var(cluster.positions[b][l], 1);
                    *slot = energy_of(&seeded).h[0][1];
                }
            }
            block
        })
        .collect();
    for (&(a, b), block) in pairs.iter().zip(&blocks) {
        for (k, row) in block.iter().enumerate() {
            for (l, value) in row.iter().enumerate() {
                hessian[(3 * a + k, 3 * b + l)] = *value;
                hessian[(3 * b + l, 3 * a + k)] = *value;
            }
        }
    }
    hessian
}

/// The multipole charge sites at the converged density, cores included.
pub(crate) fn charge_sites(
    molecule: &Molecule,
    params: &Pm3Parameters,
    basis: &Basis,
    density: &Matrix,
    periodic: &PeriodicOptions,
) -> Result<(
    Vec<crate::pbc::ewald::ChargeSite>,
    crate::pbc::ewald::EwaldParams,
)> {
    let cell = molecule.cell.expect("periodic");
    let mut sites = Vec::new();
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
        let atom_sites = AtomSites::build(elem).ok_or(Pm3Error::UnsupportedDOrbitals(atom.z))?;
        let charges = atom_sites.charges(&block, elem.core_charge);
        for (offset, charge) in atom_sites.offsets.iter().zip(&charges) {
            sites.push(crate::pbc::ewald::ChargeSite {
                position: atom.position + *offset,
                charge: *charge,
                owner: ia,
            });
        }
    }
    let ewald_params = periodic.ewald.unwrap_or_else(|| {
        crate::pbc::ewald::EwaldParams::for_cell(&cell, crate::pbc::ewald::DEFAULT_ACCURACY)
    });
    Ok((sites, ewald_params))
}

/// Zero the residual net force on the whole cell, row by row.
///
/// Every term in the Hessian is a function of interatomic displacements, so each row-block already
/// sums to zero up to rounding. This removes that rounding, which matters because the acoustic
/// frequencies come out as the square root of nearly-cancelling numbers.
pub fn enforce_acoustic_sum_rule(hessian: &mut Matrix) {
    let ndof = hessian.rows;
    let nat = ndof / 3;
    for row in 0..ndof {
        for beta in 0..3 {
            let total: f64 = (0..nat).map(|atom| hessian[(row, 3 * atom + beta)]).sum();
            let share = total / nat as f64;
            for atom in 0..nat {
                hessian[(row, 3 * atom + beta)] -= share;
            }
        }
    }
}

/// Mass-weight and diagonalize, returning frequencies in cm⁻¹ (negative = imaginary).
fn frequencies(
    molecule: &Molecule,
    params: &Pm3Parameters,
    hessian: &Matrix,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let mut masses = Vec::with_capacity(nat);
    for atom in &molecule.atoms {
        masses.push(params.element(atom.z)?.mass);
    }
    // eV/Bohr² → eV/(Å²·amu), the units the cm⁻¹ conversion is defined against.
    let per_angstrom_squared =
        crate::constants::ANGSTROM_TO_BOHR * crate::constants::ANGSTROM_TO_BOHR;
    let mut weighted = Matrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            let scale = (masses[i / 3] * masses[j / 3]).sqrt();
            weighted[(i, j)] = if scale > 0.0 {
                hessian[(i, j)] * per_angstrom_squared / scale
            } else {
                0.0
            };
        }
    }
    let (eigenvalues, _) = symmetric_eigen(&weighted)?;
    let frequencies_cm = eigenvalues
        .iter()
        .map(|value| {
            let magnitude = value.abs().sqrt() * crate::hessian::SQRT_EV_PER_ANG2_AMU_TO_CM;
            if *value < 0.0 {
                -magnitude
            } else {
                magnitude
            }
        })
        .collect();
    Ok((frequencies_cm, eigenvalues))
}

/// The CPHF response: the part of the second derivative that comes from the density moving.
fn response(
    molecule: &Molecule,
    params: &Pm3Parameters,
    periodic: &PeriodicOptions,
    scf: &PeriodicResult,
    basis: &Basis,
    hessian: &mut Matrix,
) -> Result<()> {
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let nao = basis.nao;
    let n_occ = scf.n_occ;
    let nvir = nao - n_occ;
    if n_occ == 0 || nvir == 0 {
        return Ok(());
    }

    let co = columns(&scf.mo_coeff, 0, n_occ);
    let cv = columns(&scf.mo_coeff, n_occ, nvir);
    let mut denominator = Matrix::zeros(nvir, n_occ);
    for a in 0..nvir {
        for i in 0..n_occ {
            denominator[(a, i)] = scf.mo_energies[i] - scf.mo_energies[n_occ + a];
        }
    }

    // The skeleton derivative of the Fock matrix, projected to the occupied–virtual block. This is
    // the right-hand side of the CPHF equations and also, contracted with the response, the
    // relaxation part of the Hessian.
    let kernel = super::kernel::PeriodicKernel::build(molecule, params, periodic)?;
    let g_ov: Vec<Matrix> = (0..ndof)
        .into_par_iter()
        .map(|dof| {
            let derivative =
                fock_derivative(molecule, params, periodic, scf, basis, dof / 3, dof % 3)?;
            Ok(project_ov(&derivative, &cv, &co))
        })
        .collect::<Result<Vec<_>>>()?;

    let responses: Vec<Matrix> = g_ov
        .par_iter()
        .map(|rhs| cphf(rhs, &denominator, &cv, &co, &kernel))
        .collect::<Result<Vec<_>>>()?;

    for (b, u) in responses.iter().enumerate() {
        for (a, g) in g_ov.iter().enumerate() {
            hessian[(a, b)] += 4.0 * g.frobenius_dot(u);
        }
    }
    Ok(())
}

fn columns(c: &Matrix, start: usize, count: usize) -> Matrix {
    let mut out = Matrix::zeros(c.rows, count);
    for i in 0..c.rows {
        for j in 0..count {
            out[(i, j)] = c[(i, start + j)];
        }
    }
    out
}

fn project_ov(f: &Matrix, cv: &Matrix, co: &Matrix) -> Matrix {
    let m = f.matmul_seq(co);
    cv.transpose_matmul_seq(&m)
}

/// Solve the CPHF equations for one perturbation.
fn cphf(
    g_ov: &Matrix,
    denominator: &Matrix,
    cv: &Matrix,
    co: &Matrix,
    kernel: &super::kernel::PeriodicKernel,
) -> Result<Matrix> {
    let divide = |numerator: &Matrix| -> Matrix {
        let mut u = numerator.clone();
        for (value, d) in u.as_mut_slice().iter_mut().zip(denominator.as_slice()) {
            *value = if d.abs() < 1.0e-10 { 0.0 } else { *value / *d };
        }
        u
    };
    let mut u = divide(g_ov);
    // Pulay-extrapolated and *checked*, as the molecular solvers now are. This one ran its two
    // hundred passes and returned whatever it was holding, so a periodic Hessian could come back
    // built on a response that had not converged, with nothing in the result to say so.
    let mut hist_u: Vec<Matrix> = Vec::new();
    let mut hist_e: Vec<Matrix> = Vec::new();
    let max_diis = 8;
    let mut converged = false;
    let mut residual = f64::INFINITY;
    for _ in 0..PERIODIC_CPHF_ITERATIONS {
        // The AO-basis response density for the current `U`, symmetrized and scaled the way the
        // closed-shell density is.
        let response_density = ao_response_density(&u, cv, co);
        let applied = kernel.apply(&response_density)?;
        let projected = project_ov(&applied, cv, co);
        let mut rhs = g_ov.clone();
        for (value, extra) in rhs.as_mut_slice().iter_mut().zip(projected.as_slice()) {
            *value += extra;
        }
        let next = divide(&rhs);
        let mut error = next.clone();
        let mut change = 0.0;
        for (value, old) in error.as_mut_slice().iter_mut().zip(u.as_slice()) {
            *value -= *old;
            change += *value * *value;
        }
        residual = change.sqrt();
        if residual < PERIODIC_CPHF_TOLERANCE {
            u = next;
            converged = true;
            break;
        }
        hist_u.push(next.clone());
        hist_e.push(error);
        if hist_u.len() > max_diis {
            hist_u.remove(0);
            hist_e.remove(0);
        }
        u = crate::hessian::cphf_diis(&hist_u, &hist_e).unwrap_or(next);
    }
    if !converged {
        return Err(Pm3Error::ScfNotConverged {
            iterations: PERIODIC_CPHF_ITERATIONS,
            error: residual,
        });
    }
    Ok(u)
}

/// How many passes the Γ-point periodic response takes before it has failed.
///
/// Headroom, for the same reason as [`crate::hessian::CPHF_ITERATIONS`], and with the same
/// history: at two hundred, and with no extrapolation at all, a water chain's response stopped
/// at a residual of `1.2e-7` against a declared tolerance of `1e-10` and returned it as though
/// it had converged. This loop now extrapolates with the shared Pulay solve and checks that it
/// arrived.
const PERIODIC_CPHF_ITERATIONS: usize = 400;

/// Convergence on the RMS change of the response between passes.
const PERIODIC_CPHF_TOLERANCE: f64 = 1.0e-10;

/// `∂P = 4 Cv U Coᵀ`, symmetrized — the closed-shell response density in the AO basis.
fn ao_response_density(u: &Matrix, cv: &Matrix, co: &Matrix) -> Matrix {
    let temporary = cv.matmul_seq(u);
    let mut out = temporary.matmul_transpose_seq(co);
    let n = out.rows;
    for i in 0..n {
        for j in 0..i {
            // The *sum* of the two triangles, not their mean: the diagonal below takes ×4 and
            // an off-diagonal pair takes ×2 of the sum, which is the same ×4 spread over the two
            // entries. Naming it `mean` invited the wrong fix.
            let symmetrized = out[(i, j)] + out[(j, i)];
            out[(i, j)] = 2.0 * symmetrized;
            out[(j, i)] = 2.0 * symmetrized;
        }
        out[(i, i)] *= 4.0;
    }
    out
}

/// `∂F/∂R_{atom,axis}` at fixed density, in the AO basis.
fn fock_derivative(
    molecule: &Molecule,
    params: &Pm3Parameters,
    periodic: &PeriodicOptions,
    scf: &PeriodicResult,
    basis: &Basis,
    atom: usize,
    axis: usize,
) -> Result<Matrix> {
    super::kernel::fock_derivative(molecule, params, periodic, &scf.density, basis, atom, axis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;
    use crate::pbc::gradient::periodic_gradient;

    const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";

    fn celled(xyz: &str, edge: f64) -> Molecule {
        let mut molecule = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        molecule.cell = Some(Cell::cubic(edge).unwrap());
        molecule
    }

    /// The analytic Hessian against central differences of the **analytic gradient**.
    ///
    /// This is the only test that matters for a Hessian. It exercises the skeleton second
    /// derivatives, the Ewald second derivative, the Fock derivative that forms the CPHF
    /// right-hand side, and the CPHF solve itself, against a reference that shares none of that
    /// machinery — the gradient is a first derivative and is independently validated.
    #[test]
    fn matches_finite_differences_of_the_analytic_gradient() {
        let params = Pm3Parameters::standard().unwrap();
        let options = Pm3Options::default();
        let periodic = PeriodicOptions::default();
        let base = celled(WATER, 18.0);
        let analytic = periodic_hessian(&base, &params, &options, &periodic).unwrap();

        let step = 2.0e-4;
        for atom in 0..base.atoms.len() {
            for axis in 0..3 {
                let gradient_at = |shift: f64| -> Vec<f64> {
                    let mut moved = base.clone();
                    match axis {
                        0 => moved.atoms[atom].position.x += shift,
                        1 => moved.atoms[atom].position.y += shift,
                        _ => moved.atoms[atom].position.z += shift,
                    }
                    periodic_gradient(&moved, &params, &options, &periodic)
                        .unwrap()
                        .gradient
                        .iter()
                        .flat_map(|g| g.to_array())
                        .collect()
                };
                let plus = gradient_at(step);
                let minus = gradient_at(-step);
                let row = 3 * atom + axis;
                for (column, (p, m)) in plus.iter().zip(&minus).enumerate() {
                    let numerical = (p - m) / (2.0 * step);
                    let exact = analytic[(row, column)];
                    assert!(
                        (numerical - exact).abs() < 3.0e-3,
                        "({row},{column}): analytic {exact} vs finite difference {numerical}"
                    );
                }
            }
        }
    }

    /// A second derivative of a scalar is symmetric.
    #[test]
    fn the_hessian_is_symmetric() {
        let params = Pm3Parameters::standard().unwrap();
        let hessian = periodic_hessian(
            &celled(WATER, 18.0),
            &params,
            &Pm3Options::default(),
            &PeriodicOptions::default(),
        )
        .unwrap();
        for i in 0..hessian.rows {
            for j in 0..hessian.rows {
                let difference = (hessian[(i, j)] - hessian[(j, i)]).abs();
                assert!(
                    difference < 1.0e-9,
                    "({i},{j}) asymmetric by {difference:.3e}"
                );
            }
        }
    }

    /// The acoustic sum rule, *before* it is enforced.
    ///
    /// Every term is a function of interatomic displacements, so the rows already sum to zero.
    /// Checking that here rather than trusting [`enforce_acoustic_sum_rule`] is the point: if a
    /// term were being scattered to the wrong atom, enforcement would hide it.
    #[test]
    fn the_acoustic_sum_rule_holds_before_enforcement() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = celled(WATER, 18.0);
        let hessian = periodic_hessian(
            &molecule,
            &params,
            &Pm3Options::default(),
            &PeriodicOptions::default(),
        )
        .unwrap();
        let nat = molecule.atoms.len();
        for row in 0..hessian.rows {
            for beta in 0..3 {
                let total: f64 = (0..nat).map(|atom| hessian[(row, 3 * atom + beta)]).sum();
                assert!(
                    total.abs() < 1.0e-6,
                    "row {row}, axis {beta}: sums to {total:.3e}"
                );
            }
        }
    }

    /// A molecule alone in a large cell must give the molecular Hessian.
    ///
    /// The bridge to the MOPAC-validated molecular path: whatever the periodic machinery does, in
    /// the isolated limit it has to reduce to something already checked against an external
    /// oracle.
    #[test]
    fn a_molecule_in_a_large_cell_reproduces_the_molecular_hessian() {
        let params = Pm3Parameters::standard().unwrap();
        let options = Pm3Options::default();
        let molecule = celled(WATER, 30.0);
        let periodic_result =
            periodic_hessian(&molecule, &params, &options, &PeriodicOptions::default()).unwrap();
        let molecular = crate::hessian::analytic_hessian(
            &Molecule::from_xyz_str(WATER, 0.0).unwrap(),
            &params,
            &options,
            1.0e-3,
        )
        .unwrap();
        for i in 0..molecular.rows {
            for j in 0..molecular.rows {
                let difference = (periodic_result[(i, j)] - molecular[(i, j)]).abs();
                assert!(
                    difference < 2.0e-3,
                    "({i},{j}): periodic {} vs molecular {} ({difference:.3e})",
                    periodic_result[(i, j)],
                    molecular[(i, j)]
                );
            }
        }
    }

    /// Γ-point phonons of a relaxed cell: three acoustic modes at zero and the rest real.
    #[test]
    fn phonons_have_three_acoustic_modes() {
        use crate::pbc::optimize::{relax, PeriodicOptOptions};
        let params = Pm3Parameters::standard().unwrap();
        let options = Pm3Options::default();
        let periodic = PeriodicOptions::default();
        let relaxed = relax(
            &celled(WATER, 18.0),
            &params,
            &options,
            &periodic,
            &PeriodicOptOptions {
                max_iter: 80,
                gtol: 1.0e-5,
                ..PeriodicOptOptions::default()
            },
        )
        .unwrap();
        let phonons = periodic_phonons(&relaxed.molecule, &params, &options, &periodic).unwrap();
        assert_eq!(phonons.frequencies_cm.len(), 9);
        let mut magnitudes: Vec<f64> = phonons.frequencies_cm.iter().map(|f| f.abs()).collect();
        magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Three acoustic branches at zero. That is a statement about translational invariance and
        // holds in any crystal.
        assert!(
            magnitudes[2] < 5.0,
            "the three acoustic modes should vanish, largest is {:.3} cm⁻¹",
            magnitudes[2]
        );
        assert!(
            phonons.acoustic_residual_cm < 5.0,
            "the reported residual disagrees: {:.3} cm⁻¹",
            phonons.acoustic_residual_cm
        );

        // The *librations* are also soft here, and deliberately not asserted against. An 18 Bohr
        // cell of water is essentially a gas of non-interacting molecules, so rotating one costs
        // almost nothing: three more near-zero modes, which is right rather than wrong. In a real
        // molecular crystal they would be the librational branches at a few hundred cm⁻¹.
        //
        // What must be there regardless are the molecule's three internal modes — a bend and two
        // stretches — which no amount of cell weirdness moves below 1000 cm⁻¹.
        assert!(
            magnitudes[6] > 1000.0,
            "the three internal modes should survive, lowest is {:.3} cm⁻¹",
            magnitudes[6]
        );
        // No *stiff* mode is imaginary. The soft ones are excluded on purpose: a nearly-free
        // libration has a true frequency of zero, and whether the square root of a number that
        // small comes out at `+3` or `−9 cm⁻¹` is rounding in the residual gradient, not a
        // saddle. Asserting a sign there would be asserting noise.
        for frequency in &phonons.frequencies_cm {
            if frequency.abs() > 50.0 {
                assert!(
                    *frequency > 0.0,
                    "imaginary stiff mode at a minimum: {frequency:.3} cm⁻¹"
                );
            }
        }
    }

    /// Corrections contribute, and they contribute the lattice-summed amount.
    #[test]
    fn the_corrected_variant_changes_the_hessian() {
        let params = Pm3Parameters::standard().unwrap();
        let periodic = PeriodicOptions::default();
        let molecule = celled(WATER, 18.0);
        let plain =
            periodic_hessian(&molecule, &params, &Pm3Options::default(), &periodic).unwrap();
        let corrected = periodic_hessian(
            &molecule,
            &params,
            &Pm3Options {
                variant: Variant::Pm3D3H4X,
                ..Pm3Options::default()
            },
            &periodic,
        )
        .unwrap();
        let largest = (0..plain.rows)
            .flat_map(|i| (0..plain.rows).map(move |j| (i, j)))
            .fold(0.0_f64, |m, (i, j)| {
                m.max((plain[(i, j)] - corrected[(i, j)]).abs())
            });
        assert!(largest > 1.0e-6, "the correction Hessian did nothing");
    }
}
